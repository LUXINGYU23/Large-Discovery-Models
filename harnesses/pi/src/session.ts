import { createRequire } from "node:module";
import { basename, dirname, isAbsolute, join, posix, relative, resolve, sep } from "node:path";
import { copyFile, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import {
	createSyntheticSourceInfo,
	createAgentSession,
	DefaultResourceLoader,
	ModelRuntime,
	SessionManager,
	SettingsManager,
	type AgentSession,
	type ExtensionFactory,
	type Skill,
	type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { Type, type TUnsafe } from "typebox";
import { GUEST_RESOURCE_ROOT, GondolinController } from "./gondolin.js";
import { resolveGuestRuntime, type ResolvedGuestRuntime } from "./guest-image.js";
import { McpToolBridge } from "./mcp.js";
import { PolicyController, type ToolUsageSnapshot } from "./policy.js";
import { TurnExecutionError } from "./protocol.js";
import type {
	HarnessLimits,
	HarnessProfileConfig,
	InitializeFrame,
	SessionTurnInput,
	SubmissionError,
	SubmittedArtifact,
	SubmissionValidator,
} from "./protocol.js";
import { ProviderProxy, type ProviderTurnSummary } from "./provider-proxy.js";
import {
	snapshotSubmissionArtifacts,
	verifySubmissionRecord,
	type TerminalSubmission,
} from "./submission.js";
import { atomicJson, canonicalJson, canonicalSha256, sha256 } from "./trace.js";
import { createSolPiExtension, SOL_PI_GUEST_ARCHIVE, solPiTools } from "./sol-pi.js";

const APP_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const RESOURCE_DIRECTORY = ".ldm-resources";
const MODEL_CONTEXT_WINDOW = 262_144;
const COMPACTION_SETTINGS = {
	enabled: true,
	reserveTokens: 16_384,
	keepRecentTokens: 20_000,
};
const GUEST_RUNTIME_INSTRUCTIONS = `## Guest runtime
- /workspace is an isolated writable research workspace, not a checkout of the project repository. Do not use git status to inspect the project. Git is available only for cloning public research material when useful.
- Use the registered read and write tools for files. apply_patch is not available in the guest.
- Invoke task and MCP tools directly by their registered tool names. Do not run their implementation files; runtimes such as Node.js may not be installed.`;

interface SavedSubmission {
	submission: TerminalSubmission;
	toolUsage: ToolUsageSnapshot;
}

export interface CommittedTurn {
	profileId: string;
	sessionId: string;
	turnId: string;
	roundIndex: number;
	historyFromSeq: number;
	historyToSeq: number;
	historyDigest: string;
	inputDigest: string;
	replayed: boolean;
	submissionStatus: "accepted" | "rejected";
	submissionId: string;
	submissionJson: string;
	submissionDigest: string;
	submission: Record<string, unknown>;
	submittedArtifacts: SubmittedArtifact[];
	validationErrors: SubmissionError[];
	usage: {
		providerCalls: number;
		toolCalls: Record<string, number>;
		artifactBytes: number;
	};
	toolBudget: ToolUsageSnapshot["toolBudget"];
	artifacts: {
		turn: string;
		session: string | undefined;
	};
}

function isInside(root: string, value: string): boolean {
	const path = relative(resolve(root), resolve(value));
	return path === "" || (path !== ".." && !path.startsWith(`..${sep}`) && !isAbsolute(path));
}

function packageRoot(packageName: string): string {
	const require = createRequire(import.meta.url);
	return dirname(require.resolve(`${packageName}/package.json`));
}

function selectedSkills(skills: Skill[], directories: string[]): Skill[] {
	return skills.filter((skill) => directories.some((directory) => isInside(directory, skill.filePath)));
}

function guestResourcePath(hostResourceRoot: string, value: string): string {
	if (!isInside(hostResourceRoot, value)) throw new Error(`resource path escaped snapshot: ${value}`);
	const suffix = relative(hostResourceRoot, value).split(sep).join(posix.sep);
	return suffix ? posix.join(GUEST_RESOURCE_ROOT, suffix) : GUEST_RESOURCE_ROOT;
}

async function optionalJson<T>(path: string): Promise<T | undefined> {
	try {
		return JSON.parse(await readFile(path, "utf8")) as T;
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
		throw error;
	}
}

async function directorySha256(root: string): Promise<string> {
	const files: Array<{ path: string; sha256: string }> = [];
	async function visit(directory: string): Promise<void> {
		const entries = await readdir(directory, { withFileTypes: true });
		entries.sort((left, right) => Buffer.compare(
			Buffer.from(left.name),
			Buffer.from(right.name),
		));
		for (const entry of entries) {
			const path = join(directory, entry.name);
			if (entry.isDirectory()) await visit(path);
			else if (entry.isFile()) files.push({
				path: relative(root, path).replaceAll("\\", "/"),
				sha256: sha256(await readFile(path)),
			});
			else throw new Error(`unsupported skill resource entry: ${path}`);
		}
	}
	await visit(root);
	return canonicalSha256(files);
}

export async function pinInitialization(root: string, identity: unknown): Promise<string> {
	const digest = canonicalSha256(identity);
	const manifestPath = join(root, "manifest.json");
	const previous = await optionalJson<{ initializationSha256?: string }>(manifestPath);
	if (previous) {
		if (previous.initializationSha256 !== digest) {
			throw new Error("Harness initialization identity changed; restore the original configuration and runtime or use a new artifact root");
		}
	} else {
		try {
			if ((await readdir(join(root, "sessions"))).length) {
				throw new Error("Harness sessions exist without an initialization manifest");
			}
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
		}
		await atomicJson(manifestPath, { initializationSha256: digest });
	}
	return digest;
}

async function copyDirectory(source: string, target: string): Promise<void> {
	await mkdir(target, { recursive: true });
	const entries = await readdir(source, { withFileTypes: true });
	for (const entry of entries) {
		const sourcePath = join(source, entry.name);
		const targetPath = join(target, entry.name);
		if (entry.isDirectory()) await copyDirectory(sourcePath, targetPath);
		else if (entry.isFile()) await copyFile(sourcePath, targetPath);
		else throw new Error(`unsupported skill resource entry: ${sourcePath}`);
	}
}

async function runtimePackages(): Promise<Record<string, string>> {
	const lockBody = await readFile(join(APP_ROOT, "package-lock.json"));
	const lock = JSON.parse(lockBody.toString()) as {
		packages?: Record<string, { version?: unknown; resolved?: string }>;
	};
	function version(name: string): string {
		const value = lock.packages?.[`node_modules/${name}`]?.version;
		if (typeof value !== "string") throw new Error(`package version missing from lockfile: ${name}`);
		return value;
	}
	return {
		node: process.version,
		piCodingAgent: version("@earendil-works/pi-coding-agent"),
		gondolin: version("@earendil-works/gondolin"),
		piWebAccess: version("pi-web-access"),
		context7: version("@upstash/context7-pi"),
		mcpClient: version("@modelcontextprotocol/client"),
		solPi: version("sol-pi"),
		solPiSource: lock.packages?.["node_modules/sol-pi"]?.resolved ?? "",
		packageLockSha256: sha256(lockBody),
	};
}

function sessionTools(
	context7Enabled: boolean,
	taskTools: string[],
	terminalTool: string,
	mcpTools: string[] = [],
	solPi?: Record<string, unknown>,
): string[] {
	return [
		"read",
		"write",
		"bash",
		"web_search",
		"fetch_content",
		"get_search_content",
		...(context7Enabled ? ["resolve-library-id", "query-docs"] : []),
		...taskTools,
		...mcpTools,
		...solPiTools(solPi),
		terminalTool,
	];
}

function configuredMcpToolNames(servers: InitializeFrame["mcpServers"]): string[] {
	return servers.flatMap((server) =>
		server.tools.map((tool) => "mcp__" + server.serverId + "__" + tool),
	);
}

export class SubmissionController {
	private readonly parameters: TUnsafe<Record<string, unknown>>;
	private providerRequests = 0;
	private attemptIndex = 0;
	private profileId = "";
	private turnId = "";
	private turnRoot = "";
	private value: TerminalSubmission | undefined;
	private persist: ((submission: TerminalSubmission) => Promise<void>) | undefined;
	private validate: SubmissionValidator | undefined;

	constructor(
		private readonly config: InitializeFrame,
		private readonly workspace: string,
	) {
		this.parameters = Type.Unsafe<Record<string, unknown>>(
			config.submissionContract.payloadSchema,
		);
	}

	async begin(
		profileId: string,
		turnId: string,
		turnRoot: string,
		persist: (submission: TerminalSubmission) => Promise<void>,
		validate: SubmissionValidator,
	): Promise<void> {
		this.profileId = profileId;
		this.turnId = turnId;
		this.turnRoot = turnRoot;
		this.providerRequests = 0;
		this.attemptIndex = 0;
		try {
			for (const name of await readdir(join(turnRoot, "attempts"))) {
				if (/^\d+$/.test(name)) this.attemptIndex = Math.max(this.attemptIndex, Number(name));
			}
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
		}
		this.value = undefined;
		this.persist = persist;
		this.validate = validate;
	}

	get submission(): TerminalSubmission | undefined {
		return this.value;
	}

	createExtension(): ExtensionFactory {
		return (pi) => {
			pi.on("before_provider_request", (event) => {
				const payload = event.payload;
				if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
					throw new Error("provider payload must be a JSON object");
				}
				this.providerRequests += 1;
				if (this.value) return payload;
				if (this.providerRequests > 1) return payload;
				return {
					...payload,
					tool_choice: "required",
				};
			});
		};
	}

	tool(): ToolDefinition<TUnsafe<Record<string, unknown>>> {
		const toolName = this.config.submissionContract.toolName;
		return {
			name: toolName,
			label: "Submit result",
			description: "Submit the complete structured result. Validation errors must be repaired before resubmission.",
			promptSnippet: `${toolName}: submit the complete structured result`,
			parameters: this.parameters,
			executionMode: "sequential",
			execute: async (_toolCallId, params) => {
				if (this.value) throw new Error(`${toolName} may be called only once per turn`);
				this.attemptIndex += 1;
				if (!this.validate) throw new Error("submission validator is not initialized");
				const submission = { ...params };
				const attemptRoot = join(
					this.turnRoot,
					"attempts",
					this.attemptIndex.toString().padStart(4, "0"),
				);
				let artifacts: SubmittedArtifact[];
				try {
					artifacts = await snapshotSubmissionArtifacts(
						this.config.submissionContract.artifactRules,
						this.workspace,
						this.config.artifactRoot,
						this.turnRoot,
						this.attemptIndex,
						submission,
					);
				} catch (error) {
					await rm(attemptRoot, { recursive: true, force: true });
					throw error;
				}
				const submissionJson = canonicalJson({ artifacts, submission });
				const submissionDigest = sha256(submissionJson);
				const decision = await this.validate({
					profileId: this.profileId,
					turnId: this.turnId,
					attemptIndex: this.attemptIndex,
					submission,
					artifacts,
					submissionJson,
					submissionDigest,
				});
				if ((decision.decision === "accept") !== (decision.errors.length === 0)) {
					throw new Error("submission validator returned an inconsistent decision");
				}
				await atomicJson(join(attemptRoot, "validation.json"), {
					attemptIndex: this.attemptIndex,
					decision: decision.decision,
					errors: decision.errors,
					submission,
					submissionJson,
					submissionDigest,
					artifacts,
				});
				if (decision.decision === "retry") {
					throw new Error(
						"Submission rejected by the task validator. Repair the reported paths and resubmit. "
						+ `Validation report: ${JSON.stringify(decision)}`,
					);
				}
				const terminal: TerminalSubmission = {
					submissionStatus: decision.decision === "accept" ? "accepted" : "rejected",
					submissionId: `${this.turnId}-submission-${this.attemptIndex}`,
					submissionDigest,
					submission,
					submittedArtifacts: artifacts,
					validationErrors: decision.errors,
					submissionJson,
				};
				if (!this.persist) throw new Error("submission turn is not initialized");
				await this.persist(terminal);
				this.value = terminal;
				return {
					content: [{
						type: "text",
						text: decision.decision === "accept"
							? "Submission accepted. End this turn now."
							: "Submission rejected and the turn is closed.",
					}],
					details: {
						submissionId: terminal.submissionId,
						status: terminal.submissionStatus,
					},
				};
			},
		};
	}

}

export class PersistentProfileSession {
	private readonly profileRoot: string;
	private readonly workspace: string;
	private readonly resourceRoot: string;
	private readonly sessionDirectory: string;
	private readonly policy: PolicyController;
	private readonly gondolin: GondolinController;
	private readonly mcp: McpToolBridge;
	private readonly submissions: SubmissionController;
	private session: AgentSession | undefined;
	private historyCursor = 0;
	private agentsSha256 = "";
	private skillDirSha256: string[] = [];

	constructor(
		private readonly profile: HarnessProfileConfig,
		private readonly config: InitializeFrame,
		private readonly proxy: ProviderProxy,
		guestRuntime: ResolvedGuestRuntime,
		namedSecrets: Readonly<Record<string, string>>,
	) {
		this.profileRoot = join(config.artifactRoot, "sessions", profile.profileId);
		this.workspace = join(this.profileRoot, "workspace");
		this.resourceRoot = join(this.workspace, RESOURCE_DIRECTORY);
		this.sessionDirectory = join(this.profileRoot, "pi-session");
		this.policy = new PolicyController(
			config.networkPolicy,
			config.webSearch.providers,
			config.limits.toolCallBudgets,
		);
		this.gondolin = new GondolinController(
			this.workspace,
			this.resourceRoot,
			config.networkPolicy,
			guestRuntime,
			config.solPi ? { [SOL_PI_GUEST_ARCHIVE]: join(this.sessionDirectory, "sol-pi") } : {},
		);
		this.mcp = new McpToolBridge(
			config.mcpServers,
			namedSecrets,
			`ldm-pi-${config.campaignId}-${profile.profileId}`,
		);
		this.submissions = new SubmissionController(config, this.workspace);
	}

	async initialize(): Promise<void> {
		await mkdir(this.workspace, { recursive: true });
		await mkdir(this.sessionDirectory, { recursive: true });
		if (this.config.solPi) await mkdir(join(this.sessionDirectory, "sol-pi"), { recursive: true });
		const agents = await readFile(this.profile.agentsPath, "utf8");
		this.agentsSha256 = sha256(agents);
		if (this.agentsSha256 !== this.profile.agentsSha256) {
			throw new Error(`AGENTS.md digest mismatch for profile ${this.profile.profileId}`);
		}
		this.skillDirSha256 = await Promise.all(this.profile.skillDirs.map(directorySha256));
		if (this.skillDirSha256.some((value, index) => value !== this.profile.skillDirSha256[index])) {
			throw new Error(`skill directory digest mismatch for profile ${this.profile.profileId}`);
		}
		const skillDirectories = await this.snapshotResources(agents);
		for (const extension of this.config.toolExtensions) {
			if (sha256(await readFile(extension.path)) !== extension.sha256) {
				throw new Error(`tool extension digest mismatch: ${extension.path}`);
			}
		}
		await this.mcp.initialize();
		const providerId = `ldm-harness-${this.profile.profileId}`;
		const agentDirectory = join(this.profileRoot, "pi-agent");
		await mkdir(agentDirectory, { recursive: true });
		const modelsPath = join(agentDirectory, "models.json");
		await atomicJson(modelsPath, {
			providers: {
				[providerId]: {
					baseUrl: this.proxy.baseUrl(this.profile.profileId),
					api: "openai-responses",
					apiKey: "sidecar-proxy-token",
					models: [{
						id: this.config.model,
						name: this.config.model,
						reasoning: true,
						compat: { supportsStrictMode: true },
						thinkingLevelMap: {
							off: "none",
							minimal: "minimal",
							low: "low",
							medium: "medium",
							high: "high",
							xhigh: "xhigh",
							max: "max",
						},
						input: ["text"],
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
						contextWindow: MODEL_CONTEXT_WINDOW,
						maxTokens: 32768,
					}],
				},
			},
		});

		const authPath = join(agentDirectory, "auth.json");
		let modelRuntime: ModelRuntime;
		try {
			modelRuntime = await ModelRuntime.create({
				authPath,
				modelsPath,
				refreshOnCreate: false,
			});
			await modelRuntime.setRuntimeApiKey(providerId, "sidecar-proxy-token");
		} finally {
			await rm(authPath, { force: true });
		}
		const model = modelRuntime.getModel(providerId, this.config.model);
		if (!model) throw new Error(`Pi model registration failed for ${providerId}/${this.config.model}`);

		const settings = SettingsManager.inMemory({
			compaction: COMPACTION_SETTINGS,
			retry: { enabled: false },
		});
		const webExtension = join(packageRoot("pi-web-access"), "index.ts");
		const extensionPaths = [webExtension];
		if (this.config.context7Enabled) {
			extensionPaths.push(join(packageRoot("@upstash/context7-pi"), "extensions", "context7.ts"));
		}
		extensionPaths.push(...this.config.toolExtensions.map((extension) => extension.path));
		const solPi = this.config.solPi ? await createSolPiExtension(
			this.config.solPi, agentDirectory, providerId, this.config.model,
			() => this.gondolin.toolOptions(), join(this.sessionDirectory, "sol-pi"),
		) : undefined;
		const loader = new DefaultResourceLoader({
			cwd: this.workspace,
			agentDir: agentDirectory,
			settingsManager: settings,
			additionalExtensionPaths: extensionPaths,
			additionalSkillPaths: skillDirectories,
			extensionFactories: [
				this.gondolin.createExtension(),
				this.policy.createExtension(),
				this.submissions.createExtension(),
				...(solPi ? [solPi] : []),
			],
			noPromptTemplates: true,
			noThemes: true,
			noContextFiles: true,
			skillsOverride: ({ skills, diagnostics }) => ({
				skills: selectedSkills(skills, skillDirectories).map((skill) => {
					const filePath = guestResourcePath(this.resourceRoot, skill.filePath);
					const baseDir = posix.dirname(filePath);
					return {
						...skill,
						filePath,
						baseDir,
						sourceInfo: createSyntheticSourceInfo(filePath, {
							source: "task-local",
							scope: "project",
							baseDir,
						}),
					};
				}),
				diagnostics,
			}),
			agentsFilesOverride: () => ({
				agentsFiles: [{ path: posix.join(GUEST_RESOURCE_ROOT, "AGENTS.md"), content: agents }],
			}),
			appendSystemPromptOverride: () => [GUEST_RUNTIME_INSTRUCTIONS],
		});
		await loader.reload();
		const extensionErrors = loader.getExtensions().errors;
		if (extensionErrors.length > 0) {
			throw new Error(`Pi extension load failed: ${extensionErrors.map((item) => item.error).join("; ")}`);
		}

		const manager = SessionManager.continueRecent(this.workspace, this.sessionDirectory);
		const { session } = await createAgentSession({
			cwd: this.workspace,
			agentDir: agentDirectory,
			modelRuntime,
			model,
			thinkingLevel: this.config.thinking,
			resourceLoader: loader,
			settingsManager: settings,
			sessionManager: manager,
			customTools: [...this.mcp.toolDefinitions(), this.submissions.tool()],
			tools: sessionTools(
				this.config.context7Enabled,
				this.config.toolExtensions.flatMap((extension) => extension.toolNames),
				this.config.submissionContract.toolName,
				configuredMcpToolNames(this.config.mcpServers),
				this.config.solPi,
			),
		});
		this.session = session;
		this.historyCursor = await this.recoverHistoryCursor();
	}

	private async snapshotResources(agents: string): Promise<string[]> {
		// Keep task-owned data files referenced by the persistent session.
		await rm(join(this.resourceRoot, "skills"), { recursive: true, force: true });
		await mkdir(this.resourceRoot, { recursive: true });
		const agentsPath = join(this.resourceRoot, "AGENTS.md");
		await writeFile(agentsPath, agents, "utf8");
		const skillDirectories: string[] = [];
		for (const [index, configuredDirectory] of this.profile.skillDirs.entries()) {
			const source = resolve(configuredDirectory);
			const target = join(this.resourceRoot, "skills", String(index), basename(source));
			await copyDirectory(source, target);
			if (await directorySha256(target) !== this.skillDirSha256[index]) {
				throw new Error(`skill snapshot digest mismatch for profile ${this.profile.profileId}`);
			}
			skillDirectories.push(target);
		}
		return skillDirectories;
	}

	async runTurn(input: SessionTurnInput, validate: SubmissionValidator): Promise<CommittedTurn> {
		if (!this.session) throw new Error("profile session is not initialized");
		const turnRoot = join(this.profileRoot, "turns", input.turnId);
		const commitPath = join(this.config.artifactRoot, "turns", input.turnId, "turn_committed.json");
		const priorCommit = await optionalJson<CommittedTurn>(commitPath);
		if (priorCommit) {
			if (
				priorCommit.inputDigest !== input.inputDigest
				|| priorCommit.profileId !== input.profileId
				|| priorCommit.historyFromSeq !== input.historyFromSeq
				|| priorCommit.historyToSeq !== input.historyToSeq
				|| priorCommit.historyDigest !== input.historyDigest
			) {
				throw new Error(`committed turn digest mismatch: ${input.turnId}`);
			}
			await verifySubmissionRecord(priorCommit, this.config.artifactRoot);
			this.acceptCursor(input, true);
			this.historyCursor = input.historyToSeq;
			return { ...priorCommit, replayed: true };
		}
		this.acceptCursor(input, false);

		const inputPath = join(turnRoot, "input.json");
		const priorInput = await optionalJson<SessionTurnInput>(inputPath);
		if (priorInput && priorInput.inputDigest !== input.inputDigest) {
			throw new Error(`partial turn digest mismatch: ${input.turnId}`);
		}
		if (!priorInput) await atomicJson(inputPath, input);

		const submissionPath = join(turnRoot, "submission.json");
		const savedSubmission = await optionalJson<SavedSubmission>(submissionPath);
		if (savedSubmission) {
			const provider = await this.proxy.recoveredTurnSummary(turnRoot, input.turnId);
			return this.commit(input, savedSubmission.submission, provider, savedSubmission.toolUsage, true);
		}
		await this.policy.begin(input.forbiddenQueryTerms, join(turnRoot, "tool-budget.json"));

		await this.submissions.begin(
			this.profile.profileId,
			input.turnId,
			turnRoot,
			(value) => atomicJson(submissionPath, {
				submission: value,
				toolUsage: this.policy.snapshot(),
			} satisfies SavedSubmission),
			validate,
		);
		await this.proxy.beginTurn(
			this.profile.profileId,
			this.session.sessionManager.getSessionId(),
			input.turnId,
			turnRoot,
		);
		let submission: TerminalSubmission | undefined;
		let providerSummary: ProviderTurnSummary;
		let policySummary: ToolUsageSnapshot;
		let failure: Error | undefined;
		try {
			await this.promptWithTimeout(
				`${priorInput
					? `Continue the interrupted turn from your existing research and files. The history and submission contract are unchanged. Your previous submission was not accepted; repair any reported errors and call ${this.config.submissionContract.toolName} with the complete result. Tool budgets have not reset.`
					: input.message}\n\n${this.policy.budgetMessage()}`,
				this.config.limits,
			);
			submission = this.submissions.submission;
			if (!submission) {
				const lastMessage = this.session.messages.at(-1);
				if (lastMessage?.role === "assistant" && lastMessage.stopReason === "error") {
					throw new Error(`provider response failed: ${lastMessage.errorMessage ?? "unknown provider error"}`);
				}
				throw new Error(
					`session ${this.profile.profileId} ended without ${this.config.submissionContract.toolName}`,
				);
			}
		} catch (error) {
			failure = error instanceof Error ? error : new Error(String(error));
		} finally {
			providerSummary = await this.proxy.endTurn(this.profile.profileId);
			policySummary = this.policy.end();
		}
		if (failure) {
			throw new TurnExecutionError(failure.message, [{
				profileId: input.profileId,
				turnId: input.turnId,
				usage: {
					providerCalls: providerSummary.providerCalls,
					toolCalls: policySummary.toolCalls,
					artifactBytes: providerSummary.artifactBytes,
				},
			}], /session wall-time limit reached|context_length_exceeded|stream_read_error|stream ended before a terminal response event|^provider response failed: terminated$|\b(?:server_error|internal_server_error|overloaded_error|rate_limit_exceeded|request_timeout|408|429|500|502|503|504)\b|ECONNRESET|ETIMEDOUT|fetch failed/i.test(failure.message));
		}
		if (!submission) throw new Error("turn ended without a submission");
		return this.commit(input, submission, providerSummary, policySummary);
	}

	manifestEntry(): Record<string, unknown> {
		if (!this.session) throw new Error("profile session is not initialized");
		const sessionFile = this.session.sessionManager.getSessionFile();
		return {
			profileId: this.profile.profileId,
			agentsSha256: this.agentsSha256,
			skills: this.skillDirSha256.map((value, index) => ({ directoryIndex: index, sha256: value })),
			sessionId: this.session.sessionManager.getSessionId(),
			session: sessionFile ? relative(this.config.artifactRoot, sessionFile) : undefined,
			workspace: relative(this.config.artifactRoot, this.workspace),
			environmentSnapshot: relative(
				this.config.artifactRoot,
				join(this.profileRoot, "environment_snapshot.json"),
			),
		};
	}

	mcpManifest(): Array<Record<string, unknown>> {
		return this.mcp.manifest().map((server) => ({ profileId: this.profile.profileId, ...server }));
	}

	async close(): Promise<void> {
		try {
			if (this.session) {
				this.session.abortCompaction();
				await this.session.abort();
				this.session.dispose();
				this.session = undefined;
			}
		} finally {
			try {
				const snapshot = await this.gondolin.environmentSnapshot();
				await atomicJson(join(this.profileRoot, "environment_snapshot.json"), snapshot ?? {
					status: "guest_not_started",
				});
			} catch (error) {
				await atomicJson(join(this.profileRoot, "environment_snapshot.json"), {
					error: (error as Error).message,
				});
			} finally {
				try {
					await this.gondolin.close();
				} finally {
					await this.mcp.close();
				}
			}
		}
	}

	private async promptWithTimeout(message: string, limits: HarnessLimits): Promise<void> {
		if (!this.session) throw new Error("profile session is not initialized");
		const session = this.session;
		const options = { expandPromptTemplates: false, source: "rpc" as const };
		let expired = false;
		const run = async () => {
			await session.prompt(message, options);
			while (!expired && !this.submissions.submission) {
				const lastMessage = session.messages.at(-1);
				const errorMessage = lastMessage?.role === "assistant"
					? lastMessage.errorMessage?.toLowerCase()
					: undefined;
				const interrupted = (
					lastMessage?.role === "assistant"
					&& lastMessage.stopReason === "error"
					&& errorMessage
					&& (
						errorMessage.includes("stream_read_error")
						|| errorMessage.includes("stream ended before a terminal response event")
					)
				);
				if (lastMessage?.role === "assistant" && lastMessage.stopReason === "error" && !interrupted) {
					throw new Error(`provider response failed: ${lastMessage.errorMessage ?? "unknown provider error"}`);
				}
				await session.prompt(
					interrupted
						? "The previous provider stream ended before your submission was accepted. "
							+ `Continue from your existing work, use tools to complete or repair it, and call ${this.config.submissionContract.toolName} when ready.`
						: `No submission has been accepted. Address any validation errors, use the available tools as needed, and call ${this.config.submissionContract.toolName} with the complete result.`,
					options,
				);
			}
		};
		let timer: NodeJS.Timeout | undefined;
		const timeout = new Promise<never>((_resolve, reject) => {
			timer = setTimeout(() => {
				expired = true;
				reject(new Error(`session wall-time limit reached: ${limits.wallTimeSeconds}s`));
			}, limits.wallTimeSeconds * 1000);
		});
		const pending = run();
		try {
			await Promise.race([pending, timeout]);
		} catch (error) {
			if (expired) {
				session.abortCompaction();
				await session.abort();
				await pending.catch(() => undefined);
			}
			throw error;
		} finally {
			if (timer) clearTimeout(timer);
		}
	}

	private async commit(
		input: SessionTurnInput,
		submission: TerminalSubmission,
		provider: ProviderTurnSummary,
		toolUsage: ToolUsageSnapshot,
		replayed = false,
	): Promise<CommittedTurn> {
		if (!this.session) throw new Error("profile session is not initialized");
		await verifySubmissionRecord(submission, this.config.artifactRoot);
		const sessionFile = this.session.sessionManager.getSessionFile();
		const commit: CommittedTurn = {
			profileId: this.profile.profileId,
			sessionId: this.session.sessionManager.getSessionId(),
			turnId: input.turnId,
			roundIndex: input.roundIndex,
			historyFromSeq: input.historyFromSeq,
			historyToSeq: input.historyToSeq,
			historyDigest: input.historyDigest,
			inputDigest: input.inputDigest,
			replayed,
			...submission,
			usage: {
				providerCalls: provider.providerCalls,
				toolCalls: toolUsage.toolCalls,
				artifactBytes: provider.artifactBytes,
			},
			toolBudget: toolUsage.toolBudget,
			artifacts: {
				turn: relative(this.config.artifactRoot, join(this.profileRoot, "turns", input.turnId)),
				session: sessionFile ? relative(this.config.artifactRoot, sessionFile) : undefined,
			},
		};
		await atomicJson(join(this.config.artifactRoot, "turns", input.turnId, "turn_committed.json"), commit);
		this.historyCursor = input.historyToSeq;
		return commit;
	}

	private acceptCursor(input: SessionTurnInput, replay: boolean): void {
		if (input.historyFromSeq === this.historyCursor) return;
		if (replay && input.historyToSeq === this.historyCursor) return;
		throw new Error(
			`history cursor mismatch for ${this.profile.profileId}: expected ${this.historyCursor}, received ${input.historyFromSeq}`,
		);
	}

	private async recoverHistoryCursor(): Promise<number> {
		const turnRoot = join(this.config.artifactRoot, "turns");
		let names: string[];
		try {
			names = await readdir(turnRoot);
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT") return 0;
			throw error;
		}
		const commits = (
			await Promise.all(names.map((name) => optionalJson<CommittedTurn>(join(turnRoot, name, "turn_committed.json"))))
		)
			.filter((value): value is CommittedTurn => value?.profileId === this.profile.profileId)
			.sort((left, right) => left.roundIndex - right.roundIndex);
		let cursor = 0;
		for (const commit of commits) {
			await verifySubmissionRecord(commit, this.config.artifactRoot);
			if (commit.historyFromSeq !== cursor || commit.historyToSeq < cursor) {
				throw new Error(`invalid committed history chain for profile ${this.profile.profileId}`);
			}
			cursor = commit.historyToSeq;
		}
		return cursor;
	}
}

export class PiSessionPool {
	private readonly sessions = new Map<string, PersistentProfileSession>();
	private readonly proxy: ProviderProxy;

	constructor(
		private readonly config: InitializeFrame,
		apiKey: string,
		private readonly namedSecrets: Readonly<Record<string, string>> = {},
	) {
		this.proxy = new ProviderProxy(config.baseUrl, apiKey, config.campaignId, config.providerRequestBody);
	}

	async initialize(): Promise<void> {
		await mkdir(this.config.artifactRoot, { recursive: true });
		const guestRuntime = await resolveGuestRuntime(this.config.taskId, this.config.guestRuntime);
		const packages = await runtimePackages();
		const { requestId: _requestId, artifactRoot: _artifactRoot, ...configuration } = this.config;
		const { assetDir: _assetDir, ...guestIdentity } = guestRuntime;
		const initializationSha256 = await pinInitialization(this.config.artifactRoot, {
			configuration, guestRuntime: guestIdentity, packages,
			implementationSha256: await directorySha256(join(APP_ROOT, "dist")),
			policyRunnerSha256: sha256(await readFile(join(APP_ROOT, "policy_runner.py"))),
		});
		const profileSetSha256 = canonicalSha256(this.config.profiles.map((profile) => ({
			agentsSha256: profile.agentsSha256,
			profileId: profile.profileId,
			skillDirSha256: profile.skillDirSha256,
		})));
		if (profileSetSha256 !== this.config.profileSetSha256) {
			throw new Error("profile set digest mismatch");
		}
		process.env.PI_CODING_AGENT_DIR = join(this.config.artifactRoot, "web-cache");
		await mkdir(process.env.PI_CODING_AGENT_DIR, { recursive: true });
		await writeFile(
			join(process.env.PI_CODING_AGENT_DIR, "web-search.json"),
			`${JSON.stringify({
				searchRouting: this.config.webSearch,
				fetchContent: { domainPolicy: { allow: this.config.networkPolicy.allowedHosts, deny: this.config.networkPolicy.deniedHosts } },
				fetchRouting: { providers: ["http"], allowRemoteHostedProviders: false },
				githubClone: { enabled: false },
				githubPrIssue: { enabled: false },
			}, null, 2)}\n`,
			{ encoding: "utf8", mode: 0o600 },
		);
		await this.proxy.start();
		for (const profile of this.config.profiles) {
			const session = new PersistentProfileSession(
				profile,
				this.config,
				this.proxy,
				guestRuntime,
				this.namedSecrets,
			);
			this.sessions.set(profile.profileId, session);
		}
		await Promise.all([...this.sessions.values()].map((session) => session.initialize()));
		await atomicJson(join(this.config.artifactRoot, "manifest.json"), {
			initializationSha256,
			protocolVersion: this.config.protocolVersion,
			campaignId: this.config.campaignId,
			taskId: this.config.taskId,
			caseId: this.config.caseId,
			seed: this.config.seed,
			backend: "pi",
			baseUrl: this.config.baseUrl,
			model: this.config.model,
			wireApi: this.config.wireApi,
			thinking: this.config.thinking,
			...(this.config.providerRequestBody ? { providerRequestBody: this.config.providerRequestBody } : {}),
			...(this.config.solPi ? { solPi: this.config.solPi } : {}),
			contextWindow: MODEL_CONTEXT_WINDOW,
			compaction: COMPACTION_SETTINGS,
			submissionContractSha256: this.config.submissionContractSha256,
			submissionContract: {
				contractId: this.config.submissionContract.contractId,
				toolName: this.config.submissionContract.toolName,
				artifactRules: this.config.submissionContract.artifactRules,
				maxValidationAttempts: this.config.submissionContract.maxValidationAttempts,
			},
			guestRuntime: {
				imageRef: guestRuntime.imageRef,
				recipeSha256: guestRuntime.recipeSha256,
				buildId: guestRuntime.buildId,
				manifestSha256: guestRuntime.manifestSha256,
				architecture: guestRuntime.architecture,
				rootfsSize: guestRuntime.rootfsSize,
				installPolicy: guestRuntime.installPolicy,
			},
			profileSetSha256,
			networkPolicySha256: canonicalSha256(this.config.networkPolicy),
			networkPolicy: this.config.networkPolicy,
			limits: this.config.limits,
			webSearch: this.config.webSearch,
			context7Enabled: this.config.context7Enabled,
			tools: sessionTools(
				this.config.context7Enabled,
				this.config.toolExtensions.flatMap((extension) => extension.toolNames),
				this.config.submissionContract.toolName,
				configuredMcpToolNames(this.config.mcpServers),
				this.config.solPi,
			),
			toolExtensions: this.config.toolExtensions,
			mcpServers: [...this.sessions.values()].flatMap((session) => session.mcpManifest()),
			topology: {
				profileCount: this.config.profiles.length,
			},
			packages,
			profiles: [...this.sessions.values()].map((session) => session.manifestEntry()),
		});
	}

	async runTurns(inputs: SessionTurnInput[], validate: SubmissionValidator): Promise<CommittedTurn[]> {
		const expected = new Set(this.sessions.keys());
		if (inputs.length !== expected.size || new Set(inputs.map((input) => input.profileId)).size !== inputs.length) {
			throw new Error("run_turn must contain exactly one input for every profile");
		}
		for (const input of inputs) {
			if (!expected.delete(input.profileId)) throw new Error(`unknown or duplicate profile: ${input.profileId}`);
		}
		const results = await Promise.allSettled(inputs.map(
			(input) => this.sessions.get(input.profileId)?.runTurn(input, validate) as Promise<CommittedTurn>,
		));
		const failed = results.find((result) => result.status === "rejected"
			&& !(result.reason instanceof TurnExecutionError && result.reason.retryable))
			?? results.find((result) => result.status === "rejected");
		if (failed?.status === "rejected") {
			throw new TurnExecutionError(String(failed.reason), results.flatMap((result) => {
				if (result.status === "rejected") {
					return result.reason instanceof TurnExecutionError ? result.reason.turnUsage : [];
				}
				return result.value.replayed ? [] : [{
					profileId: result.value.profileId,
					turnId: result.value.turnId,
					usage: result.value.usage,
				}];
			}), results.every((result) => result.status === "fulfilled"
				|| (result.reason instanceof TurnExecutionError && result.reason.retryable)));
		}
		return results.map((result) => (result as PromiseFulfilledResult<CommittedTurn>).value);
	}

	async close(): Promise<void> {
		const proxyClose = this.proxy.close();
		await Promise.allSettled([...this.sessions.values()].map((session) => session.close()));
		this.sessions.clear();
		await proxyClose;
	}
}
