import { createRequire } from "node:module";
import { dirname, join, relative, resolve } from "node:path";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import {
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
import { Type } from "typebox";
import { GondolinController } from "./gondolin.js";
import { PolicyController } from "./policy.js";
import type {
	HarnessLimits,
	HarnessProfileConfig,
	InitializeFrame,
	SessionTurnInput,
	SubmissionValidator,
} from "./protocol.js";
import { ProviderProxy, type ProviderTurnSummary } from "./provider-proxy.js";
import { atomicJson, canonicalSha256, sha256 } from "./trace.js";

const APP_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL_CONTEXT_WINDOW = 262_144;
const COMPACTION_SETTINGS = {
	enabled: true,
	reserveTokens: 16_384,
	keepRecentTokens: 20_000,
};

interface CandidateSubmission {
	submissionId: string;
	candidates: Array<Record<string, unknown>>;
}

const submissionSchema = Type.Object({
	candidates: Type.Array(Type.Record(Type.String(), Type.Unknown())),
}, { additionalProperties: false });

interface SavedSubmission {
	submission: CandidateSubmission;
	tools: { webCalls: number; context7Calls: number };
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
	submission: CandidateSubmission;
	usage: {
		providerCalls: number;
		webCalls: number;
		context7Calls: number;
		artifactBytes: number;
	};
	artifacts: {
		turn: string;
		session: string | undefined;
	};
}

function isInside(root: string, value: string): boolean {
	const path = relative(resolve(root), resolve(value));
	return path === "" || (!path.startsWith("..") && !path.startsWith("/"));
}

function packageRoot(packageName: string): string {
	const require = createRequire(import.meta.url);
	return dirname(require.resolve(`${packageName}/package.json`));
}

function selectedSkills(skills: Skill[], directories: string[]): Skill[] {
	return skills.filter((skill) => directories.some((directory) => isInside(directory, skill.filePath)));
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
		for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
			const path = join(directory, entry.name);
			if (entry.isDirectory()) await visit(path);
			else if (entry.isFile()) files.push({
				path: relative(root, path).replaceAll("\\", "/"),
				sha256: sha256(await readFile(path)),
			});
		}
	}
	await visit(root);
	return canonicalSha256(files);
}

async function runtimePackages(): Promise<Record<string, string>> {
	const lockBody = await readFile(join(APP_ROOT, "package-lock.json"));
	const lock = JSON.parse(lockBody.toString()) as {
		packages?: Record<string, { version?: unknown }>;
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
		packageLockSha256: sha256(lockBody),
	};
}

function sessionTools(context7Enabled: boolean, taskTools: string[]): string[] {
	return [
		"read",
		"write",
		"bash",
		"web_search",
		"fetch_content",
		"get_search_content",
		...(context7Enabled ? ["resolve-library-id", "query-docs"] : []),
		...taskTools,
		"submit_candidates",
	];
}

class SubmissionController {
	private expected = 0;
	private providerRequests = 0;
	private attemptIndex = 0;
	private submissionRequired = false;
	private profileId = "";
	private turnId = "";
	private value: CandidateSubmission | undefined;
	private persist: ((submission: CandidateSubmission) => Promise<void>) | undefined;
	private validate: SubmissionValidator | undefined;

	begin(
		profileId: string,
		turnId: string,
		expected: number,
		persist: (submission: CandidateSubmission) => Promise<void>,
		validate: SubmissionValidator,
		recovering: boolean,
	): void {
		this.profileId = profileId;
		this.turnId = turnId;
		this.expected = expected;
		this.providerRequests = 0;
		this.attemptIndex = 0;
		this.submissionRequired = recovering;
		this.value = undefined;
		this.persist = persist;
		this.validate = validate;
	}

	get submission(): CandidateSubmission | undefined {
		return this.value;
	}

	requireSubmission(): void {
		this.submissionRequired = true;
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
				if (this.submissionRequired) {
					return {
						...payload,
						tool_choice: { type: "function", name: "submit_candidates" },
					};
				}
				if (this.providerRequests > 1) return payload;
				return {
					...payload,
					tool_choice: "required",
				};
			});
		};
	}

	tool(): ToolDefinition<typeof submissionSchema> {
		return {
			name: "submit_candidates",
			label: "Submit candidates",
			description: "Submit the complete ordered candidate minibatch. Rejected entries must be replaced and resubmitted.",
			promptSnippet: "submit_candidates: submit the complete ordered candidate minibatch",
			parameters: submissionSchema,
			executionMode: "sequential",
			execute: async (_toolCallId, params) => {
				if (this.value) throw new Error("submit_candidates may be called only once per turn");
				this.submissionRequired = true;
				this.attemptIndex += 1;
				if (params.candidates.length !== this.expected) {
					throw new Error(
						`Candidate minibatch rejected: submit_candidates requires exactly ${this.expected} candidates; `
						+ `received ${params.candidates.length}. Add replacements and resubmit the complete minibatch.`,
					);
				}
				const invalidIndex = params.candidates.findIndex(
					(candidate) => !candidate || typeof candidate !== "object" || Array.isArray(candidate),
				);
				if (invalidIndex >= 0) {
					throw new Error(`Candidate minibatch rejected: candidate at index ${invalidIndex} must be an object.`);
				}
				if (!this.validate) throw new Error("submission validator is not initialized");
				const decision = await this.validate({
					profileId: this.profileId,
					turnId: this.turnId,
					attemptIndex: this.attemptIndex,
					candidates: params.candidates,
				});
				if (decision.accepted !== (decision.rejected.length === 0)) {
					throw new Error("submission validator returned an inconsistent decision");
				}
				if (!decision.accepted) {
					throw new Error(
						"Candidate minibatch rejected by the task validator. Replace every rejected entry, "
						+ `preserve the other entries, and resubmit the complete ${this.expected}-candidate minibatch. `
						+ `Rejection report: ${JSON.stringify({
							accepted: false,
							rejected: decision.rejected,
							required_replacements: decision.rejected.length,
						})}`,
					);
				}
				const submission: CandidateSubmission = {
					submissionId: `${this.turnId}-submission`,
					candidates: params.candidates,
				};
				if (!this.persist) throw new Error("submission turn is not initialized");
				await this.persist(submission);
				this.value = submission;
				this.submissionRequired = false;
				return {
					content: [{ type: "text", text: "Candidate minibatch accepted. End this turn now." }],
					details: { submissionId: this.value.submissionId, count: params.candidates.length },
				};
			},
		};
	}
}

class PersistentProfileSession {
	private readonly profileRoot: string;
	private readonly workspace: string;
	private readonly sessionDirectory: string;
	private readonly policy: PolicyController;
	private readonly gondolin: GondolinController;
	private readonly submissions = new SubmissionController();
	private session: AgentSession | undefined;
	private historyCursor = 0;
	private agentsSha256 = "";
	private skillDirSha256: string[] = [];

	constructor(
		private readonly profile: HarnessProfileConfig,
		private readonly config: InitializeFrame,
		private readonly proxy: ProviderProxy,
	) {
		this.profileRoot = join(config.artifactRoot, "sessions", profile.profileId);
		this.workspace = join(this.profileRoot, "workspace");
		this.sessionDirectory = join(this.profileRoot, "pi-session");
		this.policy = new PolicyController(config.networkPolicy, config.webProvider);
		this.gondolin = new GondolinController(this.workspace);
	}

	async initialize(): Promise<void> {
		await mkdir(this.workspace, { recursive: true });
		await mkdir(this.sessionDirectory, { recursive: true });
		const agents = await readFile(this.profile.agentsPath, "utf8");
		this.agentsSha256 = sha256(agents);
		if (this.agentsSha256 !== this.profile.agentsSha256) {
			throw new Error(`AGENTS.md digest mismatch for profile ${this.profile.profileId}`);
		}
		this.skillDirSha256 = await Promise.all(this.profile.skillDirs.map(directorySha256));
		if (this.skillDirSha256.some((value, index) => value !== this.profile.skillDirSha256[index])) {
			throw new Error(`skill directory digest mismatch for profile ${this.profile.profileId}`);
		}
		for (const extension of this.config.toolExtensions) {
			if (sha256(await readFile(extension.path)) !== extension.sha256) {
				throw new Error(`tool extension digest mismatch: ${extension.path}`);
			}
		}
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
						thinkingLevelMap: {
							off: "off",
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
		const skillDirectories = this.profile.skillDirs.map((directory) => resolve(directory));
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
			],
			noPromptTemplates: true,
			noThemes: true,
			noContextFiles: true,
			skillsOverride: ({ skills, diagnostics }) => ({
				skills: selectedSkills(skills, skillDirectories),
				diagnostics,
			}),
			agentsFilesOverride: () => ({
				agentsFiles: [{ path: this.profile.agentsPath, content: agents }],
			}),
			appendSystemPromptOverride: () => [],
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
			customTools: [this.submissions.tool()],
			tools: sessionTools(
				this.config.context7Enabled,
				this.config.toolExtensions.flatMap((extension) => extension.toolNames),
			),
		});
		this.session = session;
		this.historyCursor = await this.recoverHistoryCursor();
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
			this.acceptCursor(input, true);
			this.historyCursor = input.historyToSeq;
			return priorCommit;
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
			return this.commit(input, savedSubmission.submission, provider, savedSubmission.tools);
		}

		this.submissions.begin(
			this.profile.profileId,
			input.turnId,
			this.profile.candidatesPerTurn,
			(value) => atomicJson(submissionPath, {
				submission: value,
				tools: this.policy.snapshot(),
			} satisfies SavedSubmission),
			validate,
			priorInput !== undefined,
		);
		this.policy.begin(input.forbiddenQueryTerms);
		await this.proxy.beginTurn(
			this.profile.profileId,
			this.session.sessionManager.getSessionId(),
			input.turnId,
			turnRoot,
		);
		let submission: CandidateSubmission | undefined;
		let providerSummary: ProviderTurnSummary;
		let policySummary: { webCalls: number; context7Calls: number };
		try {
			await this.promptWithTimeout(input.message, this.config.limits);
			submission = this.submissions.submission;
			if (!submission) {
				const lastMessage = this.session.messages.at(-1);
				if (lastMessage?.role === "assistant" && lastMessage.stopReason === "error") {
					throw new Error(`provider response failed: ${lastMessage.errorMessage ?? "unknown provider error"}`);
				}
				throw new Error(`session ${this.profile.profileId} ended without submit_candidates`);
			}
		} finally {
			providerSummary = await this.proxy.endTurn(this.profile.profileId);
			policySummary = this.policy.end();
		}
		return this.commit(input, submission, providerSummary, policySummary);
	}

	manifestEntry(): Record<string, unknown> {
		if (!this.session) throw new Error("profile session is not initialized");
		const sessionFile = this.session.sessionManager.getSessionFile();
		return {
			profileId: this.profile.profileId,
			candidatesPerTurn: this.profile.candidatesPerTurn,
			agentsSha256: this.agentsSha256,
			skills: this.skillDirSha256.map((value, index) => ({ directoryIndex: index, sha256: value })),
			sessionId: this.session.sessionManager.getSessionId(),
			session: sessionFile ? relative(this.config.artifactRoot, sessionFile) : undefined,
			workspace: relative(this.config.artifactRoot, this.workspace),
		};
	}

	async close(): Promise<void> {
		if (this.session) {
			await this.session.abort();
			this.session.dispose();
			this.session = undefined;
		}
		await this.gondolin.close();
	}

	private async promptWithTimeout(message: string, limits: HarnessLimits): Promise<void> {
		if (!this.session) throw new Error("profile session is not initialized");
		const session = this.session;
		const options = { expandPromptTemplates: false, source: "rpc" as const };
		const run = async () => {
			await session.prompt(message, options);
			while (!this.submissions.submission) {
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
				this.submissions.requireSubmission();
				await session.prompt(
					interrupted
						? "The previous provider stream ended before your submission was accepted. "
							+ "Do not research again. Call submit_candidates now for the same assigned items using your prior analysis."
						: "The research phase is complete. Call submit_candidates now with the complete ordered minibatch. "
							+ "Do not perform more research or add narrative output.",
					options,
				);
			}
		};
		let timer: NodeJS.Timeout | undefined;
		const timeout = new Promise<never>((_resolve, reject) => {
			timer = setTimeout(() => reject(new Error(`session wall-time limit reached: ${limits.wallTimeSeconds}s`)), limits.wallTimeSeconds * 1000);
		});
		try {
			await Promise.race([run(), timeout]);
		} catch (error) {
			if ((error as Error).message.includes("wall-time limit")) await session.abort();
			throw error;
		} finally {
			if (timer) clearTimeout(timer);
		}
	}

	private async commit(
		input: SessionTurnInput,
		submission: CandidateSubmission,
		provider: ProviderTurnSummary,
		tools: { webCalls: number; context7Calls: number },
	): Promise<CommittedTurn> {
		if (!this.session) throw new Error("profile session is not initialized");
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
			submission,
			usage: {
				providerCalls: provider.providerCalls,
				webCalls: tools.webCalls,
				context7Calls: tools.context7Calls,
				artifactBytes: provider.artifactBytes,
			},
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

	constructor(private readonly config: InitializeFrame, apiKey: string) {
		this.proxy = new ProviderProxy(config.baseUrl, apiKey, config.campaignId);
	}

	async initialize(): Promise<void> {
		await mkdir(this.config.artifactRoot, { recursive: true });
		const profileSetSha256 = canonicalSha256(this.config.profiles.map((profile) => ({
			agentsSha256: profile.agentsSha256,
			candidatesPerTurn: profile.candidatesPerTurn,
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
				provider: this.config.webProvider,
				fetchContent: { domainPolicy: { allow: this.config.networkPolicy.allowedHosts, deny: this.config.networkPolicy.deniedHosts } },
				fetchRouting: { providers: ["http"], allowRemoteHostedProviders: false },
				githubClone: { enabled: false },
				githubPrIssue: { enabled: false },
			}, null, 2)}\n`,
			{ encoding: "utf8", mode: 0o600 },
		);
		await this.proxy.start();
		for (const profile of this.config.profiles) {
			const session = new PersistentProfileSession(profile, this.config, this.proxy);
			this.sessions.set(profile.profileId, session);
		}
		await Promise.all([...this.sessions.values()].map((session) => session.initialize()));
		await atomicJson(join(this.config.artifactRoot, "manifest.json"), {
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
			contextWindow: MODEL_CONTEXT_WINDOW,
			compaction: COMPACTION_SETTINGS,
			candidateSchemaSha256: this.config.candidateSchemaSha256,
			profileSetSha256,
			networkPolicySha256: canonicalSha256(this.config.networkPolicy),
			networkPolicy: this.config.networkPolicy,
			limits: this.config.limits,
			webProvider: this.config.webProvider,
			context7Enabled: this.config.context7Enabled,
			tools: sessionTools(
				this.config.context7Enabled,
				this.config.toolExtensions.flatMap((extension) => extension.toolNames),
			),
			toolExtensions: this.config.toolExtensions,
			topology: {
				profileCount: this.config.profiles.length,
				candidatesPerTurn: this.config.profiles.map((profile) => profile.candidatesPerTurn),
				totalCandidatesPerRound: this.config.profiles.reduce(
					(total, profile) => total + profile.candidatesPerTurn,
					0,
				),
			},
			packages: await runtimePackages(),
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
		return Promise.all(inputs.map(
			(input) => this.sessions.get(input.profileId)?.runTurn(input, validate) as Promise<CommittedTurn>,
		));
	}

	async close(): Promise<void> {
		await Promise.allSettled([...this.sessions.values()].map((session) => session.close()));
		this.sessions.clear();
		await this.proxy.close();
	}
}
