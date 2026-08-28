import { createRequire } from "node:module";
import { dirname, join, relative, resolve } from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
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
} from "./protocol.js";
import { ProviderProxy, type ProviderTurnSummary } from "./provider-proxy.js";
import { atomicJson } from "./trace.js";

interface CandidateSubmission {
	submissionId: string;
	candidates: Array<Record<string, unknown>>;
}

const submissionSchema = Type.Object({
	candidates: Type.Array(Type.Record(Type.String(), Type.Unknown())),
});

export interface CommittedTurn {
	profileId: string;
	sessionId: string;
	turnId: string;
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

class SubmissionController {
	private expected = 0;
	private providerRequests = 0;
	private turnId = "";
	private value: CandidateSubmission | undefined;
	private persist: ((submission: CandidateSubmission) => Promise<void>) | undefined;

	begin(
		turnId: string,
		expected: number,
		persist: (submission: CandidateSubmission) => Promise<void>,
		recovering: boolean,
	): void {
		this.turnId = turnId;
		this.expected = expected;
		this.providerRequests = recovering ? 1 : 0;
		this.value = undefined;
		this.persist = persist;
	}

	get submission(): CandidateSubmission | undefined {
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
				return {
					...payload,
					tool_choice: this.providerRequests === 1
						? "required"
						: { type: "function", name: "submit_candidates" },
				};
			});
		};
	}

	tool(): ToolDefinition<typeof submissionSchema> {
		return {
			name: "submit_candidates",
			label: "Submit candidates",
			description: "Submit the complete ordered candidate minibatch exactly once after research and validation.",
			promptSnippet: "submit_candidates: submit the complete ordered candidate minibatch",
			parameters: submissionSchema,
			executionMode: "sequential",
			execute: async (_toolCallId, params) => {
				if (this.value) throw new Error("submit_candidates may be called only once per turn");
				if (params.candidates.length !== this.expected) {
					throw new Error(`submit_candidates requires exactly ${this.expected} candidates`);
				}
				if (params.candidates.some((candidate) => !candidate || typeof candidate !== "object" || Array.isArray(candidate))) {
					throw new Error("each candidate must be an object");
				}
				const submission: CandidateSubmission = {
					submissionId: `${this.turnId}-submission`,
					candidates: params.candidates,
				};
				if (!this.persist) throw new Error("submission turn is not initialized");
				await this.persist(submission);
				this.value = submission;
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

	constructor(
		private readonly profile: HarnessProfileConfig,
		private readonly config: InitializeFrame,
		private readonly proxy: ProviderProxy,
	) {
		this.profileRoot = join(config.artifactRoot, "sessions", profile.profileId);
		this.workspace = join(this.profileRoot, "workspace");
		this.sessionDirectory = join(this.profileRoot, "pi-session");
		this.policy = new PolicyController(config.networkPolicy, config.limits, config.webProvider);
		this.gondolin = new GondolinController(this.workspace);
	}

	async initialize(): Promise<void> {
		await mkdir(this.workspace, { recursive: true });
		await mkdir(this.sessionDirectory, { recursive: true });
		const agents = await readFile(this.profile.agentsPath, "utf8");
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
						contextWindow: 200000,
						maxTokens: 32768,
					}],
				},
			},
		});

		const modelRuntime = await ModelRuntime.create({
			authPath: join(agentDirectory, "auth.json"),
			modelsPath,
			refreshOnCreate: false,
		});
		await modelRuntime.setRuntimeApiKey(providerId, "sidecar-proxy-token");
		const model = modelRuntime.getModel(providerId, this.config.model);
		if (!model) throw new Error(`Pi model registration failed for ${providerId}/${this.config.model}`);

		const settings = SettingsManager.inMemory({
			compaction: { enabled: false },
			retry: { enabled: false },
		});
		const webExtension = join(packageRoot("pi-web-access"), "index.ts");
		const extensionPaths = [webExtension];
		if (this.config.context7Enabled) {
			extensionPaths.push(join(packageRoot("@upstash/context7-pi"), "extensions", "context7.ts"));
		}
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
			tools: [
				"read",
				"write",
				"bash",
				"web_search",
				"fetch_content",
				"get_search_content",
				...(this.config.context7Enabled ? ["resolve-library-id", "query-docs"] : []),
				"submit_candidates",
			],
		});
		this.session = session;
	}

	async runTurn(input: SessionTurnInput): Promise<CommittedTurn> {
		if (!this.session) throw new Error("profile session is not initialized");
		const turnRoot = join(this.profileRoot, "turns", input.turnId);
		const commitPath = join(this.config.artifactRoot, "turns", input.turnId, "turn_committed.json");
		const priorCommit = await optionalJson<CommittedTurn>(commitPath);
		if (priorCommit) {
			if (priorCommit.inputDigest !== input.inputDigest || priorCommit.profileId !== input.profileId) {
				throw new Error(`committed turn digest mismatch: ${input.turnId}`);
			}
			return priorCommit;
		}

		const inputPath = join(turnRoot, "input.json");
		const priorInput = await optionalJson<SessionTurnInput>(inputPath);
		if (priorInput && priorInput.inputDigest !== input.inputDigest) {
			throw new Error(`partial turn digest mismatch: ${input.turnId}`);
		}
		if (!priorInput) await atomicJson(inputPath, input);

		const submissionPath = join(turnRoot, "submission.json");
		const savedSubmission = await optionalJson<CandidateSubmission>(submissionPath);
		if (savedSubmission) {
			return this.commit(input, savedSubmission, { providerCalls: 0, artifactBytes: 0 }, { webCalls: 0, context7Calls: 0 });
		}

		this.submissions.begin(
			input.turnId,
			this.profile.candidatesPerTurn,
			(value) => atomicJson(submissionPath, value),
			priorInput !== undefined,
		);
		this.policy.begin(input.forbiddenQueryTerms);
		await this.proxy.beginTurn(
			this.profile.profileId,
			input.turnId,
			turnRoot,
			this.config.limits.providerCalls,
			this.config.limits.artifactBytes,
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
			const lastMessage = session.messages.at(-1);
			const errorMessage = lastMessage?.role === "assistant" ? lastMessage.errorMessage?.toLowerCase() : undefined;
			if (
				!this.submissions.submission
				&& lastMessage?.role === "assistant"
				&& lastMessage.stopReason === "error"
				&& errorMessage
				&& (
					errorMessage.includes("stream_read_error")
					|| errorMessage.includes("stream ended before a terminal response event")
				)
			) {
				await session.prompt(
					"The previous provider stream ended before your submission was accepted. "
					+ "Do not research again. Call submit_candidates now for the same assigned items using your prior analysis.",
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
		return commit;
	}
}

export class PiSessionPool {
	private readonly sessions = new Map<string, PersistentProfileSession>();
	private readonly proxy: ProviderProxy;

	constructor(private readonly config: InitializeFrame, apiKey: string) {
		this.proxy = new ProviderProxy(config.baseUrl, apiKey);
	}

	async initialize(): Promise<void> {
		await mkdir(this.config.artifactRoot, { recursive: true });
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
			model: this.config.model,
			wireApi: this.config.wireApi,
			thinking: this.config.thinking,
			profiles: this.config.profiles.map((profile) => ({
				profileId: profile.profileId,
				candidatesPerTurn: profile.candidatesPerTurn,
				skills: profile.skillDirs,
			})),
		});
	}

	async runTurns(inputs: SessionTurnInput[]): Promise<CommittedTurn[]> {
		const expected = new Set(this.sessions.keys());
		if (inputs.length !== expected.size || new Set(inputs.map((input) => input.profileId)).size !== inputs.length) {
			throw new Error("run_turn must contain exactly one input for every profile");
		}
		for (const input of inputs) {
			if (!expected.delete(input.profileId)) throw new Error(`unknown or duplicate profile: ${input.profileId}`);
		}
		return Promise.all(inputs.map((input) => this.sessions.get(input.profileId)?.runTurn(input) as Promise<CommittedTurn>));
	}

	async close(): Promise<void> {
		await Promise.allSettled([...this.sessions.values()].map((session) => session.close()));
		this.sessions.clear();
		await this.proxy.close();
	}
}
