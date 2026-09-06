import { sha256 } from "./trace.js";
import { SIDECAR_RELEASE_VERSION } from "./release.js";

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

interface CommonFrame {
	type: string;
	requestId: string;
	protocolVersion: string;
	campaignId: string;
}

export interface HarnessProfileConfig {
	profileId: string;
	agentsPath: string;
	agentsSha256: string;
	skillDirs: string[];
	skillDirSha256: string[];
}

export interface HarnessArtifactRuleConfig {
	pathPointer: string;
	allowedSuffixes: string[];
	maxBytes: number;
}

export interface HarnessSubmissionContractConfig {
	contractId: string;
	toolName: string;
	payloadSchema: Record<string, unknown>;
	artifactRules: HarnessArtifactRuleConfig[];
	maxValidationAttempts: number | null;
}

export interface HarnessToolExtensionConfig {
	path: string;
	sha256: string;
	toolNames: string[];
}

export type McpInjectedValue =
	| { value: string }
	| { secretName: string; secretSource: string; prefix: string };

interface McpServerBase {
	serverId: string;
	tools: string[];
	configSha256: string;
}

export type McpServerConfig =
	| (McpServerBase & {
		transport: "stdio";
		command: string;
		args: string[];
		env: Record<string, McpInjectedValue>;
	})
	| (McpServerBase & {
		transport: "streamable_http";
		url: string;
		headers: Record<string, McpInjectedValue>;
	});

export interface NetworkPolicy {
	allowedHosts: string[];
	deniedHosts: string[];
	forbiddenQueryPatterns: string[];
}

export interface HarnessLimits {
	wallTimeSeconds: number;
	toolCallBudgets: Record<string, number>;
}

export interface GuestRuntimeConfig {
	imageRef: string;
	recipeSha256: string;
	rootfsSize: string;
	installPolicy: "session_overlay";
}

export type SearchFallbackKind = "transient" | "quota" | "network" | "invalid-response" | "unsupported";

export interface WebSearchConfig {
	providers: string[];
	fallbackOn: SearchFallbackKind[];
}

export interface BootstrapSecretFrame extends CommonFrame {
	type: "bootstrap_secret";
	apiKey: string;
	namedSecrets: Record<string, string>;
}

export interface InitializeFrame extends CommonFrame {
	type: "initialize";
	artifactRoot: string;
	baseUrl: string;
	wireApi: "responses";
	model: string;
	thinking: ThinkingLevel;
	taskId: string;
	caseId: string;
	seed: number;
	submissionContract: HarnessSubmissionContractConfig;
	submissionContractSha256: string;
	guestRuntime: GuestRuntimeConfig;
	profileSetSha256: string;
	profiles: HarnessProfileConfig[];
	toolExtensions: HarnessToolExtensionConfig[];
	mcpServers: McpServerConfig[];
	networkPolicy: NetworkPolicy;
	limits: HarnessLimits;
	webSearch: WebSearchConfig;
	context7Enabled: boolean;
}

export interface SessionTurnInput {
	profileId: string;
	turnId: string;
	roundIndex: number;
	historyFromSeq: number;
	historyToSeq: number;
	historyDigest: string;
	inputDigest: string;
	message: string;
	forbiddenQueryTerms: string[];
}

export interface RunTurnFrame extends CommonFrame {
	type: "run_turn";
	turns: SessionTurnInput[];
}

export interface SubmittedArtifact {
	pathPointer: string;
	relativePath: string;
	snapshotPath: string;
	sha256: string;
	sizeBytes: number;
}

export interface SubmissionError {
	path: string;
	code: string;
	message: string;
	hint: string;
}

export interface SubmissionValidationResultFrame extends CommonFrame {
	type: "submission_validation_result";
	validationId: string;
	submissionDigest: string;
	decision: "accept" | "retry" | "reject_turn";
	errors: SubmissionError[];
}

export interface SubmissionValidationRequest {
	profileId: string;
	turnId: string;
	attemptIndex: number;
	submission: Record<string, unknown>;
	artifacts: SubmittedArtifact[];
	submissionJson: string;
	submissionDigest: string;
}

export interface SubmissionValidationDecision {
	decision: "accept" | "retry" | "reject_turn";
	errors: SubmissionError[];
}

export type SubmissionValidator = (
	request: SubmissionValidationRequest,
) => Promise<SubmissionValidationDecision>;

export interface CloseFrame extends CommonFrame {
	type: "close";
}

export type InputFrame =
	| BootstrapSecretFrame
	| InitializeFrame
	| RunTurnFrame
	| SubmissionValidationResultFrame
	| CloseFrame;

export class TurnExecutionError extends Error {
	constructor(message: string, readonly turnUsage: Array<{
		profileId: string;
		turnId: string;
		usage: { providerCalls: number; toolCalls: Record<string, number>; artifactBytes: number };
	}>) {
		super(message);
	}
}

export class ProtocolError extends Error {
	readonly code: string;

	constructor(code: string, message: string) {
		super(message);
		this.name = "ProtocolError";
		this.code = code;
	}
}

function record(value: unknown, name: string): Record<string, unknown> {
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new ProtocolError("invalid_frame", `${name} must be an object`);
	}
	return value as Record<string, unknown>;
}

function exactKeys(data: Record<string, unknown>, expected: readonly string[], name: string): void {
	const actual = Object.keys(data).sort();
	const required = [...expected].sort();
	if (actual.length !== required.length || actual.some((key, index) => key !== required[index])) {
		throw new ProtocolError("invalid_frame", `${name} has unexpected or missing fields`);
	}
}

function string(value: unknown, name: string): string {
	if (typeof value !== "string" || value.length === 0) {
		throw new ProtocolError("invalid_frame", `${name} must be a non-empty string`);
	}
	return value;
}

function digest(value: unknown, name: string): string {
	const result = string(value, name);
	if (!/^[a-f0-9]{64}$/.test(result)) {
		throw new ProtocolError("invalid_frame", `${name} must be a lowercase SHA-256 digest`);
	}
	return result;
}

function stringArray(value: unknown, name: string): string[] {
	if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
		throw new ProtocolError("invalid_frame", `${name} must be a string array`);
	}
	return [...value] as string[];
}

function positiveInteger(value: unknown, name: string): number {
	if (!Number.isInteger(value) || (value as number) <= 0) {
		throw new ProtocolError("invalid_frame", `${name} must be a positive integer`);
	}
	return value as number;
}

function nonnegativeInteger(value: unknown, name: string): number {
	if (!Number.isInteger(value) || (value as number) < 0) {
		throw new ProtocolError("invalid_frame", `${name} must be a non-negative integer`);
	}
	return value as number;
}

function parseGuestRuntime(value: unknown): GuestRuntimeConfig {
	const data = record(value, "guestRuntime");
	exactKeys(data, ["imageRef", "recipeSha256", "rootfsSize", "installPolicy"], "guestRuntime");
	const imageRef = string(data.imageRef, "guestRuntime.imageRef");
	if (!/^ldm\/[a-z][a-z0-9-]*:[a-f0-9]{12}$/.test(imageRef)) {
		throw new ProtocolError("invalid_frame", "guestRuntime.imageRef must be a logical ldm image ref");
	}
	const rootfsSize = string(data.rootfsSize, "guestRuntime.rootfsSize");
	if (!/^[1-9][0-9]*[KMGT]$/.test(rootfsSize)) {
		throw new ProtocolError("invalid_frame", "guestRuntime.rootfsSize must be a positive size");
	}
	if (data.installPolicy !== "session_overlay") {
		throw new ProtocolError("invalid_frame", "guestRuntime.installPolicy is unsupported");
	}
	return {
		imageRef,
		recipeSha256: digest(data.recipeSha256, "guestRuntime.recipeSha256"),
		rootfsSize,
		installPolicy: "session_overlay",
	};
}

function parseProfiles(value: unknown): HarnessProfileConfig[] {
	if (!Array.isArray(value) || value.length === 0) {
		throw new ProtocolError("invalid_frame", "profiles must be a non-empty array");
	}
	const profiles = value.map((item, index) => {
		const name = `profiles[${index}]`;
		const data = record(item, name);
		exactKeys(data, [
			"profileId", "agentsPath", "agentsSha256", "skillDirs", "skillDirSha256",
		], name);
		const profileId = string(data.profileId, `${name}.profileId`);
		if (!/^[a-z][a-z0-9_]*$/.test(profileId)) {
			throw new ProtocolError("invalid_frame", `invalid profileId: ${profileId}`);
		}
		const skillDirs = stringArray(data.skillDirs, `${name}.skillDirs`);
		const skillDirSha256 = stringArray(data.skillDirSha256, `${name}.skillDirSha256`).map(
			(value, digestIndex) => digest(value, `${name}.skillDirSha256[${digestIndex}]`),
		);
		if (skillDirs.length !== skillDirSha256.length) {
			throw new ProtocolError("invalid_frame", `${name} skill directories and digests must have equal length`);
		}
		return {
			profileId,
			agentsPath: string(data.agentsPath, `${name}.agentsPath`),
			agentsSha256: digest(data.agentsSha256, `${name}.agentsSha256`),
			skillDirs,
			skillDirSha256,
		};
	});
	if (new Set(profiles.map((profile) => profile.profileId)).size !== profiles.length) {
		throw new ProtocolError("invalid_frame", "profileId values must be unique");
	}
	return profiles;
}

function parseToolExtensions(value: unknown): HarnessToolExtensionConfig[] {
	if (!Array.isArray(value)) throw new ProtocolError("invalid_frame", "toolExtensions must be an array");
	const extensions = value.map((item, index) => {
		const name = `toolExtensions[${index}]`;
		const data = record(item, name);
		exactKeys(data, ["path", "sha256", "toolNames"], name);
		const toolNames = stringArray(data.toolNames, `${name}.toolNames`);
		if (toolNames.length === 0 || toolNames.some((toolName) => !/^[a-z][a-z0-9_]*$/.test(toolName))) {
			throw new ProtocolError("invalid_frame", `${name}.toolNames must contain lowercase identifiers`);
		}
		if (new Set(toolNames).size !== toolNames.length) {
			throw new ProtocolError("invalid_frame", `${name}.toolNames must be unique`);
		}
		return {
			path: string(data.path, `${name}.path`),
			sha256: digest(data.sha256, `${name}.sha256`),
			toolNames,
		};
	});
	const names = extensions.flatMap((extension) => extension.toolNames);
	if (new Set(names).size !== names.length) {
		throw new ProtocolError("invalid_frame", "tool names must be unique across extensions");
	}
	return extensions;
}

function jsonPointer(value: unknown, name: string): string {
	const result = string(value, name);
	if (!result.startsWith("/")) {
		throw new ProtocolError("invalid_frame", `${name} must be a non-root JSON Pointer`);
	}
	return result;
}

function parseSubmissionContract(
	value: unknown,
	name: string,
): HarnessSubmissionContractConfig {
	const data = record(value, name);
	exactKeys(data, [
		"contractId", "toolName", "payloadSchema", "artifactRules", "maxValidationAttempts",
	], name);
	const contractId = string(data.contractId, `${name}.contractId`);
	if (!/^[a-z][a-z0-9_]*$/.test(contractId)) {
		throw new ProtocolError("invalid_frame", `${name}.contractId must be a lowercase identifier`);
	}
	const toolName = string(data.toolName, `${name}.toolName`);
	if (!/^[A-Za-z0-9_-]+$/.test(toolName)) {
		throw new ProtocolError("invalid_frame", `${name}.toolName must be a function identifier`);
	}
	const payloadSchema = record(data.payloadSchema, `${name}.payloadSchema`);
	if (payloadSchema.type !== "object" || payloadSchema.additionalProperties !== false) {
		throw new ProtocolError("invalid_frame", `${name}.payloadSchema must be a strict JSON object schema`);
	}
	if (!Array.isArray(data.artifactRules)) {
		throw new ProtocolError("invalid_frame", `${name}.artifactRules must be an array`);
	}
	const artifactRules = data.artifactRules.map((raw, index) => {
		const ruleName = `${name}.artifactRules[${index}]`;
		const rule = record(raw, ruleName);
		exactKeys(rule, ["pathPointer", "allowedSuffixes", "maxBytes"], ruleName);
		const allowedSuffixes = stringArray(rule.allowedSuffixes, `${ruleName}.allowedSuffixes`);
		if (
			allowedSuffixes.length === 0
			|| new Set(allowedSuffixes).size !== allowedSuffixes.length
			|| allowedSuffixes.some((suffix) => !suffix.startsWith(".") || suffix.includes("/") || suffix.includes("\\"))
		) {
			throw new ProtocolError("invalid_frame", `${ruleName}.allowedSuffixes must contain unique file suffixes`);
		}
		return {
			pathPointer: jsonPointer(rule.pathPointer, `${ruleName}.pathPointer`),
			allowedSuffixes,
			maxBytes: positiveInteger(rule.maxBytes, `${ruleName}.maxBytes`),
		};
	});
	if (new Set(artifactRules.map((rule) => rule.pathPointer)).size !== artifactRules.length) {
		throw new ProtocolError("invalid_frame", `${name}.artifactRules path pointers must be unique`);
	}
	let maxValidationAttempts: number | null = null;
	if (data.maxValidationAttempts !== null) {
		maxValidationAttempts = positiveInteger(
			data.maxValidationAttempts,
			`${name}.maxValidationAttempts`,
		);
	}
	return { contractId, toolName, payloadSchema, artifactRules, maxValidationAttempts };
}

function parseInjectedValues(value: unknown, name: string): Record<string, McpInjectedValue> {
	const data = record(value, name);
	return Object.fromEntries(Object.entries(data).map(([key, raw]) => {
		if (!key) throw new ProtocolError("invalid_frame", `${name} names must not be empty`);
		const item = record(raw, `${name}.${key}`);
		if ("value" in item) {
			exactKeys(item, ["value"], `${name}.${key}`);
			return [key, { value: string(item.value, `${name}.${key}.value`) }];
		}
		exactKeys(item, ["secretName", "secretSource", "prefix"], `${name}.${key}`);
		if (typeof item.prefix !== "string") {
			throw new ProtocolError("invalid_frame", `${name}.${key}.prefix must be a string`);
		}
		return [key, {
			secretName: string(item.secretName, `${name}.${key}.secretName`),
			secretSource: string(item.secretSource, `${name}.${key}.secretSource`),
			prefix: item.prefix,
		}];
	}));
}

function parseMcpServers(value: unknown): McpServerConfig[] {
	if (!Array.isArray(value)) throw new ProtocolError("invalid_frame", "mcpServers must be an array");
	const servers = value.map((raw, index): McpServerConfig => {
		const name = `mcpServers[${index}]`;
		const data = record(raw, name);
		const serverId = string(data.serverId, `${name}.serverId`);
		if (!/^[a-z][a-z0-9_]*$/.test(serverId)) {
			throw new ProtocolError("invalid_frame", `${name}.serverId must be a lowercase identifier`);
		}
		const tools = stringArray(data.tools, `${name}.tools`);
		if (
			tools.length === 0
			|| new Set(tools).size !== tools.length
			|| tools.some((tool) => !/^[A-Za-z0-9_-]+$/.test(tool))
		) {
			throw new ProtocolError("invalid_frame", `${name}.tools must be a unique function-name allowlist`);
		}
		const base = {
			serverId,
			tools,
			configSha256: digest(data.configSha256, `${name}.configSha256`),
		};
		if (data.transport === "stdio") {
			exactKeys(data, ["serverId", "transport", "tools", "configSha256", "command", "args", "env"], name);
			return {
				...base,
				transport: "stdio",
				command: string(data.command, `${name}.command`),
				args: stringArray(data.args, `${name}.args`),
				env: parseInjectedValues(data.env, `${name}.env`),
			};
		}
		if (data.transport === "streamable_http") {
			exactKeys(data, ["serverId", "transport", "tools", "configSha256", "url", "headers"], name);
			const url = string(data.url, `${name}.url`);
			let parsed: URL;
			try {
				parsed = new URL(url);
			} catch {
				throw new ProtocolError("invalid_frame", `${name}.url must be absolute`);
			}
			const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname);
			if (parsed.protocol !== "https:" && !(parsed.protocol === "http:" && loopback)) {
				throw new ProtocolError("invalid_frame", `${name}.url must use HTTPS except on loopback`);
			}
			if (parsed.username || parsed.password) {
				throw new ProtocolError("invalid_frame", `${name}.url must not contain credentials`);
			}
			return {
				...base,
				transport: "streamable_http",
				url,
				headers: parseInjectedValues(data.headers, `${name}.headers`),
			};
		}
		throw new ProtocolError("invalid_frame", `${name}.transport is unsupported`);
	});
	if (new Set(servers.map((server) => server.serverId)).size !== servers.length) {
		throw new ProtocolError("invalid_frame", "MCP server IDs must be unique");
	}
	const toolNames = servers.flatMap((server) => server.tools.map((tool) => `mcp__${server.serverId}__${tool}`));
	if (new Set(toolNames).size !== toolNames.length || toolNames.some((name) => name.length > 64)) {
		throw new ProtocolError("invalid_frame", "MCP tool names must be unique and at most 64 characters");
	}
	return servers;
}

function common(data: Record<string, unknown>): Omit<CommonFrame, "type"> & { type: string } {
	if (data.protocolVersion !== SIDECAR_RELEASE_VERSION) {
		throw new ProtocolError("protocol_mismatch", `expected protocol ${SIDECAR_RELEASE_VERSION}`);
	}
	return {
		type: string(data.type, "type"),
		requestId: string(data.requestId, "requestId"),
		protocolVersion: SIDECAR_RELEASE_VERSION,
		campaignId: string(data.campaignId, "campaignId"),
	};
}

export function parseFrame(line: string): InputFrame {
	let value: unknown;
	try {
		value = JSON.parse(line);
	} catch {
		throw new ProtocolError("invalid_json", "frame is not valid JSON");
	}
	const data = record(value, "frame");
	const identity = common(data);

	if (identity.type === "bootstrap_secret") {
		exactKeys(data, ["type", "requestId", "protocolVersion", "campaignId", "apiKey", "namedSecrets"], "frame");
		const namedSecrets = record(data.namedSecrets, "namedSecrets");
		return {
			...identity,
			type: "bootstrap_secret",
			apiKey: string(data.apiKey, "apiKey"),
			namedSecrets: Object.fromEntries(
				Object.entries(namedSecrets).map(([name, value]) => {
					if (!name) throw new ProtocolError("invalid_frame", "named secret names must not be empty");
					return [name, string(value, `namedSecrets.${name}`)];
				}),
			),
		};
	}
	if (identity.type === "close") {
		exactKeys(data, ["type", "requestId", "protocolVersion", "campaignId"], "frame");
		return { ...identity, type: "close" };
	}
	if (identity.type === "submission_validation_result") {
		exactKeys(data, [
			"type", "requestId", "protocolVersion", "campaignId", "validationId",
			"submissionDigest", "decision", "errors",
		], "frame");
		if (!Array.isArray(data.errors)) {
			throw new ProtocolError("invalid_frame", "errors must be an array");
		}
		const decision = string(data.decision, "decision");
		if (!["accept", "retry", "reject_turn"].includes(decision)) {
			throw new ProtocolError("invalid_frame", "unsupported submission validation decision");
		}
		const errors = data.errors.map((item, index) => {
			const name = `errors[${index}]`;
			const error = record(item, name);
			exactKeys(error, ["path", "code", "message", "hint"], name);
			if (typeof error.path !== "string" || (error.path !== "" && !error.path.startsWith("/"))) {
				throw new ProtocolError("invalid_frame", `${name}.path must be a JSON Pointer`);
			}
			const code = string(error.code, `${name}.code`);
			if (!/^[a-z][a-z0-9_]*$/.test(code)) {
				throw new ProtocolError("invalid_frame", `${name}.code must be a lowercase identifier`);
			}
			if (typeof error.hint !== "string") {
				throw new ProtocolError("invalid_frame", `${name}.hint must be a string`);
			}
			return {
				path: error.path,
				code,
				message: string(error.message, `${name}.message`),
				hint: error.hint,
			};
		});
		if ((decision === "accept") !== (errors.length === 0)) {
			throw new ProtocolError("invalid_frame", "submission validation result is inconsistent");
		}
		return {
			...identity,
			type: "submission_validation_result",
			validationId: string(data.validationId, "validationId"),
			submissionDigest: digest(data.submissionDigest, "submissionDigest"),
			decision: decision as "accept" | "retry" | "reject_turn",
			errors,
		};
	}
	if (identity.type === "run_turn") {
		exactKeys(data, ["type", "requestId", "protocolVersion", "campaignId", "turns"], "frame");
		if (!Array.isArray(data.turns) || data.turns.length === 0) {
			throw new ProtocolError("invalid_frame", "turns must be a non-empty array");
		}
		return {
			...identity,
			type: "run_turn",
			turns: data.turns.map((item, index) => {
				const name = `turns[${index}]`;
				const turn = record(item, name);
				exactKeys(turn, [
					"profileId", "turnId", "roundIndex", "historyFromSeq", "historyToSeq",
					"historyDigest", "inputDigest", "message", "forbiddenQueryTerms",
				], name);
				const turnId = string(turn.turnId, `${name}.turnId`);
				if (!/^[A-Za-z0-9_-]+$/.test(turnId)) {
					throw new ProtocolError("invalid_frame", `invalid turnId: ${turnId}`);
				}
				const historyFromSeq = nonnegativeInteger(turn.historyFromSeq, `${name}.historyFromSeq`);
				const historyToSeq = nonnegativeInteger(turn.historyToSeq, `${name}.historyToSeq`);
				if (historyToSeq < historyFromSeq) {
					throw new ProtocolError("invalid_frame", `${name}.historyToSeq precedes historyFromSeq`);
				}
				return {
					profileId: string(turn.profileId, `${name}.profileId`),
					turnId,
					roundIndex: nonnegativeInteger(turn.roundIndex, `${name}.roundIndex`),
					historyFromSeq,
					historyToSeq,
					historyDigest: digest(turn.historyDigest, `${name}.historyDigest`),
					inputDigest: digest(turn.inputDigest, `${name}.inputDigest`),
					message: string(turn.message, `${name}.message`),
					forbiddenQueryTerms: stringArray(turn.forbiddenQueryTerms, `${name}.forbiddenQueryTerms`),
				};
			}),
		};
	}
	if (identity.type !== "initialize") {
		throw new ProtocolError("unknown_frame", `unknown frame type: ${identity.type}`);
	}

	exactKeys(data, [
		"type", "requestId", "protocolVersion", "campaignId", "artifactRoot", "baseUrl", "wireApi",
		"model", "thinking", "taskId", "caseId", "seed", "submissionContractJson", "submissionContractSha256", "profileSetSha256",
		"guestRuntime", "profiles", "toolExtensions", "mcpServers", "networkPolicy", "limits", "webSearch", "context7Enabled",
	], "frame");
	const submissionContractJson = string(data.submissionContractJson, "submissionContractJson");
	const submissionContractSha256 = digest(data.submissionContractSha256, "submissionContractSha256");
	if (sha256(submissionContractJson) !== submissionContractSha256) {
		throw new ProtocolError("invalid_frame", "submissionContract digest mismatch");
	}
	let parsedSubmissionContract: unknown;
	try {
		parsedSubmissionContract = JSON.parse(submissionContractJson);
	} catch {
		throw new ProtocolError("invalid_frame", "submissionContractJson is not valid JSON");
	}
	const submissionContract = parseSubmissionContract(
		parsedSubmissionContract,
		"submissionContractJson",
	);
	const guestRuntime = parseGuestRuntime(data.guestRuntime);
	const policy = record(data.networkPolicy, "networkPolicy");
	exactKeys(policy, ["allowedHosts", "deniedHosts", "forbiddenQueryPatterns"], "networkPolicy");
	const limits = record(data.limits, "limits");
	exactKeys(limits, ["wallTimeSeconds", "toolCallBudgets"], "limits");
	const toolCallBudgets = record(limits.toolCallBudgets, "limits.toolCallBudgets");
	for (const [toolName, limit] of Object.entries(toolCallBudgets)) {
		if (!/^[A-Za-z0-9_-]+$/.test(toolName)) {
			throw new ProtocolError("invalid_frame", `invalid tool budget name: ${toolName}`);
		}
		nonnegativeInteger(limit, `limits.toolCallBudgets.${toolName}`);
	}
	if (submissionContract.toolName in toolCallBudgets) {
		throw new ProtocolError(
			"invalid_frame",
			`${submissionContract.toolName} cannot have a tool call budget`,
		);
	}
	const webSearch = record(data.webSearch, "webSearch");
	exactKeys(webSearch, ["providers", "fallbackOn"], "webSearch");
	const providers = stringArray(webSearch.providers, "webSearch.providers");
	if (
		providers.length === 0
		|| new Set(providers).size !== providers.length
		|| providers.some((provider) => !/^[a-z][a-z0-9-]*$/.test(provider) || provider === "auto" || provider === "all")
	) {
		throw new ProtocolError(
			"invalid_frame",
			"webSearch.providers must contain unique resolved lowercase provider names",
		);
	}
	const fallbackOn = stringArray(webSearch.fallbackOn, "webSearch.fallbackOn");
	const fallbackKinds: SearchFallbackKind[] = ["transient", "quota", "network", "invalid-response", "unsupported"];
	if (
		fallbackOn.length === 0
		|| new Set(fallbackOn).size !== fallbackOn.length
		|| fallbackOn.some((kind) => !fallbackKinds.includes(kind as SearchFallbackKind))
	) {
		throw new ProtocolError("invalid_frame", "webSearch.fallbackOn contains invalid or duplicate kinds");
	}
	const thinking = string(data.thinking, "thinking") as ThinkingLevel;
	const thinkingLevels: ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
	if (!thinkingLevels.includes(thinking)) throw new ProtocolError("invalid_frame", `invalid thinking level: ${thinking}`);
	if (data.wireApi !== "responses") throw new ProtocolError("invalid_frame", "wireApi must be responses");
	if (typeof data.context7Enabled !== "boolean") {
		throw new ProtocolError("invalid_frame", "context7Enabled must be boolean");
	}

	const profiles = parseProfiles(data.profiles);
	const toolExtensions = parseToolExtensions(data.toolExtensions);
	const mcpServers = parseMcpServers(data.mcpServers);
	const availableTools = new Set([
		"read", "write", "bash", "web_search", "fetch_content", "get_search_content",
		...toolExtensions.flatMap((extension) => extension.toolNames),
		...mcpServers.flatMap((server) => server.tools.map((tool) => `mcp__${server.serverId}__${tool}`)),
		...(data.context7Enabled ? ["resolve-library-id", "query-docs"] : []),
	]);
	if (availableTools.has(submissionContract.toolName)) {
		throw new ProtocolError("invalid_frame", "terminal tool conflicts with another available tool");
	}
	const unavailableBudgets = Object.keys(toolCallBudgets).filter((name) => !availableTools.has(name));
	if (unavailableBudgets.length > 0) {
		throw new ProtocolError(
			"invalid_frame",
			`tool budgets reference unavailable tools: ${unavailableBudgets.sort().join(", ")}`,
		);
	}

	return {
		...identity,
		type: "initialize",
		artifactRoot: string(data.artifactRoot, "artifactRoot"),
		baseUrl: string(data.baseUrl, "baseUrl"),
		wireApi: "responses",
		model: string(data.model, "model"),
		thinking,
		taskId: string(data.taskId, "taskId"),
		caseId: string(data.caseId, "caseId"),
		seed: nonnegativeInteger(data.seed, "seed"),
		submissionContract,
		submissionContractSha256,
		guestRuntime,
		profileSetSha256: digest(data.profileSetSha256, "profileSetSha256"),
		profiles,
		toolExtensions,
		mcpServers,
		networkPolicy: {
			allowedHosts: stringArray(policy.allowedHosts, "networkPolicy.allowedHosts"),
			deniedHosts: stringArray(policy.deniedHosts, "networkPolicy.deniedHosts"),
			forbiddenQueryPatterns: stringArray(policy.forbiddenQueryPatterns, "networkPolicy.forbiddenQueryPatterns"),
		},
		limits: {
			wallTimeSeconds: positiveInteger(limits.wallTimeSeconds, "limits.wallTimeSeconds"),
			toolCallBudgets: Object.fromEntries(
				Object.entries(toolCallBudgets).map(([name, limit]) => [
					name,
					nonnegativeInteger(limit, `limits.toolCallBudgets.${name}`),
				]),
			),
		},
		webSearch: {
			providers,
			fallbackOn: fallbackOn as SearchFallbackKind[],
		},
		context7Enabled: data.context7Enabled,
	};
}
