import { sha256 } from "./trace.js";

export const PROTOCOL_VERSION = 5;

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

interface CommonFrame {
	type: string;
	requestId: string;
	protocolVersion: number;
	campaignId: string;
}

export interface HarnessProfileConfig {
	profileId: string;
	agentsPath: string;
	agentsSha256: string;
	skillDirs: string[];
	skillDirSha256: string[];
	candidatesPerTurn: number;
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
	candidateSchema: Record<string, unknown>;
	candidateSchemaSha256: string;
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

export interface SubmissionRejection {
	index: number;
	code: string;
	message: string;
}

export interface SubmissionValidationResultFrame extends CommonFrame {
	type: "submission_validation_result";
	validationId: string;
	accepted: boolean;
	rejected: SubmissionRejection[];
	requiredReplacements: number;
}

export interface SubmissionValidationRequest {
	profileId: string;
	turnId: string;
	attemptIndex: number;
	candidates: Array<Record<string, unknown>>;
}

export interface SubmissionValidationDecision {
	accepted: boolean;
	rejected: SubmissionRejection[];
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

function parseProfiles(value: unknown): HarnessProfileConfig[] {
	if (!Array.isArray(value) || value.length === 0) {
		throw new ProtocolError("invalid_frame", "profiles must be a non-empty array");
	}
	const profiles = value.map((item, index) => {
		const name = `profiles[${index}]`;
		const data = record(item, name);
		exactKeys(data, [
			"profileId", "agentsPath", "agentsSha256", "skillDirs", "skillDirSha256", "candidatesPerTurn",
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
			candidatesPerTurn: positiveInteger(data.candidatesPerTurn, `${name}.candidatesPerTurn`),
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
	if (new Set(toolNames).size !== toolNames.length) {
		throw new ProtocolError("invalid_frame", "MCP tool names must be unique");
	}
	return servers;
}

function common(data: Record<string, unknown>): Omit<CommonFrame, "type"> & { type: string } {
	if (data.protocolVersion !== PROTOCOL_VERSION) {
		throw new ProtocolError("protocol_mismatch", `expected protocol ${PROTOCOL_VERSION}`);
	}
	return {
		type: string(data.type, "type"),
		requestId: string(data.requestId, "requestId"),
		protocolVersion: PROTOCOL_VERSION,
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
			"accepted", "rejected", "requiredReplacements",
		], "frame");
		if (typeof data.accepted !== "boolean") {
			throw new ProtocolError("invalid_frame", "accepted must be boolean");
		}
		if (!Array.isArray(data.rejected)) {
			throw new ProtocolError("invalid_frame", "rejected must be an array");
		}
		const rejected = data.rejected.map((item, index) => {
			const name = `rejected[${index}]`;
			const rejection = record(item, name);
			exactKeys(rejection, ["index", "code", "message"], name);
			const code = string(rejection.code, `${name}.code`);
			if (!/^[a-z][a-z0-9_]*$/.test(code)) {
				throw new ProtocolError("invalid_frame", `${name}.code must be a lowercase identifier`);
			}
			return {
				index: nonnegativeInteger(rejection.index, `${name}.index`),
				code,
				message: string(rejection.message, `${name}.message`),
			};
		});
		if (new Set(rejected.map((item) => item.index)).size !== rejected.length) {
			throw new ProtocolError("invalid_frame", "submission rejection indices must be unique");
		}
		const requiredReplacements = nonnegativeInteger(
			data.requiredReplacements,
			"requiredReplacements",
		);
		if (requiredReplacements !== rejected.length || data.accepted !== (rejected.length === 0)) {
			throw new ProtocolError("invalid_frame", "submission validation result is inconsistent");
		}
		return {
			...identity,
			type: "submission_validation_result",
			validationId: string(data.validationId, "validationId"),
			accepted: data.accepted,
			rejected,
			requiredReplacements,
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
		"model", "thinking", "taskId", "caseId", "seed", "candidateSchemaJson", "candidateSchemaSha256", "profileSetSha256",
		"profiles", "toolExtensions", "mcpServers", "networkPolicy", "limits", "webSearch", "context7Enabled",
	], "frame");
	const candidateSchemaJson = string(data.candidateSchemaJson, "candidateSchemaJson");
	const candidateSchemaSha256 = digest(data.candidateSchemaSha256, "candidateSchemaSha256");
	if (sha256(candidateSchemaJson) !== candidateSchemaSha256) {
		throw new ProtocolError("invalid_frame", "candidateSchema digest mismatch");
	}
	let parsedCandidateSchema: unknown;
	try {
		parsedCandidateSchema = JSON.parse(candidateSchemaJson);
	} catch {
		throw new ProtocolError("invalid_frame", "candidateSchemaJson is not valid JSON");
	}
	const candidateSchema = record(parsedCandidateSchema, "candidateSchemaJson");
	if (candidateSchema.type !== "object" || candidateSchema.additionalProperties !== false) {
		throw new ProtocolError("invalid_frame", "candidateSchema must be a strict JSON object schema");
	}
	const policy = record(data.networkPolicy, "networkPolicy");
	exactKeys(policy, ["allowedHosts", "deniedHosts", "forbiddenQueryPatterns"], "networkPolicy");
	const limits = record(data.limits, "limits");
	exactKeys(limits, ["wallTimeSeconds"], "limits");
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
		candidateSchema,
		candidateSchemaSha256,
		profileSetSha256: digest(data.profileSetSha256, "profileSetSha256"),
		profiles: parseProfiles(data.profiles),
		toolExtensions: parseToolExtensions(data.toolExtensions),
		mcpServers: parseMcpServers(data.mcpServers),
		networkPolicy: {
			allowedHosts: stringArray(policy.allowedHosts, "networkPolicy.allowedHosts"),
			deniedHosts: stringArray(policy.deniedHosts, "networkPolicy.deniedHosts"),
			forbiddenQueryPatterns: stringArray(policy.forbiddenQueryPatterns, "networkPolicy.forbiddenQueryPatterns"),
		},
		limits: {
			wallTimeSeconds: positiveInteger(limits.wallTimeSeconds, "limits.wallTimeSeconds"),
		},
		webSearch: {
			providers,
			fallbackOn: fallbackOn as SearchFallbackKind[],
		},
		context7Enabled: data.context7Enabled,
	};
}
