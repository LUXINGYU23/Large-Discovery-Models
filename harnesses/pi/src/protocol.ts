export const PROTOCOL_VERSION = 2;

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

export interface NetworkPolicy {
	allowedHosts: string[];
	deniedHosts: string[];
	forbiddenQueryPatterns: string[];
}

export interface HarnessLimits {
	wallTimeSeconds: number;
	providerCalls: number;
	webCalls: number;
	context7Calls: number;
	artifactBytes: number;
}

export interface BootstrapSecretFrame extends CommonFrame {
	type: "bootstrap_secret";
	apiKey: string;
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
	candidateSchemaSha256: string;
	profileSetSha256: string;
	profiles: HarnessProfileConfig[];
	networkPolicy: NetworkPolicy;
	limits: HarnessLimits;
	webProvider: "anysearch";
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

export interface CloseFrame extends CommonFrame {
	type: "close";
}

export type InputFrame = BootstrapSecretFrame | InitializeFrame | RunTurnFrame | CloseFrame;

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
		exactKeys(data, ["type", "requestId", "protocolVersion", "campaignId", "apiKey"], "frame");
		return { ...identity, type: "bootstrap_secret", apiKey: string(data.apiKey, "apiKey") };
	}
	if (identity.type === "close") {
		exactKeys(data, ["type", "requestId", "protocolVersion", "campaignId"], "frame");
		return { ...identity, type: "close" };
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
		"model", "thinking", "taskId", "caseId", "seed", "candidateSchemaSha256", "profileSetSha256",
		"profiles", "networkPolicy", "limits", "webProvider", "context7Enabled",
	], "frame");
	const policy = record(data.networkPolicy, "networkPolicy");
	exactKeys(policy, ["allowedHosts", "deniedHosts", "forbiddenQueryPatterns"], "networkPolicy");
	const limits = record(data.limits, "limits");
	exactKeys(limits, ["wallTimeSeconds", "providerCalls", "webCalls", "context7Calls", "artifactBytes"], "limits");
	const thinking = string(data.thinking, "thinking") as ThinkingLevel;
	const thinkingLevels: ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
	if (!thinkingLevels.includes(thinking)) throw new ProtocolError("invalid_frame", `invalid thinking level: ${thinking}`);
	if (data.wireApi !== "responses") throw new ProtocolError("invalid_frame", "wireApi must be responses");
	if (data.webProvider !== "anysearch") throw new ProtocolError("invalid_frame", "webProvider must be anysearch");
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
		candidateSchemaSha256: digest(data.candidateSchemaSha256, "candidateSchemaSha256"),
		profileSetSha256: digest(data.profileSetSha256, "profileSetSha256"),
		profiles: parseProfiles(data.profiles),
		networkPolicy: {
			allowedHosts: stringArray(policy.allowedHosts, "networkPolicy.allowedHosts"),
			deniedHosts: stringArray(policy.deniedHosts, "networkPolicy.deniedHosts"),
			forbiddenQueryPatterns: stringArray(policy.forbiddenQueryPatterns, "networkPolicy.forbiddenQueryPatterns"),
		},
		limits: {
			wallTimeSeconds: positiveInteger(limits.wallTimeSeconds, "limits.wallTimeSeconds"),
			providerCalls: positiveInteger(limits.providerCalls, "limits.providerCalls"),
			webCalls: positiveInteger(limits.webCalls, "limits.webCalls"),
			context7Calls: positiveInteger(limits.context7Calls, "limits.context7Calls"),
			artifactBytes: positiveInteger(limits.artifactBytes, "limits.artifactBytes"),
		},
		webProvider: "anysearch",
		context7Enabled: data.context7Enabled,
	};
}
