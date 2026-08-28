export const PROTOCOL_VERSION = 1;

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

export interface HarnessProfileConfig {
	profileId: string;
	agentsPath: string;
	skillDirs: string[];
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

export interface BootstrapSecretFrame {
	type: "bootstrap_secret";
	requestId: string;
	apiKey: string;
}

export interface InitializeFrame {
	type: "initialize";
	requestId: string;
	protocolVersion: number;
	artifactRoot: string;
	baseUrl: string;
	wireApi: "responses";
	model: string;
	thinking: ThinkingLevel;
	profiles: HarnessProfileConfig[];
	networkPolicy: NetworkPolicy;
	limits: HarnessLimits;
	webProvider: "anysearch";
	context7Enabled: boolean;
}

export interface SessionTurnInput {
	profileId: string;
	turnId: string;
	inputDigest: string;
	message: string;
	forbiddenQueryTerms: string[];
}

export interface RunTurnFrame {
	type: "run_turn";
	requestId: string;
	turns: SessionTurnInput[];
}

export interface CloseFrame {
	type: "close";
	requestId: string;
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

function string(value: unknown, name: string): string {
	if (typeof value !== "string" || value.length === 0) {
		throw new ProtocolError("invalid_frame", `${name} must be a non-empty string`);
	}
	return value;
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

function parseProfiles(value: unknown): HarnessProfileConfig[] {
	if (!Array.isArray(value) || value.length === 0) {
		throw new ProtocolError("invalid_frame", "profiles must be a non-empty array");
	}
	const profiles = value.map((item, index) => {
		const data = record(item, `profiles[${index}]`);
		const profileId = string(data.profileId, `profiles[${index}].profileId`);
		if (!/^[a-z][a-z0-9_]*$/.test(profileId)) {
			throw new ProtocolError("invalid_frame", `invalid profileId: ${profileId}`);
		}
		return {
			profileId,
			agentsPath: string(data.agentsPath, `profiles[${index}].agentsPath`),
			skillDirs: stringArray(data.skillDirs, `profiles[${index}].skillDirs`),
			candidatesPerTurn: positiveInteger(
				data.candidatesPerTurn,
				`profiles[${index}].candidatesPerTurn`,
			),
		};
	});
	if (new Set(profiles.map((profile) => profile.profileId)).size !== profiles.length) {
		throw new ProtocolError("invalid_frame", "profileId values must be unique");
	}
	return profiles;
}

export function parseFrame(line: string): InputFrame {
	let value: unknown;
	try {
		value = JSON.parse(line);
	} catch {
		throw new ProtocolError("invalid_json", "frame is not valid JSON");
	}
	const data = record(value, "frame");
	const type = string(data.type, "type");
	const requestId = string(data.requestId, "requestId");

	if (type === "bootstrap_secret") {
		return { type, requestId, apiKey: string(data.apiKey, "apiKey") };
	}
	if (type === "close") return { type, requestId };
	if (type === "run_turn") {
		if (!Array.isArray(data.turns) || data.turns.length === 0) {
			throw new ProtocolError("invalid_frame", "turns must be a non-empty array");
		}
		return {
			type,
			requestId,
			turns: data.turns.map((item, index) => {
				const turn = record(item, `turns[${index}]`);
				const turnId = string(turn.turnId, `turns[${index}].turnId`);
				const inputDigest = string(turn.inputDigest, `turns[${index}].inputDigest`);
				if (!/^[A-Za-z0-9_-]+$/.test(turnId)) {
					throw new ProtocolError("invalid_frame", `invalid turnId: ${turnId}`);
				}
				if (!/^[a-f0-9]{64}$/.test(inputDigest)) {
					throw new ProtocolError("invalid_frame", "inputDigest must be a lowercase SHA-256 digest");
				}
				return {
					profileId: string(turn.profileId, `turns[${index}].profileId`),
					turnId,
					inputDigest,
					message: string(turn.message, `turns[${index}].message`),
					forbiddenQueryTerms: stringArray(
						turn.forbiddenQueryTerms,
						`turns[${index}].forbiddenQueryTerms`,
					),
				};
			}),
		};
	}
	if (type !== "initialize") throw new ProtocolError("unknown_frame", `unknown frame type: ${type}`);

	const policy = record(data.networkPolicy, "networkPolicy");
	const limits = record(data.limits, "limits");
	const thinking = string(data.thinking, "thinking") as ThinkingLevel;
	const thinkingLevels: ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
	if (!thinkingLevels.includes(thinking)) throw new ProtocolError("invalid_frame", `invalid thinking level: ${thinking}`);
	if (data.protocolVersion !== PROTOCOL_VERSION) {
		throw new ProtocolError("protocol_mismatch", `expected protocol ${PROTOCOL_VERSION}`);
	}
	if (data.wireApi !== "responses") {
		throw new ProtocolError("invalid_frame", "wireApi must be responses");
	}
	if (data.webProvider !== "anysearch") {
		throw new ProtocolError("invalid_frame", "webProvider must be anysearch");
	}
	if (typeof data.context7Enabled !== "boolean") {
		throw new ProtocolError("invalid_frame", "context7Enabled must be boolean");
	}

	return {
		type,
		requestId,
		protocolVersion: PROTOCOL_VERSION,
		artifactRoot: string(data.artifactRoot, "artifactRoot"),
		baseUrl: string(data.baseUrl, "baseUrl"),
		wireApi: "responses",
		model: string(data.model, "model"),
		thinking,
		profiles: parseProfiles(data.profiles),
		networkPolicy: {
			allowedHosts: stringArray(policy.allowedHosts, "networkPolicy.allowedHosts"),
			deniedHosts: stringArray(policy.deniedHosts, "networkPolicy.deniedHosts"),
			forbiddenQueryPatterns: stringArray(
				policy.forbiddenQueryPatterns,
				"networkPolicy.forbiddenQueryPatterns",
			),
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
