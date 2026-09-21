import assert from "node:assert/strict";
import test from "node:test";
import { ProtocolError, parseFrame } from "./protocol.js";
import { SIDECAR_RELEASE_VERSION } from "./release.js";
import { sha256 } from "./trace.js";

test("parseFrame accepts the explicit responses configuration", () => {
	const candidateSchema = {
		type: "object",
		properties: { value: { type: "number", enum: [1, 0.000001] } },
		required: ["value"],
		additionalProperties: false,
	};
	const submissionContract = {
		contractId: "candidate_batch",
		toolName: "submit_candidates",
		payloadSchema: {
			type: "object",
			properties: {
				candidates: { type: "array", minItems: 16, maxItems: 16, items: candidateSchema },
			},
			required: ["candidates"],
			additionalProperties: false,
		},
		artifactRules: [],
		maxValidationAttempts: null,
	};
	const submissionContractJson = JSON.stringify(submissionContract);
	const input = {
		type: "initialize",
		requestId: "init-1",
		protocolVersion: SIDECAR_RELEASE_VERSION,
		campaignId: "campaign-1",
		artifactRoot: "/run/harness",
		baseUrl: "https://provider.example",
		wireApi: "responses",
		model: "model",
		thinking: "max",
		providerRequestBody: { temperature: 0.5, reasoning: { effort: "max" } },
		taskId: "synthonbench",
		caseId: "surrogate:1M:kif11",
		seed: 1,
		submissionContractJson,
		submissionContractSha256: sha256(submissionContractJson),
		guestRuntime: {
			imageRef: "ldm/synthonbench-research:aaaaaaaaaaaa",
			recipeSha256: "a".repeat(64),
			rootfsSize: "8G",
			installPolicy: "session_overlay",
		},
		profileSetSha256: "c".repeat(64),
		profiles: [{
			profileId: "target_sar",
			agentsPath: "/resources/AGENTS.md",
			agentsSha256: "a".repeat(64),
			skillDirs: [],
			skillDirSha256: [],
		}],
		toolExtensions: [],
		mcpServers: [{
			serverId: "literature",
			transport: "streamable_http",
			url: "https://mcp.example/mcp",
			headers: {
				Authorization: {
					secretName: "mcp.literature.header.auth",
					secretSource: "secret_env:LITERATURE_TOKEN",
					prefix: "Bearer ",
				},
			},
			tools: ["search"],
			configSha256: "d".repeat(64),
		}],
		networkPolicy: { allowedHosts: ["pubmed.ncbi.nlm.nih.gov"], deniedHosts: ["example.invalid"], forbiddenQueryPatterns: ["benchmark score"] },
		limits: { wallTimeSeconds: 60, toolCallBudgets: { web_search: 4 } },
		webSearch: {
			providers: ["parallel-mcp", "exa", "duckduckgo"],
			fallbackOn: ["transient", "quota", "network", "invalid-response", "unsupported"],
		},
		context7Enabled: true,
	};
	const frame = parseFrame(JSON.stringify(input));
	assert.equal(frame.type, "initialize");
	if (frame.type === "initialize") {
		assert.equal(frame.thinking, "max");
		assert.deepEqual(frame.providerRequestBody, { temperature: 0.5, reasoning: { effort: "max" } });
		assert.deepEqual(frame.submissionContract, submissionContract);
		assert.deepEqual(frame.webSearch.providers, ["parallel-mcp", "exa", "duckduckgo"]);
		assert.equal(frame.guestRuntime.imageRef, "ldm/synthonbench-research:aaaaaaaaaaaa");
		assert.equal(frame.mcpServers[0]?.serverId, "literature");
	}
	const { providerRequestBody: _options, ...withoutOptions } = input;
	assert.equal(parseFrame(JSON.stringify(withoutOptions)).type, "initialize");
	const plugin = parseFrame(JSON.stringify({
		...input,
		solPi: { version: 1, actionFusion: true, observationPack: true, onlineContextCompact: true },
		limits: { ...input.limits, toolCallBudgets: { obs_recall: 3 } },
	}));
	assert.equal(plugin.type, "initialize");
	if (plugin.type === "initialize") assert.equal(plugin.solPi?.actionFusion, true);
});

test("parseFrame rejects a submission contract with a changed digest", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "initialize",
			requestId: "init-1",
			protocolVersion: SIDECAR_RELEASE_VERSION,
			campaignId: "campaign-1",
			artifactRoot: "/run/harness",
			baseUrl: "https://provider.example",
			wireApi: "responses",
			model: "model",
			thinking: "max",
			taskId: "fixture",
			caseId: "case",
			seed: 1,
			submissionContractJson: JSON.stringify({
				contractId: "fixture",
				toolName: "submit_fixture",
				payloadSchema: { type: "object", properties: {}, additionalProperties: false },
				artifactRules: [],
				maxValidationAttempts: null,
			}),
			submissionContractSha256: "b".repeat(64),
			guestRuntime: {
				imageRef: "ldm/fixture-research:aaaaaaaaaaaa",
				recipeSha256: "a".repeat(64),
				rootfsSize: "4G",
				installPolicy: "session_overlay",
			},
			profileSetSha256: "c".repeat(64),
			profiles: [{
				profileId: "chemist",
				agentsPath: "/resources/AGENTS.md",
				agentsSha256: "a".repeat(64),
				skillDirs: [],
				skillDirSha256: [],
			}],
			toolExtensions: [],
			mcpServers: [],
			networkPolicy: { allowedHosts: [], deniedHosts: [], forbiddenQueryPatterns: [] },
			limits: { wallTimeSeconds: 60, toolCallBudgets: {} },
			webSearch: {
				providers: ["parallel-mcp", "exa", "duckduckgo"],
				fallbackOn: ["quota", "network"],
			},
			context7Enabled: true,
		})),
		(error: unknown) => error instanceof ProtocolError && error.code === "invalid_frame",
	);
});

test("parseFrame rejects a different sidecar release version", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "initialize",
			requestId: "init-other-release",
			protocolVersion: "0.0.0",
			campaignId: "campaign-1",
		})),
		(error: unknown) => error instanceof ProtocolError && error.code === "protocol_mismatch",
	);
});

test("parseFrame requires named secret bootstrap values", () => {
	const frame = parseFrame(JSON.stringify({
		type: "bootstrap_secret",
		requestId: "secret-1",
		protocolVersion: SIDECAR_RELEASE_VERSION,
		campaignId: "campaign-1",
		apiKey: "provider-secret",
		namedSecrets: { "mcp.remote.header.auth": "mcp-secret" },
	}));
	assert.equal(frame.type, "bootstrap_secret");
	if (frame.type === "bootstrap_secret") {
		assert.equal(frame.namedSecrets["mcp.remote.header.auth"], "mcp-secret");
	}
});

test("parseFrame rejects path traversal turn identifiers", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "run_turn",
			requestId: "turn-1",
			protocolVersion: SIDECAR_RELEASE_VERSION,
			campaignId: "campaign-1",
			turns: [{
				profileId: "target_sar",
				turnId: "../escape",
				roundIndex: 0,
				historyFromSeq: 0,
				historyToSeq: 0,
				historyDigest: "b".repeat(64),
				inputDigest: "a".repeat(64),
				message: "message",
				forbiddenQueryTerms: [],
			}],
		})),
		(error: unknown) => error instanceof ProtocolError && error.code === "invalid_frame",
	);
});

test("parseFrame rejects unknown fields instead of silently ignoring them", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "close",
			requestId: "close-1",
			protocolVersion: SIDECAR_RELEASE_VERSION,
			campaignId: "campaign-1",
			unexpectedField: true,
		})),
		(error: unknown) => error instanceof ProtocolError && error.code === "invalid_frame",
	);
});

test("parseFrame accepts a consistent submission validation result", () => {
	const frame = parseFrame(JSON.stringify({
		type: "submission_validation_result",
		requestId: "turn-1",
		protocolVersion: SIDECAR_RELEASE_VERSION,
		campaignId: "campaign-1",
		validationId: "validation-1",
		submissionDigest: "d".repeat(64),
		decision: "retry",
		errors: [{
			path: "/candidates/2",
			code: "historical_duplicate",
			message: "The candidate was already evaluated.",
			hint: "Replace this entry.",
		}],
	}));
	assert.equal(frame.type, "submission_validation_result");
	if (frame.type === "submission_validation_result") {
		assert.equal(frame.errors[0]?.code, "historical_duplicate");
	}
});

test("parseFrame rejects inconsistent submission validation results", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "submission_validation_result",
			requestId: "turn-1",
			protocolVersion: SIDECAR_RELEASE_VERSION,
			campaignId: "campaign-1",
			validationId: "validation-1",
			submissionDigest: "d".repeat(64),
			decision: "accept",
			errors: [{
				path: "/candidates/0",
				code: "invalid_candidate",
				message: "Invalid.",
				hint: "Replace this entry.",
			}],
		})),
		(error: unknown) => error instanceof ProtocolError && error.code === "invalid_frame",
	);
});
