import assert from "node:assert/strict";
import test from "node:test";
import { PROTOCOL_VERSION, ProtocolError, parseFrame } from "./protocol.js";
import { sha256 } from "./trace.js";

test("parseFrame accepts the explicit responses configuration", () => {
	const candidateSchema = {
		type: "object",
		properties: { value: { type: "number", enum: [1, 0.000001] } },
		required: ["value"],
		additionalProperties: false,
	};
	const candidateSchemaJson = '{"additionalProperties":false,"properties":{"value":{"enum":[1.0,0.000001],"type":"number"}},"required":["value"],"type":"object"}';
	const frame = parseFrame(JSON.stringify({
		type: "initialize",
		requestId: "init-1",
		protocolVersion: PROTOCOL_VERSION,
		campaignId: "campaign-1",
		artifactRoot: "/run/harness",
		baseUrl: "https://provider.example",
		wireApi: "responses",
		model: "model",
		thinking: "max",
		taskId: "synthonbench",
		caseId: "surrogate:1M:kif11",
		seed: 1,
		candidateSchemaJson,
		candidateSchemaSha256: sha256(candidateSchemaJson),
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
			candidatesPerTurn: 16,
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
	}));
	assert.equal(frame.type, "initialize");
	if (frame.type === "initialize") {
		assert.equal(frame.thinking, "max");
		assert.deepEqual(frame.candidateSchema, candidateSchema);
		assert.deepEqual(frame.webSearch.providers, ["parallel-mcp", "exa", "duckduckgo"]);
		assert.equal(frame.guestRuntime.imageRef, "ldm/synthonbench-research:aaaaaaaaaaaa");
		assert.equal(frame.mcpServers[0]?.serverId, "literature");
	}
});

test("parseFrame rejects a candidate schema with a changed digest", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "initialize",
			requestId: "init-1",
			protocolVersion: PROTOCOL_VERSION,
			campaignId: "campaign-1",
			artifactRoot: "/run/harness",
			baseUrl: "https://provider.example",
			wireApi: "responses",
			model: "model",
			thinking: "max",
			taskId: "fixture",
			caseId: "case",
			seed: 1,
			candidateSchemaJson: '{"additionalProperties":false,"properties":{},"type":"object"}',
			candidateSchemaSha256: "b".repeat(64),
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
				candidatesPerTurn: 1,
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

test("parseFrame rejects protocol v6 before sidecar initialization", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "initialize",
			requestId: "init-v6",
			protocolVersion: 6,
			campaignId: "campaign-1",
		})),
		(error: unknown) => error instanceof ProtocolError && error.code === "protocol_mismatch",
	);
});

test("parseFrame requires named secret bootstrap values", () => {
	const frame = parseFrame(JSON.stringify({
		type: "bootstrap_secret",
		requestId: "secret-1",
		protocolVersion: PROTOCOL_VERSION,
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
			protocolVersion: PROTOCOL_VERSION,
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
			protocolVersion: PROTOCOL_VERSION,
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
		protocolVersion: PROTOCOL_VERSION,
		campaignId: "campaign-1",
		validationId: "validation-1",
		accepted: false,
		rejected: [{
			index: 2,
			code: "historical_duplicate",
			message: "The candidate was already evaluated.",
		}],
		requiredReplacements: 1,
	}));
	assert.equal(frame.type, "submission_validation_result");
	if (frame.type === "submission_validation_result") {
		assert.equal(frame.rejected[0]?.code, "historical_duplicate");
	}
});

test("parseFrame rejects inconsistent submission validation results", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "submission_validation_result",
			requestId: "turn-1",
			protocolVersion: PROTOCOL_VERSION,
			campaignId: "campaign-1",
			validationId: "validation-1",
			accepted: true,
			rejected: [{ index: 0, code: "invalid_candidate", message: "Invalid." }],
			requiredReplacements: 1,
		})),
		(error: unknown) => error instanceof ProtocolError && error.code === "invalid_frame",
	);
});
