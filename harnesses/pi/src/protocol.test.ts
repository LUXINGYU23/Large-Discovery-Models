import assert from "node:assert/strict";
import test from "node:test";
import { ProtocolError, parseFrame } from "./protocol.js";

test("parseFrame accepts the explicit responses configuration", () => {
	const frame = parseFrame(JSON.stringify({
		type: "initialize",
		requestId: "init-1",
		protocolVersion: 1,
		artifactRoot: "/run/harness",
		baseUrl: "https://provider.example",
		wireApi: "responses",
		model: "model",
		thinking: "max",
		profiles: [{ profileId: "target_sar", agentsPath: "/resources/AGENTS.md", skillDirs: [], candidatesPerTurn: 16 }],
		networkPolicy: { allowedHosts: ["pubmed.ncbi.nlm.nih.gov"], deniedHosts: ["example.invalid"], forbiddenQueryPatterns: ["benchmark score"] },
		limits: { wallTimeSeconds: 60, providerCalls: 8, webCalls: 4, context7Calls: 2, artifactBytes: 1_000_000 },
		webProvider: "anysearch",
		context7Enabled: true,
	}));
	assert.equal(frame.type, "initialize");
	if (frame.type === "initialize") assert.equal(frame.thinking, "max");
});

test("parseFrame rejects path traversal turn identifiers", () => {
	assert.throws(
		() => parseFrame(JSON.stringify({
			type: "run_turn",
			requestId: "turn-1",
			turns: [{
				profileId: "target_sar",
				turnId: "../escape",
				inputDigest: "a".repeat(64),
				message: "message",
				forbiddenQueryTerms: [],
			}],
		})),
		(error: unknown) => error instanceof ProtocolError && error.code === "invalid_frame",
	);
});
