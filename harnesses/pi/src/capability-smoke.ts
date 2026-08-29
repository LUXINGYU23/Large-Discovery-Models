import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PiSessionPool } from "./session.js";
import { PROTOCOL_VERSION, type InitializeFrame } from "./protocol.js";
import { canonicalSha256, sha256 } from "./trace.js";

function usage() {
	return {
		input_tokens: 10,
		output_tokens: 10,
		total_tokens: 20,
		input_tokens_details: { cached_tokens: 0 },
		output_tokens_details: { reasoning_tokens: 0 },
	};
}

function toolEvents(index: number, name: string, argumentsJson: string): unknown[] {
	const responseId = `resp_${index}`;
	const item = {
		type: "function_call",
		id: `fc_${index}`,
		call_id: `call_${index}`,
		name,
		arguments: argumentsJson,
		status: "completed",
	};
	return [
		{ type: "response.created", response: { id: responseId, status: "in_progress", output: [] } },
		{ type: "response.output_item.added", output_index: 0, item: { ...item, arguments: "" } },
		{ type: "response.function_call_arguments.done", output_index: 0, arguments: argumentsJson },
		{ type: "response.output_item.done", output_index: 0, item },
		{ type: "response.completed", response: { id: responseId, status: "completed", output: [item], usage: usage() } },
	];
}

function textEvents(index: number, text: string): unknown[] {
	const responseId = `resp_${index}`;
	const item = {
		type: "message",
		id: `msg_${index}`,
		role: "assistant",
		status: "completed",
		content: [{ type: "output_text", text, annotations: [] }],
	};
	return [
		{ type: "response.created", response: { id: responseId, status: "in_progress", output: [] } },
		{ type: "response.output_item.added", output_index: 0, item: { ...item, content: [] } },
		{ type: "response.output_item.done", output_index: 0, item },
		{ type: "response.completed", response: { id: responseId, status: "completed", output: [item], usage: usage() } },
	];
}

function failureEvents(index: number, message: string): unknown[] {
	const response = {
		id: `resp_${index}`,
		status: "failed",
		output: [],
		error: { code: "upstream_error", message },
	};
	return [
		{ type: "response.created", response: { ...response, status: "in_progress", error: null } },
		{ type: "response.failed", response },
	];
}

function writeEvents(response: import("node:http").ServerResponse, events: unknown[]): void {
	response.writeHead(200, { "content-type": "text/event-stream" });
	for (const event of events) response.write(`data: ${JSON.stringify(event)}\n\n`);
	response.end("data: [DONE]\n\n");
}

async function main(): Promise<void> {
	const root = await mkdtemp(join(tmpdir(), "ldm-pi-capability-"));
	const secret = "capability-smoke-secret";
	let call = 0;
	const requestBodies: string[] = [];
	const provider = createServer(async (request, response) => {
		assert.equal(request.url, "/v1/responses");
		assert.equal(request.headers.authorization, `Bearer ${secret}`);
		let body = "";
		for await (const chunk of request) body += chunk.toString();
		requestBodies.push(body);
		call += 1;
		if (call === 1) {
			writeEvents(response, toolEvents(call, "bash", JSON.stringify({
				command: "printf 'sandbox-ok' > proof.txt; cat proof.txt; if command -v wget >/dev/null && wget -qO- -T 2 https://example.com >/dev/null 2>&1; then exit 9; fi; if command -v curl >/dev/null && curl -fsS --max-time 2 https://example.com >/dev/null 2>&1; then exit 9; fi",
			})));
		} else if (call === 2) {
			writeEvents(response, toolEvents(call, "read", JSON.stringify({ path: "proof.txt" })));
		} else if (call === 3) {
			writeEvents(response, toolEvents(call, "submit_candidates", JSON.stringify({
				candidates: [{ reaction_id: "r1", synthon_ids: ["a", "b"] }, { reaction_id: "r2", synthon_ids: ["c", "d"] }],
			})));
		} else if (call === 4) {
			writeEvents(response, toolEvents(call, "submit_candidates", JSON.stringify({
				candidates: [{ reaction_id: "r5", synthon_ids: ["i", "j"] }, { reaction_id: "r2", synthon_ids: ["c", "d"] }],
			})));
		} else if (call === 5) {
			writeEvents(response, textEvents(call, "Corrected submission accepted."));
		} else if (call === 6) {
			writeEvents(response, failureEvents(call, "stream_read_error"));
		} else if (call === 7) {
			writeEvents(response, toolEvents(call, "submit_candidates", JSON.stringify({
				candidates: [{ reaction_id: "r3", synthon_ids: ["e", "f"] }, { reaction_id: "r4", synthon_ids: ["g", "h"] }],
			})));
		} else {
			writeEvents(response, textEvents(call, "Recovered submission accepted."));
		}
	});
	await new Promise<void>((resolve) => provider.listen(0, "127.0.0.1", resolve));
	const address = provider.address();
	assert(address && typeof address !== "string");

	const agentsPath = join(root, "AGENTS.md");
	const agents = "You are the capability-smoke researcher. Use the available tools and submit exactly two candidates.\n";
	await writeFile(agentsPath, agents);
	const profile = {
		profileId: "target_sar",
		agentsPath,
		agentsSha256: sha256(agents),
		skillDirs: [],
		skillDirSha256: [],
		candidatesPerTurn: 2,
	};
	const config: InitializeFrame = {
		type: "initialize",
		requestId: "initialize",
		protocolVersion: PROTOCOL_VERSION,
		campaignId: "capability-campaign",
		artifactRoot: join(root, "harness"),
		baseUrl: `http://127.0.0.1:${address.port}/v1`,
		wireApi: "responses",
		model: "fake-responses-model",
		thinking: "max",
		taskId: "capability",
		caseId: "local-smoke",
		seed: 1,
		candidateSchemaSha256: sha256("candidate-schema"),
		profileSetSha256: canonicalSha256([{
			agentsSha256: profile.agentsSha256,
			candidatesPerTurn: profile.candidatesPerTurn,
			profileId: profile.profileId,
			skillDirSha256: profile.skillDirSha256,
		}]),
		profiles: [profile],
		toolExtensions: [],
		networkPolicy: {
			allowedHosts: ["example.com", "context7.com"],
			deniedHosts: ["github.com"],
			forbiddenQueryPatterns: ["benchmark score"],
		},
		limits: { wallTimeSeconds: 120 },
		webProvider: "anysearch",
		context7Enabled: true,
	};
	const pool = new PiSessionPool(config, secret);
	const validate = async (request: { turnId: string; attemptIndex: number }) => {
		if (request.turnId === "capability_turn" && request.attemptIndex === 1) {
			return {
				accepted: false,
				rejected: [{
					index: 0,
					code: "historical_duplicate",
					message: "Candidate r1 was already evaluated; replace index 0.",
				}],
			};
		}
		return { accepted: true, rejected: [] };
	};
	try {
		await pool.initialize();
		const inputDigest = sha256("capability-turn");
		const [turn] = await pool.runTurns([{
			profileId: "target_sar",
			turnId: "capability_turn",
			roundIndex: 0,
			historyFromSeq: 0,
			historyToSeq: 1,
			historyDigest: sha256("history-0"),
			inputDigest,
			message: "Verify the sandbox with bash and read, then submit exactly two candidates.",
			forbiddenQueryTerms: ["candidate-secret-id"],
		}], validate);
		assert(turn);
		assert.equal(turn.submission.candidates.length, 2);
		assert.equal(turn.usage.providerCalls, 5);
		assert(requestBodies[0]?.includes("capability-smoke researcher"));
		assert(requestBodies[0]?.includes('"effort":"max"'));
		const payloads = requestBodies.map((body) => JSON.parse(body) as { tool_choice?: unknown });
		assert.equal(payloads[0]?.tool_choice, "required");
		const submissionChoice = { type: "function", name: "submit_candidates" };
		assert.equal(payloads[1]?.tool_choice, undefined);
		assert.equal(payloads[2]?.tool_choice, undefined);
		assert.deepEqual(payloads[3]?.tool_choice, submissionChoice);
		assert.equal(payloads[4]?.tool_choice, undefined);

		const recoveryInput = {
			profileId: "target_sar",
			turnId: "capability_recovery_turn",
			roundIndex: 1,
			historyFromSeq: 1,
			historyToSeq: 2,
			historyDigest: sha256("history-1"),
			inputDigest: sha256("capability-recovery-turn"),
			message: "Submit exactly two more candidates.",
			forbiddenQueryTerms: ["candidate-secret-id"],
		};
		const [recovered] = await pool.runTurns([recoveryInput], validate);
		assert(recovered);
		assert.equal(recovered.submission.candidates.length, 2);
		assert.equal(recovered.usage.providerCalls, 3);
		assert.deepEqual((JSON.parse(requestBodies[6] as string) as { tool_choice?: unknown }).tool_choice, submissionChoice);
		const [replayed] = await pool.runTurns([recoveryInput], validate);
		assert.deepEqual(replayed, recovered);
		assert.equal(call, 8);
		await assert.rejects(
			pool.runTurns(
				[{ ...recoveryInput, turnId: "cursor_mismatch", inputDigest: sha256("cursor-mismatch") }],
				validate,
			),
			/history cursor mismatch/,
		);


		const sessionFiles = await readdir(join(root, "harness", "sessions", "target_sar", "pi-session"));
		assert.equal(sessionFiles.filter((name) => name.endsWith(".jsonl")).length, 1);
		const session = await readFile(join(root, "harness", "sessions", "target_sar", "pi-session", sessionFiles[0] as string), "utf8");
		const readResult = session
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line) as { message?: { role?: string; toolName?: string; content?: unknown; isError?: boolean } })
			.find((entry) => entry.message?.role === "toolResult" && entry.message.toolName === "read")
			?.message;
		assert(readResult);
		assert.equal(readResult.isError, false, JSON.stringify(readResult.content));
		assert.match(JSON.stringify(readResult.content), /sandbox-ok/);
		assert.equal(await readFile(join(root, "harness", "sessions", "target_sar", "workspace", "proof.txt"), "utf8"), "sandbox-ok");
		assert.match(session, /submit_candidates/);
		assert.match(session, /historical_duplicate/);
		assert.match(session, /already evaluated; replace index 0/);
		assert.match(session, /previous provider stream ended/);
		assert.match(session, /sandbox-ok/);
		const providerIndex = await readFile(join(root, "harness", "sessions", "target_sar", "turns", "capability_turn", "provider_index.jsonl"), "utf8");
		assert.equal(providerIndex.trim().split("\n").length, 5);
		assert.equal((await readdir(join(root, "harness", "sessions", "target_sar", "pi-agent"))).includes("auth.json"), false);
		const manifest = JSON.parse(await readFile(join(root, "harness", "manifest.json"), "utf8")) as {
			campaignId?: unknown;
			profileSetSha256?: unknown;
			contextWindow?: unknown;
			compaction?: unknown;
			profiles?: Array<{ sessionId?: unknown }>;
		};
		assert.equal(manifest.campaignId, config.campaignId);
		assert.equal(manifest.profileSetSha256, config.profileSetSha256);
		assert.equal(manifest.contextWindow, 262_144);
		assert.deepEqual(manifest.compaction, {
			enabled: true,
			reserveTokens: 16_384,
			keepRecentTokens: 20_000,
		});
		assert.equal(manifest.profiles?.[0]?.sessionId, turn.sessionId);

		const recoveryRoot = join(root, "harness", "sessions", "target_sar", "turns", "capability_recovery_turn");
		const recoveryIndex = await readFile(join(recoveryRoot, "provider_index.jsonl"), "utf8");
		assert.equal(recoveryIndex.trim().split("\n").length, 3);
		const recoveryArtifacts = await readdir(join(recoveryRoot, "provider"));
		assert.equal(recoveryArtifacts.filter((name) => name.endsWith(".request.bin")).length, 3);
		assert.equal(recoveryArtifacts.filter((name) => name.endsWith(".response.bin")).length, 3);
		assert.doesNotMatch(session, new RegExp(secret));
		assert.doesNotMatch(providerIndex, new RegExp(secret));
		process.stdout.write(`${JSON.stringify({ status: "ok", providerCalls: call, sessionEntries: session.trim().split("\n").length })}\n`);
	} finally {
		await pool.close();
		await new Promise<void>((resolve, reject) => provider.close((error) => (error ? reject(error) : resolve())));
		await rm(root, { recursive: true, force: true });
	}
}

await main();
