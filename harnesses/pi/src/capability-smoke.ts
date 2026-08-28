import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PiSessionPool } from "./session.js";
import type { InitializeFrame } from "./protocol.js";
import { sha256 } from "./trace.js";

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
			writeEvents(response, textEvents(call, "Submitted."));
		} else if (call === 5) {
			writeEvents(response, failureEvents(call, "stream_read_error"));
		} else if (call === 6) {
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
	await writeFile(agentsPath, "You are the capability-smoke researcher. Use the available tools and submit exactly two candidates.\n");
	const config: InitializeFrame = {
		type: "initialize",
		requestId: "initialize",
		protocolVersion: 1,
		artifactRoot: join(root, "harness"),
		baseUrl: `http://127.0.0.1:${address.port}/v1`,
		wireApi: "responses",
		model: "fake-responses-model",
		thinking: "max",
		profiles: [{ profileId: "target_sar", agentsPath, skillDirs: [], candidatesPerTurn: 2 }],
		networkPolicy: {
			allowedHosts: ["example.com", "context7.com"],
			deniedHosts: ["github.com"],
			forbiddenQueryPatterns: ["benchmark score"],
		},
		limits: { wallTimeSeconds: 120, providerCalls: 6, webCalls: 2, context7Calls: 2, artifactBytes: 5_000_000 },
		webProvider: "anysearch",
		context7Enabled: true,
	};
	const pool = new PiSessionPool(config, secret);
	try {
		await pool.initialize();
		const inputDigest = sha256("capability-turn");
		const [turn] = await pool.runTurns([{
			profileId: "target_sar",
			turnId: "capability_turn",
			inputDigest,
			message: "Verify the sandbox with bash and read, then submit exactly two candidates.",
			forbiddenQueryTerms: ["candidate-secret-id"],
		}]);
		assert(turn);
		assert.equal(turn.submission.candidates.length, 2);
		assert.equal(turn.usage.providerCalls, 4);
		assert(requestBodies[0]?.includes("capability-smoke researcher"));
		assert(requestBodies[0]?.includes('"effort":"max"'));
		const payloads = requestBodies.map((body) => JSON.parse(body) as { tool_choice?: unknown });
		assert.equal(payloads[0]?.tool_choice, "required");
		const submissionChoice = { type: "function", name: "submit_candidates" };
		assert.deepEqual(payloads[1]?.tool_choice, submissionChoice);
		assert.deepEqual(payloads[2]?.tool_choice, submissionChoice);
		assert.equal(payloads[3]?.tool_choice, undefined);

		const recoveryInput = {
			profileId: "target_sar",
			turnId: "capability_recovery_turn",
			inputDigest: sha256("capability-recovery-turn"),
			message: "Submit exactly two more candidates.",
			forbiddenQueryTerms: ["candidate-secret-id"],
		};
		const [recovered] = await pool.runTurns([recoveryInput]);
		assert(recovered);
		assert.equal(recovered.submission.candidates.length, 2);
		assert.equal(recovered.usage.providerCalls, 3);
		assert.deepEqual((JSON.parse(requestBodies[5] as string) as { tool_choice?: unknown }).tool_choice, submissionChoice);


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
		assert.match(session, /previous provider stream ended/);
		assert.match(session, /sandbox-ok/);
		const providerIndex = await readFile(join(root, "harness", "sessions", "target_sar", "turns", "capability_turn", "provider_index.jsonl"), "utf8");
		assert.equal(providerIndex.trim().split("\n").length, 4);

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
