import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { ProviderProxy } from "./provider-proxy.js";
import { streamSimple } from "@earendil-works/pi-ai/api/openai-responses";

test("provider proxy captures raw stream chunks while redacting credentials", async () => {
	const secret = "unit-test-provider-secret";
	let receivedAuthorization = "";
	let receivedBody = "";
	const upstream = createServer(async (request, response) => {
		receivedAuthorization = request.headers.authorization ?? "";
		for await (const chunk of request) receivedBody += chunk.toString();
		response.writeHead(200, {
			"content-type": "text/event-stream",
			"set-cookie": `provider_session=${secret}`,
		});
		response.write("data: first\n\n");
		response.write(`data: ${secret.slice(0, 8)}`);
		response.write(`${secret.slice(8)}\n\n`);
		response.end("data: [DONE]\n\n");
	});
	await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
	const address = upstream.address();
	assert(address && typeof address !== "string");
	const root = await mkdtemp(join(tmpdir(), "ldm-provider-proxy-"));
	const proxy = new ProviderProxy(`http://127.0.0.1:${address.port}/v1`, secret, "campaign-1");
	try {
		await proxy.start();
		await proxy.beginTurn("target_sar", "session-1", "turn_1", root, async () => true);
		const response = await fetch(`${proxy.baseUrl("target_sar")}/responses`, {
			method: "POST",
			headers: { authorization: "Bearer sidecar-proxy-token", "content-type": "application/json" },
			body: JSON.stringify({ model: "fake", marker: secret }),
		});
		const body = await response.text();
		const summary = await proxy.endTurn("target_sar");
		assert.equal(response.status, 200);
		assert.match(body, new RegExp(secret));
		assert.equal(receivedAuthorization, `Bearer ${secret}`);
		assert.match(receivedBody, new RegExp(secret));
		assert.equal(summary.providerCalls, 1);

		const requestTrace = await readFile(join(root, "provider", "turn_1-provider-1.request.bin"), "utf8");
		const responseTrace = await readFile(join(root, "provider", "turn_1-provider-1.response.bin"), "utf8");
		const metadata = await readFile(join(root, "provider_index.jsonl"), "utf8");
		assert.doesNotMatch(requestTrace, new RegExp(secret));
		assert.doesNotMatch(responseTrace, new RegExp(secret));
		assert.doesNotMatch(metadata, new RegExp(secret));
		assert.match(responseTrace, /\[REDACTED\]/);
		const index = JSON.parse(metadata.trim()) as {
			campaignId: string; profileId: string; sessionId: string;
			response: { chunks: number; headers: Record<string, string> };
		};
		assert.deepEqual([index.campaignId, index.profileId, index.sessionId], ["campaign-1", "target_sar", "session-1"]);
		assert.equal(index.response.chunks, 4);
		assert.equal(index.response.headers["set-cookie"], "[REDACTED]");

		await proxy.beginTurn("target_sar", "session-1", "turn_1", root, async () => true);
		const recovered = await fetch(proxy.baseUrl("target_sar") + "/responses", {
			method: "POST",
			body: "{}",
		});
		await recovered.text();
		const recoveredSummary = await proxy.endTurn("target_sar");
		assert.equal(recoveredSummary.providerCalls, 2);
		assert.match(await readFile(join(root, "provider", "turn_1-provider-1.request.bin"), "utf8"), /fake/);
		assert.equal(await readFile(join(root, "provider", "turn_1-provider-2.request.bin"), "utf8"), "{}");
	} finally {
		await proxy.close();
		await new Promise<void>((resolve, reject) => upstream.close((error) => (error ? reject(error) : resolve())));
	}
});

test("provider proxy rejects credentials embedded in the base URL", () => {
	assert.throws(
		() => new ProviderProxy("https://user:secret@provider.example/v1", "secret", "campaign-1"),
		/base URL must not contain credentials/,
	);
});

test("provider proxy permits only the authorized request and never reuses its ID", async () => {
	let upstreamCalls = 0;
	const upstream = createServer((_request, response) => {
		upstreamCalls += 1;
		response.end("ok");
	});
	await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
	const address = upstream.address();
	assert(address && typeof address !== "string");
	const root = await mkdtemp(join(tmpdir(), "ldm-provider-authorization-"));
	const proxy = new ProviderProxy(`http://127.0.0.1:${address.port}/v1`, "fixture", "campaign");
	let balance = 1;
	const requests: Array<{ profileId: string; providerRequestId: string; requestDigest: string }> = [];
	const authorize = async (request: typeof requests[number]) => {
		requests.push(request);
		if (balance === 0) return false;
		balance -= 1;
		return true;
	};
	try {
		await proxy.start();
		await proxy.beginTurn("a", "session-a", "turn-a", join(root, "a"), authorize);
		await proxy.beginTurn("b", "session-b", "turn-b", join(root, "b"), authorize);
		const responses = await Promise.all(["a", "b"].map((profile) => fetch(proxy.baseUrl(profile) + "/responses", {
			method: "POST", body: "{}",
		})));
		assert.deepEqual(responses.map((response) => response.status).sort(), [200, 403]);
		await Promise.all(responses.map((response) => response.text()));
		assert.equal(upstreamCalls, 1);
		const a = await proxy.endTurn("a");
		const b = await proxy.endTurn("b");
		assert.equal(a.providerCalls + b.providerCalls, 1);
		assert.equal(requests.length, 2);
		assert(requests.every((request) => /^[a-f0-9]{64}$/.test(request.requestDigest)));
		const denied = a.providerCalls === 0 ? "a" : "b";
		await proxy.beginTurn(denied, `session-${denied}`, `turn-${denied}`, join(root, denied), authorize);
		const retry = await fetch(proxy.baseUrl(denied) + "/responses", { method: "POST", body: "{}" });
		assert.equal(retry.status, 403);
		await retry.text();
		assert.equal((await proxy.endTurn(denied)).providerCalls, 0);
		assert.equal(requests.at(-1)?.providerRequestId, `turn-${denied}-provider-2`);
		assert.equal(upstreamCalls, 1);
		let release!: (authorized: boolean) => void;
		let ready!: () => void;
		const authorizationReady = new Promise<void>((resolve) => { ready = resolve; });
		await proxy.beginTurn("late", "session-late", "turn-late", join(root, "late"), async () => {
			ready();
			return new Promise<boolean>((resolve) => { release = resolve; });
		});
		const lateRequest = fetch(proxy.baseUrl("late") + "/responses", { method: "POST", body: "{}" });
		await authorizationReady;
		await proxy.endTurn("late");
		release(true);
		const lateResponse = await lateRequest;
		assert.equal(lateResponse.status, 409);
		await lateResponse.text();
		assert.equal(upstreamCalls, 1);
	} finally {
		await proxy.close();
		await new Promise<void>((resolve) => upstream.close(() => resolve()));
	}
});

test("provider proxy applies and traces the declared generation settings", async () => {
	let received: Record<string, unknown> = {};
	const upstream = createServer(async (request, response) => {
		let body = "";
		for await (const chunk of request) body += chunk.toString();
		received = JSON.parse(body);
		assert.equal(Number(request.headers["content-length"]), Buffer.byteLength(body));
		response.end("ok");
	});
	await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
	const address = upstream.address();
	assert(address && typeof address !== "string");
	const root = await mkdtemp(join(tmpdir(), "ldm-provider-settings-"));
	const settings = { temperature: 0.4, max_output_tokens: 1024, reasoning: { effort: "high" }, top_p: 0.9, tool_choice: "auto" };
	const proxy = new ProviderProxy(`http://127.0.0.1:${address.port}/v1`, "unit-test-token", "campaign", settings);
	try {
		await proxy.start();
		await proxy.beginTurn("research", "session", "turn", root, async () => true);
		const response = await fetch(proxy.baseUrl("research") + "/responses", {
			method: "POST", body: JSON.stringify({ model: "fixture", input: [], temperature: 1, tool_choice: "required" }),
		});
		assert.equal(await response.text(), "ok");
		await proxy.endTurn("research");
		assert.deepEqual(received, { model: "fixture", input: [], ...settings });
		assert.deepEqual(JSON.parse(await readFile(join(root, "provider/turn-provider-1.request.bin"), "utf8")), received);
	} finally {
		await proxy.close();
		await new Promise<void>((resolve) => upstream.close(() => resolve()));
	}
	assert.throws(() => new ProviderProxy("https://provider.example/v1", "fixture", "campaign", { tools: [] }), /protocol fields/);
});

for (const [thinking, effort] of [["off", "none"], ["high", "high"]] as const) {
	test(`actual Pi SDK ${thinking} request preserves the explicit Direct reasoning effort`, async () => {
		let received: Record<string, unknown> = {};
		const upstream = createServer(async (request, response) => {
			let body = "";
			for await (const chunk of request) body += chunk.toString();
			received = JSON.parse(body);
			response.writeHead(200, { "content-type": "text/event-stream" });
			response.end(`data: ${JSON.stringify({ type: "response.completed", response: {
				id: "fixture", status: "completed", output: [], usage: { input_tokens: 1, output_tokens: 0 },
			} })}\n\n`);
		});
		await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
		const address = upstream.address();
		assert(address && typeof address !== "string");
		const root = await mkdtemp(join(tmpdir(), "ldm-reasoning-parity-"));
		const proxy = new ProviderProxy(`http://127.0.0.1:${address.port}/v1`, "fixture", "campaign", {
			reasoning: { effort },
		});
		try {
			await proxy.start();
			await proxy.beginTurn("research", "session", "turn", root, async () => true);
			const stream = streamSimple({
				id: "fixture", name: "fixture", api: "openai-responses", provider: "ldm-harness-research",
				baseUrl: proxy.baseUrl("research"), reasoning: true, thinkingLevelMap: { off: "none", high: "high" },
				input: ["text"], contextWindow: 4096, maxTokens: 128,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			}, { messages: [{ role: "user", content: "OK", timestamp: 0 }] }, {
				apiKey: "sidecar-proxy-token", ...(thinking === "off" ? {} : { reasoning: thinking }), maxRetries: 0,
			});
			const result = await stream.result();
			assert.notEqual(result.stopReason, "error", result.errorMessage);
			assert.deepEqual(received.reasoning, { effort });
			assert.equal((await proxy.endTurn("research")).providerCalls, 1);
		} finally {
			await proxy.close();
			await new Promise<void>((resolve) => upstream.close(() => resolve()));
		}
	});
}
