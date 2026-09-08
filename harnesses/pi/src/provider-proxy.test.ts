import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { ProviderProxy } from "./provider-proxy.js";

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
		await proxy.beginTurn("target_sar", "session-1", "turn_1", root);
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

		await proxy.beginTurn("target_sar", "session-1", "turn_1", root);
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
