import assert from "node:assert/strict";
import test from "node:test";
import { DeepSeekWebTools } from "./deepseek-web.js";
import type { NetworkPolicy } from "./protocol.js";

const policy: NetworkPolicy = { allowedHosts: [], deniedHosts: [], forbiddenQueryPatterns: [] };
const source = {
	title: "Paper", url: "https://nature.com/paper", snippet: "Short", summary: "Long summary",
	content: "Full page", rerank_score: 2.4, authority: 3, site_name: "Nature", site_icon: null, date: null,
};
function execute(client: DeepSeekWebTools, name: string, input: Record<string, unknown>, signal?: AbortSignal,
	networkPolicy = policy) {
	return client.toolDefinitions(networkPolicy).find((tool) => tool.name === name)!
		.execute("call", input, signal, undefined, {} as never);
}

test("Search sends the official contract and exposes only allowed sources", async (t) => {
	const calls: Array<{ url: string; init?: RequestInit }> = [];
	t.mock.method(globalThis, "fetch", async (url: string, init?: RequestInit) => {
		calls.push({ url, ...(init ? { init } : {}) });
		return Response.json({ results: [source, { ...source, url: "https://blocked.nature.com/secret" },
			{ ...source, url: "https://other.org/paper" }] });
	});
	const result = await execute(new DeepSeekWebTools("search-secret"), "web_search", {
		query: 'reaction "design" site:nature.com', count: 3, ori_question: "How does it work?", need_content: true,
	}, undefined, { ...policy, allowedHosts: ["nature.com"], deniedHosts: ["blocked.nature.com"] });
	assert.equal(calls[0]?.url, "https://api.deepseek.com/v1/tools/search");
	assert.equal(calls[0]?.init?.method, "POST");
	assert.equal(calls[0]?.init?.redirect, "error");
	assert.deepEqual(calls[0]?.init?.headers, { Authorization: "Bearer search-secret", "Content-Type": "application/json" });
	assert.deepEqual(JSON.parse(calls[0]?.init?.body as string), {
		query: 'reaction "design" site:nature.com', count: 3, ori_question: "How does it work?", need_content: true,
	});
	assert.deepEqual(JSON.parse((result.content[0] as { text: string }).text), { results: [source] });
	assert.deepEqual(result.details, { provider: "deepseek" });
});

test("Fetch uses the API and returns Markdown without local page fetching", async (t) => {
	const calls: string[] = [];
	t.mock.method(globalThis, "fetch", async (url: string, init: RequestInit) => {
		calls.push(url);
		assert.deepEqual(JSON.parse(init.body as string), { url: source.url });
		return Response.json({ title: "Paper", content: "# Paper\nMarkdown" });
	});
	const result = await execute(new DeepSeekWebTools("key"), "fetch_content", { url: source.url });
	assert.deepEqual(calls, ["https://api.deepseek.com/v1/tools/fetch"]);
	assert.match(JSON.stringify(result.content), /Markdown/);
});

test("invalid requests, missing keys and forbidden domains never reach the API", async (t) => {
	const mock = t.mock.method(globalThis, "fetch", async () => { throw new Error("must not call"); });
	const client = new DeepSeekWebTools("key");
	for (const input of [{ query: " " }, { query: "x".repeat(129) }, { query: "a", count: 51 },
		{ query: "a", count: 1.5 }, { query: "a", provider: "auto" }, { query: "a", ori_question: " " }]) {
		await assert.rejects(execute(client, "web_search", input));
	}
	for (const url of ["file:///etc/passwd", "https://user:pass@nature.com", "relative", "https://nature.com/" + "a".repeat(8192)]) {
		await assert.rejects(execute(client, "fetch_content", { url }));
	}
	await assert.rejects(execute(client, "fetch_content", { url: source.url }, undefined,
		{ ...policy, deniedHosts: ["nature.com"] }));
	await assert.rejects(execute(new DeepSeekWebTools(undefined), "web_search", { query: "a" }), /DEEPSEEK_API_KEY/);
	assert.equal(mock.mock.callCount(), 0);
});

test("Unicode queries use code points and empty search results are valid", async (t) => {
	t.mock.method(globalThis, "fetch", async () => Response.json({ results: [] }));
	const result = await execute(new DeepSeekWebTools("key"), "web_search", { query: "🧪".repeat(128) });
	assert.deepEqual(JSON.parse((result.content[0] as { text: string }).text), { results: [] });
});

test("transient HTTP errors retry the same API and honor cancellation", async (t) => {
	let count = 0;
	t.mock.method(globalThis, "fetch", async () => {
		count++;
		return count < 3 ? new Response("busy", { status: count === 1 ? 429 : 503, headers: { "retry-after": "0" } })
			: Response.json({ results: [] });
	});
	const client = new DeepSeekWebTools("key");
	await execute(client, "web_search", { query: "a" });
	assert.equal(count, 3);
	await assert.rejects(execute(client, "web_search", { query: "b" }, AbortSignal.abort()), /abort/i);
	assert.equal(count, 3);
});

test("permanent errors are not retried or exposed with provider secrets", async (t) => {
	for (const status of [400, 401, 402, 403, 404]) {
		const mock = t.mock.method(globalThis, "fetch", async () => new Response("secret-key", { status }));
		await assert.rejects(execute(new DeepSeekWebTools("secret-key"), "web_search", { query: "a" }),
			(error: Error) => error.message.includes(`HTTP ${status}`) && !error.message.includes("secret-key"));
		assert.equal(mock.mock.callCount(), 1);
		mock.mock.restore();
	}
});

test("malformed success responses and empty fetch content are rejected", async (t) => {
	for (const body of [{}, { results: [{}] }, { results: "bad" }]) {
		const mock = t.mock.method(globalThis, "fetch", async () => Response.json(body));
		await assert.rejects(execute(new DeepSeekWebTools("key"), "web_search", { query: "a" }), /invalid response/);
		mock.mock.restore();
	}
	t.mock.method(globalThis, "fetch", async () => Response.json({ title: "", content: " " }));
	await assert.rejects(execute(new DeepSeekWebTools("key"), "fetch_content", { url: source.url }), /no page content/);
});
