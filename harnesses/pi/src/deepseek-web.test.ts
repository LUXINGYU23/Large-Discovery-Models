import assert from "node:assert/strict";
import test from "node:test";
import { validateToolCall } from "@earendil-works/pi-ai";
import { DeepSeekWebTools } from "./deepseek-web.js";
import type { NetworkPolicy } from "./protocol.js";
import { EXTERNAL_WEB_CONTENT_NOTICE as notice } from "./vendor/deepseek-web/contract.js";

const policy: NetworkPolicy = { allowedHosts: [], deniedHosts: [], forbiddenQueryPatterns: [] };
const source = { title: "Paper", url: "https://nature.com/paper", snippet: "Short", date: "2026-10-18",
	summary: "Not for model", content: "Full page excluded", rerank_score: 2.4, authority: 3 };
function execute(name: string, input: Record<string, unknown>, options: {
	signal?: AbortSignal; policy?: NetworkPolicy; question?: string; key?: string;
} = {}) {
	return new DeepSeekWebTools(options.key ?? "search-secret")
		.toolDefinitions(options.policy ?? policy, () => options.question).find((tool) => tool.name === name)!
		.execute("call", input, options.signal, undefined, {} as never);
}
function text(result: Awaited<ReturnType<typeof execute>>) { return (result.content[0] as { text: string }).text; }
function metadata(result: Awaited<ReturnType<typeof execute>>) {
	const { requests, ...data } = result.details as Record<string, unknown>;
	assert(Array.isArray(requests));
	assert(requests.every((item) => item.dispatched && item.queueMs >= 0 && item.durationMs >= 0));
	return data;
}

test("official request projection and rendered sources omit API summaries and full content", async (t) => {
	t.mock.method(globalThis, "fetch", async (url: string, init: RequestInit) => {
		assert.equal(url, "https://api.deepseek.com/v1/tools/search");
		assert.equal(init.method, "POST");
		assert.equal(init.redirect, "error");
		assert.equal((init.headers as Record<string, string>).authorization, "Bearer search-secret");
		assert.deepEqual(JSON.parse(init.body as string), {
			query: "reaction design", count: 8, need_content: false, ori_question: "How does it work?",
		});
		return Response.json({ results: [source, { ...source, url: "https://blocked.nature.com/secret" },
			{ ...source, url: "https://other.org/paper" }] });
	});
	const result = await execute("web_search", { queries: ["reaction design"] }, {
		question: "  How\n does it work? ", policy: { ...policy, allowedHosts: ["nature.com"], deniedHosts: ["blocked.nature.com"] },
	});
	assert.equal(text(result), `${notice}\n\nSources:\n- [Paper](https://nature.com/paper) — Short (2026-10-18)\n\nCite the relevant URLs above as markdown links in your answer.`);
	assert.deepEqual(metadata(result), { sources: [{ title: "Paper", url: source.url, snippet: "Short", publishedAt: source.date }], truncated: false });
});

test("queries run concurrently, exact duplicates collapse, merge is round-robin capped at eight", async (t) => {
	let calls = 0;
	let release!: () => void;
	const bothStarted = new Promise<void>((resolve) => { release = resolve; });
	t.mock.method(globalThis, "fetch", async (_url: string, init: RequestInit) => {
		const body = JSON.parse(init.body as string);
		assert.equal(body.ori_question, undefined);
		if (++calls === 2) release();
		await bothStarted;
		return Response.json({ results: Array.from({ length: 8 }, (_, i) => ({ ...source,
			url: i === 0 ? "https://nature.com/shared" : `https://nature.com/${body.query}/${i}` })) });
	});
	const result = await execute("web_search", { queries: ["a", "b", "a"] });
	assert.equal(calls, 2);
	assert.deepEqual((result.details as { sources: { url: string }[] }).sources.map((item) => item.url),
		["shared", "a/1", "b/1", "a/2", "b/2", "a/3", "b/3", "a/4"].map((path) => `https://nature.com/${path}`));
	assert.match(text(result), /Showing the first 8 sources/);
});

test("a failed query cancels siblings, waits for cleanup and returns no partial result", async (t) => {
	let cancelled = false;
	t.mock.method(globalThis, "fetch", async (_url: string, init: RequestInit) => {
		if (JSON.parse(init.body as string).query === "bad") return new Response("secret", { status: 429 });
		return await new Promise<Response>((_resolve, reject) => {
			init.signal!.addEventListener("abort", () => { cancelled = true; reject(init.signal!.reason); }, { once: true });
		});
	});
	await assert.rejects(execute("web_search", { queries: ["slow", "bad"] }), /HTTP 429/);
	assert.equal(cancelled, true);
});

test("schema and query-count errors never reach API; bound precedes deduplication", async (t) => {
	const tools = new DeepSeekWebTools("search-secret").toolDefinitions(policy);
	const mock = t.mock.method(globalThis, "fetch", async () => { throw new Error("must not call"); });
	for (const input of [{ query: "a" }, { queries: [] }, { queries: [" "] }, { queries: [1] },
		{ queries: Array(5).fill("same") }, { queries: ["a"], need_content: true }]) {
		await assert.rejects(execute("web_search", input));
	}
	for (const queries of [[], Array(5).fill("same"), [""], [" \n "]]) {
		assert.throws(() => validateToolCall(tools, { type: "toolCall", id: "invalid", name: "web_search", arguments: { queries } }));
	}
	for (const count of [1, 4]) {
		assert.doesNotThrow(() => validateToolCall(tools, { type: "toolCall", id: "valid", name: "web_search",
			arguments: { queries: Array(count).fill("research") } }));
	}
	for (const url of [" ", "file:///etc/passwd", "https://user:pass@nature.com", "relative"]) {
		await assert.rejects(execute("web_fetch", { url }));
	}
	await assert.rejects(execute("web_fetch", { url: source.url }, { policy: { ...policy, deniedHosts: ["nature.com"] } }));
	await assert.rejects(execute("web_search", { queries: ["a"] }, { key: "" }), /DEEPSEEK_API_KEY/);
	await assert.rejects(execute("web_search", { queries: ["a"] }, { signal: AbortSignal.abort() }), /abort/i);
	assert.equal(mock.mock.callCount(), 0);
});

test("fetch URL length is bounded in the schema and runtime before dispatch", async (t) => {
	const tools = new DeepSeekWebTools("search-secret").toolDefinitions(policy);
	const urls: string[] = [];
	t.mock.method(globalThis, "fetch", async (_url: string, init: RequestInit) => {
		urls.push(JSON.parse(init.body as string).url);
		return Response.json({ content: "Paper" });
	});
	for (const character of ["a", "研", "🧪"]) {
		const prefix = "https://example.org/";
		const valid = prefix + character.repeat(8192 - prefix.length);
		const invalid = valid + character;
		assert.throws(() => validateToolCall(tools, { type: "toolCall", id: "overlong", name: "web_fetch", arguments: { url: invalid } }));
		await assert.rejects(execute("web_fetch", { url: invalid }), /8192 characters/);
		assert.doesNotThrow(() => validateToolCall(tools, { type: "toolCall", id: "valid", name: "web_fetch", arguments: { url: valid } }));
		await execute("web_fetch", { url: valid });
		assert.equal(urls.at(-1), valid);
	}
	assert.equal(urls.length, 3);
});

test("original question is Unicode-bounded and blank context is omitted", async (t) => {
	const bodies: Record<string, unknown>[] = [];
	t.mock.method(globalThis, "fetch", async (_url: string, init: RequestInit) => {
		bodies.push(JSON.parse(init.body as string));
		return Response.json({ results: [] });
	});
	const result = await execute("web_search", { queries: ["a"] }, { question: "🧪".repeat(129) });
	await execute("web_search", { queries: ["a"] }, { question: " \n " });
	assert.equal(bodies[0]!.ori_question, "🧪".repeat(128));
	assert.equal(bodies[1]!.ori_question, undefined);
	assert.equal(text(result), `${notice}\n\nNo results found.\n\nCite the relevant URLs above as markdown links in your answer.`);
});

test("queries enforce the Unicode character limit before dispatch", async (t) => {
	const tools = new DeepSeekWebTools("search-secret").toolDefinitions(policy);
	const queries: string[] = [];
	t.mock.method(globalThis, "fetch", async (_url: string, init: RequestInit) => {
		queries.push(JSON.parse(init.body as string).query);
		return Response.json({ results: [] });
	});
	for (const character of ["a", "研", "🧪"]) {
		assert.throws(() => validateToolCall(tools, { type: "toolCall", id: "overlong", name: "web_search",
			arguments: { queries: [character.repeat(129)] } }));
		assert.doesNotThrow(() => validateToolCall(tools, { type: "toolCall", id: "valid", name: "web_search",
			arguments: { queries: [character.repeat(128)] } }));
		await assert.rejects(execute("web_search", { queries: [character.repeat(129)] }), /128 Unicode characters/);
		await execute("web_search", { queries: [character.repeat(128)] });
	}
	assert.deepEqual(queries, ["a", "研", "🧪"].map((character) => character.repeat(128)));
});

test("HTTP and network failures are classified, never retried and do not disclose response bodies", async (t) => {
	for (const [name, status, code] of [["web_search", 401, "WEB_AUTH"], ["web_search", 503, "WEB_HTTP_ERROR"],
		["web_fetch", 403, "WEB_POLICY_DENIED"]] as const) {
		const mock = t.mock.method(globalThis, "fetch", async () => new Response("search-secret", { status }));
		await assert.rejects(execute(name, name === "web_search" ? { queries: ["a"] } : { url: source.url }),
			(error: Error) => {
				const failure = JSON.parse(error.message);
				assert.equal(failure.code, code);
				assert.equal(failure.httpStatus, status);
				assert.equal(failure.requests[0].errorCode, code);
				return !error.message.includes("search-secret");
			});
		assert.equal(mock.mock.callCount(), 1);
		mock.mock.restore();
	}
	const mock = t.mock.method(globalThis, "fetch", async () => { throw new Error("search-secret"); });
	await assert.rejects(execute("web_search", { queries: ["a"] }), /network request failed/);
	assert.equal(mock.mock.callCount(), 1);
});

test("malformed payloads fail while optional API fields are not required", async (t) => {
	for (const body of [{}, { results: [{}] }, { results: "bad" }]) {
		const mock = t.mock.method(globalThis, "fetch", async () => Response.json(body));
		await assert.rejects(execute("web_search", { queries: ["a"] }), /invalid response/);
		mock.mock.restore();
	}
	t.mock.method(globalThis, "fetch", async () => Response.json({ results: [{ url: source.url, title: "", snippet: "", date: null }] }));
	assert.match(text(await execute("web_search", { queries: ["a"] })), /\[nature.com\]/);
});

test("fetch renders Markdown unchanged and caps the complete output including footer", async (t) => {
	let content = "# Paper\nMarkdown";
	t.mock.method(globalThis, "fetch", async (url: string, init: RequestInit) => {
		assert.equal(url, "https://api.deepseek.com/v1/tools/fetch");
		assert.deepEqual(JSON.parse(init.body as string), { url: source.url });
		return Response.json({ title: "API title is not copied", content });
	});
	const result = await execute("web_fetch", { url: source.url });
	assert.equal(text(result), `Fetched ${source.url} (HTTP 200)\n\n${notice}\n\n${content}`);
	assert.deepEqual(metadata(result), { url: source.url, statusCode: 200, truncated: false });
	content = "x".repeat(220_000);
	const long = await execute("web_fetch", { url: source.url });
	assert.equal(text(long).length, 200_000);
	assert.match(text(long), /\n\n\(Content truncated\. Fetch a more specific URL or section for the full text\.\)$/);
	assert.deepEqual(metadata(long), { url: source.url, statusCode: 200, truncated: true });
	assert(!JSON.stringify(long.details).includes(content));
	const headerLength = `Fetched ${source.url} (HTTP 200)\n\n${notice}\n\n`.length;
	content = "x".repeat(200_000 - headerLength);
	const exact = await execute("web_fetch", { url: source.url });
	assert.equal(text(exact).length, 200_000);
	assert.equal(metadata(exact).truncated, false);
	content += "x";
	const overflow = await execute("web_fetch", { url: source.url });
	assert.equal(text(overflow).length, 200_000);
	assert.equal(metadata(overflow).truncated, true);
});
