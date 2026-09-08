import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { ExtensionAPI, ToolCallEvent, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import { assertAllowedUrl, PolicyController } from "./policy.js";

const networkPolicy = {
	allowedHosts: ["nih.gov", "nature.com"],
	deniedHosts: ["blocked.nih.gov"],
	forbiddenQueryPatterns: [],
};

function policyHandlers(controller: PolicyController) {
	const handlers = new Map<string, (event: never) => unknown>();
	controller.createExtension()({
		on(name: string, handler: (event: never) => unknown) {
			handlers.set(name, handler);
		},
	} as unknown as ExtensionAPI);
	return {
		call: async (event: ToolCallEvent) => await handlers.get("tool_call")?.(event as never) as
			{ block?: boolean; reason?: string } | undefined,
		result: async (event: ToolResultEvent) => await handlers.get("tool_result")?.(event as never) as
			{ content?: ToolResultEvent["content"]; isError?: boolean } | undefined,
	};
}

function controller(budgets: Record<string, number> = {}): PolicyController {
	return new PolicyController(
		networkPolicy,
		["parallel-mcp", "exa", "duckduckgo"],
		budgets,
	);
}

function call(toolName: string, input: Record<string, unknown> = {}): ToolCallEvent {
	return { type: "tool_call", toolCallId: `${toolName}-${Math.random()}`, toolName, input } as ToolCallEvent;
}

test("URL policy allows declared domains and rejects unsafe URLs", () => {
	assert.doesNotThrow(() => assertAllowedUrl("https://pubmed.ncbi.nlm.nih.gov/123", networkPolicy));
	assert.throws(() => assertAllowedUrl("https://blocked.nih.gov/private", networkPolicy));
	assert.throws(() => assertAllowedUrl("https://example.com", networkPolicy));
	assert.throws(() => assertAllowedUrl("/mnt/data/oracle.csv", networkPolicy));
});

test("network policy filters results and keeps content ids session-local", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-policy-"));
	const owner = controller();
	const other = controller();
	const ownerHandlers = policyHandlers(owner);
	const otherHandlers = policyHandlers(other);
	try {
		await owner.begin(["synthonbench"], join(root, "owner.json"));
		await other.begin(["synthonbench"], join(root, "other.json"));
		const hidden = await ownerHandlers.result({
			type: "tool_result",
			toolCallId: "hidden",
			toolName: "web_search",
			input: { query: "reaction design" },
			content: [{ type: "text", text: "Hidden SynthonBench answer" }],
			details: { searchId: "blocked-id" },
			isError: false,
		} as ToolResultEvent);
		assert.equal(hidden?.isError, true);

		await ownerHandlers.result({
			type: "tool_result",
			toolCallId: "allowed",
			toolName: "web_search",
			input: { query: "reaction design" },
			content: [{ type: "text", text: "https://www.nature.com/articles/example" }],
			details: { searchId: "owner-search" },
			isError: false,
		} as ToolResultEvent);
		assert.equal(await ownerHandlers.call(call("get_search_content", { responseId: "owner-search" })), undefined);
		assert.equal((await otherHandlers.call(call("get_search_content", { responseId: "owner-search" })))?.block, true);
	} finally {
		owner.end();
		other.end();
		await rm(root, { recursive: true, force: true });
	}
});

test("invalid search inputs are rejected before usage is reserved", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-policy-"));
	const subject = controller({ web_search: 2 });
	const handlers = policyHandlers(subject);
	try {
		await subject.begin([], join(root, "budget.json"));
		const rejected = await handlers.call(call("web_search", {
			query: "reaction design",
			provider: "tavily",
		}));
		assert.equal(rejected?.block, true);
		assert.match(rejected?.reason ?? "", /"reason":"provider_not_allowed"/);
		assert.equal(subject.snapshot().toolCalls.web_search, undefined);

		const allowed = call("web_search", { query: "reaction design", provider: "DuckDuckGo" });
		assert.equal(await handlers.call(allowed), undefined);
		assert.deepEqual(allowed.input, { query: "reaction design", provider: "duckduckgo", workflow: "none", domainFilter: ["nih.gov", "nature.com"] });
	} finally {
		subject.end();
		await rm(root, { recursive: true, force: true });
	}
});

test("tool budgets reserve atomically, report remaining calls, and leave unlisted tools unlimited", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-policy-"));
	const subject = controller({ web_search: 1, mcp__literature__search: 0 });
	const handlers = policyHandlers(subject);
	try {
		await subject.begin([], join(root, "budget.json"));
		assert.match(subject.budgetMessage(), /web_search: 1 of 1 calls remaining/);
		const concurrent = await Promise.all([
			handlers.call(call("web_search", { query: "first" })),
			handlers.call(call("web_search", { query: "second" })),
		]);
		assert.equal(concurrent.filter((result) => result?.block).length, 1);
		assert.match(concurrent.find((result) => result?.block)?.reason ?? "", /tool_budget_exhausted/);
		assert.match((await handlers.call(call("mcp__literature__search")))?.reason ?? "", /tool_budget_exhausted/);
		for (let index = 0; index < 20; index += 1) {
			assert.equal(await handlers.call(call("read", { path: `${index}.txt` })), undefined);
		}
		const result = await handlers.result({
			type: "tool_result",
			toolCallId: "search-result",
			toolName: "web_search",
			input: { query: "first" },
			content: [{ type: "text", text: "failed" }],
			details: {},
			isError: true,
		} as ToolResultEvent);
		assert.match(JSON.stringify(result?.content), /0 of 1 calls remaining/);
		assert.deepEqual(subject.snapshot().toolCalls, { read: 20, web_search: 1 });

		const otherAgent = controller({ web_search: 1 });
		await otherAgent.begin([], join(root, "other-agent.json"));
		assert.equal(
			await policyHandlers(otherAgent).call(call("web_search", { query: "independent" })),
			undefined,
		);
		otherAgent.end();
	} finally {
		subject.end();
		await rm(root, { recursive: true, force: true });
	}
});

test("persisted reservations survive restart while a new turn receives a fresh budget", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-policy-"));
	const budgetPath = join(root, "turn-1.json");
	try {
		const first = controller({ "query-docs": 1 });
		await first.begin([], budgetPath);
		await policyHandlers(first).call(call("query-docs"));
		first.end();

		const resumed = controller({ "query-docs": 1 });
		await resumed.begin([], budgetPath);
		assert.match((await policyHandlers(resumed).call(call("query-docs")))?.reason ?? "", /tool_budget_exhausted/);
		resumed.end();

		const nextTurn = controller({ "query-docs": 1 });
		await nextTurn.begin([], join(root, "turn-2.json"));
		assert.equal(await policyHandlers(nextTurn).call(call("query-docs")), undefined);
		nextTurn.end();
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});
