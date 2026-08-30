import assert from "node:assert/strict";
import test from "node:test";
import type { ExtensionAPI, ToolCallEvent, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import { assertAllowedUrl, PolicyController } from "./policy.js";

const policy = {
	allowedHosts: ["nih.gov", "nature.com"],
	deniedHosts: ["blocked.nih.gov"],
	forbiddenQueryPatterns: [],
};

test("URL policy allows declared domains and their subdomains", () => {
	assert.doesNotThrow(() => assertAllowedUrl("https://pubmed.ncbi.nlm.nih.gov/123", policy));
});

test("URL policy rejects denied, undeclared, and local paths", () => {
	assert.throws(() => assertAllowedUrl("https://blocked.nih.gov/private", policy));
	assert.throws(() => assertAllowedUrl("https://example.com", policy));
	assert.throws(() => assertAllowedUrl("/mnt/data/oracle.csv", policy));
});

function policyHandlers(controller: PolicyController) {
	const handlers = new Map<string, (event: never) => unknown>();
	controller.createExtension()({
		on(name: string, handler: (event: never) => unknown) {
			handlers.set(name, handler);
		},
	} as unknown as ExtensionAPI);
	return {
		call: (event: ToolCallEvent) => handlers.get("tool_call")?.(event as never) as { block?: boolean; reason?: string } | undefined,
		result: (event: ToolResultEvent) => handlers.get("tool_result")?.(event as never) as { isError?: boolean } | undefined,
	};
}

function controller(): PolicyController {
	return new PolicyController(policy, ["parallel-mcp", "exa", "duckduckgo"]);
}

test("web results are filtered before their responseId becomes session-readable", () => {
	const subject = controller();
	const handlers = policyHandlers(subject);
	subject.begin(["synthonbench"]);
	const blocked = handlers.result({
		type: "tool_result",
		toolCallId: "search-1",
		toolName: "web_search",
		input: { query: "reaction design" },
		content: [{ type: "text", text: "Hidden SynthonBench answer" }],
		details: { responseId: "blocked-id" },
		isError: false,
	} as ToolResultEvent);
	assert.equal(blocked?.isError, true);
	assert.equal(handlers.call({
		type: "tool_call",
		toolCallId: "read-1",
		toolName: "get_search_content",
		input: { responseId: "blocked-id" },
	} as ToolCallEvent)?.block, true);
	subject.end();
});

test("responseIds remain private to the policy controller that received them", () => {
	const owner = controller();
	const other = controller();
	const ownerHandlers = policyHandlers(owner);
	const otherHandlers = policyHandlers(other);
	owner.begin(["synthonbench"]);
	other.begin(["synthonbench"]);
	assert.equal(ownerHandlers.result({
		type: "tool_result",
		toolCallId: "search-1",
		toolName: "web_search",
		input: { query: "reaction design" },
		content: [{ type: "text", text: "https://www.nature.com/articles/example" }],
		details: { responseId: "owner-id" },
		isError: false,
	} as ToolResultEvent), undefined);
	const read = {
		type: "tool_call",
		toolCallId: "read-1",
		toolName: "get_search_content",
		input: { responseId: "owner-id" },
	} as ToolCallEvent;
	assert.equal(ownerHandlers.call(read), undefined);
	assert.equal(otherHandlers.call(read)?.block, true);
	owner.end();
	owner.begin(["synthonbench"]);
	assert.equal(ownerHandlers.call(read), undefined);
	owner.end();
	other.end();
});

test("tool usage is counted without a call limit", () => {
	const subject = controller();
	const handlers = policyHandlers(subject);
	subject.begin([]);
	for (let index = 0; index < 20; index += 1) {
		assert.equal(handlers.call({
			type: "tool_call",
			toolCallId: `search-${index}`,
			toolName: "web_search",
			input: { query: `reaction design ${index}` },
		} as ToolCallEvent), undefined);
	}
	for (let index = 0; index < 10; index += 1) {
		assert.equal(handlers.call({
			type: "tool_call",
			toolCallId: `context-${index}`,
			toolName: "query-docs",
			input: { query: `library API ${index}` },
		} as ToolCallEvent), undefined);
	}
	assert.deepEqual(subject.end(), { webCalls: 20, context7Calls: 10 });
});

test("web search preserves allowed provider choices and rejects providers outside the route", () => {
	const subject = controller();
	const handlers = policyHandlers(subject);
	subject.begin([]);
	const allowed = {
		type: "tool_call",
		toolCallId: "search-allowed",
		toolName: "web_search",
		input: { query: "reaction design", provider: "DuckDuckGo" },
	} as ToolCallEvent;
	assert.equal(handlers.call(allowed), undefined);
	assert.equal((allowed.input as Record<string, unknown>).provider, "duckduckgo");

	const blocked = handlers.call({
		type: "tool_call",
		toolCallId: "search-blocked",
		toolName: "web_search",
		input: { query: "reaction design", provider: "tavily" },
	} as ToolCallEvent);
	assert.equal(blocked?.block, true);
	assert.match(blocked?.reason ?? "", /"reason":"provider_not_allowed"/);
	assert.match(blocked?.reason ?? "", /"allowed_providers":\["parallel-mcp","exa","duckduckgo"\]/);
	subject.end();
});

test("unrestricted search keeps Agent domain filters and defaults to automatic routing", () => {
	const subject = new PolicyController(
		{ allowedHosts: [], deniedHosts: [], forbiddenQueryPatterns: [] },
		["parallel-mcp", "exa", "duckduckgo"],
	);
	const handlers = policyHandlers(subject);
	subject.begin([]);
	const call = {
		type: "tool_call",
		toolCallId: "search-auto",
		toolName: "web_search",
		input: { query: "reaction design", domainFilter: ["pubmed.ncbi.nlm.nih.gov"] },
	} as ToolCallEvent;
	assert.equal(handlers.call(call), undefined);
	assert.deepEqual(call.input, {
		query: "reaction design",
		domainFilter: ["pubmed.ncbi.nlm.nih.gov"],
		provider: "auto",
		workflow: "none",
	});
	subject.end();
});
