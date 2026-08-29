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
	return new PolicyController(policy, "anysearch");
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
