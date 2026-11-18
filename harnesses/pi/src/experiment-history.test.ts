import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import {
	deliveredSeq, deliveredTaskMessages, ExperimentHistory, HISTORY_MESSAGE_TYPE, STEERING_MODE, TASK_MESSAGE_TYPE,
	VISIBILITY_ENTRY,
} from "./experiment-history.js";
import { SubmissionController } from "./submission.js";
import { parseFrame } from "./protocol.js";
import type {
	HarnessHistoryConfig, HistoryNotice, HistoryQueryAnswer, HistoryQueryRequest, InitializeFrame, SubmissionValidator,
} from "./protocol.js";
import { SIDECAR_RELEASE_VERSION } from "./release.js";
import { sha256 } from "./trace.js";

const CONFIG: HarnessHistoryConfig = {
	toolName: "experiment_history",
	fields: [{ name: "x", type: "integer", filterable: true, sortable: true },
		{ name: "objective", type: "number", filterable: true, sortable: true, direction: "maximize" }],
	objective: "objective",
	pageSizeDefault: 20,
	pageSizeMax: 100,
	responseMaxBytes: 32768,
	inlineLimit: 3,
	maxCasRetries: 2,
};

/** Host stand-in: a log head that tests advance, notices rendered for (from, to]. */
class FakeHost {
	head = 0;
	readonly requests: HistoryQueryRequest[] = [];
	/** Runs after a query answer is computed: results booked while the tool call is in flight. */
	afterQuery: (() => void) | undefined;

	notice(from: number, to: number): HistoryNotice {
		return { from_seq: from, to_seq: to, count: to - from, text: `[experiment history] ${to - from} new (seq ${from + 1}-${to})` };
	}

	querier = async (request: HistoryQueryRequest): Promise<HistoryQueryAnswer> => {
		this.requests.push(request);
		const head = this.head;
		const notice = request.operation !== "head" && request.visibleSeq < head ? this.notice(request.visibleSeq, head) : null;
		if (request.operation === "query") this.afterQuery?.();
		if (request.operation === "query" && request.query?.view === "bogus") {
			return { headSeq: head, result: null, notice, error: { code: "invalid_query", message: "bad view" } };
		}
		return { headSeq: head, result: request.operation === "query" ? { view: "summary", watermark: head } : null,
			notice, error: null };
	};
}

function fakePi() {
	const handlers = new Map<string, (event: unknown) => Promise<void> | void>();
	const sent: Array<{ message: Record<string, unknown>; options: Record<string, unknown> | undefined }> = [];
	const entries: Array<{ type: string; data: unknown }> = [];
	const api = {
		on: (name: string, handler: (event: unknown) => Promise<void> | void) => { handlers.set(name, handler); },
		sendMessage: (message: Record<string, unknown>, options?: Record<string, unknown>) => { sent.push({ message, options }); },
		appendEntry: (type: string, data: unknown) => { entries.push({ type, data }); },
	};
	return { api, handlers, sent, entries };
}

test("head hints are merged into one steer notice per step boundary and recorded before delivery", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-exp-history-"));
	try {
		const host = new FakeHost();
		host.head = 4;
		const history = new ExperimentHistory(CONFIG);
		const pi = fakePi();
		history.createExtension()(pi.api as never);
		await history.begin("researcher", "turn-1", 4, root, host.querier);
		assert.equal(history.visibleSeq, 4);
		await pi.handlers.get("turn_end")!({});
		assert.equal(pi.sent.length, 0, "nothing new: no notice");
		host.head = 5; history.advanceHead(5);
		host.head = 7; history.advanceHead(7);
		await pi.handlers.get("turn_end")!({});
		assert.equal(pi.sent.length, 1, "two head hints before one step give one merged notice");
		assert.equal(pi.sent[0]!.options?.deliverAs, "steer");
		assert.equal(pi.sent[0]!.message.customType, HISTORY_MESSAGE_TYPE);
		assert.match(String(pi.sent[0]!.message.content), /seq 5-7/);
		assert.equal(history.visibleSeq, 7);
		const records = await history.records();
		assert.deepEqual(records.map((r) => r.kind), ["turn_begin", "step_notice"]);
		assert.deepEqual([records[1]!.fromSeq, records[1]!.toSeq], [4, 7]);
		assert(pi.entries.every((entry) => entry.type === VISIBILITY_ENTRY));
		history.closeTurn();
		host.head = 9; history.advanceHead(9);
		await pi.handlers.get("turn_end")!({});
		assert.equal(pi.sent.length, 1, "no steer after the terminal submission (it would start another model call)");
		assert.equal(await history.end(), 7);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("history tool responses carry and adopt the gap notice; query errors are readable results", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-exp-history-"));
	try {
		const host = new FakeHost();
		const history = new ExperimentHistory(CONFIG);
		await history.begin("researcher", "turn-1", 0, root, host.querier);
		host.head = 2;
		const tool = history.tool();
		assert.equal(tool.name, "experiment_history");
		assert.match(tool.description, /Filterable fields: x, objective/);
		const result = await tool.execute("call-1", { view: "summary" }, undefined, undefined, {} as never);
		const text = (result.content[0] as { text: string }).text;
		assert.match(text, /"watermark":2/);
		assert.match(text, /seq 1-2/);
		assert.equal(history.visibleSeq, 2);
		assert.deepEqual(result.details, { headSeq: 2, historyNotice: { fromSeq: 0, toSeq: 2, count: 2 } });
		host.head = 3;
		const failed = await tool.execute("call-2", { view: "bogus" }, undefined, undefined, {} as never);
		assert.match((failed.content[0] as { text: string }).text, /Query error invalid_query: bad view[\s\S]*seq 3-3/);
		assert.deepEqual(failed.details, { headSeq: 3, error: "invalid_query", historyNotice: { fromSeq: 2, toSeq: 3, count: 1 } });
		assert.equal(history.visibleSeq, 3, "the notice in an error response is still delivered (and persisted)");
		assert.equal(history.seenSeq, 0, "the model has not read either response yet");
		assert.deepEqual(host.requests.map((r) => [r.operation, r.visibleSeq]), [["head", 0], ["query", 0], ["query", 2]]);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("a crash between the intent record and delivery re-delivers the range instead of assuming it", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-exp-crash-"));
	try {
		const host = new FakeHost();
		host.head = 2;
		const history = new ExperimentHistory(CONFIG);
		const pi = fakePi();
		pi.api.sendMessage = () => { throw new Error("process killed before the steer message reached the session"); };
		history.createExtension()(pi.api as never);
		await history.begin("researcher", "turn-1", 2, root, host.querier);
		host.head = 5; history.advanceHead(5);
		await assert.rejects(async () => { await pi.handlers.get("turn_end")!({}); }, /process killed/);
		const intents = await history.records();
		assert.deepEqual([intents.at(-1)!.kind, intents.at(-1)!.toSeq], ["step_notice", 5], "the intent was recorded first");

		// Restart: the persisted session has no notice, so the watermark is the frozen base, not the intent.
		const restarted = new ExperimentHistory(CONFIG);
		const pi2 = fakePi();
		restarted.createExtension()(pi2.api as never);
		await restarted.begin("researcher", "turn-1", 2, root, host.querier, [{ type: "message", message: { role: "user" } }]);
		assert.equal(restarted.visibleSeq, 2);
		assert.deepEqual(restarted.submissionWatermark(), { visibleSeq: 2, casRetries: 0 });
		assert.equal(await restarted.deliverAtStart(), true, "the unconfirmed range is delivered again before the first call");
		assert.equal(pi2.sent.length, 1);
		assert.deepEqual(pi2.sent[0]!.message.details, { historyNotice: { fromSeq: 2, toSeq: 5, count: 3 } });
		assert.deepEqual([restarted.visibleSeq, restarted.seenSeq], [5, 5]);
		// Once Pi has persisted it, a later recovery trusts it.
		const persisted = [{ type: "custom_message", customType: HISTORY_MESSAGE_TYPE, details: pi2.sent[0]!.message.details }];
		const again = new ExperimentHistory(CONFIG);
		await again.begin("researcher", "turn-1", 2, root, host.querier, persisted);
		assert.equal(again.visibleSeq, 5);
		assert.equal(deliveredSeq([{ type: "custom_message", customType: "other", details: { historyNotice: { toSeq: 9 } } }]), 0);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("results booked after the frozen base reach the agent before its first model call", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-exp-start-"));
	try {
		const host = new FakeHost();
		host.head = 6;
		const history = new ExperimentHistory(CONFIG);
		const pi = fakePi();
		history.createExtension()(pi.api as never);
		await history.begin("researcher", "turn-1", 4, root, host.querier);
		assert.equal(history.visibleSeq, 4);
		assert.equal(await history.deliverAtStart(), true);
		assert.match(String(pi.sent[0]!.message.content), /seq 5-6/);
		assert.equal(await history.deliverAtStart(), false, "nothing new: no second notice");
		assert.deepEqual((await history.records()).map((r) => r.kind), ["turn_begin", "start_notice"]);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("a notice returned in the same step as a submission does not count as seen for CAS", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-exp-seen-"));
	try {
		const host = new FakeHost();
		const history = new ExperimentHistory(CONFIG);
		const pi = fakePi();
		history.createExtension()(pi.api as never);
		await history.begin("researcher", "turn-1", 0, root, host.querier);
		host.head = 3;
		await history.tool().execute("q", { view: "summary" }, undefined, undefined, {} as never);
		assert.deepEqual(history.submissionWatermark(), { visibleSeq: 0, casRetries: 0 },
			"a parallel submit in the same assistant message has not read the query result");
		pi.handlers.get("before_provider_request")!({});
		assert.deepEqual(history.submissionWatermark(), { visibleSeq: 3, casRetries: 0 });
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("task messages: own message type between steps, deduplicated, never history, proven by the session", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-exp-task-"));
	try {
		const host = new FakeHost();
		host.head = 3;
		const history = new ExperimentHistory(CONFIG);
		const pi = fakePi();
		history.createExtension()(pi.api as never);
		const hint = { messageId: "hook:5:0", text: "Consider the region near 70.", source: "hook:hint" };
		history.queueTaskMessage(hint); // arrived while the pool was idle: kept for the turn
		await history.begin("researcher", "turn-2", 3, root, host.querier,
			[{ type: "custom_message", customType: TASK_MESSAGE_TYPE, details: { taskMessage: { messageId: "hook:1:0", turnId: "turn-1", source: "x" } } }]);
		history.queueTaskMessage({ ...hint, messageId: "hook:1:0" }); // delivered in an earlier turn: ignored
		history.queueTaskMessage(hint); // a resend: ignored
		assert.equal(await history.deliverAtStart(), false, "no history notice: nothing new");
		assert.equal(pi.sent.length, 1, "the queued task message goes out before the first model call");
		assert.equal(pi.sent[0]!.message.customType, TASK_MESSAGE_TYPE);
		assert.match(String(pi.sent[0]!.message.content), /\[task message from hook:hint\] Consider the region near 70\.[\s\S]*not a measurement/);
		assert.deepEqual(pi.sent[0]!.message.details, { taskMessage: { messageId: "hook:5:0", turnId: "turn-2", source: "hook:hint" } });
		assert.deepEqual([history.visibleSeq, history.seenSeq], [3, 3], "task messages never move the watermarks");
		history.queueTaskMessage(hint);
		history.queueTaskMessage({ ...hint, messageId: "hook:9:0", text: "Second hint." });
		host.head = 4; history.advanceHead(4);
		await pi.handlers.get("turn_end")!({});
		assert.deepEqual(pi.sent.slice(1).map((s) => [s.message.customType, s.options?.deliverAs]),
			[[HISTORY_MESSAGE_TYPE, "steer"], [TASK_MESSAGE_TYPE, "steer"]], "one notice and the one new message");
		assert.deepEqual(history.submissionWatermark(), { visibleSeq: 3, casRetries: 0 }, "CAS ignores task messages");
		history.closeTurn();
		history.queueTaskMessage({ ...hint, messageId: "hook:11:0" });
		await pi.handlers.get("turn_end")!({});
		assert.equal(pi.sent.length, 3, "nothing after the terminal submission");
		await history.end();
		const records = await history.records();
		assert.deepEqual(records.map((r) => [r.kind, r.messageId]), [["turn_begin", undefined], ["task_message", "hook:5:0"],
			["step_notice", undefined], ["task_message", "hook:9:0"], ["turn_end", undefined]]);
		// The commit reports what the persisted session shows for this turn only.
		const entries = pi.sent.map((s) => ({ type: "custom_message", customType: s.message.customType, details: s.message.details }));
		assert.deepEqual(history.taskMessagesFor("turn-2", entries), ["hook:5:0", "hook:9:0"]);
		assert.deepEqual(history.taskMessagesFor("turn-1", entries), []);
		// The undelivered one was dropped at the end of the turn: the host resends it or puts it into the next input.
		await history.begin("researcher", "turn-3", 4, join(root, "t3"), host.querier, entries);
		assert.equal(await history.deliverTaskMessages(), 0);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("protocol frames: task messages are one-way frames with an id, text and source", () => {
	const common = { requestId: "run-1", protocolVersion: SIDECAR_RELEASE_VERSION, campaignId: "c" };
	const frame = { ...common, type: "task_message", messageId: "hook:3:0", text: "hint", source: "hook:h" };
	assert.deepEqual(parseFrame(JSON.stringify(frame)), frame);
	assert.throws(() => parseFrame(JSON.stringify({ ...frame, text: "" })), /text/);
	assert.throws(() => parseFrame(JSON.stringify({ ...frame, extra: 1 })), /unexpected or missing/);
	assert.throws(() => parseFrame(JSON.stringify({ ...frame, messageId: "x".repeat(201) })), /too long/);
});

function submissionConfig(): InitializeFrame {
	return {
		artifactRoot: "/unused",
		history: CONFIG,
		submissionContract: { toolName: "submit_candidates", payloadSchema: { type: "object" }, artifactRules: [] },
	} as unknown as InitializeFrame;
}

test("submission CAS: stale answers notify, are not validation attempts and are bounded; acceptance is not an experiment", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-exp-cas-"));
	try {
		const turnRoot = join(root, "turns", "turn-1");
		const host = new FakeHost();
		host.head = 3;
		const history = new ExperimentHistory(CONFIG);
		const pi = fakePi();
		history.createExtension()(pi.api as never);
		const modelCall = () => pi.handlers.get("before_provider_request")!({});
		await history.begin("researcher", "turn-1", 3, turnRoot, host.querier);
		const seen: Array<[number, number | undefined, number | undefined]> = [];
		const validate: SubmissionValidator = async (request) => {
			seen.push([request.attemptIndex, request.visibleSeq, request.casRetries]);
			if (request.visibleSeq! < host.head && request.casRetries! < 2) {
				const notice = host.notice(request.visibleSeq!, host.head);
				host.head += 1; // results keep arriving while the agent repairs
				return { decision: "stale", errors: [], headSeq: notice.to_seq, notice };
			}
			return { decision: "accept", errors: [], historySeq: host.head, receipt: "2 proposals recorded for generation 1." };
		};
		const persisted: unknown[] = [];
		const controller = new SubmissionController(submissionConfig(), join(root, "workspace"), history);
		await controller.begin("researcher", "turn-1", turnRoot, async (value) => { persisted.push(value); }, validate);
		const tool = controller.tool();
		host.head = 4;
		const first = await tool.execute("c1", { candidates: [1, 2] }, undefined, undefined, {} as never);
		assert.match((first.content[0] as { text: string }).text,
			/not validated and does not count[\s\S]*seq 4-4[\s\S]*resubmitting unchanged is allowed/);
		assert.deepEqual(first.details, { status: "stale", historyNotice: { fromSeq: 3, toSeq: 4, count: 1 } });
		assert.equal(controller.submission?.submissionId, undefined, "a stale answer does not end the turn");
		assert.equal(history.visibleSeq, 4);
		const results = [first];
		modelCall();
		results.push(await tool.execute("c2", { candidates: [1, 2] }, undefined, undefined, {} as never));
		assert.match((results[1]!.content[0] as { text: string }).text, /seq 5-5/);
		modelCall();
		const accepted = await tool.execute("c3", { candidates: [1, 2] }, undefined, undefined, {} as never);
		assert.deepEqual(seen, [[1, 3, 0], [1, 4, 1], [1, 5, 2]], "stale answers keep attempt 1; the third goes through");
		const text = (accepted.content[0] as { text: string }).text;
		assert.match(text, /2 proposals recorded for generation 1\./);
		assert.match(text, /A proposal is not an experiment/);
		assert.doesNotMatch(text, /experiment_id|exp-/);
		assert.equal(controller.submission?.submissionId, "turn-1-submission-1");
		const report = JSON.parse(await readFile(join(turnRoot, "attempts/0001/validation.json"), "utf8"));
		assert.deepEqual([report.visibleSeq, report.casRetries, report.historySeq], [5, 2, 6]);
		assert.deepEqual((await history.records()).map((r) => r.kind), ["turn_begin", "cas_stale", "cas_stale", "validated"]);
		// Resume after a crash: the watermark comes from the persisted tool results, not from intent records.
		const persistedResults = results.map((r) => ({ type: "message", message: { role: "toolResult", details: r.details } }));
		const resumed = new ExperimentHistory(CONFIG);
		await resumed.begin("researcher", "turn-1", 3, turnRoot, host.querier, persistedResults);
		assert.equal(resumed.visibleSeq, 5);
		assert.deepEqual(resumed.submissionWatermark(), { visibleSeq: 5, casRetries: 0 });
		const lost = new ExperimentHistory(CONFIG);
		await lost.begin("researcher", "turn-1", 3, turnRoot, host.querier, persistedResults.slice(0, 1));
		assert.equal(lost.visibleSeq, 4, "the second stale result never reached the session file: not trusted");
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("protocol frames: history config, head hint, query result and stale validation results", () => {
	const common = { requestId: "run-1", protocolVersion: SIDECAR_RELEASE_VERSION, campaignId: "c" };
	const head = parseFrame(JSON.stringify({ ...common, type: "history_head_advanced", headSeq: 12 }));
	assert.deepEqual(head, { ...common, type: "history_head_advanced", headSeq: 12 });
	assert.throws(() => parseFrame(JSON.stringify({ ...common, type: "history_head_advanced", headSeq: -1 })), /headSeq/);
	const notice = { from_seq: 3, to_seq: 5, count: 2, text: "two new", inline: null };
	const result = parseFrame(JSON.stringify({ ...common, type: "history_query_result", queryId: "q1", headSeq: 5,
		result: { view: "summary" }, notice, error: null }));
	assert.equal(result.type === "history_query_result" && result.notice?.to_seq, 5);
	assert.throws(() => parseFrame(JSON.stringify({ ...common, type: "history_query_result", queryId: "q1", headSeq: 5,
		result: null, notice: { ...notice, to_seq: 3 }, error: null })), /non-empty range/);
	const base = { ...common, type: "submission_validation_result", validationId: "v1", submissionDigest: "a".repeat(64) };
	const stale = parseFrame(JSON.stringify({ ...base, decision: "stale", errors: [], headSeq: 5, notice }));
	assert.equal(stale.type === "submission_validation_result" && stale.decision, "stale");
	assert.throws(() => parseFrame(JSON.stringify({ ...base, decision: "stale", errors: [] })), /inconsistent/);
	assert.throws(() => parseFrame(JSON.stringify({ ...base, decision: "retry", errors: [{ path: "/a", code: "already_pending",
		message: "m", hint: "" }], receipt: "x" })), /inconsistent/);
	const accepted = parseFrame(JSON.stringify({ ...base, decision: "accept", errors: [], historySeq: 5, receipt: "ok" }));
	assert.equal(accepted.type === "submission_validation_result" && accepted.historySeq, 5);
	const legacy = parseFrame(JSON.stringify({ ...base, decision: "accept", errors: [] }));
	assert.deepEqual(Object.keys(legacy).sort(), ["campaignId", "decision", "errors", "protocolVersion", "requestId",
		"submissionDigest", "type", "validationId"]);

	const contract = { contractId: "c", toolName: "submit_candidates", payloadSchema: { type: "object", additionalProperties: false },
		artifactRules: [], maxValidationAttempts: 3 };
	const contractJson = JSON.stringify(contract);
	const init = { ...common, type: "initialize", artifactRoot: "/a", baseUrl: "https://p", wireApi: "responses", model: "m",
		thinking: "low", taskId: "t", caseId: "k", seed: 0, submissionContractJson: contractJson,
		submissionContractSha256: sha256(contractJson), profileSetSha256: "c".repeat(64),
		guestRuntime: { imageRef: "ldm/t:aaaaaaaaaaaa", recipeSha256: "a".repeat(64), rootfsSize: "4G", installPolicy: "session_overlay" },
		profiles: [{ profileId: "r", agentsPath: "/A.md", agentsSha256: "a".repeat(64), skillDirs: [], skillDirSha256: [] }],
		toolExtensions: [], mcpServers: [], networkPolicy: { allowedHosts: [], deniedHosts: [], forbiddenQueryPatterns: [] },
		limits: { wallTimeSeconds: 60, toolCallBudgets: { experiment_history: 8 } }, context7Enabled: false };
	assert.throws(() => parseFrame(JSON.stringify(init)), /unavailable tools: experiment_history/);
	const parsed = parseFrame(JSON.stringify({ ...init, history: CONFIG }));
	assert.equal(parsed.type === "initialize" && parsed.history?.toolName, "experiment_history");
	assert.throws(() => parseFrame(JSON.stringify({ ...init, history: { ...CONFIG, toolName: "submit_candidates" },
		limits: { ...init.limits, toolCallBudgets: {} } })), /conflicts/);
	assert.throws(() => parseFrame(JSON.stringify({ ...init, history: { ...CONFIG, extra: 1 } })), /history/);
});

test("a real Pi session delivers the steer notice after the step's tool results and before the next model call", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-exp-steer-"));
	const requests: Array<{ input: unknown[] }> = [];
	const sse = (events: unknown[]) => events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n";
	const usage = { input_tokens: 10, output_tokens: 5, total_tokens: 15, input_tokens_details: { cached_tokens: 0 },
		output_tokens_details: { reasoning_tokens: 0 } };
	const server = createServer(async (request, response) => {
		let body = "";
		for await (const chunk of request) body += chunk.toString();
		requests.push(JSON.parse(body));
		response.writeHead(200, { "content-type": "text/event-stream" });
		if (requests.length === 1) {
			const call = { type: "function_call", id: "fc_1", call_id: "call_1", name: "experiment_history",
				arguments: JSON.stringify({ view: "summary" }), status: "completed" };
			response.end(sse([
				{ type: "response.created", response: { id: "r1", status: "in_progress", output: [] } },
				{ type: "response.output_item.added", output_index: 0, item: { ...call, arguments: "" } },
				{ type: "response.function_call_arguments.done", output_index: 0, arguments: call.arguments },
				{ type: "response.output_item.done", output_index: 0, item: call },
				{ type: "response.completed", response: { id: "r1", status: "completed", output: [call], usage } },
			]));
			return;
		}
		const item = { type: "message", id: "msg_2", role: "assistant", status: "completed",
			content: [{ type: "output_text", text: "Noted the new results.", annotations: [] }] };
		response.end(sse([
			{ type: "response.created", response: { id: "r2", status: "in_progress", output: [] } },
			{ type: "response.output_item.added", output_index: 0, item: { ...item, content: [] } },
			{ type: "response.output_item.done", output_index: 0, item },
			{ type: "response.completed", response: { id: "r2", status: "completed", output: [item], usage } },
		]));
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address();
	assert(address && typeof address !== "string");
	let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
	try {
		const agentDir = join(root, "agent");
		await mkdir(agentDir);
		const modelsPath = join(agentDir, "models.json");
		await writeFile(modelsPath, JSON.stringify({ providers: { fixture: {
			baseUrl: `http://127.0.0.1:${address.port}`, api: "openai-responses", apiKey: "fixture-key",
			models: [{ id: "fixture", name: "fixture", reasoning: false, input: ["text"], contextWindow: 262144, maxTokens: 4096,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
		} } }));
		const runtime = await ModelRuntime.create({ authPath: join(agentDir, "auth.json"), modelsPath, refreshOnCreate: false });
		await runtime.setRuntimeApiKey("fixture", "fixture-key");
		const model = runtime.getModel("fixture", "fixture");
		assert(model);
		const host = new FakeHost();
		host.head = 2;
		const history = new ExperimentHistory(CONFIG);
		const settings = SettingsManager.inMemory({ retry: { enabled: false }, steeringMode: STEERING_MODE });
		const loader = new DefaultResourceLoader({ cwd: root, agentDir, settingsManager: settings, noExtensions: true,
			noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
			extensionFactories: [history.createExtension()] });
		await loader.reload();
		const manager = SessionManager.create(root, join(root, "sessions"));
		({ session } = await createAgentSession({ cwd: root, agentDir, modelRuntime: runtime, model, thinkingLevel: "off",
			sessionManager: manager, settingsManager: settings, resourceLoader: loader,
			customTools: [history.tool()], tools: ["experiment_history"] }));
		await session.bindExtensions({});
		await history.begin("researcher", "turn-1", 2, join(root, "turn"), host.querier);
		// Two results are booked while the agent's history query runs, and the host sends its head hint.
		// ...and a task message frame arrives meanwhile (design 5.9 notify_agents).
		host.afterQuery = () => {
			host.head = 4; history.advanceHead(4);
			history.queueTaskMessage({ messageId: "hook:7:0", text: "Try the upper half too.", source: "hook:hint" });
		};
		await session.prompt("Look at the history.", { expandPromptTemplates: false, source: "rpc" });
		assert.equal(requests.length, 2);
		const second = JSON.stringify(requests[1]!.input);
		const toolOutput = second.indexOf("function_call_output");
		const notice = second.indexOf("seq 3-4");
		assert(toolOutput >= 0 && notice > toolOutput, "the notice follows the tool result in the next model request");
		assert(second.indexOf("Try the upper half too.") > toolOutput, "so does the task message");
		assert.equal(JSON.stringify(requests[0]!.input).includes("seq 3-4"), false);
		const custom = manager.getBranch().filter((entry) => entry.type === "custom_message");
		assert.deepEqual(custom.map((entry) => (entry as { customType?: string }).customType),
			[HISTORY_MESSAGE_TYPE, TASK_MESSAGE_TYPE], "both are part of the persisted session (forks inherit them)");
		const reopened = SessionManager.open(manager.getSessionFile()!, join(root, "sessions"));
		assert.equal(deliveredSeq(reopened.getBranch()), 4, "recovery finds the notice in the session file");
		assert.deepEqual(deliveredTaskMessages(reopened.getBranch(), "turn-1"), ["hook:7:0"]);
		assert.equal(history.visibleSeq, 4, "the task message did not move the watermark");
		assert.deepEqual((await history.records()).map((r) => r.kind), ["turn_begin", "step_notice", "task_message"]);
	} finally {
		session?.dispose();
		await new Promise<void>((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
		await rm(root, { recursive: true, force: true });
	}
});
