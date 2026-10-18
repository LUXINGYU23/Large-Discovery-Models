import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import { httpError, runWebTool, scheduleWebRequest, WebToolError, WEB_RUNTIME } from "./web-runtime.js";
import { DeepSeekWebTools } from "./deepseek-web.js";

test("errors retain codes, retry advice and safe HTTP metadata", () => {
	for (const [status, code, retryable] of [[401, "WEB_AUTH", false], [402, "WEB_QUOTA", false],
		[403, "WEB_POLICY_DENIED", false], [404, "WEB_HTTP_ERROR", false], [429, "WEB_RATE_LIMIT", true], [503, "WEB_HTTP_ERROR", true]] as const) {
		const failure = httpError(status, "2");
		assert.equal(failure.failure.code, code);
		assert.equal(failure.failure.retryable, retryable);
		assert.equal(JSON.parse(failure.message).retryAfterMs, 2000);
	}
	assert.equal(httpError(429, "invalid").failure.retryAfterMs, undefined);
});

test("deadline and caller cancellation are distinct, and both wait for task cleanup", async () => {
	let release!: () => void;
	const drained = new Promise<void>((resolve) => { release = resolve; });
	let observedAbort!: () => void;
	const aborted = new Promise<void>((resolve) => { observedAbort = resolve; });
	let settled = false;
	const pending = runWebTool(undefined, async (signal) => {
		await new Promise<void>((resolve) => signal.addEventListener("abort", () => { observedAbort(); resolve(); }, { once: true }));
		await drained;
	}, 10).finally(() => { settled = true; });
	const check = assert.rejects(pending, (error: unknown) => error instanceof WebToolError && error.failure.code === "TOOL_TIMEOUT");
	await aborted;
	assert.equal(settled, false);
	release();
	await check;
	await assert.rejects(runWebTool(AbortSignal.abort(new DOMException("outer deadline", "TimeoutError")), async () => {}),
		(error: unknown) => error instanceof WebToolError && error.failure.code === "WEB_ABORTED");
});

test("credential pool is bounded, FIFO, abortable while queued and retains slots until cleanup", async () => {
	const key = "queue-test";
	const limit = WEB_RUNTIME.concurrency;
	const initialOrder = Array.from({ length: limit }, (_, index) => index);
	const controller = new AbortController();
	const releases: Array<() => void> = [];
	const order: number[] = [];
	let active = 0;
	let peak = 0;
	const task = (index: number) => async () => {
		order.push(index); peak = Math.max(peak, ++active);
		await new Promise<void>((resolve) => { releases[index] = resolve; });
		active--;
	};
	const running = initialOrder.map((index) => scheduleWebRequest(key, controller.signal, task(index)));
	await setImmediate();
	const queuedAbort = new AbortController();
	const cancelled = scheduleWebRequest(key, queuedAbort.signal, task(limit));
	const check = assert.rejects(cancelled, (error: unknown) => error instanceof WebToolError && error.failure.code === "WEB_ABORTED");
	const next = scheduleWebRequest(key, new AbortController().signal, task(limit + 1));
	queuedAbort.abort();
	controller.abort();
	await check;
	await setImmediate();
	assert.equal(active, limit);
	assert.deepEqual(order, initialOrder);
	releases[0]!();
	await setImmediate();
	assert.deepEqual(order, [...initialOrder, limit + 1]);
	for (const release of releases) release?.();
	await Promise.all([...running, next]);
	assert.equal(peak, limit);
});

test("queue saturation rejects before dispatch and cancelled waiters release capacity", async () => {
	let release!: () => void;
	const held = new Promise<void>((resolve) => { release = resolve; });
	const running = Array.from({ length: WEB_RUNTIME.concurrency }, () => scheduleWebRequest("saturation", new AbortController().signal, () => held));
	const controllers = Array.from({ length: WEB_RUNTIME.maxQueued }, () => new AbortController());
	let dispatched = 0;
	const queued = controllers.map((controller) => scheduleWebRequest("saturation", controller.signal, async () => { dispatched++; }));
	const drained = Promise.allSettled(queued);
	await assert.rejects(scheduleWebRequest("saturation", new AbortController().signal, async () => {}),
		(error: unknown) => error instanceof WebToolError && error.failure.code === "WEB_QUEUE_FULL");
	for (const controller of controllers) controller.abort();
	await drained;
	assert.equal(dispatched, 0);
	release();
	await Promise.all(running);
	await scheduleWebRequest("saturation", new AbortController().signal, async () => { dispatched++; });
	assert.equal(dispatched, 1);
});

test("separate clients with the same credential share network concurrency and record failures", async (t) => {
	let active = 0;
	let peak = 0;
	t.mock.method(globalThis, "fetch", async () => {
		peak = Math.max(peak, ++active);
		await setImmediate();
		active--;
		return new Response("private provider diagnostic", { status: 429, headers: { "retry-after": "3" } });
	});
	const clients = Array.from({ length: WEB_RUNTIME.concurrency + 5 }, () => new DeepSeekWebTools("shared-credential"));
	const results = await Promise.allSettled(clients.map((client) => client.toolDefinitions({
		allowedHosts: [], deniedHosts: [], forbiddenQueryPatterns: [],
	})[0]!.execute("call", { queries: ["research"] }, undefined, undefined, {} as never)));
	assert.equal(peak, WEB_RUNTIME.concurrency);
	for (const result of results) {
		assert.equal(result.status, "rejected");
		if (result.status !== "rejected") continue;
		const error = result.reason as WebToolError;
		assert.equal(error.failure.code, "WEB_RATE_LIMIT");
		assert.equal(error.failure.retryAfterMs, 3000);
		assert.equal(error.failure.requests?.length, 1);
		assert.equal(error.failure.requests?.[0]?.httpStatus, 429);
		assert(!error.message.includes("private provider diagnostic"));
		assert(!error.message.includes("shared-credential"));
	}
});
