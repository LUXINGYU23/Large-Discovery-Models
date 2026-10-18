import PQueue from "p-queue";
import { sha256 } from "./trace.js";

export const WEB_RUNTIME = { concurrency: 4, maxQueued: 128, timeoutMs: 30_000 } as const;
export type WebErrorCode = "WEB_AUTH" | "WEB_QUOTA" | "WEB_RATE_LIMIT" | "WEB_HTTP_ERROR"
	| "WEB_NETWORK" | "WEB_INVALID_RESPONSE" | "WEB_INVALID_REQUEST" | "WEB_POLICY_DENIED"
	| "WEB_QUEUE_FULL" | "WEB_ABORTED" | "TOOL_TIMEOUT" | "WEB_INTERNAL";
export interface WebRequestMetric {
	endpoint: "search" | "fetch";
	queueMs: number;
	durationMs: number;
	dispatched: boolean;
	httpStatus?: number;
	errorCode?: WebErrorCode;
}
export interface WebFailure {
	type: "web_error";
	code: WebErrorCode;
	message: string;
	retryable: boolean;
	httpStatus?: number;
	retryAfterMs?: number;
	requests?: WebRequestMetric[];
}

/** Pi serializes thrown messages, so carry the stable error envelope through that boundary. */
export class WebToolError extends Error {
	constructor(readonly failure: WebFailure) {
		super(JSON.stringify(failure));
		this.name = "WebToolError";
	}
}
export function webError(code: WebErrorCode, message: string, retryable = false,
	fields: Partial<Pick<WebFailure, "httpStatus" | "retryAfterMs">> = {}): WebToolError {
	return new WebToolError({ type: "web_error", code, message, retryable, ...fields });
}
export function abortError(signal: AbortSignal): WebToolError {
	return signal.reason instanceof WebToolError && signal.reason.failure.code === "TOOL_TIMEOUT"
		? signal.reason : webError("WEB_ABORTED", "Web operation was cancelled");
}
export function httpError(status: number, retryAfter: string | null): WebToolError {
	const code = status === 401 || status === 403 ? "WEB_AUTH" : status === 402 ? "WEB_QUOTA"
		: status === 429 ? "WEB_RATE_LIMIT" : "WEB_HTTP_ERROR";
	const seconds = retryAfter === null ? NaN : Number(retryAfter);
	const wait = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(retryAfter ?? "") - Date.now();
	return webError(code, `DeepSeek request failed (HTTP ${status})`, status === 408 || status === 429 || status >= 500,
		{ httpStatus: status, ...(Number.isFinite(wait) ? { retryAfterMs: Math.max(0, wait) } : {}) });
}

// Shared by every profile/client using the same credential in this sidecar process.
// Keys are hashed; idle queues are removed. Other processes have their own pools.
const queues = new Map<string, PQueue>();
function credentialQueue(key: string): PQueue {
	const id = sha256(key);
	let queue = queues.get(id);
	if (!queue) {
		queue = new PQueue({ concurrency: WEB_RUNTIME.concurrency });
		queues.set(id, queue);
		const created = queue;
		queue.addEventListener("idle", () => { if (queues.get(id) === created) queues.delete(id); }, { once: true });
	}
	return queue;
}

export async function scheduleWebRequest<T>(key: string, signal: AbortSignal, task: () => Promise<T>): Promise<T> {
	if (signal.aborted) throw abortError(signal);
	const queue = credentialQueue(key);
	if (queue.size >= WEB_RUNTIME.maxQueued) throw webError("WEB_QUEUE_FULL", "Web request queue is full", true);
	const waiting = new AbortController();
	const cancelWaiting = () => waiting.abort(abortError(signal));
	signal.addEventListener("abort", cancelWaiting, { once: true });
	try {
		return await queue.add(async () => {
			// PQueue may reject a running task early on abort. Only its waiting phase uses
			// that signal: the running task must drain before its concurrency slot is freed.
			signal.removeEventListener("abort", cancelWaiting);
			if (signal.aborted) throw abortError(signal);
			return await task();
		}, { signal: waiting.signal });
	} finally {
		signal.removeEventListener("abort", cancelWaiting);
	}
}

export async function runWebTool<T>(signal: AbortSignal | undefined,
	action: (signal: AbortSignal, requests: WebRequestMetric[]) => Promise<T>,
	timeoutMs: number = WEB_RUNTIME.timeoutMs): Promise<T> {
	const deadline = new AbortController();
	const timer = setTimeout(() => deadline.abort(webError("TOOL_TIMEOUT", `Web tool timed out after ${timeoutMs}ms`, true)), timeoutMs);
	const active = signal ? AbortSignal.any([signal, deadline.signal]) : deadline.signal;
	const requests: WebRequestMetric[] = [];
	try {
		if (active.aborted) throw abortError(active);
		const result = await action(active, requests);
		if (active.aborted) throw abortError(active);
		return result;
	} catch (error) {
		const normalized = active.aborted ? abortError(active)
			: error instanceof WebToolError ? error : webError("WEB_INTERNAL", "Web tool failed unexpectedly");
		throw new WebToolError({ ...normalized.failure, requests });
	} finally {
		clearTimeout(timer);
	}
}
