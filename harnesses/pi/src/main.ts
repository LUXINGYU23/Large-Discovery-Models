import { createInterface } from "node:readline";
import { PiSessionPool } from "./session.js";
import {
	ProtocolError,
	TurnExecutionError,
	parseFrame,
	type HistoryHeadAdvancedFrame,
	type HistoryQueryAnswer,
	type HistoryQueryRequest,
	type HistoryQueryResultFrame,
	type InputFrame,
	type ProviderAuthorizationRequest,
	type ProviderAuthorizationResultFrame,
	type SubmissionValidationDecision,
	type SubmissionValidationRequest,
	type SubmissionValidationResultFrame,
	type TaskMessageFrame,
} from "./protocol.js";
import { SIDECAR_RELEASE_VERSION } from "./release.js";
import { Redactor } from "./trace.js";

let apiKey: string | undefined;
let namedSecrets: Record<string, string> | undefined;
let campaignId: string | undefined;
let pool: PiSessionPool | undefined;
let redactor = new Redactor([]);
let closing = false;

type CommandFrame = Exclude<InputFrame, SubmissionValidationResultFrame | ProviderAuthorizationResultFrame
	| HistoryHeadAdvancedFrame | HistoryQueryResultFrame | TaskMessageFrame>;

class SubmissionValidationBroker {
	private readonly pending = new Map<string, {
		requestId: string;
		submissionDigest: string;
		resolve: (decision: SubmissionValidationDecision) => void;
		reject: (error: Error) => void;
	}>();
	private nextId = 0;
	private closed: Error | undefined;

	request(
		frame: { requestId: string; campaignId: string },
		request: SubmissionValidationRequest,
	): Promise<SubmissionValidationDecision> {
		if (this.closed) return Promise.reject(this.closed);
		this.nextId += 1;
		const validationId = `${frame.requestId}-validation-${this.nextId.toString().padStart(6, "0")}`;
		const result = new Promise<SubmissionValidationDecision>((resolve, reject) => {
			this.pending.set(validationId, {
				requestId: frame.requestId,
				submissionDigest: request.submissionDigest,
				resolve,
				reject,
			});
		});
		respondTo(frame, "submission_validation_requested", { validationId, ...request });
		return result;
	}

	resolve(frame: SubmissionValidationResultFrame): void {
		const pending = this.pending.get(frame.validationId);
		if (
			!pending
			|| pending.requestId !== frame.requestId
			|| pending.submissionDigest !== frame.submissionDigest
		) {
			throw new ProtocolError("invalid_state", `unknown submission validation: ${frame.validationId}`);
		}
		this.pending.delete(frame.validationId);
		const { type: _type, requestId: _requestId, protocolVersion: _version, campaignId: _campaign,
			validationId: _validationId, submissionDigest: _digest, ...decision } = frame;
		pending.resolve(decision);
	}

	rejectAll(error: Error): void {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}

	/** Shutdown: reject pending and later requests (an aborted turn's handlers may still ask). */
	close(error: Error): void {
		this.closed = error;
		this.rejectAll(error);
	}
}

const validations = new SubmissionValidationBroker();

class ProviderAuthorizationBroker {
	private readonly pending = new Map<string, {
		requestId: string;
		campaignId: string;
		providerRequestId: string;
		requestDigest: string;
		resolve: (authorized: boolean) => void;
		reject: (error: Error) => void;
	}>();
	private nextId = 0;
	private closed: Error | undefined;

	request(frame: { requestId: string; campaignId: string }, request: ProviderAuthorizationRequest): Promise<boolean> {
		if (this.closed) return Promise.reject(this.closed);
		this.nextId += 1;
		const authorizationId = `${frame.requestId}-authorization-${this.nextId.toString().padStart(6, "0")}`;
		const result = new Promise<boolean>((resolve, reject) => {
			this.pending.set(authorizationId, {
				requestId: frame.requestId,
				campaignId: frame.campaignId,
				providerRequestId: request.providerRequestId,
				requestDigest: request.requestDigest,
				resolve,
				reject,
			});
		});
		respondTo(frame, "provider_authorization_requested", { authorizationId, ...request });
		return result;
	}

	resolve(frame: ProviderAuthorizationResultFrame): void {
		const pending = this.pending.get(frame.authorizationId);
		if (
			!pending
			|| pending.requestId !== frame.requestId
			|| pending.campaignId !== frame.campaignId
			|| pending.providerRequestId !== frame.providerRequestId
			|| pending.requestDigest !== frame.requestDigest
		) {
			throw new ProtocolError("invalid_state", `unknown provider authorization: ${frame.authorizationId}`);
		}
		this.pending.delete(frame.authorizationId);
		pending.resolve(frame.authorized);
	}

	rejectAll(error: Error): void {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}

	/** Shutdown: reject pending and later requests (an aborted turn's handlers may still ask). */
	close(error: Error): void {
		this.closed = error;
		this.rejectAll(error);
	}
}

const authorizations = new ProviderAuthorizationBroker();

class HistoryQueryBroker {
	private readonly pending = new Map<string, {
		requestId: string;
		resolve: (answer: HistoryQueryAnswer) => void;
		reject: (error: Error) => void;
	}>();
	private nextId = 0;
	private closed: Error | undefined;

	request(frame: { requestId: string; campaignId: string }, request: HistoryQueryRequest): Promise<HistoryQueryAnswer> {
		if (this.closed) return Promise.reject(this.closed);
		this.nextId += 1;
		const queryId = `${frame.requestId}-history-${this.nextId.toString().padStart(6, "0")}`;
		const result = new Promise<HistoryQueryAnswer>((resolve, reject) => {
			this.pending.set(queryId, { requestId: frame.requestId, resolve, reject });
		});
		respondTo(frame, "history_query_requested", { queryId, ...request });
		return result;
	}

	resolve(frame: HistoryQueryResultFrame): void {
		const pending = this.pending.get(frame.queryId);
		if (!pending || pending.requestId !== frame.requestId) {
			throw new ProtocolError("invalid_state", `unknown history query: ${frame.queryId}`);
		}
		this.pending.delete(frame.queryId);
		pending.resolve({ headSeq: frame.headSeq, result: frame.result, notice: frame.notice, error: frame.error });
	}

	rejectAll(error: Error): void {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}

	/** Shutdown: reject pending and later requests (an aborted turn's handlers may still ask). */
	close(error: Error): void {
		this.closed = error;
		this.rejectAll(error);
	}
}

const historyQueries = new HistoryQueryBroker();

function respond(value: unknown): void {
	process.stdout.write(`${JSON.stringify(redactor.value(value))}\n`);
}

function respondTo(
	frame: { requestId: string; campaignId: string },
	type: string,
	fields: Record<string, unknown> = {},
): void {
	respond({
		type,
		requestId: frame.requestId,
		protocolVersion: SIDECAR_RELEASE_VERSION,
		campaignId: frame.campaignId,
		...fields,
	});
}

function errorCode(error: unknown): string {
	if (error instanceof TurnExecutionError && error.retryable) return "recoverable_turn_error";
	return error instanceof ProtocolError ? error.code : "sidecar_error";
}

async function close(): Promise<void> {
	if (closing) return;
	closing = true;
	validations.close(new Error("harness sidecar is closing"));
	authorizations.close(new Error("harness sidecar is closing"));
	historyQueries.close(new Error("harness sidecar is closing"));
	const activePool = pool;
	pool = undefined;
	try {
		await activePool?.close();
	} finally {
		apiKey = undefined;
		namedSecrets = undefined;
		campaignId = undefined;
		redactor = new Redactor([]);
	}
}

process.on("SIGTERM", () => {
	void close().finally(() => process.exit(143));
});
process.on("SIGINT", () => {
	void close().finally(() => process.exit(130));
});

respond({ type: "ready", protocolVersion: SIDECAR_RELEASE_VERSION });

async function handle(frame: CommandFrame): Promise<boolean> {
	if (frame.type === "bootstrap_secret") {
		if (apiKey || namedSecrets || pool) throw new ProtocolError("invalid_state", "bootstrap_secret is accepted exactly once before initialize");
		apiKey = frame.apiKey;
		namedSecrets = frame.namedSecrets;
		campaignId = frame.campaignId;
		redactor = new Redactor([apiKey, ...Object.values(namedSecrets)]);
		respondTo(frame, "secret_bootstrapped");
		return true;
	}
	if (frame.campaignId !== campaignId) throw new ProtocolError("invalid_state", "campaignId changed after bootstrap");
	if (frame.type === "initialize") {
		if (!apiKey || !namedSecrets) throw new ProtocolError("invalid_state", "bootstrap_secret must precede initialize");
		if (pool) throw new ProtocolError("invalid_state", "sidecar is already initialized");
		pool = new PiSessionPool(frame, apiKey, namedSecrets);
		try {
			await pool.initialize();
		} catch (error) {
			await pool.close();
			pool = undefined;
			throw error;
		}
		apiKey = undefined;
		namedSecrets = undefined;
		respondTo(frame, "initialized", {
			profiles: frame.profiles.map((profile) => profile.profileId),
			manifest: "manifest.json",
		});
		return true;
	}
	if (frame.type === "run_turn") {
		if (!pool) throw new ProtocolError("invalid_state", "sidecar is not initialized");
		const turns = await pool.runTurns(
			frame.turns,
			(request) => validations.request(frame, request),
			(request) => authorizations.request(frame, request),
			(request) => historyQueries.request(frame, request),
		);
		respondTo(frame, "turn_committed", { turns });
		return true;
	}
	if (frame.type === "close") {
		await close();
		respondTo(frame, "closed");
		return false;
	}
	return true;
}

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity, terminal: false });
let activeCommand: Promise<void> | undefined;
for await (const line of lines) {
	if (!line.trim()) continue;
	let requestId: string | undefined;
	let requestCampaignId: string | undefined;
	let requestType: string | undefined;
	let frame: InputFrame | undefined;
	try {
		const parsed = JSON.parse(line) as { requestId?: unknown; campaignId?: unknown; type?: unknown };
		if (typeof parsed.requestId === "string") requestId = parsed.requestId;
		if (typeof parsed.type === "string") requestType = parsed.type;
		if (typeof parsed.campaignId === "string") {
			requestCampaignId = parsed.campaignId;
		}
	} catch {
		// parseFrame returns the authoritative protocol error below.
	}
	try {
		frame = parseFrame(line);
	} catch (error) {
		if (requestType === "provider_authorization_result" && activeCommand) {
			authorizations.rejectAll(error as Error);
			continue;
		}
		if (requestType === "history_query_result" && activeCommand) {
			historyQueries.rejectAll(error as Error);
			continue;
		}
		// Hints are never answered, even when malformed: the host resends undelivered task messages or puts them
		// into the next turn input.
		if (requestType === "history_head_advanced" || requestType === "task_message") continue;
		respond({
			type: "error",
			requestId,
			protocolVersion: SIDECAR_RELEASE_VERSION,
			campaignId: requestCampaignId ?? campaignId,
			error: { code: errorCode(error), message: redactor.text((error as Error).message) },
		});
		continue;
	}
	if (frame.type === "submission_validation_result") {
		try {
			validations.resolve(frame);
		} catch (error) {
			respondTo(frame, "error", {
				error: { code: errorCode(error), message: redactor.text((error as Error).message) },
			});
		}
		continue;
	}
	if (frame.type === "provider_authorization_result") {
		try {
			authorizations.resolve(frame);
		} catch (error) {
			authorizations.rejectAll(error as Error);
		}
		continue;
	}
	if (frame.type === "history_head_advanced") {
		// Not a command: allowed while a turn runs; idle sessions simply remember the head.
		if (frame.campaignId === campaignId) pool?.advanceHistoryHead(frame.headSeq);
		continue;
	}
	if (frame.type === "task_message") {
		// Not a command either: queued per session, delivered between steps, deduplicated by messageId.
		if (frame.campaignId === campaignId) pool?.queueTaskMessage(frame);
		continue;
	}
	if (frame.type === "history_query_result") {
		try {
			historyQueries.resolve(frame);
		} catch (error) {
			historyQueries.rejectAll(error as Error);
		}
		continue;
	}
	if (activeCommand) {
		respondTo(frame, "error", {
			error: { code: "invalid_state", message: "sidecar is processing another command" },
		});
		continue;
	}
	const operation = (async () => {
		try {
			if (!(await handle(frame))) lines.close();
		} catch (error) {
			const failure = {
				code: errorCode(error),
				message: redactor.text((error as Error).message),
				...(error instanceof TurnExecutionError ? { turnUsage: error.turnUsage } : {}),
			};
			validations.rejectAll(error as Error);
			authorizations.rejectAll(error as Error);
			historyQueries.rejectAll(error as Error);
			respondTo(frame, "error", {
				error: failure,
			});
		}
	})();
	activeCommand = operation;
	void operation.finally(() => {
		if (activeCommand === operation) activeCommand = undefined;
	});
}

// Input ended. A host that closes normally waits for "closed" first; otherwise it is gone and can no longer authorize
// provider requests or answer validations and queries, so fail them now (as SIGTERM does) instead of letting an
// orphaned turn hold the session until its wall time. The persisted session is recovered by the next sidecar.
// Every step is bounded and the process exits explicitly: a guest VM or connection that does not release its handle
// must not keep an orphaned sidecar (and its session directory) alive. Sessions are persisted as they run.
async function settle(step: string, promise: Promise<unknown> | undefined, ms = 15_000): Promise<void> {
	let timer: NodeJS.Timeout | undefined;
	const expired = new Promise<boolean>((resolve) => { timer = setTimeout(() => resolve(true), ms); });
	const timedOut = await Promise.race([Promise.resolve(promise).then(() => false, () => false), expired]);
	clearTimeout(timer);
	if (timedOut) process.stderr.write(`sidecar shutdown: ${step} did not finish within ${ms} ms\n`);
}
if (activeCommand) await settle("close after input end", close());
await settle("active command", activeCommand);
await settle("close", close());
process.exit(0);
