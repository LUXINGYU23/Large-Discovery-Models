import { appendFile, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { Type, type TUnsafe } from "typebox";
import type { ExtensionAPI, ExtensionFactory, ToolDefinition } from "@earendil-works/pi-coding-agent";
import type {
	HarnessHistoryConfig,
	HistoryNotice,
	HistoryQuerier,
	HistoryQueryAnswer,
	SubmissionValidationDecision,
} from "./protocol.js";

/** Custom message type of history notices; part of the session, so forks and later turns inherit it. */
export const HISTORY_MESSAGE_TYPE = "ldm-experiment-history";
/** Custom entry (not model context) mirroring each visibility record line. */
export const VISIBILITY_ENTRY = "ldm-history-visibility-v1";
const VISIBILITY_FILE = "visibility.jsonl";

/** Custom message type of task messages (design 5.9 ``notify_agents``): hints, never experiment history. */
export const TASK_MESSAGE_TYPE = "ldm-task-message";
/**
 * Pi steering mode of research sessions. Pi defaults to "one-at-a-time", which would hold a task message queued
 * at the same step boundary as a history notice until one more model call; "all" delivers both before the next
 * call. The extension is the only steering source in a sidecar session (there is no interactive user).
 */
export const STEERING_MODE = "all";

export type NoticeChannel = "start_notice" | "step_notice" | "query_notice" | "cas_stale";

export interface TaskMessage {
	messageId: string;
	text: string;
	source: string;
}

export interface TaskMessageDetails {
	taskMessage: { messageId: string; turnId: string; source: string };
}

/** Task messages the persisted session contains (optionally only those delivered in one turn). */
export function deliveredTaskMessages(entries: readonly unknown[], turnId?: string): string[] {
	const ids: string[] = [];
	for (const raw of entries) {
		const entry = raw as { type?: string; customType?: string; details?: unknown };
		if (entry.type !== "custom_message" || entry.customType !== TASK_MESSAGE_TYPE) continue;
		const item = (entry.details as Partial<TaskMessageDetails> | undefined)?.taskMessage;
		if (typeof item?.messageId !== "string" || (turnId !== undefined && item.turnId !== turnId)) continue;
		if (!ids.includes(item.messageId)) ids.push(item.messageId);
	}
	return ids;
}

export interface VisibilityRecord {
	kind: "turn_begin" | NoticeChannel | "validated" | "turn_end" | "task_message";
	messageId?: string;
	visibleSeq: number;
	seenSeq: number;
	headSeq: number;
	fromSeq?: number;
	toSeq?: number;
	count?: number;
	historySeq?: number;
	casRetries?: number;
	at: string;
}

/** Machine-readable trace of a notice inside a persisted session entry (custom message or tool result). */
export interface HistoryNoticeDetails {
	historyNotice: { fromSeq: number; toSeq: number; count: number };
}

/**
 * Largest watermark the session itself proves the agent was told: notices are recovered from what Pi
 * persisted (steer/start custom messages and tool results), never from our own intent records.
 */
export function deliveredSeq(entries: readonly unknown[]): number {
	let best = 0;
	for (const raw of entries) {
		const entry = raw as { type?: string; customType?: string; details?: unknown; message?: { role?: string; details?: unknown } };
		const details = entry.type === "custom_message" && entry.customType === HISTORY_MESSAGE_TYPE ? entry.details
			: entry.type === "message" && entry.message?.role === "toolResult" ? entry.message.details : undefined;
		const toSeq = (details as Partial<HistoryNoticeDetails> | undefined)?.historyNotice?.toSeq;
		if (Number.isInteger(toSeq) && (toSeq as number) > best) best = toSeq as number;
	}
	return best;
}

function historySchema(config: HarnessHistoryConfig): Record<string, unknown> {
	const range = { type: "object", properties: { min: { type: "number" }, max: { type: "number" } }, additionalProperties: false };
	return {
		type: "object",
		additionalProperties: false,
		properties: {
			view: { type: "string", enum: ["summary", "observations", "pending", "proposals"],
				description: "summary (default): counts and best; observations: measured results; pending: experiments in flight (whole table); proposals: fate of submitted proposals." },
			filter: {
				type: "object",
				additionalProperties: false,
				properties: {
					status: { type: "array", items: { type: "string" } },
					since_seq: { type: "integer", minimum: 0, description: "Only entries with seq > since_seq (increments)." },
					until_seq: { type: "integer", minimum: 0 },
					generation: range,
					keys: { type: "array", items: { type: "string" }, description: "Candidate keys: has this been measured?" },
					fields: { type: "object", description: "Field conditions: {name: {eq} | {in: [...]} | {min, max}}." },
					mine: { type: "boolean", description: "proposals view: only this session's proposals." },
				},
			},
			sort: {
				type: "object",
				additionalProperties: false,
				properties: {
					by: { type: "string", description: "seq (default, newest first), objective, or a sortable field." },
					order: { type: "string", enum: ["asc", "desc"] },
				},
			},
			page_size: { type: "integer", minimum: 1, maximum: config.pageSizeMax },
			cursor: { type: "string", description: "next_cursor of the previous page of the same query." },
		},
	};
}

function fieldSummary(config: HarnessHistoryConfig): string {
	const describe = (flag: string) => config.fields.filter((field) => field[flag] === true).map((field) => String(field.name));
	return `Filterable fields: ${describe("filterable").join(", ") || "none"}. Sortable: seq, `
		+ `${config.objective ? "objective, " : ""}${describe("sortable").join(", ") || "none"}.`;
}

function noticeDetails(notice: HistoryNotice): HistoryNoticeDetails {
	return { historyNotice: { fromSeq: notice.from_seq, toSeq: notice.to_seq, count: notice.count } };
}

/**
 * Sidecar half of the shared experiment history (design 5.8): the query tool, notices and the session's
 * watermarks. It never reads the log; every read is a host query.
 *
 * Two watermarks: ``visible`` is what has been handed to the session (a notice in a tool result, a queued
 * steer message); ``seen`` is ``visible`` as of the start of the latest model call, i.e. what the model has
 * actually read. Submission CAS uses ``seen``. After a crash both are rebuilt from the persisted session,
 * so an unconfirmed notice is delivered again rather than assumed (at least once: duplicates are harmless,
 * a missed notice would let CAS pass a submission the agent was never told about).
 */
export class ExperimentHistory {
	private headSeq = 0;
	private visible = 0;
	private seen = 0;
	private consecutiveStale = 0;
	private active = false;
	private closed = false;
	private profileId = "";
	private turnId = "";
	private turnRoot = "";
	private query: HistoryQuerier | undefined;
	private pi: ExtensionAPI | undefined;
	private taskQueue: TaskMessage[] = [];
	private taskDelivered = new Set<string>();
	private readonly parameters: TUnsafe<Record<string, unknown>>;

	constructor(private readonly config: HarnessHistoryConfig) {
		this.parameters = Type.Unsafe<Record<string, unknown>>(historySchema(config));
	}

	get visibleSeq(): number {
		return this.visible;
	}

	get seenSeq(): number {
		return this.seen;
	}

	get knownHeadSeq(): number {
		return this.headSeq;
	}

	/** One-way host hint ("log head advanced"). Delivery happens at the next step boundary. */
	advanceHead(headSeq: number): void {
		this.headSeq = Math.max(this.headSeq, headSeq);
	}

	/**
	 * Start or resume a turn. The frozen turn input describes the log up to historyToSeq; anything later
	 * counts only if the persisted session (``sessionEntries``) contains the notice.
	 */
	async begin(profileId: string, turnId: string, historyToSeq: number, turnRoot: string, query: HistoryQuerier,
		sessionEntries: readonly unknown[] = []): Promise<void> {
		this.profileId = profileId;
		this.turnId = turnId;
		this.turnRoot = turnRoot;
		this.query = query;
		this.closed = false;
		this.visible = Math.max(historyToSeq, deliveredSeq(sessionEntries));
		this.seen = this.visible;
		this.consecutiveStale = 0;
		// Task messages: the session proves delivery; queued frames may have arrived before the turn began.
		this.taskDelivered = new Set(deliveredTaskMessages(sessionEntries));
		this.taskQueue = this.taskQueue.filter((message) => !this.taskDelivered.has(message.messageId));
		for (const item of await this.records()) {
			if (item.kind === "cas_stale") this.consecutiveStale += 1;
			if (item.kind === "validated") this.consecutiveStale = 0;
		}
		this.headSeq = Math.max(this.headSeq, this.visible);
		this.active = true;
		this.advanceHead((await this.ask("head", null)).headSeq);
		await this.record({ kind: "turn_begin" });
	}

	/**
	 * Before the first model call: the agent is idle, so the notice of (visible, head] is appended to the
	 * session directly (and persisted) instead of waiting for the first step boundary.
	 */
	async deliverAtStart(): Promise<boolean> {
		const delivered = await this.deliver("start_notice");
		if (delivered) this.seen = this.visible;
		await this.deliverTaskMessages();
		return delivered;
	}

	async end(): Promise<number> {
		if (this.active) await this.record({ kind: "turn_end" });
		this.active = false;
		this.query = undefined;
		// Undelivered task messages are the host's to resend (or to put into the next turn input).
		this.taskQueue = [];
		return this.visible;
	}

	/** One-way host frame; duplicates of a queued or delivered id are ignored. Delivered at a step boundary. */
	queueTaskMessage(message: TaskMessage): void {
		if (this.taskDelivered.has(message.messageId)
			|| this.taskQueue.some((item) => item.messageId === message.messageId)) return;
		this.taskQueue.push({ messageId: message.messageId, text: message.text, source: message.source });
	}

	/** Ids of task messages delivered in ``turnId`` according to the persisted session (the turn result field). */
	taskMessagesFor(turnId: string, sessionEntries: readonly unknown[]): string[] {
		return deliveredTaskMessages(sessionEntries, turnId);
	}

	/**
	 * Task messages go out as their own custom message type (a hint, marked as such): they never advance the
	 * visibility watermark and never reach the history tool or the submission CAS. Recorded, then delivered.
	 */
	async deliverTaskMessages(): Promise<number> {
		let count = 0;
		while (this.active && !this.closed && this.pi && this.taskQueue.length > 0) {
			const message = this.taskQueue.shift()!;
			if (this.taskDelivered.has(message.messageId)) continue;
			await this.record({ kind: "task_message", messageId: message.messageId });
			const details: TaskMessageDetails = {
				taskMessage: { messageId: message.messageId, turnId: this.turnId, source: message.source },
			};
			this.pi.sendMessage({
				customType: TASK_MESSAGE_TYPE,
				content: `[task message from ${message.source}] ${message.text}\nThis is a hint from the task, `
					+ "not a measurement; the experiment history is unchanged.",
				display: true,
				details,
			}, { deliverAs: "steer" });
			this.taskDelivered.add(message.messageId);
			count += 1;
		}
		return count;
	}

	/** No notifications after the terminal submission: a steer message would start another model call. */
	closeTurn(): void {
		this.closed = true;
	}

	createExtension(): ExtensionFactory {
		return (pi) => {
			this.pi = pi;
			// turn_end handlers are awaited before the agent loop polls the steering queue, so a steer
			// message queued here is delivered after this step's tool results and before the next model call.
			pi.on("turn_end", async () => {
				await this.deliverPending();
				await this.deliverTaskMessages();
			});
			// Everything handed to the session so far is in the context of this request.
			pi.on("before_provider_request", () => {
				this.seen = this.visible;
			});
		};
	}

	async deliverPending(): Promise<boolean> {
		return this.deliver("step_notice");
	}

	private async deliver(kind: "start_notice" | "step_notice"): Promise<boolean> {
		if (!this.active || this.closed || !this.pi || this.headSeq <= this.visible) return false;
		const answer = await this.ask("notify", null);
		if (!answer.notice || this.closed) return false;
		await this.accept(answer.notice, kind);
		this.pi.sendMessage({
			customType: HISTORY_MESSAGE_TYPE,
			content: answer.notice.text,
			display: true,
			details: noticeDetails(answer.notice),
		}, { deliverAs: "steer" });
		return true;
	}

	tool(): ToolDefinition<TUnsafe<Record<string, unknown>>> {
		const name = this.config.toolName;
		return {
			name,
			label: "Experiment history",
			description: "Query the authoritative experiment history of this campaign at a fixed watermark: measured "
				+ "results, experiments in flight and the fate of submitted proposals. Every response states its "
				+ "watermark and how many newer entries exist (unseen). Prefer summary, increments (filter.since_seq), "
				+ `top results (sort by objective) and key lookups over paging. ${fieldSummary(this.config)}`,
			promptSnippet: `${name}: query measured results, pending experiments and proposal outcomes`,
			parameters: this.parameters,
			executionMode: "sequential",
			execute: async (_toolCallId, params) => {
				const answer = await this.ask("query", { ...params });
				let notice = "";
				if (answer.notice) {
					await this.accept(answer.notice, "query_notice");
					notice = `\n\n${answer.notice.text}`;
				}
				// Errors are ordinary results so that the notice they carry is persisted with its details.
				const text = answer.error
					? `Query error ${answer.error.code}: ${answer.error.message}`
					: JSON.stringify(answer.result);
				return {
					content: [{ type: "text", text: `${text}${notice}` }],
					details: { headSeq: answer.headSeq, ...(answer.error ? { error: answer.error.code } : {}),
						...(answer.notice ? noticeDetails(answer.notice) : {}) },
				};
			},
		};
	}

	/** Fields the submission controller adds to its validation request: what the model has read. */
	submissionWatermark(): { visibleSeq: number; casRetries: number } {
		return { visibleSeq: this.seen, casRetries: this.consecutiveStale };
	}

	/** A stale answer: record and adopt the notice; returns the text and details for the tool result. */
	async stale(decision: SubmissionValidationDecision): Promise<{ text: string; details: HistoryNoticeDetails }> {
		if (decision.decision !== "stale" || !decision.notice || decision.headSeq === undefined) {
			throw new Error("stale submission answer lacks its notice");
		}
		this.advanceHead(decision.headSeq);
		this.consecutiveStale += 1;
		await this.accept(decision.notice, "cas_stale");
		return { text: decision.notice.text, details: noticeDetails(decision.notice) };
	}

	async validated(historySeq: number | undefined): Promise<void> {
		if (historySeq === undefined) throw new Error("history-enabled validation result lacks historySeq");
		this.consecutiveStale = 0;
		this.advanceHead(historySeq);
		await this.record({ kind: "validated", historySeq });
	}

	private async ask(operation: "query" | "notify" | "head", query: Record<string, unknown> | null): Promise<HistoryQueryAnswer> {
		if (!this.query) throw new Error("experiment history is not active for a turn");
		const answer = await this.query({ profileId: this.profileId, turnId: this.turnId, operation, query, visibleSeq: this.visible });
		if (answer.headSeq < this.visible) throw new Error("host history head precedes the session watermark");
		this.advanceHead(answer.headSeq);
		return answer;
	}

	private async accept(notice: HistoryNotice, kind: NoticeChannel): Promise<void> {
		if (notice.from_seq > this.visible || notice.to_seq > this.headSeq) {
			throw new Error(`history notice (${notice.from_seq}, ${notice.to_seq}] does not continue watermark ${this.visible}`);
		}
		// An intent record for audit, written before the hand-off. Recovery does not trust it: only the
		// persisted session proves delivery (see deliveredSeq).
		await this.record({ kind, fromSeq: notice.from_seq, toSeq: notice.to_seq, count: notice.count,
			...(kind === "cas_stale" ? { casRetries: this.consecutiveStale } : {}) });
		this.visible = Math.max(this.visible, notice.to_seq);
	}

	private async record(fields: Partial<VisibilityRecord> & Pick<VisibilityRecord, "kind">): Promise<void> {
		const item: VisibilityRecord = { visibleSeq: this.visible, seenSeq: this.seen, headSeq: this.headSeq,
			at: new Date().toISOString(), ...fields };
		await mkdir(this.turnRoot, { recursive: true });
		await appendFile(join(this.turnRoot, VISIBILITY_FILE), `${JSON.stringify(item)}\n`);
		this.pi?.appendEntry(VISIBILITY_ENTRY, { profileId: this.profileId, turnId: this.turnId, ...item });
	}

	async records(): Promise<VisibilityRecord[]> {
		let body: string;
		try {
			body = await readFile(join(this.turnRoot, VISIBILITY_FILE), "utf8");
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
			throw error;
		}
		// Only complete lines count; a torn final line from a crash is ignored.
		return body.split("\n").slice(0, -1).filter((line) => line.trim()).map((line) => JSON.parse(line) as VisibilityRecord);
	}
}
