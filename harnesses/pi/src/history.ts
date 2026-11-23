import { appendFile, readFile } from "node:fs/promises";
import { join } from "node:path";
import { compact, type ExtensionFactory, type SessionManager } from "@earendil-works/pi-coding-agent";
import { atomicJson } from "./trace.js";

// Leave room for a full research response plus newly returned tool data.
export const HISTORY_POLICY = {
	compaction: { enabled: true, reserveTokens: 65_536, keepRecentTokens: 20_000 },
	summaryReserveTokens: 40_960,
	summaryThinking: "high" as const,
	summaryTimeoutMs: 180_000,
	summaryRetry: { enabled: true, maxRetries: 1, baseDelayMs: 1_000 },
	maxContinuations: 2,
};

export class HistoryRecoveryError extends Error {}

const TRUNCATED_SUMMARY = /generation hit the token cap and the summary is incomplete/i;
const WALL_TIME_LIMIT = /^Session wall-time limit reached: \d+s;/;
const CONCISE_SUMMARY = "Write a concise summary of at most 2,000 words. Preserve decisions, relevant evidence, constraints, and unfinished work.";

interface RecoveryState {
	continuations: number;
	failure?: { message: string; compactionId: string | null };
}

/** Native Pi owns the session tree; this controller owns bounded recovery and diagnostics. */
export class HistoryController {
	private root: string | undefined;
	private manager: SessionManager | undefined;
	private state: RecoveryState = { continuations: 0 };
	private repairing = false;

	async begin(
		root: string, manager: SessionManager, repair?: () => Promise<unknown>, recoverTimedOutTurn = false,
		workerPreflightRepaired = false,
	): Promise<void> {
		this.root = root;
		this.manager = manager;
		try {
			this.state = JSON.parse(await readFile(join(root, "history-recovery.json"), "utf8"));
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
			this.state = { continuations: 0 };
		}
		const failure = this.state.failure;
		if (workerPreflightRepaired && failure && WALL_TIME_LIMIT.test(failure.message)) {
			// The host has independently verified the timed-out worker never launched a model.
			await this.record({ type: "history_repaired", reason: "worker_preflight_repaired",
				compactionId: this.compactionId(), previousFailure: failure.message });
			this.state = { continuations: this.state.continuations };
			await this.save();
		} else if (repair && failure && failure.compactionId === this.compactionId()
			&& (TRUNCATED_SUMMARY.test(failure.message)
				|| (recoverTimedOutTurn && WALL_TIME_LIMIT.test(failure.message)))) {
			await this.repairHistory(repair);
		}
		// A new native compaction is the durable proof that the history was repaired.
		if (this.state.failure && this.state.failure.compactionId !== this.compactionId()) {
			this.state = { continuations: 0 };
			await this.save();
			await this.record({ type: "history_repaired", compactionId: this.compactionId() });
		}
		this.assertReady();
	}

	private async repairHistory(repair: () => Promise<unknown>): Promise<void> {
		const blocked = this.state;
		this.state = { continuations: blocked.continuations };
		this.repairing = true;
		try {
			await repair();
		} finally {
			// The on-disk marker remains until a native summary has been committed.
			this.state = blocked;
			this.repairing = false;
			await this.save();
		}
	}

	assertReady(): void {
		if (this.state.failure) throw new HistoryRecoveryError(this.state.failure.message);
	}

	async fail(message: string): Promise<void> {
		if (!this.state.failure) {
			this.state.failure = { message, compactionId: this.compactionId() };
			await this.save();
			await this.record({ type: "history_blocked", ...this.state.failure });
		}
	}

	async continueTurn(stopReason: string | undefined, recoverLength?: () => Promise<void>): Promise<void> {
		this.assertReady();
		if (this.state.continuations >= HISTORY_POLICY.maxContinuations) {
			await this.fail("Submission recovery exhausted: no accepted submission after two continuation prompts.");
			this.assertReady();
		}
		if (stopReason === "length") {
			if (!recoverLength) {
				await this.fail("Truncated output requires successful native compaction before continuation.");
				this.assertReady();
			}
			// Charge before recovery so a restart cannot grant another attempt.
			this.state.continuations += 1;
			await this.save();
			try {
				await recoverLength!();
				this.assertReady();
			} catch (error) {
				await this.fail(`Truncated-output compaction failed: ${error instanceof Error ? error.message : String(error)}`);
				this.assertReady();
			}
		} else {
			this.state.continuations += 1;
			await this.save();
		}
		await this.record({ type: "continuation", attempt: this.state.continuations, stopReason });
	}

	compactedAfter(timestamp: number): boolean {
		const entry = this.manager?.getBranch().findLast((entry) => entry.type === "compaction");
		return !!entry && new Date(entry.timestamp).getTime() >= timestamp;
	}

	createExtension(proxyApiKey: string): ExtensionFactory {
		return (pi) => {
			pi.on("session_before_compact", async (event, ctx) => {
				if (this.state.failure) return { cancel: true };
				const started = Date.now();
				await this.record({ type: "compaction_started", reason: event.reason,
					tokensBefore: event.preparation.tokensBefore,
					thinking: this.repairing ? "low" : HISTORY_POLICY.summaryThinking });
				try {
					if (!ctx.model) throw new Error("Compaction requires a model");
					const model = { ...ctx.model, baseUrl: `${ctx.model.baseUrl}/compaction` };
					const summarize = (thinking: "high" | "low", instructions?: string) => compact({ ...event.preparation,
						settings: { ...event.preparation.settings, reserveTokens: HISTORY_POLICY.summaryReserveTokens },
					}, model, proxyApiKey, undefined, instructions,
					AbortSignal.any([event.signal, AbortSignal.timeout(HISTORY_POLICY.summaryTimeoutMs)]),
						thinking, undefined, undefined, HISTORY_POLICY.summaryRetry, {
							onRetryScheduled: (attempt, maxAttempts, delayMs, errorMessage) =>
								this.record({ type: "compaction_retry", attempt, maxAttempts, delayMs, errorMessage }),
						});
					const conciseInstructions = [event.customInstructions, CONCISE_SUMMARY].filter(Boolean).join("\n\n");
					let result;
					if (this.repairing) {
						result = await summarize("low", conciseInstructions);
					} else {
						try {
							result = await summarize(HISTORY_POLICY.summaryThinking, event.customInstructions);
						} catch (error) {
							if (event.signal.aborted || !TRUNCATED_SUMMARY.test(String(error))) throw error;
							await this.record({ type: "compaction_retry", reason: "truncated_summary", thinking: "low" });
							result = await summarize("low", conciseInstructions);
						}
					}
					await this.record({ type: "compaction_generated", reason: event.reason,
						durationMs: Date.now() - started, tokensBefore: result.tokensBefore,
						summaryChars: result.summary.length, usage: result.usage });
					return { compaction: result };
				} catch (error) {
					// Returning cancel prevents Pi from falling back to the same failed summary call.
					if (!event.signal.aborted) {
						await this.fail(`History compaction failed: ${error instanceof Error ? error.message : String(error)}`);
					}
					await this.record({ type: "compaction_failed", durationMs: Date.now() - started,
						aborted: event.signal.aborted });
					return { cancel: true };
				}
			});
			pi.on("session_compact", async (event) => {
				await this.record({ type: "compaction_committed", reason: event.reason,
					compactionId: event.compactionEntry.id, firstKeptEntryId: event.compactionEntry.firstKeptEntryId });
			});
			pi.on("session_compact_failed", async (event) => {
				if (!event.aborted) await this.fail(`Native history recovery failed: ${event.errorMessage ?? "unknown error"}`);
			});
		};
	}

	private compactionId(): string | null {
		return this.manager?.getBranch().findLast((entry) => entry.type === "compaction")?.id ?? null;
	}

	private async save(): Promise<void> {
		if (!this.root) throw new Error("History recovery requires an active turn");
		await atomicJson(join(this.root, "history-recovery.json"), this.state);
	}

	private async record(event: Record<string, unknown>): Promise<void> {
		if (!this.root) throw new Error("History event requires an active turn");
		await appendFile(join(this.root, "history-events.jsonl"), `${JSON.stringify({ at: new Date().toISOString(), ...event })}\n`, { mode: 0o600 });
	}
}
