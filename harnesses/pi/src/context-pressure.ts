import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { ContextMeter, type ContextEnvelope } from "./context-meter.js";
import { WebEvidenceArchive } from "./web-evidence.js";

export const CONTEXT_MEASUREMENT_ENTRY = "ldm-context-measurement-v1";
const ANCHOR_ENTRY = "ldm-context-anchor-v1";

/** Final context projection, shared by plain Pi and SoL-Pi, without altering canonical messages. */
export class ContextPressureController {
	readonly meter = new ContextMeter();
	readonly archive: WebEvidenceArchive;
	constructor(root: string, private readonly reserveTokens: number) { this.archive = new WebEvidenceArchive(root); }

	private envelope(pi: ExtensionAPI, ctx: ExtensionContext): ContextEnvelope {
		const active = new Set(pi.getActiveTools());
		return { provider: ctx.model?.provider ?? "", model: ctx.model?.id ?? "",
			systemPrompt: ctx.getSystemPrompt(),
			tools: pi.getAllTools().filter((tool) => active.has(tool.name)).map((tool) => ({
				name: tool.name, description: tool.description, parameters: tool.parameters,
			})),
			contextWindow: ctx.model?.contextWindow ?? 0, reserveTokens: this.reserveTokens };
	}

	createExtension(getMessages: () => AgentMessage[]): ExtensionFactory {
		return (pi) => {
			pi.on("session_start", (_event, ctx) => {
				const entry = ctx.sessionManager.getBranch().findLast((entry) => entry.type === "custom" && entry.customType === ANCHOR_ENTRY);
				if (entry?.type === "custom") this.meter.restore(entry.data);
			});
			pi.on("context", async (event, ctx) => {
				const envelope = this.envelope(pi, ctx);
				const before = this.meter.measure(event.messages, envelope);
				let projected = { messages: event.messages, archived: 0 };
				try { projected = await this.archive.project(event.messages, before.underPressure); }
				catch {
					// Pi swallows context-hook exceptions. Explicitly retain the original
					// evidence and record the failure instead of silently losing content.
					pi.appendEntry(CONTEXT_MEASUREMENT_ENTRY, { phase: "archive-failed", retainedOriginals: true });
				}
				const after = this.meter.measure(projected.messages, envelope);
				pi.appendEntry(CONTEXT_MEASUREMENT_ENTRY, { phase: "request", before, after,
					archivedResults: projected.archived, estimator: "pi-estimateTokens; provider input anchor only for matching envelope and prefix" });
				this.meter.prepare(projected.messages, envelope);
				return { messages: projected.messages };
			});
			pi.on("message_end", (event) => {
				if (event.message.role !== "assistant") return;
				const anchor = this.meter.observe(event.message);
				if (anchor) pi.appendEntry(ANCHOR_ENTRY, anchor);
				pi.appendEntry(CONTEXT_MEASUREMENT_ENTRY, { phase: "provider-usage", provider: event.message.provider,
					model: event.message.model, stopReason: event.message.stopReason, usage: event.message.usage });
			});
			pi.on("session_before_compact", async (event, ctx) => {
				const envelope = this.envelope(pi, ctx);
				const messages = getMessages();
				const before = this.meter.measure(messages, envelope);
				try {
					const projected = await this.archive.project(messages, before.underPressure || event.reason === "overflow");
					const after = this.meter.measure(projected.messages, envelope);
					const skipSummary = event.reason === "threshold" && projected.archived > 0 && !after.underPressure;
					pi.appendEntry(CONTEXT_MEASUREMENT_ENTRY, { phase: "before-compaction", reason: event.reason,
						before, after, archivedResults: projected.archived, skipSummary });
					if (skipSummary) return { cancel: true };
					// Preserve recall pointers in the native summarizer's input too. Manual
					// and overflow compaction still run even if our estimate looks smaller.
					event.preparation.messagesToSummarize = (await this.archive.project(event.preparation.messagesToSummarize, true)).messages;
					event.preparation.turnPrefixMessages = (await this.archive.project(event.preparation.turnPrefixMessages, true)).messages;
				} catch {
					pi.appendEntry(CONTEXT_MEASUREMENT_ENTRY, { phase: "archive-failed", retainedOriginals: true });
				}
			});
			pi.on("session_compact", (event) => {
				pi.appendEntry(CONTEXT_MEASUREMENT_ENTRY, { phase: "compaction-complete", reason: event.reason,
					usage: event.compactionEntry.usage, tokensBefore: event.compactionEntry.tokensBefore });
			});
		};
	}
}
