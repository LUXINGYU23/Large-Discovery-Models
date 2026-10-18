import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import { convertToLlm, estimateTokens } from "@earendil-works/pi-coding-agent";
import { canonicalSha256 } from "./trace.js";

export interface ContextEnvelope {
	provider: string;
	model: string;
	systemPrompt: string;
	tools: unknown[];
	contextWindow: number;
	reserveTokens: number;
}
export interface ContextAnchor {
	envelopeHash: string;
	prefixHash: string;
	prefixLength: number;
	inputTokens: number;
}
export interface ContextMeasurement {
	estimatedInputTokens: number;
	basis: "heuristic" | "provider-anchored";
	contextWindow: number;
	reserveTokens: number;
	underPressure: boolean;
	heuristic: { system: number; tools: number; messages: number; webResults: number };
}
function textTokens(text: string): number {
	return text ? estimateTokens({ role: "user", content: text, timestamp: 0 }) : 0;
}
function prefixHash(messages: AgentMessage[]): string { return canonicalSha256(convertToLlm(messages)); }
function envelopeHash(envelope: ContextEnvelope): string { return canonicalSha256(envelope); }

/** One estimator for pressure, pruning decisions, and recorded diagnostics. Counts are estimates, not billing. */
export class ContextMeter {
	private anchor: ContextAnchor | undefined;
	private pending: Omit<ContextAnchor, "inputTokens"> | undefined;

	restore(value: unknown): void {
		if (!value || typeof value !== "object") return;
		const data = value as ContextAnchor;
		if (/^[a-f0-9]{64}$/.test(data.envelopeHash) && /^[a-f0-9]{64}$/.test(data.prefixHash)
			&& Number.isSafeInteger(data.prefixLength) && data.prefixLength >= 0
			&& Number.isSafeInteger(data.inputTokens) && data.inputTokens > 0) this.anchor = data;
	}

	measure(messages: AgentMessage[], envelope: ContextEnvelope): ContextMeasurement {
		const sizes = messages.map(estimateTokens);
		const heuristic = {
			system: textTokens(envelope.systemPrompt), tools: textTokens(JSON.stringify(envelope.tools)),
			messages: sizes.reduce((sum, size) => sum + size, 0),
			webResults: messages.reduce((sum, message, index) => sum +
				(message.role === "toolResult" && ["web_search", "web_fetch"].includes(message.toolName) ? sizes[index]! : 0), 0),
		};
		let estimatedInputTokens = heuristic.system + heuristic.tools + heuristic.messages;
		let basis: ContextMeasurement["basis"] = "heuristic";
		const anchor = this.anchor;
		if (anchor && anchor.envelopeHash === envelopeHash(envelope) && anchor.prefixLength <= messages.length
			&& anchor.prefixHash === prefixHash(messages.slice(0, anchor.prefixLength))) {
			estimatedInputTokens = anchor.inputTokens + sizes.slice(anchor.prefixLength).reduce((sum, size) => sum + size, 0);
			basis = "provider-anchored";
		}
		return { estimatedInputTokens, basis, heuristic, contextWindow: envelope.contextWindow,
			reserveTokens: envelope.reserveTokens, underPressure: estimatedInputTokens + envelope.reserveTokens >= envelope.contextWindow };
	}

	prepare(messages: AgentMessage[], envelope: ContextEnvelope): void {
		this.pending = { envelopeHash: envelopeHash(envelope), prefixHash: prefixHash(messages), prefixLength: messages.length };
	}

	observe(message: AssistantMessage): ContextAnchor | undefined {
		const pending = this.pending;
		this.pending = undefined;
		if (!pending || message.stopReason === "error" || message.stopReason === "aborted") return undefined;
		const usage = message.usage;
		const inputTokens = usage.input + usage.cacheRead + usage.cacheWrite;
		if (!Number.isSafeInteger(inputTokens) || inputTokens <= 0) return undefined;
		this.anchor = { ...pending, inputTokens };
		return this.anchor;
	}
}
