import { readFile } from "node:fs/promises";
import type {
	ExtensionFactory,
	ToolCallEvent,
	ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import type { NetworkPolicy } from "./protocol.js";
import { atomicJson } from "./trace.js";
import { boundedQuestion } from "./vendor/deepseek-web/contract.js";
import { webError } from "./web-runtime.js";

interface ActivePolicyTurn {
	forbiddenTerms: string[];
	budgetPath: string;
	used: Record<string, number>;
	persist: Promise<void>;
}

export interface ToolUsageSnapshot {
	toolCalls: Record<string, number>;
	toolBudget: Record<string, { limit: number; used: number; remaining: number }>;
}

function normalizedHost(value: string): string {
	return value.trim().toLowerCase().replace(/^\*\./, "").replace(/\.$/, "");
}

function hostMatches(hostname: string, configured: string): boolean {
	const host = normalizedHost(hostname);
	const expected = normalizedHost(configured);
	return host === expected || host.endsWith(`.${expected}`);
}

export function assertAllowedUrl(value: string, policy: NetworkPolicy): void {
	let url: URL;
	try {
		url = new URL(value);
	} catch {
		throw new Error("web_fetch only accepts absolute HTTP(S) URLs");
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") {
		throw new Error("web_fetch only accepts HTTP(S) URLs");
	}
	if (url.username || url.password) throw new Error("web_fetch does not accept URLs with credentials");
	if (policy.deniedHosts.some((host) => hostMatches(url.hostname, host))) {
		throw new Error(`network policy denies host: ${url.hostname}`);
	}
	if (policy.allowedHosts.length > 0 && !policy.allowedHosts.some((host) => hostMatches(url.hostname, host))) {
		throw new Error(`network policy does not allow host: ${url.hostname}`);
	}
}

function values(value: unknown): string[] {
	if (typeof value === "string") return [value];
	if (Array.isArray(value)) return value.flatMap(values);
	return [];
}

function queryText(input: Record<string, unknown>): string {
	return values(input.queries).join("\n");
}

function strings(value: unknown, seen = new Set<object>()): string[] {
	if (typeof value === "string") return [value];
	if (value === null || typeof value !== "object" || seen.has(value)) return [];
	seen.add(value);
	if (Array.isArray(value)) return value.flatMap((item) => strings(item, seen));
	return Object.values(value).flatMap((item) => strings(item, seen));
}

function urls(value: unknown): string[] {
	const matches = strings(value).flatMap((text) => text.match(/https?:\/\/[^\s<>"'`]+/gi) ?? []);
	return matches.map((url) => url.replace(/[),.;!?\]}]+$/, ""));
}

function assertAllowedQuery(query: string, policy: NetworkPolicy, forbiddenTerms: string[]): void {
	const normalized = query.toLowerCase();
	for (const term of forbiddenTerms) {
		if (term.length > 0 && normalized.includes(term.toLowerCase())) {
			throw new Error("network policy blocks benchmark identifier searches");
		}
	}
	for (const pattern of policy.forbiddenQueryPatterns) {
		if (new RegExp(pattern, "i").test(query)) {
			throw new Error("network policy blocks this search query");
		}
	}
}

function fetchUrls(input: Record<string, unknown>): string[] {
	return [...values(input.url), ...values(input.urls)];
}

export class PolicyController {
	private active: ActivePolicyTurn | undefined;

	constructor(
		private readonly policy: NetworkPolicy,
		private readonly budgets: Readonly<Record<string, number>> = {},
	) {}

	async begin(forbiddenTerms: string[], budgetPath: string): Promise<void> {
		if (this.active) throw new Error("policy controller already has an active turn");
		const used = await readBudget(budgetPath, this.budgets);
		this.active = { forbiddenTerms, budgetPath, used, persist: Promise.resolve() };
		await this.persist(this.active);
	}

	end(): ToolUsageSnapshot {
		const active = this.active;
		this.active = undefined;
		return active ? snapshot(active.used, this.budgets) : snapshot({}, this.budgets);
	}

	snapshot(): ToolUsageSnapshot {
		return snapshot(this.active?.used ?? {}, this.budgets);
	}

	/** Optional search context must not bypass the benchmark query boundary. */
	searchContext(question: string): string | undefined {
		try {
			assertAllowedQuery(question, this.policy, this.active?.forbiddenTerms ?? []);
			const bounded = boundedQuestion(question);
			if (bounded !== undefined) assertAllowedQuery(bounded, this.policy, this.active?.forbiddenTerms ?? []);
			return bounded;
		} catch {
			return undefined;
		}
	}

	budgetMessage(): string {
		const finite = Object.entries(this.snapshot().toolBudget);
		if (finite.length === 0) return "Tool call budget for this turn: all available tools are unlimited.";
		return [
			"Tool call budget for this turn:",
			...finite.map(([name, state]) => `- ${name}: ${state.remaining} of ${state.limit} calls remaining`),
			"- Other available tools: unlimited",
		].join("\n");
	}

	createExtension(): ExtensionFactory {
		return (pi) => {
			pi.on("tool_call", (event) => this.toolCall(event));
			pi.on("tool_result", (event) => this.toolResult(event));
		};
	}

	private async toolCall(event: ToolCallEvent): Promise<{ block?: boolean; reason?: string; terminate?: boolean } | void> {
		const active = this.active;
		if (!active) return;
		const input = event.input as Record<string, unknown>;
		try {
			if (event.toolName === "web_search") {
				assertAllowedQuery(queryText(input), this.policy, active.forbiddenTerms);
			} else if (event.toolName === "web_fetch") {
				assertAllowedQuery(strings(input).join("\n"), this.policy, active.forbiddenTerms);
				for (const url of fetchUrls(input)) {
					assertAllowedUrl(url, this.policy);
				}
			} else if (event.toolName === "resolve-library-id" || event.toolName === "query-docs") {
				assertAllowedQuery(strings(input).join("\n"), this.policy, active.forbiddenTerms);
			}
		} catch (error) {
			return { block: true, reason: ["web_search", "web_fetch"].includes(event.toolName)
				? webError("WEB_POLICY_DENIED", (error as Error).message).message : (error as Error).message };
		}
		const limit = this.budgets[event.toolName];
		const used = active.used[event.toolName] ?? 0;
		if (limit !== undefined && used >= limit) {
			return {
				block: true,
				reason: JSON.stringify({
					ok: false,
					reason: "tool_budget_exhausted",
					tool: event.toolName,
					limit,
					used,
					remaining: 0,
					scope: "this_agent_this_turn",
					instruction: "Continue with existing evidence or use another available tool.",
				}),
			};
		}
		active.used[event.toolName] = used + 1;
		await this.persist(active);
	}

	private toolResult(event: ToolResultEvent): {
		content?: ToolResultEvent["content"];
		details?: unknown;
		isError?: boolean;
	} | void {
		const active = this.active;
		if (!active) return;
		let content = event.content;
		let details = event.details;
		let isError = event.isError;
		if (isError && ["web_search", "web_fetch"].includes(event.toolName)) {
			try {
				const block = event.content[0];
				const failure = JSON.parse(block?.type === "text" ? block.text : "");
				if (failure.type === "web_error") details = { error: failure };
			} catch { /* Non-web errors, such as the budget guard, keep their original details. */ }
		}
		if (["web_search", "web_fetch", "resolve-library-id", "query-docs"].includes(event.toolName)) {
			try {
				assertAllowedQuery(strings([event.content, event.details]).join("\n"), this.policy, active.forbiddenTerms);
				if (event.toolName === "web_search" || event.toolName === "web_fetch") {
					for (const url of urls([event.content, event.details])) {
						const parsed = new URL(url);
						if (this.policy.deniedHosts.some((host) => hostMatches(parsed.hostname, host))) {
							throw new Error(`network policy denies host: ${parsed.hostname}`);
						}
					}
				}
			} catch {
				content = [{ type: "text", text: "Network policy removed this tool result." }];
				details = { blocked: true };
				isError = true;
			}
		}
		const limit = this.budgets[event.toolName];
		if (limit !== undefined) {
			const used = active.used[event.toolName] ?? 0;
			content = [...content, {
				type: "text",
				text: `[Tool budget: ${event.toolName} has ${Math.max(0, limit - used)} of ${limit} calls remaining in this turn.]`,
			}];
		}
		if (content !== event.content || details !== event.details || isError !== event.isError) {
			return { content, details, isError };
		}
	}

	private async persist(active: ActivePolicyTurn): Promise<void> {
		active.persist = active.persist.then(() => atomicJson(
			active.budgetPath,
			persistedBudget(active.used, this.budgets),
		));
		await active.persist;
	}
}

function snapshot(
	used: Readonly<Record<string, number>>,
	limits: Readonly<Record<string, number>>,
): ToolUsageSnapshot {
	return {
		toolCalls: Object.fromEntries(Object.entries(used).sort(([left], [right]) => left.localeCompare(right))),
		toolBudget: Object.fromEntries(Object.entries(limits)
			.sort(([left], [right]) => left.localeCompare(right))
			.map(([name, limit]) => {
				const count = used[name] ?? 0;
				return [name, { limit, used: count, remaining: Math.max(0, limit - count) }];
			})),
	};
}

function persistedBudget(
	used: Readonly<Record<string, number>>,
	limits: Readonly<Record<string, number>>,
): Record<string, Record<string, number>> {
	return {
		limits: { ...limits },
		used: { ...used },
		remaining: Object.fromEntries(Object.entries(limits).map(([name, limit]) => [
			name,
			Math.max(0, limit - (used[name] ?? 0)),
		])),
	};
}

async function readBudget(
	path: string,
	limits: Readonly<Record<string, number>>,
): Promise<Record<string, number>> {
	let raw: unknown;
	try {
		raw = JSON.parse(await readFile(path, "utf8"));
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return {};
		throw error;
	}
	if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("invalid persisted tool budget");
	const data = raw as Record<string, unknown>;
	if (Object.keys(data).sort().join() !== "limits,remaining,used") throw new Error("invalid persisted tool budget");
	for (const key of ["limits", "used", "remaining"]) {
		if (!data[key] || typeof data[key] !== "object" || Array.isArray(data[key])) {
			throw new Error("invalid persisted tool budget");
		}
	}
	const savedLimits = data.limits as Record<string, unknown>;
	const used = data.used as Record<string, unknown>;
	const remaining = data.remaining as Record<string, unknown>;
	const limitNames = Object.keys(limits).sort();
	if (
		Object.keys(savedLimits).sort().join() !== limitNames.join()
		|| Object.keys(remaining).sort().join() !== limitNames.join()
		|| limitNames.some((name) => savedLimits[name] !== limits[name])
	) {
		throw new Error("persisted tool budget limits changed");
	}
	for (const [name, count] of Object.entries(used)) {
		if (!/^[A-Za-z0-9_-]+$/.test(name) || !Number.isInteger(count) || (count as number) < 0) {
			throw new Error("invalid persisted tool budget usage");
		}
	}
	for (const [name, limit] of Object.entries(limits)) {
		const count = (used[name] as number | undefined) ?? 0;
		if (count > limit || remaining[name] !== limit - count) {
			throw new Error("invalid persisted tool budget remaining count");
		}
	}
	return used as Record<string, number>;
}
