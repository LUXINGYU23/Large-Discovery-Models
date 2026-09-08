import { readFile } from "node:fs/promises";
import type {
	ExtensionFactory,
	ToolCallEvent,
	ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import type { NetworkPolicy } from "./protocol.js";
import { atomicJson } from "./trace.js";

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
		throw new Error("fetch_content only accepts absolute HTTP(S) URLs");
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") {
		throw new Error("fetch_content only accepts HTTP(S) URLs");
	}
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
	return [...values(input.query), ...values(input.queries)].join("\n");
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

function providerRejection(
	reason: "invalid_provider_selection" | "provider_not_allowed",
	requested: unknown,
	allowed: readonly string[],
): Error {
	return new Error(`Web search provider rejected: ${JSON.stringify({
		accepted: false,
		reason,
		requested_provider: requested ?? null,
		allowed_providers: allowed,
		guidance: "Use provider auto or select only from allowed_providers.",
	})}`);
}

function normalizeProviderSelection(value: unknown, allowed: readonly string[]): "auto" | string | string[] {
	if (value === undefined || (typeof value === "string" && value.trim().toLowerCase() === "auto")) {
		return "auto";
	}
	const requested = Array.isArray(value) ? value : [value];
	if (
		requested.length === 0
		|| requested.some((provider) => typeof provider !== "string" || provider.trim().length === 0)
	) {
		throw providerRejection("invalid_provider_selection", value, allowed);
	}
	const normalized = (requested as string[]).map((provider) => provider.trim().toLowerCase());
	if (new Set(normalized).size !== normalized.length) {
		throw providerRejection("invalid_provider_selection", value, allowed);
	}
	if (normalized.some((provider) => !allowed.includes(provider))) {
		throw providerRejection("provider_not_allowed", value, allowed);
	}
	return Array.isArray(value) ? normalized : normalized[0] as string;
}

export class PolicyController {
	private active: ActivePolicyTurn | undefined;
	private readonly contentIds = new Set<string>();

	constructor(
		private readonly policy: NetworkPolicy,
		private readonly webProviders: readonly string[],
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
				input.provider = normalizeProviderSelection(input.provider, this.webProviders);
				input.workflow = "none";
				if (this.policy.allowedHosts.length > 0) {
					input.domainFilter = [...this.policy.allowedHosts];
				}
			} else if (event.toolName === "fetch_content") {
				assertAllowedQuery(strings(input).join("\n"), this.policy, active.forbiddenTerms);
				for (const url of fetchUrls(input)) {
					assertAllowedUrl(url, this.policy);
				}
			} else if (event.toolName === "get_search_content") {
				const responseId = input.responseId;
				if (typeof responseId !== "string" || !this.contentIds.has(responseId)) {
					throw new Error("responseId does not belong to this session");
				}
				assertAllowedQuery(strings(input).join("\n"), this.policy, active.forbiddenTerms);
			} else if (event.toolName === "resolve-library-id" || event.toolName === "query-docs") {
				assertAllowedQuery(strings(input).join("\n"), this.policy, active.forbiddenTerms);
			}
		} catch (error) {
			return { block: true, reason: (error as Error).message };
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
		if (["web_search", "fetch_content", "get_search_content", "resolve-library-id", "query-docs"].includes(event.toolName)) {
			try {
				assertAllowedQuery(strings([event.content, event.details]).join("\n"), this.policy, active.forbiddenTerms);
				if (event.toolName === "web_search" || event.toolName === "fetch_content" || event.toolName === "get_search_content") {
					for (const url of urls([event.content, event.details])) {
						const parsed = new URL(url);
						if (this.policy.deniedHosts.some((host) => hostMatches(parsed.hostname, host))) {
							throw new Error(`network policy denies host: ${parsed.hostname}`);
						}
					}
				}
				if ((event.toolName === "web_search" || event.toolName === "fetch_content") && !event.isError) {
					const resultDetails = event.details as {
						responseId?: unknown;
						searchId?: unknown;
						fetchId?: unknown;
					} | undefined;
					for (const id of [resultDetails?.responseId, resultDetails?.searchId, resultDetails?.fetchId]) {
						if (typeof id === "string") this.contentIds.add(id);
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
