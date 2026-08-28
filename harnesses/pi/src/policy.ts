import type {
	ExtensionFactory,
	ToolCallEvent,
	ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import type { HarnessLimits, NetworkPolicy } from "./protocol.js";

interface ActivePolicyTurn {
	forbiddenTerms: string[];
	webCalls: number;
	context7Calls: number;
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

export class PolicyController {
	private active: ActivePolicyTurn | undefined;
	private readonly responseIds = new Set<string>();

	constructor(
		private readonly policy: NetworkPolicy,
		private readonly limits: HarnessLimits,
		private readonly webProvider: string,
	) {}

	begin(forbiddenTerms: string[]): void {
		if (this.active) throw new Error("policy controller already has an active turn");
		this.active = { forbiddenTerms, webCalls: 0, context7Calls: 0 };
	}

	end(): { webCalls: number; context7Calls: number } {
		const active = this.active;
		this.active = undefined;
		return { webCalls: active?.webCalls ?? 0, context7Calls: active?.context7Calls ?? 0 };
	}

	snapshot(): { webCalls: number; context7Calls: number } {
		return {
			webCalls: this.active?.webCalls ?? 0,
			context7Calls: this.active?.context7Calls ?? 0,
		};
	}

	createExtension(): ExtensionFactory {
		return (pi) => {
			pi.on("tool_call", (event) => this.toolCall(event));
			pi.on("tool_result", (event) => this.toolResult(event));
		};
	}

	private toolCall(event: ToolCallEvent): { block?: boolean; reason?: string; terminate?: boolean } | void {
		const active = this.active;
		if (!active) return;
		const input = event.input as Record<string, unknown>;
		let rejection: string | undefined;
		try {
			if (event.toolName === "web_search") {
				active.webCalls += 1;
				if (active.webCalls > this.limits.webCalls) throw new Error("web tool call limit reached");
				assertAllowedQuery(queryText(input), this.policy, active.forbiddenTerms);
				input.provider = this.webProvider;
				input.workflow = "none";
				input.domainFilter = [...this.policy.allowedHosts];
			} else if (event.toolName === "fetch_content") {
				active.webCalls += 1;
				if (active.webCalls > this.limits.webCalls) throw new Error("web tool call limit reached");
				assertAllowedQuery(strings(input).join("\n"), this.policy, active.forbiddenTerms);
				for (const url of fetchUrls(input)) {
					assertAllowedUrl(url, this.policy);
				}
			} else if (event.toolName === "get_search_content") {
				active.webCalls += 1;
				if (active.webCalls > this.limits.webCalls) throw new Error("web tool call limit reached");
				const responseId = input.responseId;
				if (typeof responseId !== "string" || !this.responseIds.has(responseId)) {
					throw new Error("responseId does not belong to this session");
				}
				assertAllowedQuery(strings(input).join("\n"), this.policy, active.forbiddenTerms);
			} else if (event.toolName === "resolve-library-id" || event.toolName === "query-docs") {
				active.context7Calls += 1;
				if (active.context7Calls > this.limits.context7Calls) throw new Error("Context7 call limit reached");
				assertAllowedQuery(strings(input).join("\n"), this.policy, active.forbiddenTerms);
			}
		} catch (error) {
			rejection = (error as Error).message;
		}
		if (rejection) return { block: true, reason: rejection };
	}

	private toolResult(event: ToolResultEvent): {
		content?: ToolResultEvent["content"];
		details?: unknown;
		isError?: boolean;
	} | void {
		const active = this.active;
		if (!active || !["web_search", "fetch_content", "get_search_content", "resolve-library-id", "query-docs"].includes(event.toolName)) {
			return;
		}
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
				const details = event.details as { responseId?: unknown } | undefined;
				if (typeof details?.responseId === "string") this.responseIds.add(details.responseId);
			}
		} catch {
			return {
				content: [{ type: "text", text: "Network policy removed this tool result." }],
				details: { blocked: true },
				isError: true,
			};
		}
	}
}
