import { setTimeout as delay } from "node:timers/promises";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Type, type TUnsafe } from "typebox";
import { z } from "zod";
import { assertAllowedUrl } from "./policy.js";
import type { NetworkPolicy } from "./protocol.js";

const API_ROOT = "https://api.deepseek.com/v1/tools";
const query = z.string().refine((value) => value.trim().length > 0 && [...value].length <= 128,
	"must contain 1–128 Unicode characters and not be blank");
const searchInput = z.object({
	query,
	count: z.number().int().min(1).max(50).default(10),
	ori_question: query.optional(),
	need_content: z.boolean().default(false),
}).strict();
const fetchInput = z.object({ url: z.string().refine((value) => {
	try {
		const url = new URL(value);
		return [...value].length <= 8192 && ["http:", "https:"].includes(url.protocol)
			&& !!url.hostname && !url.username && !url.password;
	} catch { return false; }
}, "must be an HTTP(S) URL without credentials, at most 8192 characters") }).strict();
const searchOutput = z.object({ results: z.array(z.object({
	title: z.string(), url: z.string().url(), snippet: z.string(), summary: z.string(),
	content: z.string(), rerank_score: z.number(), authority: z.number().int(),
	site_name: z.string().nullable(), site_icon: z.string().nullable(), date: z.string().nullable(),
})) });
const fetchOutput = z.object({ title: z.string(), content: z.string() });

/** One instance per pool shares independent Search/Fetch request pacing across profiles. */
export class DeepSeekWebTools {
	private readonly nextRequest = { search: 0, fetch: 0 };

	constructor(private readonly apiKey: string | undefined) {}

	requireKey(): string {
		if (!this.apiKey?.trim()) throw new Error("DEEPSEEK_API_KEY is required for Pi Search and Fetch");
		return this.apiKey;
	}

	toolDefinitions(policy: NetworkPolicy): Array<ToolDefinition<TUnsafe<Record<string, unknown>>>> {
		return [
			{
				name: "web_search", label: "Search the web",
				description: "Search with DeepSeek. Use a single query of at most 128 Unicode characters. Supports site:, -site:, lang:, after:, before: and quoted phrases. Returns ranked sources and summaries; need_content includes available page text. Use fetch_content for full Markdown.",
				parameters: Type.Unsafe<Record<string, unknown>>({
					type: "object", additionalProperties: false, required: ["query"],
					properties: {
						query: { type: "string", minLength: 1, maxLength: 128 },
						count: { type: "integer", minimum: 1, maximum: 50, default: 10 },
						ori_question: { type: "string", minLength: 1, maxLength: 128 },
						need_content: { type: "boolean", default: false },
					},
				}),
				executionMode: "sequential",
				execute: async (_id, input, signal) => {
					const body = searchInput.parse(input);
					const response = searchOutput.safeParse(await this.request("search", body, signal));
					if (!response.success) throw new Error("DeepSeek Search returned an invalid response");
					// Filter before returning either text or details to the model/history.
					const results = response.data.results.filter((result) => {
						try { assertAllowedUrl(result.url, policy); return true; } catch { return false; }
					});
					return this.result({ results });
				},
			},
			{
				name: "fetch_content", label: "Fetch web content",
				description: "Fetch one HTTP(S) page with DeepSeek and return its title and full Markdown content.",
				parameters: Type.Unsafe<Record<string, unknown>>({
					type: "object", additionalProperties: false, required: ["url"],
					properties: { url: { type: "string", minLength: 1, maxLength: 8192 } },
				}),
				executionMode: "sequential",
				execute: async (_id, input, signal) => {
					const body = fetchInput.parse(input);
					assertAllowedUrl(body.url, policy);
					const response = fetchOutput.safeParse(await this.request("fetch", body, signal));
					if (!response.success) throw new Error("DeepSeek Fetch returned an invalid response");
					if (!response.data.content.trim()) throw new Error("DeepSeek Fetch returned no page content");
					return this.result({ url: body.url, ...response.data });
				},
			},
		];
	}

	private result(data: unknown) {
		return { content: [{ type: "text" as const, text: JSON.stringify(data) }], details: { provider: "deepseek" } };
	}

	private async request(endpoint: "search" | "fetch", body: unknown, signal?: AbortSignal): Promise<unknown> {
		const key = this.requireKey();
		const timeout = AbortSignal.timeout(60_000);
		const activeSignal = signal ? AbortSignal.any([signal, timeout]) : timeout;
		for (let attempt = 0; attempt < 3; attempt++) {
			activeSignal.throwIfAborted();
			const now = Date.now();
			const start = Math.max(now, this.nextRequest[endpoint]);
			this.nextRequest[endpoint] = start + 210;
			await delay(start - now, undefined, { signal: activeSignal });
			let response: Response;
			try {
				response = await fetch(`${API_ROOT}/${endpoint}`, {
					method: "POST", redirect: "error", signal: activeSignal,
					headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
					body: JSON.stringify(body),
				});
			} catch {
				activeSignal.throwIfAborted();
				if (attempt === 2) throw new Error(`DeepSeek ${endpoint} network request failed`);
				await delay(1000 * 2 ** attempt, undefined, { signal: activeSignal });
				continue;
			}
			if (response.status === 200) {
				try { return await response.json(); }
				catch { throw new Error(`DeepSeek ${endpoint} returned invalid JSON`); }
			}
			await response.body?.cancel();
			const retryable = response.status === 408 || response.status === 429 || response.status >= 500;
			if (!retryable || attempt === 2) {
				throw new Error(`DeepSeek ${endpoint} failed (HTTP ${response.status})`);
			}
			const retryAfter = response.headers.get("retry-after");
			const seconds = Number(retryAfter);
			const wait = retryAfter === null ? 1000 * 2 ** attempt
				: Number.isFinite(seconds) ? seconds * 1000 : Date.parse(retryAfter) - Date.now();
			await delay(Number.isFinite(wait) ? Math.min(30_000, Math.max(0, wait)) : 1000 * 2 ** attempt,
				undefined, { signal: activeSignal });
		}
		throw new Error(`DeepSeek ${endpoint} exhausted retries`);
	}
}
