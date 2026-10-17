import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { DeepSeekWebTools } from "./deepseek-web.js";

// Accept the key on stdin so SSH callers need not place it in argv or on disk.
const key = process.env.DEEPSEEK_API_KEY?.trim() || readFileSync(0, "utf8").trim();
const tools = new DeepSeekWebTools(key).toolDefinitions({
	allowedHosts: ["deepseek.com"], deniedHosts: [], forbiddenQueryPatterns: [],
});
const search = await tools[0]!.execute("smoke-search", {
	query: "DeepSeek Search Fetch API site:api-docs.deepseek.com", count: 3, need_content: true,
}, undefined, undefined, {} as never);
const results = JSON.parse((search.content[0] as { text: string }).text).results as Array<{ url: string }>;
assert(results.length > 0, "Search must return at least one allowed result");
const page = await tools[1]!.execute("smoke-fetch", {
	url: "https://api-docs.deepseek.com/guides/search/",
}, undefined, undefined, {} as never);
const document = JSON.parse((page.content[0] as { text: string }).text) as { title: string; content: string };
assert(document.content.length > 0);
console.log(JSON.stringify({ provider: "deepseek", searchResults: results.length,
	urls: results.map((result) => result.url), fetchTitle: document.title, fetchCharacters: document.content.length }));
