import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { HistoryController, HistoryRecoveryError, HISTORY_POLICY } from "./history.js";
import { ProviderProxy } from "./provider-proxy.js";
import { ContextPressureController } from "./context-pressure.js";
import { WEB_EVIDENCE_MARKER } from "./web-evidence.js";
import { sha256 } from "./trace.js";

test("continuation allowance and blocked history survive a process restart", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-history-resume-"));
	try {
		const manager = SessionManager.create(root, join(root, "sessions"));
		manager.appendMessage({ role: "user", content: "preserve evidence", timestamp: Date.now() });
		for (let i = 0; i < 2; i += 1) {
			const history = new HistoryController();
			await history.begin(root, manager);
			await history.continueTurn("stop");
		}
		const exhausted = new HistoryController();
		await exhausted.begin(root, manager);
		await assert.rejects(exhausted.continueTurn("stop"), HistoryRecoveryError);
		manager.appendMessage({ role: "user", content: "resume again", timestamp: Date.now() });
		await assert.rejects(new HistoryController().begin(root, manager), HistoryRecoveryError);
		manager.appendCompaction("Repaired native summary", manager.getLeafId()!, 100);
		const repaired = new HistoryController();
		await repaired.begin(root, manager);
		await repaired.continueTurn("stop");
		assert.equal(manager.getBranch().filter(e => e.type === "message").length, 2);
	} finally { await rm(root, { recursive: true, force: true }); }
});

for (const mode of ["success", "incomplete", "automatic_failure", "transient"] as const) {
	const fail = mode === "incomplete" || mode === "automatic_failure";
	test(`native compaction hook: ${mode}`, async () => {
		const root = await mkdtemp(join(tmpdir(), "ldm-history-native-"));
		const requests: Array<Record<string, unknown>> = [];
		const server = createServer(async (request, response) => {
			assert.equal(request.url, "/responses");
			let body = "";
			for await (const chunk of request) body += chunk.toString();
			requests.push(JSON.parse(body));
			if (mode === "transient" && requests.length === 1) {
				response.writeHead(200, { "content-type": "text/event-stream" });
				response.end(`data: ${JSON.stringify({ type: "response.failed", response: {
					id: "resp_retry", status: "failed", output: [], error: { code: "server_error", message: "temporary 502" },
				} })}\n\ndata: [DONE]\n\n`);
				return;
			}
			const item = { type: "message", id: "msg_summary", role: "assistant", status: "completed",
				content: [{ type: "output_text", text: "Summary of archived evidence.", annotations: [] }] };
			const events = [
				{ type: "response.created", response: { id: "resp_summary", status: "in_progress", output: [] } },
				{ type: "response.output_item.added", output_index: 0, item: { ...item, content: [] } },
				{ type: "response.output_item.done", output_index: 0, item },
				{ type: fail ? "response.incomplete" : "response.completed", response: {
					id: "resp_summary", status: fail ? "incomplete" : "completed", output: [item],
					...(fail ? { incomplete_details: { reason: "max_output_tokens" } } : {}),
					usage: { input_tokens: 100, output_tokens: 20, total_tokens: 120,
						input_tokens_details: { cached_tokens: 0 }, output_tokens_details: { reasoning_tokens: 0 } },
				} },
			];
			response.writeHead(200, { "content-type": "text/event-stream" });
			for (const event of events) response.write(`data: ${JSON.stringify(event)}\n\n`);
			response.end("data: [DONE]\n\n");
		});
		await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
		const address = server.address();
		assert(address && typeof address !== "string");
		let session: Awaited<ReturnType<typeof createAgentSession>>["session"] | undefined;
		const proxy = new ProviderProxy(`http://127.0.0.1:${address.port}`, "fixture-key", "test-history", {
			reasoning: { effort: "max" }, max_output_tokens: 1024, tool_choice: "required",
		});
		try {
			await proxy.start();
			const agentDir = join(root, "agent");
			await mkdir(agentDir);
			const modelsPath = join(agentDir, "models.json");
			await writeFile(modelsPath, JSON.stringify({ providers: { fixture: {
				baseUrl: proxy.baseUrl("research"), api: "openai-responses", apiKey: "fixture-key",
				models: [{ id: "fixture", name: "fixture", reasoning: true, input: ["text"],
					thinkingLevelMap: { low: "low", max: "max" },
					contextWindow: 262144, maxTokens: 32768, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
			} } }));
			const runtime = await ModelRuntime.create({ authPath: join(agentDir, "auth.json"), modelsPath, refreshOnCreate: false });
			await runtime.setRuntimeApiKey("fixture", "fixture-key");
			const model = runtime.getModel("fixture", "fixture");
			assert(model);
			const settings = SettingsManager.inMemory({ compaction: HISTORY_POLICY.compaction, retry: { enabled: false } });
			const history = new HistoryController();
			const pressure = new ContextPressureController(join(root, ".web-evidence"), HISTORY_POLICY.compaction.reserveTokens);
			const loader = new DefaultResourceLoader({ cwd: root, agentDir, settingsManager: settings,
				noExtensions: true, noSkills: true, noContextFiles: true, noPromptTemplates: true, noThemes: true,
				extensionFactories: [pressure.createExtension(() => session?.messages ?? []), history.createExtension("fixture-key")] });
			await loader.reload();
			const manager = SessionManager.create(root, join(root, "sessions"));
			manager.appendMessage({ role: "user", content: "archived evidence ".repeat(10000), timestamp: Date.now() });
			const webEvidence = "Full paper evidence beyond the receipt preview.\n".repeat(1000);
			if (mode === "success") manager.appendMessage({ role: "toolResult", toolName: "web_fetch", toolCallId: "paper",
				content: [{ type: "text", text: webEvidence }], isError: false, timestamp: Date.now() });
			manager.appendMessage({ role: "assistant", content: [{ type: "text", text: "Recorded evidence" }],
				api: "openai-responses", provider: "fixture", model: "fixture", stopReason: "stop", timestamp: Date.now(),
				usage: { input: 210000, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 210010,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
			manager.appendMessage({ role: "user", content: "recent evidence ".repeat(7000), timestamp: Date.now() });
			({ session } = await createAgentSession({ cwd: root, agentDir, modelRuntime: runtime, model,
				thinkingLevel: "max", sessionManager: manager, settingsManager: settings, resourceLoader: loader, tools: [] }));
			await session.bindExtensions({});
			await history.begin(root, manager);
			await proxy.beginTurn("research", manager.getSessionId(), "turn", root, async () => { history.assertReady(); return true; });
			const oldIds = manager.getBranch().map(e => e.id);
			if (fail) {
				if (mode === "automatic_failure") await session.prompt("Continue research");
				else await assert.rejects(session.compact());
				assert.throws(() => history.assertReady(), HistoryRecoveryError);
				assert.equal(manager.getBranch().filter(e => e.type === "compaction").length, 0);
				await assert.rejects(new HistoryController().begin(root, manager), HistoryRecoveryError);
			} else {
				await session.compact();
				history.assertReady();
				assert.equal(manager.getBranch().filter(e => e.type === "compaction").length, 1);
				assert.deepEqual(manager.getBranch().slice(0, oldIds.length).map(e => e.id), oldIds);
				assert.match(await readFile(join(root, "history-events.jsonl"), "utf8"), /compaction_committed/);
			}
			assert.equal(requests.length, mode === "transient" ? 2 : 1, "no fallback or research dispatch on failed compaction");
			assert.equal((requests[0]!.reasoning as { effort: string }).effort, "low");
			assert.equal(requests[0]!.max_output_tokens, 32768);
			assert.notEqual(requests[0]!.tool_choice, "required");
			assert.equal(session.thinkingLevel, "max", "research reasoning must remain unchanged");
			if (mode === "success") {
				assert(JSON.stringify(requests[0]).includes(WEB_EVIDENCE_MARKER), "summarizer must receive evidence recall receipts");
				assert(!JSON.stringify(requests[0]).includes(webEvidence.slice(0, 2000)));
				assert.equal(await readFile(join(pressure.archive.root, `${sha256(webEvidence)}.md`), "utf8"), webEvidence);
				await session.prompt("Continue research after summarizing");
				assert.deepEqual(requests.at(-1)!.reasoning, { effort: "max" });
				assert.equal(requests.at(-1)!.max_output_tokens, 1024);
				assert.equal(requests.at(-1)!.tool_choice, "required");
			}
		} finally {
			await session?.dispose();
			await proxy.endTurn("research");
			await proxy.close();
			await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
			await rm(root, { recursive: true, force: true });
		}
	});
}
