import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { createAgentSession, DefaultResourceLoader, SessionManager, SettingsManager, estimateTokens,
	type AgentSession, type ExtensionAPI, type ExtensionContext, type SessionBeforeCompactEvent } from "@earendil-works/pi-coding-agent";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { Type } from "typebox";
import { ContextMeter, type ContextEnvelope } from "./context-meter.js";
import { ContextPressureController, CONTEXT_MEASUREMENT_ENTRY } from "./context-pressure.js";
import { WebEvidenceArchive, WEB_EVIDENCE_GUEST, WEB_EVIDENCE_MARKER } from "./web-evidence.js";
import { sha256 } from "./trace.js";
import { createSolPiExtension } from "./sol-pi.js";

const envelope: ContextEnvelope = { provider: "test", model: "model", systemPrompt: "system",
	tools: [{ name: "web_fetch", parameters: { type: "object" } }], contextWindow: 1000, reserveTokens: 100 };
const user: AgentMessage = { role: "user", content: "Find evidence", timestamp: 1 };
function paper(text: string): AgentMessage {
	return { role: "toolResult", toolName: "web_fetch", toolCallId: "fetch-1", content: [{ type: "text", text }],
		isError: false, timestamp: 2 };
}
const body = "# Paper\n" + "Introduction evidence\n".repeat(1000) + "## Methods\nMIDDLE_RESULT=42\n" + "Results evidence\n".repeat(1000) + "## References\nCitation\n";

test("meter counts prompt, schemas and web text; usage anchors require unchanged envelope and prefix", () => {
	const meter = new ContextMeter();
	const first = meter.measure([user, paper(body)], envelope);
	assert(first.underPressure && first.heuristic.system > 0 && first.heuristic.tools > 0);
	assert.equal(first.heuristic.webResults, estimateTokens(paper(body)));
	meter.prepare([user], envelope);
	const response = fauxAssistantMessage("Answer");
	response.usage = { ...response.usage, input: 100, cacheRead: 20, cacheWrite: 5, output: 7, totalTokens: 132 };
	const anchor = meter.observe(response);
	assert(anchor);
	const measured = meter.measure([user, response], envelope);
	assert.equal(measured.basis, "provider-anchored");
	assert.equal(measured.estimatedInputTokens, 125 + estimateTokens(response));
	for (const changed of [{ ...envelope, model: "different" }, { ...envelope, systemPrompt: "changed" },
		{ ...envelope, tools: [] }]) assert.equal(meter.measure([user, response], changed).basis, "heuristic");
	assert.equal(meter.measure([{ ...user, content: "replaced prefix" }], envelope).basis, "heuristic");
	const restored = new ContextMeter();
	restored.restore(JSON.parse(JSON.stringify(anchor)));
	assert.deepEqual(restored.measure([user, response], envelope), measured);
	meter.prepare([user], envelope);
	assert.equal(meter.observe(fauxAssistantMessage("Error", { stopReason: "error" })), undefined);
});

test("pressure archive is lossless, retains middle sections, stays projected on resume, and detects corruption", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-evidence-"));
	try {
		const archive = new WebEvidenceArchive(root);
		const message = paper(body + "🧪".repeat(4));
		const original = structuredClone(message);
		assert.deepEqual(await archive.project([message], false), { messages: [message], archived: 0 });
		const packed = await archive.project([message], true);
		assert.equal(packed.archived, 1);
		assert.deepEqual(message, original);
		const text = (packed.messages[0] as { content: { text: string }[] }).content[0]!.text;
		assert(text.startsWith(WEB_EVIDENCE_MARKER));
		assert.match(text, /1002: ## Methods/);
		assert(text.length < 8192);
		const digest = sha256(body + "🧪".repeat(4));
		assert.equal(await readFile(join(root, `${digest}.md`), "utf8"), body + "🧪".repeat(4));
		assert.deepEqual(await new WebEvidenceArchive(root).project([message], false), packed);
		assert.deepEqual((await archive.project(packed.messages, true)).messages, packed.messages);
		await writeFile(join(root, `${digest}.md`), "corrupted");
		await assert.rejects(archive.project([message], false), /integrity/);
	} finally { await rm(root, { recursive: true, force: true }); }
});

test("threshold compaction is skipped after relief; manual and overflow keep native summarization", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-pressure-"));
	try {
		const controller = new ContextPressureController(root, 100);
		const handlers = new Map<string, (...args: never[]) => unknown>();
		const entries: unknown[] = [];
		const pi = { on: (name: string, handler: (...args: never[]) => unknown) => handlers.set(name, handler),
			appendEntry: (_type: string, data: unknown) => entries.push(data), getActiveTools: () => [], getAllTools: () => [] };
		controller.createExtension(() => [user, paper(body)])(pi as unknown as ExtensionAPI);
		const ctx = { model: { provider: "test", id: "model", contextWindow: 5000 }, getSystemPrompt: () => "system" } as ExtensionContext;
		const invoke = (reason: "threshold" | "manual" | "overflow") => {
			const event = { reason, preparation: { messagesToSummarize: [paper(body)], turnPrefixMessages: [] } } as unknown as SessionBeforeCompactEvent;
			return { event, pending: handlers.get("session_before_compact")!(event as never, ctx as never) as Promise<unknown> };
		};
		assert.deepEqual(await invoke("threshold").pending, { cancel: true });
		for (const reason of ["manual", "overflow"] as const) {
			const call = invoke(reason);
			assert.equal(await call.pending, undefined);
			assert.match(JSON.stringify(call.event.preparation.messagesToSummarize), /ldm_web_evidence_v1/);
		}
		assert(entries.length >= 3);
	} finally { await rm(root, { recursive: true, force: true }); }
});

for (const [withSolPi, underPressure] of [[false, true], [true, true], [true, false]] as const)
test(`actual Pi SDK preserves/recalls evidence (SoL-Pi=${withSolPi}, pressure=${underPressure})`, async () => {
	const cwd = await mkdtemp(join(tmpdir(), "ldm-context-sdk-"));
	const agentDir = join(cwd, "agent");
	await mkdir(agentDir);
	let session: AgentSession | undefined;
	try {
		const faux = fauxProvider({ provider: `pressure-test-${withSolPi}`, api: `pressure-api-${withSolPi}`,
			models: [{ id: "pressure-model", contextWindow: underPressure ? 8000 : 100_000, maxTokens: 1000 }] });
		const controller = new ContextPressureController(join(agentDir, "web-evidence"), 1000);
		const path = `${WEB_EVIDENCE_GUEST}/${sha256(body)}.md`;
		faux.setResponses([
			fauxAssistantMessage(fauxToolCall("web_fetch", { url: "https://nature.com/paper" }), { stopReason: "toolUse" }),
			(context) => {
				const web = context.messages.find((message) => message.role === "toolResult" && message.toolName === "web_fetch");
				assert(web);
				if (!underPressure) {
					assert.match(JSON.stringify(web), /MIDDLE_RESULT=42/);
					return fauxAssistantMessage(fauxToolCall("noop", {}), { stopReason: "toolUse" });
				}
				assert.match(JSON.stringify(web), /ldm_web_evidence_v1/);
				assert(!JSON.stringify(web).includes("MIDDLE_RESULT=42"));
				return fauxAssistantMessage(fauxToolCall("read", { path, offset: 1002, limit: 2 }), { stopReason: "toolUse" });
			},
			(context) => {
				if (!underPressure) return fauxAssistantMessage(fauxToolCall("noop", {}), { stopReason: "toolUse" });
				assert.match(JSON.stringify(context.messages.at(-1)), /MIDDLE_RESULT=42/);
				return fauxAssistantMessage("Verified methods: 42.");
			},
		]);
		if (!underPressure) faux.appendResponses([(context) => {
			const web = context.messages.find((message) => message.role === "toolResult" && message.toolName === "web_fetch");
			assert.match(JSON.stringify(web), /MIDDLE_RESULT=42/);
			assert(!JSON.stringify(web).includes(WEB_EVIDENCE_MARKER));
			return fauxAssistantMessage("Full evidence retained after repeated requests.");
		}]);
		const settingsManager = SettingsManager.inMemory({ retry: { enabled: false }, compaction: { enabled: false } });
		const sol = withSolPi ? await createSolPiExtension({ version: 1, actionFusion: false, observationPack: true,
			evidencePreservingReducer: false, onlineContextCompact: false, cacheWriteReadRatio: 50 }, agentDir,
			faux.getModel().provider, faux.getModel().id, async () => ({}), join(agentDir, "sessions", "sol-pi")) : undefined;
		const loader = new DefaultResourceLoader({ cwd, agentDir, settingsManager, noExtensions: true, noSkills: true,
			noPromptTemplates: true, noThemes: true, noContextFiles: true, extensionFactories: [
				(pi) => pi.registerProvider(faux.provider), ...(sol ? [sol] : []), controller.createExtension(() => session?.messages ?? []),
			] });
		await loader.reload();
		assert.deepEqual(loader.getExtensions().errors, []);
		({ session } = await createAgentSession({ cwd, agentDir, model: faux.getModel(), resourceLoader: loader,
			settingsManager, sessionManager: SessionManager.create(cwd, join(agentDir, "sessions")), customTools: [
				{ name: "noop", label: "Noop", description: "Continue inspection", parameters: Type.Object({}),
					execute: async () => ({ content: [{ type: "text", text: "Continue" }], details: {} }) },
				{ name: "web_fetch", label: "Fetch", description: "Fetch paper", parameters: Type.Object({ url: Type.String() }),
					execute: async () => ({ content: [{ type: "text", text: body }], details: {} }) },
				{ name: "read", label: "Read", description: "Recall evidence", parameters: Type.Object({ path: Type.String(), offset: Type.Number(), limit: Type.Number() }),
					execute: async (_id, args) => {
						const input = args as { path: string; offset: number; limit: number };
						assert.equal(input.path, path);
						const content = await readFile(join(controller.archive.root, `${sha256(body)}.md`), "utf8");
						return { content: [{ type: "text", text: content.split("\n").slice(input.offset - 1, input.offset - 1 + input.limit).join("\n") }], details: {} };
					} },
			] }));
		await session.bindExtensions({});
		await session.prompt("Read the paper's methods", { expandPromptTemplates: false });
		assert.equal(faux.state.callCount, underPressure ? 3 : 4);
		assert.equal(session.messages.filter((message) => message.role === "toolResult" && message.toolName === "web_fetch").length, 1);
		const originalWeb = session.messages.find((message) => message.role === "toolResult" && message.toolName === "web_fetch");
		assert(originalWeb?.role === "toolResult" && originalWeb.content[0]?.type === "text");
		assert.equal(originalWeb.content[0].text, body);
		const log = await readFile(session.sessionManager.getSessionFile()!, "utf8");
		assert(log.includes(CONTEXT_MEASUREMENT_ENTRY));
		assert(log.includes("MIDDLE_RESULT=42"));
		assert.equal(log.includes('"archivedResults":1'), underPressure);
	} finally { session?.dispose(); await rm(cwd, { recursive: true, force: true }); }
});
