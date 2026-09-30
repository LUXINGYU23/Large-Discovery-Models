import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createAgentSession, DefaultResourceLoader, SessionManager, SettingsManager, type AgentSession } from "@earendil-works/pi-coding-agent";
import { fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import test from "node:test";
import { createSolPiExtension } from "./sol-pi.js";

test("SoL-Pi registers all mechanisms and fused commands use the supplied guest operations", async () => {
	const cwd = await mkdtemp(join(tmpdir(), "ldm-sol-pi-"));
	const agentDir = join(cwd, "agent");
	await mkdir(agentDir);
	let session: AgentSession | undefined;
	const commands: string[] = [];
	try {
		const faux = fauxProvider({ provider: "sol-test", api: "sol-test-api" });
		faux.setResponses([
			fauxAssistantMessage(fauxToolCall("write", {
				path: "/workspace/proof.txt", content: "research",
				then_run: { command: "guest-only-command" },
			}), { stopReason: "toolUse" }),
			fauxAssistantMessage(fauxToolCall("update_plan", {
				steps: [{ id: "research", goal: "Review measurements", status: "in_progress" }],
			}), { stopReason: "toolUse" }),
			fauxAssistantMessage("Done"),
		]);
		const extension = await createSolPiExtension({
			version: 1, actionFusion: true, observationPack: true,
			evidencePreservingReducer: true, onlineContextCompact: true, cacheWriteReadRatio: 50,
		}, agentDir, faux.getModel().provider, faux.getModel().id, async () => ({
			write: { operations: {
				writeFile: async (path, content) => { await writeFile(path, content); },
				mkdir: async (path) => { await mkdir(path, { recursive: true }); },
			} },
			edit: { operations: { readFile, writeFile: async (path, content) => { await writeFile(path, content); }, access } },
			bash: { operations: { exec: async (command, _cwd, { onData }) => {
				commands.push(command);
				onData(Buffer.from("guest-result"));
				return { exitCode: 0 };
			} } },
		}), join(agentDir, "sessions", "sol-pi"));
		const settingsManager = SettingsManager.inMemory({ retry: { enabled: false } });
		const loader = new DefaultResourceLoader({
			cwd, agentDir, settingsManager, noExtensions: true, noSkills: true,
			noPromptTemplates: true, noThemes: true, noContextFiles: true,
			extensionFactories: [(pi) => pi.registerProvider(faux.provider), extension],
		});
		await loader.reload();
		assert.deepEqual(loader.getExtensions().errors, []);
		({ session } = await createAgentSession({
			cwd, agentDir, model: faux.getModel(), resourceLoader: loader, settingsManager,
			sessionManager: SessionManager.create(cwd, join(agentDir, "sessions")),
		}));
		await session.bindExtensions({});
		await session.prompt("Perform the research check", { expandPromptTemplates: false });
		assert.deepEqual(commands, ["guest-only-command"]);
		assert.equal(await readFile(join(cwd, "proof.txt"), "utf8"), "research");
		const results = session.messages.filter((message) => message.role === "toolResult");
		assert.equal(results.length, 2);
		assert(results.every((result) => !result.isError), JSON.stringify(results));
		assert.match(JSON.stringify(results), /then_run:succeeded/);
		for (const tool of ["write", "edit", "obs_recall", "update_plan"]) {
			assert(session.getActiveToolNames().includes(tool), tool);
		}
		const mechanisms = loader.getExtensions().extensions.flatMap((extension) => [...extension.handlers.keys()]);
		assert(mechanisms.includes("tool_result"));
		assert(mechanisms.includes("agent_settled"));
	} finally {
		session?.dispose();
		await rm(cwd, { recursive: true, force: true });
	}
});
