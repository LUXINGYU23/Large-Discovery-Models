import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { McpToolBridge } from "./mcp.js";
import type { McpServerConfig } from "./protocol.js";

test("built-in policy MCP reads the active snapshot and runs draft tools", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-policy-mcp-"));
	const workspace = join(root, "sessions", "policy_architect", "workspace");
	const round = join(root, "rounds", "round_001");
	await mkdir(workspace, { recursive: true });
	await mkdir(round, { recursive: true });
	await writeFile(join(workspace, "optimization_policy.py"), "POLICY_API_VERSION = 1\n");
	await writeFile(join(round, "contract.json"), JSON.stringify({ feature_names: ["a", "b"] }));
	await writeFile(join(round, "research_snapshot.json"), JSON.stringify({
		task_objective: "fixture",
		reference_diagnostics: { baseline_score: 0.5 },
	}));
	await writeFile(join(root, "active_round.json"), JSON.stringify({
		round_index: 1,
		round_path: "rounds/round_001",
		input_sha256: "a".repeat(64),
		contract_sha256: "b".repeat(64),
		active_policy: null,
	}));
	const fakeRunner = join(root, "fake-runner.mjs");
	await writeFile(fakeRunner, `
const command = process.argv[2];
console.log(JSON.stringify(command === "inspect"
  ? { status: "ok", inspection: { policy_api_version: 1 } }
  : { status: "ok", stage: "test", alpha: 1, eta: 2, prior_summary: {} }));
`);
	const entrypoint = fileURLToPath(new URL("./policy-mcp.js", import.meta.url));
	const config: McpServerConfig = {
		serverId: "ldm_policy",
		transport: "stdio",
		command: process.execPath,
		args: [entrypoint, "stdio"],
		env: {
			LDM_POLICY_ROOT: { value: root },
			LDM_POLICY_WORKSPACE: { value: workspace },
			LDM_POLICY_PYTHON: { value: process.execPath },
			LDM_POLICY_RUNNER: { value: fakeRunner },
		},
		tools: [
			"inspect_policy_contract",
			"validate_policy_draft",
			"evaluate_policy_draft",
		],
		configSha256: "c".repeat(64),
	};
	const bridge = new McpToolBridge([config], {}, "policy-test");
	try {
		await bridge.initialize();
		const tools = bridge.toolDefinitions();
		const inspect = await tools[0]!.execute("inspect", {}, undefined, undefined, {} as never);
		const validation = await tools[1]!.execute(
			"validate",
			{ artifact_path: "optimization_policy.py" },
			undefined,
			undefined,
			{} as never,
		);
		const evaluation = await tools[2]!.execute(
			"evaluate",
			{ artifact_path: "optimization_policy.py" },
			undefined,
			undefined,
			{} as never,
		);
		assert.equal((inspect.details as any).structuredContent.round_index, 1);
		assert.equal((validation.details as any).structuredContent.status, "ok");
		assert.deepEqual(
			(evaluation.details as any).structuredContent.reference_diagnostics,
			{ baseline_score: 0.5 },
		);
		await assert.rejects(
			(tools[1] as ToolDefinition).execute(
				"escape",
				{ artifact_path: "../escape.py" },
				undefined,
				undefined,
				{} as never,
			),
			/artifact_path must be a relative Python file/,
		);
	} finally {
		await bridge.close();
		await rm(root, { recursive: true, force: true });
	}
});
