import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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
	const input = {
		execution_context: {
			mean_context: { target_location: 2, target_scale: 0.5 },
			weight_context: { history_size: 4 },
		},
	};
	const research = { task_objective: "fixture", history: Array(10_000).fill({ utility: 2, notes: "measured evidence" }) };
	await writeFile(join(round, "input.json"), JSON.stringify(input));
	await writeFile(join(round, "research_snapshot.json"), JSON.stringify(research));
	await writeFile(join(round, "arrays.npz"), "fixture arrays");
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
if (command === "execute") {
  if (process.argv[process.argv.indexOf("--diagnostics") + 1] !== process.env.LDM_POLICY_DIAGNOSTICS
      || process.argv[process.argv.indexOf("--diagnostics-sha256") + 1] !== "d".repeat(64)) {
    throw new Error("Task diagnostic registration was not forwarded");
  }
} else if (process.argv.includes("--diagnostics")) {
  throw new Error("Inspection must not execute task diagnostics");
}
console.log(JSON.stringify(command === "inspect"
  ? { status: "ok", inspection: { policy_api_version: 1 } }
  : { status: "ok", stage: "test", alpha: 1, eta: 2, draft_diagnostics: { draft_gp_rmse: 0.25 } }));
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
			LDM_POLICY_DIAGNOSTICS: { value: join(root, "task_diagnostics.py") },
			LDM_POLICY_DIAGNOSTICS_SHA256: { value: "d".repeat(64) },
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
		assert.deepEqual((inspect.details as any).structuredContent.guest_snapshot, {
			directory: "/workspace/.ldm-resources/policy/round_001", arrays: "arrays.npz",
			contract: "contract.json", input: "input.json", research: "research_snapshot.json", read_only: true,
		});
		assert.equal(await readFile(join(workspace, ".ldm-resources/policy/round_001/arrays.npz"), "utf8"), "fixture arrays");
		assert.ok(JSON.stringify(inspect.content).length < 2000);
		assert.deepEqual(
			JSON.parse(await readFile(join(workspace, ".ldm-resources/policy/round_001/input.json"), "utf8")), input,
		);
		assert.deepEqual(
			JSON.parse(await readFile(join(workspace, ".ldm-resources/policy/round_001/research_snapshot.json"), "utf8")), research,
		);
		assert.equal((validation.details as any).structuredContent.status, "ok");
		assert.equal(
			(evaluation.details as any).structuredContent.draft_diagnostics.draft_gp_rmse,
			0.25,
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
