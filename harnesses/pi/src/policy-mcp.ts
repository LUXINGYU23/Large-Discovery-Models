import { execFile } from "node:child_process";
import {
	copyFile,
	mkdir,
	mkdtemp,
	readFile,
	realpath,
	rm,
	stat,
} from "node:fs/promises";
import {
	basename,
	isAbsolute,
	join,
	relative,
	resolve,
	sep,
} from "node:path";
import { promisify } from "node:util";
import { McpServer } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";

const executeFile = promisify(execFile);

interface PolicyMcpConfig {
	root: string;
	workspace: string;
	python: string;
	runner: string;
	diagnostics?: { path: string; sha256: string };
}

interface ActiveRound {
	round_index: number;
	round_path: string;
	input_sha256: string;
	contract_sha256: string;
	active_policy: unknown;
}

interface RoundSnapshot {
	pointer: ActiveRound;
	directory: string;
	contract: Record<string, unknown>;
}

function createPolicyServer(config = environmentConfig()): McpServer {
	const server = new McpServer({ name: "ldm-policy", version: "1.0.0" });
	server.registerTool(
		"inspect_policy_contract",
		{
			description: "Inspect the authoritative optimization-policy contract and export current snapshot files to the guest. Read execution_context from input.json and query research_snapshot.json with local scripts; do not print whole history files.",
			inputSchema: z.object({}),
		},
		async () => response(await inspectSnapshot(config)),
	);
	server.registerTool(
		"validate_policy_draft",
		{
			description: "Statically inspect a draft optimization_policy.py against the current contract.",
			inputSchema: z.object({ artifact_path: z.string() }),
		},
		async ({ artifact_path }, context) => {
			const snapshot = await loadSnapshot(config);
			const artifact = await draftArtifact(config.workspace, artifact_path);
			return response(await runPolicy(config, [
				"inspect",
				"--artifact", artifact,
				"--contract", join(snapshot.directory, "contract.json"),
			], context.mcpReq.signal));
		},
	);
	server.registerTool(
		"evaluate_policy_draft",
		{
			description: "Execute a draft optimization policy and return task-defined diagnostics from the authoritative round snapshot. No unseen oracle labels are available.",
			inputSchema: z.object({ artifact_path: z.string() }),
		},
		async ({ artifact_path }, context) => {
			const snapshot = await loadSnapshot(config);
			const artifact = await draftArtifact(config.workspace, artifact_path);
			const output = await mkdtemp(join(config.workspace, ".policy-eval-"));
			try {
				return response(await runPolicy(config, [
					"execute", "--artifact", artifact,
					"--input", snapshot.directory, "--output", output,
					...(config.diagnostics ? [
						"--diagnostics", config.diagnostics.path,
						"--diagnostics-sha256", config.diagnostics.sha256,
					] : []),
				], context.mcpReq.signal));
			} finally {
				await rm(output, { recursive: true, force: true });
			}
		},
	);
	return server;
}

async function inspectSnapshot(config: PolicyMcpConfig): Promise<Record<string, unknown>> {
	const snapshot = await loadSnapshot(config);
	const resourcePath = join(".ldm-resources", "policy", basename(snapshot.directory));
	await mkdir(join(config.workspace, resourcePath), { recursive: true });
	const exported = await containedPath(config.workspace, resourcePath, false);
	for (const name of ["arrays.npz", "contract.json", "input.json", "research_snapshot.json"]) {
		await copyFile(join(snapshot.directory, name), join(exported, name));
	}
	return {
		round_index: snapshot.pointer.round_index,
		input_sha256: snapshot.pointer.input_sha256,
		contract_sha256: snapshot.pointer.contract_sha256,
		contract: snapshot.contract,
		active_policy: snapshot.pointer.active_policy,
		guest_snapshot: {
			directory: `/workspace/${resourcePath.split(sep).join("/")}`,
			arrays: "arrays.npz",
			contract: "contract.json",
			input: "input.json",
			research: "research_snapshot.json",
			read_only: true,
		},
	};
}

async function loadSnapshot(config: PolicyMcpConfig): Promise<RoundSnapshot> {
	const pointer = await jsonFile(join(config.root, "active_round.json")) as unknown as ActiveRound;
	if (
		!Number.isInteger(pointer.round_index)
		|| pointer.round_index < 0
		|| typeof pointer.round_path !== "string"
		|| typeof pointer.input_sha256 !== "string"
		|| typeof pointer.contract_sha256 !== "string"
	) {
		throw new Error("active policy round pointer is invalid");
	}
	const directory = await containedPath(config.root, pointer.round_path, false);
	return {
		pointer,
		directory,
		contract: await jsonFile(join(directory, "contract.json")),
	};
}

async function draftArtifact(workspace: string, value: string): Promise<string> {
	if (
		!value
		|| value.includes("\\")
		|| isAbsolute(value)
		|| value.split("/").some((part) => part === "" || part === "." || part === "..")
		|| !value.endsWith(".py")
	) {
		throw new Error("artifact_path must be a relative Python file inside the policy workspace");
	}
	return containedPath(workspace, value, true);
}

async function containedPath(root: string, value: string, requireFile: boolean): Promise<string> {
	const candidate = resolve(root, value);
	if (!isInside(root, candidate)) throw new Error("path escapes the configured policy root");
	let resolved: string;
	try {
		resolved = await realpath(candidate);
	} catch {
		throw new Error(`policy path does not exist: ${basename(value)}`);
	}
	if (!isInside(root, resolved)) throw new Error("policy path resolves outside the configured root");
	const metadata = await stat(resolved);
	if (requireFile ? !metadata.isFile() : !metadata.isDirectory()) {
		throw new Error(requireFile ? "policy artifact must be a regular file" : "policy round must be a directory");
	}
	return resolved;
}

async function runPolicy(
	config: PolicyMcpConfig,
	args: string[],
	signal?: AbortSignal,
): Promise<Record<string, unknown>> {
	let stdout = "";
	let stderr = "";
	try {
		const result = await executeFile(config.python, [config.runner, ...args], {
			signal,
			timeout: 15_000,
			maxBuffer: 1024 * 1024,
			env: {
				...process.env,
				PYTHONHASHSEED: "0",
				OMP_NUM_THREADS: "1",
				OPENBLAS_NUM_THREADS: "1",
				MKL_NUM_THREADS: "1",
				NUMEXPR_NUM_THREADS: "1",
			},
		});
		stdout = result.stdout;
		stderr = result.stderr;
	} catch (error) {
		const failure = error as Error & { stdout?: string; stderr?: string };
		stdout = failure.stdout ?? "";
		stderr = failure.stderr ?? failure.message;
	}
	for (const line of stdout.trim().split(/\r?\n/).reverse()) {
		try {
			const value = JSON.parse(line) as unknown;
			if (value && typeof value === "object" && !Array.isArray(value)) {
				return value as Record<string, unknown>;
			}
		} catch {
			// Continue to an earlier output line.
		}
	}
	throw new Error(`policy runner returned no structured result: ${(stderr || stdout).slice(-1000)}`);
}

async function jsonFile(path: string): Promise<Record<string, unknown>> {
	const value = JSON.parse(await readFile(path, "utf8")) as unknown;
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error(`policy file must contain a JSON object: ${basename(path)}`);
	}
	return value as Record<string, unknown>;
}

function response(value: Record<string, unknown>) {
	return {
		content: [{ type: "text" as const, text: JSON.stringify(value) }],
		structuredContent: value,
	};
}

function environmentConfig(): PolicyMcpConfig {
	const root = process.env.LDM_POLICY_ROOT;
	const workspace = process.env.LDM_POLICY_WORKSPACE;
	const diagnosticsPath = process.env.LDM_POLICY_DIAGNOSTICS;
	const diagnosticsSha256 = process.env.LDM_POLICY_DIAGNOSTICS_SHA256;
	if ((diagnosticsPath !== undefined || diagnosticsSha256 !== undefined) && (
		!diagnosticsPath || !isAbsolute(diagnosticsPath)
		|| !diagnosticsSha256 || !/^[a-f0-9]{64}$/.test(diagnosticsSha256)
	)) {
		throw new Error("Task diagnostics require an absolute path and SHA-256 digest");
	}
	if (!root || !workspace || !isAbsolute(root) || !isAbsolute(workspace)) {
		throw new Error("LDM_POLICY_ROOT and LDM_POLICY_WORKSPACE must be absolute paths");
	}
	return {
		root,
		workspace,
		python: process.env.LDM_POLICY_PYTHON ?? "python3",
		runner: process.env.LDM_POLICY_RUNNER ?? "/app/policy_runner.py",
		...(diagnosticsPath && diagnosticsSha256
			? { diagnostics: { path: diagnosticsPath, sha256: diagnosticsSha256 } } : {}),
	};
}

function isInside(root: string, value: string): boolean {
	const path = relative(resolve(root), resolve(value));
	return path === "" || (path !== ".." && !path.startsWith(`..${sep}`) && !isAbsolute(path));
}

if (process.argv[2] === "stdio") {
	serveStdio(() => createPolicyServer());
}
