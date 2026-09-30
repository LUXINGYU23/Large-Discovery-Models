import { createRequire } from "node:module";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { createJiti } from "jiti";
import type { ExtensionFactory, ToolsOptions } from "@earendil-works/pi-coding-agent";
import { guestPath } from "./gondolin.js";
import { atomicJson } from "./trace.js";

export const SOL_PI_GUEST_ARCHIVE = "/workspace/.sol-pi";

interface SolPiConfig {
	version: 1;
	actionFusion: boolean;
	observationPack: boolean;
	evidencePreservingReducer: boolean;
	evidencePreservingReducerProvider: string;
	evidencePreservingReducerModel: string;
	onlineContextCompact: boolean;
	cacheWriteReadRatio: number;
}

export function solPiTools(config?: Record<string, unknown>): string[] {
	return [
		...(config?.actionFusion ? ["edit"] : []),
		...(config?.observationPack ? ["obs_recall"] : []),
		...(config?.onlineContextCompact ? ["update_plan"] : []),
	];
}

export async function createSolPiExtension(
	settings: Record<string, unknown>,
	agentDirectory: string,
	provider: string,
	model: string,
	toolOptions: () => Promise<ToolsOptions>,
	archiveRoot: string,
): Promise<ExtensionFactory> {
	const require = createRequire(import.meta.url);
	const root = join(dirname(require.resolve("sol-pi/package.json")), "src", "sol-pi");
	const jiti = createJiti(import.meta.url);
	const { loadSolPiConfig } = await jiti.import<{
		loadSolPiConfig(cwd: string, agentDir: string, trusted: boolean): SolPiConfig;
	}>(join(root, "config.ts"));
	await atomicJson(join(agentDirectory, "sol-pi.json"), {
		evidencePreservingReducerProvider: provider,
		evidencePreservingReducerModel: model,
		...settings,
	});
	const config = loadSolPiConfig(agentDirectory, agentDirectory, false);
	if (config.evidencePreservingReducer && (
		config.evidencePreservingReducerProvider !== provider || config.evidencePreservingReducerModel !== model
	)) throw new Error("SoL-Pi reducer must use this session's traced provider and model");
	const { registerConfiguredFeatures } = await jiti.import<{
		registerConfiguredFeatures: (pi: Parameters<ExtensionFactory>[0], config: SolPiConfig) => void;
	}>(join(root, "index.ts"));
	const { createActionFusionExtension } = await jiti.import<{
		createActionFusionExtension(options: {
			bashOptions?: ToolsOptions["bash"];
			editOptions?: ToolsOptions["edit"];
			writeOptions?: ToolsOptions["write"];
		}): ExtensionFactory;
	}>(join(root, "extensions", "action-fusion", "index.ts"));
	return (pi) => {
		// Keep upstream mechanisms intact; only bind file/command I/O to the guest.
		registerConfiguredFeatures(pi, { ...config, actionFusion: false });
		if (config.actionFusion) pi.on("session_start", async (_event, ctx) => {
			const options = await toolOptions();
			await createActionFusionExtension({
				bashOptions: options.bash,
				editOptions: options.edit,
				writeOptions: options.write,
			})({
				...pi,
				registerTool: (tool) => pi.registerTool({
					...tool,
					execute: (id, input, signal, update, context) => {
						let path = String(input.path);
						if (path.startsWith("@")) path = path.slice(1);
						if (path.startsWith("file://")) path = fileURLToPath(path);
						const guest = guestPath(ctx.cwd, path);
						const host = join(ctx.cwd, relative("/workspace", guest));
						return tool.execute(id, { ...input, path: host }, signal, update, context);
					},
				}),
			});
		});
		// Receipts retain exact evidence and refer to the guest's read-only archive.
		pi.on("context", ({ messages }) => ({
			messages: messages.map((message) => message.role === "toolResult" ? {
				...message,
				content: message.content.map((block) => block.type === "text"
					? { ...block, text: block.text.replaceAll(archiveRoot, SOL_PI_GUEST_ARCHIVE) }
					: block),
			} : message),
		}));
	};
}
