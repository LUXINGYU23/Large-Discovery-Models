import { createRequire } from "node:module";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { createJiti } from "jiti";
import type { ExtensionFactory, ToolsOptions, ExtensionAPI, ContextEvent, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { isWebResult } from "./web-evidence.js";
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
	const { registerObservationPack } = await jiti.import<{ registerObservationPack: (pi: ExtensionAPI) => void }>(
		join(root, "extensions", "observation-pack", "index.ts"));
	return (pi) => {
		// Bind guest I/O and reserve web observations for the common pressure projection.
		registerConfiguredFeatures(pi, { ...config, actionFusion: false, observationPack: false });
		if (config.observationPack) registerObservationPack(webAwareObservationApi(pi));
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

/** Web evidence has one owner: the pressure projection. SoL-Pi continues packing other observations. */
function webAwareObservationApi(pi: ExtensionAPI): ExtensionAPI {
	return { ...pi, on: ((name: string, handler: unknown) => {
		if (name !== "context") {
			(pi.on as (name: string, handler: unknown) => void)(name, handler);
			return;
		}
		const onContext = handler as (event: ContextEvent, ctx: ExtensionContext) => Promise<{ messages?: ContextEvent["messages"] } | void>;
		pi.on("context", async (event, ctx) => {
			const result = await onContext({ ...event, messages: event.messages.map((message) =>
				message.role === "toolResult" && isWebResult(message) ? { ...message, content: [] } : message) }, ctx);
			if (!result?.messages) return;
			if (result.messages.length !== event.messages.length) throw new Error("SoL-Pi observation projection changed message count");
			return { messages: result.messages.map((message, index) => isWebResult(event.messages[index]!) ? event.messages[index]! : message) };
		});
	}) as ExtensionAPI["on"] };
}
