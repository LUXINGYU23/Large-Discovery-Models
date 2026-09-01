import {
	Client,
	StreamableHTTPClientTransport,
	type Tool as McpTool,
	type Transport,
} from "@modelcontextprotocol/client";
import {
	getDefaultEnvironment,
	StdioClientTransport,
} from "@modelcontextprotocol/client/stdio";
import type { AgentToolResult, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Type, type TUnsafe } from "typebox";
import type { McpInjectedValue, McpServerConfig } from "./protocol.js";
import { canonicalSha256, Redactor } from "./trace.js";

interface McpConnection {
	config: McpServerConfig;
	client: Client;
	transport: Transport;
	tools: McpTool[];
}

export class McpToolBridge {
	private readonly connections: McpConnection[] = [];
	private readonly redactor: Redactor;
	private initialized = false;

	constructor(
		private readonly configs: readonly McpServerConfig[],
		private readonly namedSecrets: Readonly<Record<string, string>>,
		private readonly clientName: string,
	) {
		this.redactor = new Redactor(Object.values(namedSecrets));
	}

	async initialize(): Promise<void> {
		if (this.initialized) throw new Error("MCP bridge is already initialized");
		this.initialized = true;
		try {
			for (const config of this.configs) {
				this.connections.push(await this.connect(config));
			}
		} catch (error) {
			await this.close();
			throw new Error(`MCP initialization failed: ${this.redactor.text((error as Error).message)}`);
		}
	}

	toolNames(): string[] {
		return this.connections.flatMap((connection) =>
			connection.tools.map((tool) => piToolName(connection.config.serverId, tool.name))
		);
	}

	toolDefinitions(): Array<ToolDefinition<TUnsafe<Record<string, unknown>>>> {
		return this.connections.flatMap((connection) => connection.tools.map((tool) => ({
			name: piToolName(connection.config.serverId, tool.name),
			label: tool.title ?? tool.name,
			description: tool.description ?? `MCP tool ${tool.name}`,
			promptSnippet: `${piToolName(connection.config.serverId, tool.name)}: ${tool.description ?? tool.name}`,
			parameters: Type.Unsafe<Record<string, unknown>>(tool.inputSchema),
			executionMode: "sequential" as const,
			execute: async (_toolCallId: string, params: Record<string, unknown>) => {
				try {
					const result = await connection.client.callTool({
						name: tool.name,
						arguments: params,
					});
					const safe = this.redactor.value(result) as typeof result;
					if (safe.isError === true) {
						throw new Error(`MCP tool returned an error: ${JSON.stringify(safe.content)}`);
					}
					return {
						content: mcpContent(safe.content),
						details: {
							structuredContent: safe.structuredContent,
							mcpContent: safe.content,
						},
					};
				} catch (error) {
					throw new Error(this.redactor.text((error as Error).message));
				}
			},
		})));
	}

	manifest(): Array<Record<string, unknown>> {
		return this.connections.map((connection) => ({
			serverId: connection.config.serverId,
			transport: connection.config.transport,
			configSha256: connection.config.configSha256,
			serverInfo: connection.client.getServerVersion(),
			capabilitiesSha256: canonicalSha256(connection.client.getServerCapabilities() ?? {}),
			secretSources: secretSources(connection.config),
			tools: connection.tools.map((tool) => ({
				mcpName: tool.name,
				piName: piToolName(connection.config.serverId, tool.name),
				descriptionSha256: canonicalSha256(tool.description ?? ""),
				inputSchemaSha256: canonicalSha256(tool.inputSchema),
			})),
		}));
	}

	async close(): Promise<void> {
		const connections = this.connections.splice(0);
		await Promise.allSettled(connections.map(async ({ client, transport }) => {
			if (transport instanceof StreamableHTTPClientTransport) {
				await transport.terminateSession().catch(() => undefined);
			}
			await client.close();
		}));
	}

	private async connect(config: McpServerConfig): Promise<McpConnection> {
		const client = new Client({ name: this.clientName, version: "1.0.0" });
		const transport = this.transport(config);
		if (transport instanceof StdioClientTransport && transport.stderr) {
			const stderr = this.redactor.stream();
			transport.stderr.on("data", (chunk: Buffer | string) => {
				process.stderr.write(stderr.update(Buffer.from(chunk)));
			});
			transport.stderr.on("end", () => process.stderr.write(stderr.end()));
		}
		await client.connect(transport);
		const available = await client.listTools();
		const byName = new Map(available.tools.map((tool) => [tool.name, tool]));
		const missing = config.tools.filter((name) => !byName.has(name));
		if (missing.length > 0) {
			await client.close();
			throw new Error(`MCP server ${config.serverId} is missing allowlisted tools: ${missing.join(", ")}`);
		}
		const tools = config.tools.map((name) => byName.get(name) as McpTool);
		for (const tool of tools) {
			if (
				!tool.inputSchema
				|| typeof tool.inputSchema !== "object"
				|| Array.isArray(tool.inputSchema)
				|| tool.inputSchema.type !== "object"
			) {
				await client.close();
				throw new Error(`MCP tool ${config.serverId}/${tool.name} requires an object input schema`);
			}
		}
		return { config, client, transport, tools };
	}

	private transport(config: McpServerConfig): Transport {
		if (config.transport === "stdio") {
			return new StdioClientTransport({
				command: config.command,
				args: config.args,
				env: {
					...getDefaultEnvironment(),
					...resolveValues(config.env, this.namedSecrets),
				},
				stderr: "pipe",
			});
		}
		return new StreamableHTTPClientTransport(new URL(config.url), {
			requestInit: { headers: resolveValues(config.headers, this.namedSecrets) },
		});
	}
}

function resolveValues(
	values: Record<string, McpInjectedValue>,
	secrets: Readonly<Record<string, string>>,
): Record<string, string> {
	return Object.fromEntries(Object.entries(values).map(([name, value]) => {
		if ("value" in value) return [name, value.value];
		const secret = secrets[value.secretName];
		if (!secret) throw new Error(`MCP named secret is unavailable: ${value.secretName}`);
		return [name, `${value.prefix}${secret}`];
	}));
}

function piToolName(serverId: string, toolName: string): string {
	return `mcp__${serverId}__${toolName}`;
}

function secretSources(config: McpServerConfig): Array<{ target: string; source: string }> {
	const values = config.transport === "stdio" ? config.env : config.headers;
	return Object.entries(values).flatMap(([target, value]) => (
		"secretName" in value ? [{ target, source: value.secretSource }] : []
	));
}

function mcpContent(content: readonly unknown[]): AgentToolResult<unknown>["content"] {
	const blocks = content as Array<Record<string, unknown>>;
	return blocks.map((block) => {
		if (block.type === "text" && typeof block.text === "string") {
			return { type: "text", text: block.text };
		}
		if (
			block.type === "image"
			&& typeof block.data === "string"
			&& typeof block.mimeType === "string"
		) {
			return { type: "image", data: block.data, mimeType: block.mimeType };
		}
		return { type: "text", text: JSON.stringify(block) };
	});
}
