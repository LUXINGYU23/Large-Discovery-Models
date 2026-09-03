import { McpServer } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";

export function createFixtureServer(secret: string): McpServer {
	const server = new McpServer({ name: "ldm-mcp-fixture", version: "1.0.0" });
	server.registerTool(
		"echo",
		{
			description: "Echo a value for MCP bridge tests.",
			inputSchema: z.object({ value: z.string() }),
		},
		async ({ value }) => ({
			content: [{ type: "text", text: `${value}:${secret}:${process.pid}` }],
			structuredContent: { value },
		}),
	);
	server.registerTool(
		"image",
		{ inputSchema: z.object({}) },
		async () => ({
			content: [{ type: "image", data: "aW1hZ2U=", mimeType: "image/png" }],
		}),
	);
	server.registerTool(
		"fail",
		{ inputSchema: z.object({}) },
		async () => ({
			content: [{ type: "text", text: `failure:${secret}` }],
			isError: true,
		}),
	);
	return server;
}

if (process.argv[2] === "stdio") {
	serveStdio(() => createFixtureServer(process.env.FIXTURE_SECRET ?? ""));
}
