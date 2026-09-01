import assert from "node:assert/strict";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { NodeStreamableHTTPServerTransport } from "@modelcontextprotocol/node";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { createFixtureServer } from "./mcp-fixture.js";
import { McpToolBridge } from "./mcp.js";
import type { McpServerConfig } from "./protocol.js";

async function call(tool: ToolDefinition, value: string) {
	return tool.execute("call-1", { value }, undefined, undefined, {} as never);
}

test("stdio MCP tools are allowlisted, callable, and secret-safe", async () => {
	const secret = "stdio-fixture-secret";
	const fixture = fileURLToPath(new URL("./mcp-fixture.js", import.meta.url));
	const config: McpServerConfig = {
		serverId: "local",
		transport: "stdio",
		command: process.execPath,
		args: [fixture, "stdio"],
		env: {
			FIXTURE_SECRET: {
				secretName: "mcp.local.env.fixture",
				secretSource: "test",
				prefix: "",
			},
		},
		tools: ["echo", "image", "fail"],
		configSha256: "a".repeat(64),
	};
	const bridge = new McpToolBridge(
		[config],
		{ "mcp.local.env.fixture": secret },
		"stdio-test",
	);
	try {
		await bridge.initialize();
		assert.deepEqual(bridge.toolNames(), [
			"mcp__local__echo",
			"mcp__local__image",
			"mcp__local__fail",
		]);
		const tools = bridge.toolDefinitions();
		const result = await call(tools[0] as ToolDefinition, "hello");
		assert.match(JSON.stringify(result), /hello:\[REDACTED\]:\d+/);
		assert.doesNotMatch(JSON.stringify(result), new RegExp(secret));
		assert.deepEqual(result.details, {
			structuredContent: { value: "hello" },
			mcpContent: [{ type: "text", text: (result.content[0] as { text: string }).text }],
		});
		const image = await call(tools[1] as ToolDefinition, "unused");
		assert.deepEqual(image.content, [{ type: "image", data: "aW1hZ2U=", mimeType: "image/png" }]);
		await assert.rejects(call(tools[2] as ToolDefinition, "unused"), /failure:\[REDACTED\]/);
		const manifest = bridge.manifest();
		assert.equal(manifest[0]?.serverId, "local");
		assert.doesNotMatch(JSON.stringify(manifest), new RegExp(secret));
	} finally {
		await bridge.close();
	}
});

test("stdio MCP clients use independent server processes", async () => {
	const fixture = fileURLToPath(new URL("./mcp-fixture.js", import.meta.url));
	const config: McpServerConfig = {
		serverId: "isolated",
		transport: "stdio",
		command: process.execPath,
		args: [fixture, "stdio"],
		env: {},
		tools: ["echo"],
		configSha256: "c".repeat(64),
	};
	const first = new McpToolBridge([config], {}, "first-session");
	const second = new McpToolBridge([config], {}, "second-session");
	try {
		await Promise.all([first.initialize(), second.initialize()]);
		const results = await Promise.all([
			call(first.toolDefinitions()[0] as ToolDefinition, "first"),
			call(second.toolDefinitions()[0] as ToolDefinition, "second"),
		]);
		const pids = results.map((result) => {
			const text = (result.content[0] as { text: string }).text;
			return text.split(":").at(-1);
		});
		assert.notEqual(pids[0], pids[1]);
	} finally {
		await Promise.all([first.close(), second.close()]);
	}
});

test("Streamable HTTP MCP forwards configured headers and rejects missing tools", async () => {
	const secret = "http-fixture-secret";
	const transport = new NodeStreamableHTTPServerTransport({ sessionIdGenerator: undefined });
	const fixture = createFixtureServer(secret);
	await fixture.connect(transport);
	const http = createServer(async (request, response) => {
		if (request.headers.authorization !== `Bearer ${secret}`) {
			response.writeHead(401).end();
			return;
		}
		await transport.handleRequest(request, response);
	});
	await new Promise<void>((resolve) => http.listen(0, "127.0.0.1", resolve));
	const address = http.address();
	assert(address && typeof address !== "string");
	const base: McpServerConfig = {
		serverId: "remote",
		transport: "streamable_http",
		url: `http://127.0.0.1:${address.port}/mcp`,
		headers: {
			Authorization: {
				secretName: "mcp.remote.header.authorization",
				secretSource: "test",
				prefix: "Bearer ",
			},
		},
		tools: ["echo"],
		configSha256: "b".repeat(64),
	};
	const secrets = { "mcp.remote.header.authorization": secret };
	const bridge = new McpToolBridge([base], secrets, "http-test");
	try {
		await bridge.initialize();
		const result = await call(bridge.toolDefinitions()[0] as ToolDefinition, "hello");
		assert.match(JSON.stringify(result), /hello:\[REDACTED\]/);
		const missing = new McpToolBridge(
			[{ ...base, serverId: "missing", tools: ["absent"] }],
			secrets,
			"missing-test",
		);
		await assert.rejects(missing.initialize(), /missing allowlisted tools: absent/);
	} finally {
		await bridge.close();
		await fixture.close();
		await new Promise<void>((resolve, reject) => http.close((error) => error ? reject(error) : resolve()));
	}
});
