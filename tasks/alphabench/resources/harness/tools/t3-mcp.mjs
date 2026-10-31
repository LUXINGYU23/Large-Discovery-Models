import { createConnection } from "node:net";
import { McpServer } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";

const socketPath = process.env.LDM_T3_HOST_SOCKET;
if (!socketPath?.startsWith("/")) throw new Error("LDM_T3_HOST_SOCKET must be absolute");

function host(tool, arguments_) {
	return new Promise((resolve, reject) => {
		const connection = createConnection(socketPath);
		let response = "";
		connection.setTimeout(180_000);
		connection.on("connect", () => connection.write(`${JSON.stringify({ tool, arguments: arguments_ })}\n`));
		connection.on("data", (chunk) => {
			response += chunk.toString("utf8");
			if (response.length > 4_000_000) {
				connection.destroy(new Error("Host tool response exceeds the size limit"));
				return;
			}
			const newline = response.indexOf("\n");
			if (newline < 0) return;
			connection.end();
			try {
				const message = JSON.parse(response.slice(0, newline));
				if (!message.ok) throw new Error(message.error ?? "Host tool rejected the request");
				resolve(message.result);
			} catch (error) {
				reject(error);
			}
		});
		connection.on("timeout", () => connection.destroy(new Error("Host tool timed out")));
		connection.on("error", reject);
		connection.on("close", () => {
			if (!response.includes("\n")) reject(new Error("Host tool closed without a result"));
		});
	});
}

function result(value) {
	return { content: [{ type: "text", text: JSON.stringify(value) }], structuredContent: value };
}

const server = new McpServer({ name: "alphabench-t3", version: "1.0.0" });
function tool(name, description, inputSchema) {
	server.registerTool(name, { description, inputSchema }, async (arguments_) => result(await host(name, arguments_)));
}

tool("get_contract", "Inspect the frozen T3 backend, market, fields, operators, filter and remaining Host check/query allowances.", z.object({}));
tool("validate_expression", "Parse and canonicalize a T3 expression without an Oracle request.", z.object({ expression: z.string() }));
tool("check_expression", "Run a paid dynamic or lint check. The Host reserves its Oracle budget before execution.", z.object({ expression: z.string() }));
tool("get_history", "Read a bounded page of public, previously evaluated search observations.", z.object({ offset: z.number().int().min(0).default(0), limit: z.number().int().min(1).max(100).default(20) }));
tool("get_observation", "Read one public search observation by candidate ID.", z.object({ candidate_id: z.string() }));
tool("query_surrogate", "Query the frozen start-of-round GP posterior for a factor; no Oracle label is revealed.", z.object({ expression: z.string(), snapshot_id: z.string() }));

serveStdio(() => server);
