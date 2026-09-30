import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";

const client = new Client({ name: "t3-smoke", version: "1.0.0" });
const transport = new StdioClientTransport({ command: "node", args: ["/app/t3-mcp.mjs", "stdio"], env: process.env });
await client.connect(transport);
try {
	async function call(name, arguments_) {
		const response = await client.callTool({ name, arguments: arguments_ });
		if (response.isError) throw new Error(JSON.stringify(response.content));
		return response.structuredContent;
	}
	const contract = await call("get_contract", {});
	const validated = await call("validate_expression", { expression: "Mean($close,5)" });
	const history = await call("get_history", { offset: 0, limit: 1 });
	const checked = await call("check_expression", { expression: "Mean($close,5)" });
	const query = process.argv[2] === "none" ? null : await call("query_surrogate", {
		expression: "Mean($close,5)", snapshot_id: process.argv[2],
	});
	process.stdout.write(`${JSON.stringify({ contract, validated, history, checked, query })}\n`);
} finally {
	await client.close();
}
