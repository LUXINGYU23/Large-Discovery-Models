import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import test from "node:test";

test("the sidecar exits on its own when its input ends (an orphaned host)", async () => {
	const child = spawn(process.execPath, [fileURLToPath(new URL("./main.js", import.meta.url))], {
		stdio: ["pipe", "pipe", "pipe"],
	});
	const [first] = await once(child.stdout, "data") as [Buffer];
	assert.equal(JSON.parse(first.toString().split("\n")[0] ?? "").type, "ready");
	const started = Date.now();
	child.stdin.end();
	const timer = setTimeout(() => child.kill("SIGKILL"), 10_000);
	const [code] = await once(child, "exit") as [number | null];
	clearTimeout(timer);
	assert.equal(code, 0);
	assert(Date.now() - started < 5_000);
});
