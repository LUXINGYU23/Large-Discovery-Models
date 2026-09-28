import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { configureGuestCache, resolveGuestRuntime } from "/app/dist/guest-image.js";
import { loadTaskGuestRecipe } from "/app/dist/task-guest-recipe.js";
import { GondolinController } from "/app/dist/gondolin.js";

configureGuestCache("/runtime-home/.cache/gondolin");
assert.equal((await readFile("/artifacts/private/secret.canary", "utf8")).trim(), "host_only");
const recipe = await loadTaskGuestRecipe("alphabench");
const guest = await resolveGuestRuntime(recipe.taskId, recipe.guestRuntime);
const controller = new GondolinController(
	"/artifacts/sessions/isolation/workspace",
	"/artifacts/sessions/isolation/workspace/.ldm-resources",
	{ allowedHosts: ["offline.invalid"], deniedHosts: [], forbiddenQueryPatterns: [] },
	guest,
);
try {
	const options = await controller.toolOptions();
	await assert.rejects(options.read.operations.readFile("/artifacts/private/secret.canary"));
	let output = "";
	const command = [
		"test -r .ldm-resources/AGENTS.md",
		"! (printf altered > .ldm-resources/AGENTS.md)",
		"test ! -e /artifacts/private/secret.canary",
		"test ! -e /t3-host/host.sock",
		"python3 -c 'from pathlib import Path; Path(\"candidate.json\").write_text(\"ok\")'",
	].join(" && ");
	const result = await options.bash.operations.exec(command, "/workspace", {
		onData: (chunk) => { output += chunk; }, timeout: 30,
	});
	assert.equal(result.exitCode, 0, output);
	let networkOutput = "";
	const network = await options.bash.operations.exec(
		"python3 -c 'import urllib.request; urllib.request.urlopen(\"https://example.com\", timeout=10)'",
		"/workspace", { onData: (chunk) => { networkOutput += chunk; }, timeout: 20 },
	);
	assert.notEqual(network.exitCode, 0, networkOutput);
	assert.match(networkOutput, /403|denied|blocked|not allowed|forbidden/i);
	process.stdout.write(`${JSON.stringify({ status: "ok", imageRef: guest.imageRef })}\n`);
} finally {
	await controller.close();
}
