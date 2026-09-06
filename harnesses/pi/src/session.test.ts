import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import { PiSessionPool } from "./session.js";

test("a failed parallel turn drains other sessions before recovery", async () => {
	const pool = new PiSessionPool({ baseUrl: "https://example.test", campaignId: "test" } as never, "test");
	const sessions = (pool as unknown as { sessions: Map<string, { runTurn: () => Promise<unknown> }> }).sessions;
	let first = true;
	let release!: () => void;
	const pending = new Promise<void>((resolve) => { release = resolve; });
	sessions.set("a", { runTurn: async () => {
		if (first) { first = false; throw new Error("provider 502"); }
		return { sessionId: "a" };
	} });
	sessions.set("b", { runTurn: async () => { await pending; return { sessionId: "b" }; } });
	const inputs = [{ profileId: "a" }, { profileId: "b" }] as never;
	const validate = async () => ({} as never);
	let settled = false;
	const failed = pool.runTurns(inputs, validate).finally(() => { settled = true; });
	const assertion = assert.rejects(failed, /provider 502/);
	await setImmediate();
	assert.equal(settled, false);
	release();
	await assertion;
	assert.deepEqual(await pool.runTurns(inputs, validate), [{ sessionId: "a" }, { sessionId: "b" }]);
});
