import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PiSessionPool, PersistentProfileSession, SubmissionController, pinInitialization } from "./session.js";
import type { InitializeFrame, SubmissionValidator } from "./protocol.js";
import { TurnExecutionError } from "./protocol.js";

test("rejected artifacts can be edited and resubmitted across partial-turn recovery", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-repair-"));
	try {
		const workspace = join(root, "workspace");
		const turnRoot = join(root, "turns", "turn-1");
		await mkdir(workspace);
		const path = join(workspace, "policy.py");
		await writeFile(path, "VALUE = 1\n");
		const config = {
			artifactRoot: root,
			submissionContract: {
				toolName: "submit_policy",
				payloadSchema: { type: "object" },
				artifactRules: [{ pathPointer: "/artifact_path", allowedSuffixes: [".py"], maxBytes: 128 }],
			},
		} as unknown as InitializeFrame;
		const attempts: number[] = [];
		const validate: SubmissionValidator = async (request) => {
			attempts.push(request.attemptIndex);
			return request.attemptIndex === 1
				? { decision: "retry", errors: [{ path: "/artifact_path", code: "wrong_shape", message: "Expected two objectives", hint: "Edit the array shape and validate again" }] }
				: { decision: "accept", errors: [] };
		};
		const first = new SubmissionController(config, workspace);
		await first.begin("policy", "turn-1", turnRoot, async () => {}, validate);
		let providerHook!: (event: { payload: object }) => unknown;
		first.createExtension()({ on: (_name: string, hook: typeof providerHook) => { providerHook = hook; } } as never);
		assert.deepEqual(providerHook({ payload: {} }), { tool_choice: "required" });
		await assert.rejects(first.tool().execute("call-1", { artifact_path: "policy.py" }, undefined, undefined, {} as never), /wrong_shape.*Expected two objectives/);
		assert.equal(first.submission, undefined);
		assert.deepEqual(providerHook({ payload: { tool_choice: "auto" } }), { tool_choice: "auto" });
		await writeFile(path, "VALUE = [1, 2]\n");
		const recovered = new SubmissionController(config, workspace);
		await recovered.begin("policy", "turn-1", turnRoot, async () => {}, validate);
		await recovered.tool().execute("call-2", { artifact_path: "policy.py" }, undefined, undefined, {} as never);
		assert.deepEqual(attempts, [1, 2]);
		assert.equal(recovered.submission?.submissionStatus, "accepted");
		const firstReport = JSON.parse(await readFile(join(turnRoot, "attempts/0001/validation.json"), "utf8"));
		const secondReport = JSON.parse(await readFile(join(turnRoot, "attempts/0002/validation.json"), "utf8"));
		assert.equal(firstReport.decision, "retry");
		assert.notEqual(firstReport.submissionDigest, secondReport.submissionDigest);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("session resume pins configuration and runtime identity without rewriting provenance", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-identity-"));
	try {
		const identity = { model: "original", profileSha256: "a", runtime: "image", contract: "two-objective" };
		const digest = await pinInitialization(root, identity);
		assert.equal(await pinInitialization(root, { ...identity }), digest);
		const original = await readFile(join(root, "manifest.json"), "utf8");
		for (const key of Object.keys(identity)) {
			await assert.rejects(pinInitialization(root, { ...identity, [key]: "changed" }), /identity changed/);
		}
		assert.equal(await readFile(join(root, "manifest.json"), "utf8"), original);
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("a failed parallel turn drains other sessions before recovery", async () => {
	const pool = new PiSessionPool({ baseUrl: "https://example.test", campaignId: "test" } as never, "test");
	const sessions = (pool as unknown as { sessions: Map<string, { runTurn: () => Promise<unknown> }> }).sessions;
	let first = true;
	const usage = { providerCalls: 3, toolCalls: { bash: 2 }, artifactBytes: 120 };
	let release!: () => void;
	const pending = new Promise<void>((resolve) => { release = resolve; });
	sessions.set("a", { runTurn: async () => {
		if (first) { first = false; throw new TurnExecutionError("provider 502", [{ profileId: "a", turnId: "a-1", usage }], true); }
		return { sessionId: "a" };
	} });
	const committed = { sessionId: "b", profileId: "b", turnId: "b-1", usage };
	sessions.set("b", { runTurn: async () => { await pending; return committed; } });
	const inputs = [{ profileId: "a", turnId: "a-1" }, { profileId: "b", turnId: "b-1" }] as never;
	const validate = async () => ({} as never);
	let settled = false;
	const failed = pool.runTurns(inputs, validate).finally(() => { settled = true; });
	const assertion = assert.rejects(failed, (error: unknown) => {
		assert.ok(error instanceof TurnExecutionError);
		assert.match(error.message, /provider 502/);
		assert.equal(error.retryable, true);
		assert.deepEqual(error.turnUsage, [
			{ profileId: "a", turnId: "a-1", usage }, { profileId: "b", turnId: "b-1", usage },
		]);
		return true;
	});
	await setImmediate();
	assert.equal(settled, false);
	release();
	await assertion;
	assert.deepEqual(await pool.runTurns(inputs, validate), [{ sessionId: "a" }, committed]);
});

test("partial-turn continuation keeps history and classifies execution failures", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-failed-turn-"));
	try {
		const messages: string[] = [];
		const failures = [
			"session wall-time limit reached: 1800s",
			"provider response failed: server_error: An error occurred while processing your request.",
			"provider response failed: internal_server_error: Try again later.",
			"provider response failed: overloaded_error: Try again later.",
			"provider response failed: rate_limit_exceeded: Slow down.",
			"provider response failed: request_timeout: Try again later.",
			"provider response failed: terminated",
			"provider response failed: 401 unauthorized",
			"provider response failed: insufficient_quota: Check billing.",
		];
		const profile = Object.assign(Object.create(PersistentProfileSession.prototype), {
			profileRoot: join(root, "sessions/research"),
			profile: { profileId: "research" }, config: { artifactRoot: root, limits: {}, submissionContract: { toolName: "submit_candidates" } },
			session: { sessionManager: { getSessionId: () => "session-1" } }, historyCursor: 0,
			policy: { begin: async () => {}, budgetMessage: () => "", end: () => ({ toolCalls: { bash: 2 } }) },
			submissions: { begin: async () => {} },
			proxy: { beginTurn: async () => {}, endTurn: async () => ({ providerCalls: 3, artifactBytes: 120 }) },
			promptWithTimeout: async (message: string) => {
				messages.push(message);
				throw new Error(failures[messages.length - 1]);
			},
		});
		await assert.rejects(profile.runTurn({
			profileId: "research", turnId: "turn-1", inputDigest: "digest", historyFromSeq: 0, historyToSeq: 1, message: "ORIGINAL_HISTORY",
		}, async () => ({})), (error: unknown) => {
			assert.ok(error instanceof TurnExecutionError);
			assert.equal(error.retryable, true);
			assert.deepEqual(error.turnUsage, [{ profileId: "research", turnId: "turn-1",
				usage: { providerCalls: 3, toolCalls: { bash: 2 }, artifactBytes: 120 } }]);
			return true;
		});
		for (let index = 1; index < failures.length; index += 1) {
			await assert.rejects(profile.runTurn({
				profileId: "research", turnId: "turn-1", inputDigest: "digest", historyFromSeq: 0, historyToSeq: 1, message: "ORIGINAL_HISTORY",
			}, async () => ({})), (error: unknown) => {
				assert.ok(error instanceof TurnExecutionError);
				assert.equal(error.retryable, index < failures.length - 2, failures[index]);
				return true;
			});
		}
		assert.match(messages[0]!, /ORIGINAL_HISTORY/);
		assert.doesNotMatch(messages[1]!, /ORIGINAL_HISTORY/);
		assert.match(messages[1]!, /Continue the interrupted turn.*Tool budgets have not reset/);
		assert.equal(profile.historyCursor, 0);
		await assert.rejects(readFile(join(root, "turns/turn-1/turn_committed.json")), { code: "ENOENT" });
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});

test("a fatal failure is not masked by another session's recoverable failure", async () => {
	const pool = new PiSessionPool({ baseUrl: "https://example.test", campaignId: "test" } as never, "test");
	const sessions = (pool as unknown as { sessions: Map<string, { runTurn: () => Promise<unknown> }> }).sessions;
	sessions.set("a", { runTurn: async () => { throw new TurnExecutionError("timeout", [], true); } });
	sessions.set("b", { runTurn: async () => { throw new Error("digest mismatch"); } });
	await assert.rejects(pool.runTurns([{ profileId: "a" }, { profileId: "b" }] as never, async () => ({} as never)),
		(error: unknown) => error instanceof TurnExecutionError && !error.retryable && error.message.includes("digest mismatch"));
});

test("wall-time cancellation drains the current prompt without starting another request", async () => {
	let release!: () => void;
	const pending = new Promise<void>((resolve) => { release = resolve; });
	let requests = 0;
	let aborted = false;
	let compactionAborted = false;
	const context = {
		session: {
			messages: [],
			prompt: async () => { requests += 1; await pending; },
			abortCompaction: () => { compactionAborted = true; release(); },
			abort: async () => { aborted = true; await pending; },
		},
		submissions: { submission: undefined },
	};
	const prototype = PersistentProfileSession.prototype as unknown as {
		promptWithTimeout(message: string, limits: { wallTimeSeconds: number }): Promise<void>;
	};
	await assert.rejects(
		prototype.promptWithTimeout.call(context, "research", { wallTimeSeconds: 0.01 }),
		/wall-time limit/,
	);
	await setImmediate();
	assert.equal(aborted, true);
	assert.equal(compactionAborted, true);
	assert.equal(requests, 1);
});
