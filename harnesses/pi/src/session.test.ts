import assert from "node:assert/strict";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PiSessionPool, PersistentProfileSession } from "./session.js";
import { TurnExecutionError } from "./protocol.js";
import { HistoryController, HistoryRecoveryError } from "./history.js";
import { canonicalJson, sha256 } from "./trace.js";

test("history recovery orders same-round commits and rejects corrupt chains", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-history-cursor-"));
	const profile = Object.assign(Object.create(PersistentProfileSession.prototype), {
		config: { artifactRoot: root }, profile: { profileId: "research" },
	}) as { recoverHistoryCursor(): Promise<number> };
	const body = "verified artifact";
	const artifact = { pathPointer: "/artifact", relativePath: "artifact.txt", snapshotPath: "artifact.txt", sha256: sha256(body), sizeBytes: Buffer.byteLength(body) };
	const submission = { artifact: "artifact.txt" };
	const submissionJson = canonicalJson({ artifacts: [artifact], submission });
	const record = { profileId: "research", roundIndex: 1, submissionStatus: "accepted", submission,
		submittedArtifacts: [artifact], validationErrors: [], submissionJson, submissionDigest: sha256(submissionJson) };
	const run = async (chain: Array<[number, number, number]>, expected: number | RegExp, corrupt = false) => {
		await rm(join(root, "turns"), { recursive: true, force: true });
		// Reverse names ensure directory enumeration cannot determine chain order.
		for (const [index, [roundIndex, historyFromSeq, historyToSeq]] of chain.entries()) {
			const dir = join(root, "turns", String(chain.length - index).padStart(3, "0"));
			await mkdir(dir, { recursive: true });
			await writeFile(join(dir, "turn_committed.json"), JSON.stringify({ ...record, roundIndex, historyFromSeq, historyToSeq,
				...(corrupt && index === 0 ? { submissionDigest: "0".repeat(64) } : {}) }));
		}
		if (typeof expected === "number") assert.equal(await profile.recoverHistoryCursor(), expected);
		else await assert.rejects(profile.recoverHistoryCursor(), expected);
	};
	try {
		await writeFile(join(root, "artifact.txt"), body);
		for (const chain of [
			[[1, 0, 0], [1, 0, 16], [1, 16, 16]],
			[[1, 16, 16], [1, 0, 16], [1, 0, 0]],
			[[1, 0, 16], [1, 16, 16], [1, 16, 32], [2, 32, 32]],
		] as Array<Array<[number, number, number]>>) await run(chain, chain.length === 4 ? 32 : 16);
		await run([[1, 0, 16], [1, 17, 32]], /invalid committed history chain/);
		await run([[1, 0, 16], [1, 15, 32]], /invalid committed history chain/);
		await run([[1, 0, 16], [1, 0, 16]], /invalid committed history chain/);
		await run([[1, 0, 16]], /terminal submission digest mismatch/, true);
		await rm(join(root, "artifact.txt"));
		await run([[1, 0, 16]], /ENOENT/);
	} finally { await rm(root, { recursive: true, force: true }); }
});

test("a failed parallel turn drains other sessions before recovery", async () => {
	const pool = new PiSessionPool({ baseUrl: "https://example.test", campaignId: "test" } as never, "test");
	const sessions = (pool as unknown as { sessions: Map<string, { runTurn: () => Promise<unknown> }> }).sessions;
	let first = true;
	const usage = { providerCalls: 3, toolCalls: { bash: 2 }, artifactBytes: 120, tokens: { input: 0, cachedInput: 0, output: 0, reasoning: 0 } };
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
	const authorize = async () => true;
	let settled = false;
	const failed = pool.runTurns(inputs, validate, authorize).finally(() => { settled = true; });
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
	assert.deepEqual(await pool.runTurns(inputs, validate, authorize), [{ sessionId: "a" }, committed]);
});

test("partial-turn continuation keeps history and classifies execution failures", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-failed-turn-"));
	try {
		const messages: string[] = [];
		const failures = [
			"provider response failed: server_error: An error occurred while processing your request.",
			"provider response failed: internal_server_error: Try again later.",
			"provider response failed: overloaded_error: Try again later.",
			"provider response failed: rate_limit_exceeded: Slow down.",
			"provider response failed: request_timeout: Try again later.",
			"provider response failed: terminated",
			"session wall-time limit reached: 1800s",
			"provider response failed: 401 unauthorized",
			"provider response failed: insufficient_quota: Check billing.",
		];
		const profile = Object.assign(Object.create(PersistentProfileSession.prototype), {
			history: new HistoryController(),
			profileRoot: join(root, "sessions/research"),
			profile: { profileId: "research" }, config: { artifactRoot: root, limits: {}, submissionContract: { toolName: "submit_candidates" } },
			session: { sessionManager: { getSessionId: () => "session-1" } }, historyCursor: 0,
			policy: { begin: async () => {}, budgetMessage: () => "", end: () => ({ toolCalls: { bash: 2 } }) },
			submissions: { begin: async () => {} },
			proxy: { beginTurn: async () => {}, endTurn: async () => ({ providerCalls: 3, artifactBytes: 120, tokens: { input: 9, cachedInput: 4, output: 2, reasoning: 1 } }) },
			promptWithTimeout: async (message: string) => {
				messages.push(message);
				throw new Error(failures[messages.length - 1]);
			},
		});
		await assert.rejects(profile.runTurn({
			profileId: "research", turnId: "turn-1", inputDigest: "digest", historyFromSeq: 0, historyToSeq: 1, message: "ORIGINAL_HISTORY",
		}, async () => ({}), async () => true), (error: unknown) => {
			assert.ok(error instanceof TurnExecutionError);
			assert.equal(error.retryable, true);
			assert.deepEqual(error.turnUsage, [{ profileId: "research", turnId: "turn-1",
				usage: { providerCalls: 3, toolCalls: { bash: 2 }, artifactBytes: 120,
					tokens: { input: 9, cachedInput: 4, output: 2, reasoning: 1 } } }]);
			return true;
		});
		for (let index = 1; index < failures.length; index += 1) {
			await assert.rejects(profile.runTurn({
				profileId: "research", turnId: "turn-1", inputDigest: "digest", historyFromSeq: 0, historyToSeq: 1, message: "ORIGINAL_HISTORY",
			}, async () => ({}), async () => true), (error: unknown) => {
				assert.ok(error instanceof TurnExecutionError);
				assert.equal(error.retryable, index < failures.length - 3, failures[index]);
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
	await assert.rejects(pool.runTurns([{ profileId: "a" }, { profileId: "b" }] as never, async () => ({} as never), async () => true),
		(error: unknown) => error instanceof TurnExecutionError && !error.retryable && error.message.includes("digest mismatch"));
});

test("wall-time cancellation drains the current prompt without starting another request", async () => {
	let release!: () => void;
	const pending = new Promise<void>((resolve) => { release = resolve; });
	let requests = 0;
	let aborted = false;
	let compactionAborted = false;
	const context = {
		history: { assertReady: () => {}, fail: async () => {} },
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

test("compaction failures are terminal even when their text contains a retryable provider code", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-history-failure-"));
	try {
		let upstreamCalls = 0;
		const history = new HistoryController();
		const manager = { getSessionId: () => "session-1", getBranch: () => [] };
		const profile = Object.assign(Object.create(PersistentProfileSession.prototype), {
			profileRoot: join(root, "sessions/research"), history,
			profile: { profileId: "research" }, historyCursor: 0,
			config: { artifactRoot: root, limits: {}, submissionContract: { toolName: "submit_candidates" } },
			session: { sessionManager: manager },
			policy: { begin: async () => {}, budgetMessage: () => "", end: () => ({ toolCalls: {} }) },
			submissions: { begin: async () => {} },
			proxy: { beginTurn: async () => {}, endTurn: async () => ({ providerCalls: upstreamCalls, artifactBytes: 0 }) },
			promptWithTimeout: async () => {
				upstreamCalls += 1;
				await history.fail("Compaction failed: context_length_exceeded 502");
				history.assertReady();
			},
		});
		const turn = { profileId: "research", turnId: "turn", inputDigest: "digest", historyFromSeq: 0, historyToSeq: 1, message: "research" };
		for (let i = 0; i < 2; i += 1) {
			await assert.rejects(profile.runTurn(turn, async () => ({}), async () => true),
				(error: unknown) => error instanceof TurnExecutionError && !error.retryable);
		}
		assert.equal(upstreamCalls, 1, "same failed history must not be sent again after resume");
	} finally { await rm(root, { recursive: true, force: true }); }
});

test("failed native compaction of a length stop never receives another submission nudge", async () => {
	const root = await mkdtemp(join(tmpdir(), "ldm-history-length-"));
	try {
		const history = new HistoryController();
		await history.begin(root, { getBranch: () => [] } as never);
		let calls = 0;
		const context = {
			history, config: { submissionContract: { toolName: "submit_candidates" } },
			session: { messages: [{ role: "assistant", stopReason: "length", timestamp: Date.now() }], prompt: async () => { calls += 1; }, compact: async () => { throw new Error("summary failed"); } },
			submissions: { submission: undefined },
		};
		const prototype = PersistentProfileSession.prototype as unknown as {
			promptWithTimeout(message: string, limits: { wallTimeSeconds: number }): Promise<void>;
		};
		await assert.rejects(prototype.promptWithTimeout.call(context, "research", { wallTimeSeconds: 2 }), HistoryRecoveryError);
		assert.equal(calls, 1);
	} finally { await rm(root, { recursive: true, force: true }); }
});


test("a length stop after successful native compaction gets one bounded continuation", async () => {
 const root = await mkdtemp(join(tmpdir(), "ldm-compacted-length-"));
 try {
  const manager = SessionManager.create(root, join(root,"sessions"));
  const timestamp = Date.now()-1000;
  manager.appendMessage({role:"user",content:"research",timestamp});
  manager.appendCompaction("Preserved work", manager.getLeafId()!, 100);
  const history = new HistoryController();await history.begin(root,manager);
  let calls=0;const submissions:{submission:unknown}={submission:undefined};
  const context={history,config:{submissionContract:{toolName:"submit_candidates"}},submissions,
   session:{messages:[{role:"assistant",stopReason:"length",timestamp}],
    compact:async()=>{throw new Error("Must reuse completed native compaction");},
    prompt:async(message:string)=>{calls++;if(calls===2){assert.match(message,/output-token budget/);submissions.submission={accepted:true};}}}};
  const prototype=PersistentProfileSession.prototype as unknown as {promptWithTimeout(message:string,limits:{wallTimeSeconds:number}):Promise<void>};
  await prototype.promptWithTimeout.call(context,"research",{wallTimeSeconds:2});
  assert.equal(calls,2);assert.equal(JSON.parse(await readFile(join(root,"history-recovery.json"),"utf8")).continuations,1);
 } finally {await rm(root,{recursive:true,force:true});}
});
