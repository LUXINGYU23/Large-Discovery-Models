import assert from "node:assert/strict";
import test from "node:test";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { directorySha256, initializeFork } from "./session.js";
import { sha256 } from "./trace.js";

test("per-profile forks restore different historical parents and reject changed identities", async () => {
    const root = await mkdtemp(join(tmpdir(), "ldm-multiple-forks-"));
    try {
        for (const [profile, cursor] of [["left", 1], ["right", 17]] as const) {
            const snapshot = join(root, "fork-sources", profile);
            await mkdir(join(snapshot, "workspace"), { recursive: true });
            await writeFile(join(snapshot, "workspace", "research.md"), profile);
            const header = { type: "session", version: 3, id: profile, timestamp: new Date().toISOString(), cwd: "/parent" };
            const entry = { type: "message", id: `leaf-${profile}`, parentId: null, timestamp: new Date().toISOString(),
                message: { role: "user", content: profile, timestamp: Date.now() } };
            await writeFile(join(snapshot, "session.jsonl"), [header, entry].map(x => JSON.stringify(x)).join("\n") + "\n");
            const source = { sessionSha256: sha256(await readFile(join(snapshot, "session.jsonl"))),
                workspaceSha256: await directorySha256(join(snapshot, "workspace")), historyCursor: cursor };
            const child = join(root, "sessions", profile);
            await initializeFork(snapshot, child, source);
            const files = await readdir(join(child, "pi-session"));
            const manager = SessionManager.open(join(child, "pi-session", files[0]!));
            assert.equal(files.length, 1);
            assert.notEqual(manager.getSessionId(), profile);
            assert.equal(manager.getHeader()?.parentSession, join(snapshot, "session.jsonl"));
            assert.equal(manager.getLeafId(), `leaf-${profile}`);
            assert.equal(await readFile(join(child, "workspace", "research.md"), "utf8"), profile);
            await writeFile(join(child, "workspace", "research.md"), "child-only");
            await initializeFork(snapshot, child, source);
            assert.equal(await readFile(join(child, "workspace", "research.md"), "utf8"), "child-only");
            assert.equal(await readFile(join(snapshot, "workspace", "research.md"), "utf8"), profile);
            assert.deepEqual(await readdir(join(child, "pi-session")), files);
            await assert.rejects(initializeFork(snapshot, child, { ...source, historyCursor: cursor + 1 }), /identity changed/);
            await writeFile(join(snapshot, "workspace", "research.md"), "corruption");
            await assert.rejects(initializeFork(snapshot, child, source), /digest mismatch/);
        }
    } finally { await rm(root, { recursive: true, force: true }); }
});
