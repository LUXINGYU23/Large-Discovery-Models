import assert from "node:assert/strict";
import test from "node:test";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { directorySha256, initializeFork } from "./session.js";
import { sha256 } from "./trace.js";

test("native forks preserve ancestry and files, isolate siblings and resume without resetting", async () => {
    const root = await mkdtemp(join(tmpdir(), "ldm-native-fork-"));
    try {
        const source = join(root, "fork-source");
        await mkdir(join(source, "workspace"), { recursive: true });
        await writeFile(join(source, "workspace", "research.md"), "shared evidence");
        const header = { type: "session", version: 3, id: "parent", timestamp: new Date().toISOString(), cwd: "/main" };
        const entry = { type: "message", id: "known-leaf", parentId: null, timestamp: new Date().toISOString(),
            message: { role: "user", content: "previous research evidence", timestamp: Date.now() } };
        await writeFile(join(source, "session.jsonl"), [header, entry].map(x => JSON.stringify(x)).join("\n") + "\n");
        const identity = { sessionSha256: sha256(await readFile(join(source, "session.jsonl"))),
            workspaceSha256: await directorySha256(join(source, "workspace")), historyCursor: 17 };
        const left = join(root, "sessions", "left");
        const right = join(root, "sessions", "right");
        await initializeFork(root, left, identity);
        await initializeFork(root, right, identity);
        const files = await readdir(join(left, "pi-session"));
        assert.equal(files.length, 1);
        const manager = SessionManager.open(join(left, "pi-session", files[0]!));
        assert.equal(manager.getLeafId(), "known-leaf");
        assert.notEqual(manager.getSessionId(), "parent");
        assert.equal(manager.getHeader()?.parentSession, join(source, "session.jsonl"));
        await writeFile(join(left, "workspace", "research.md"), "left-only");
        await initializeFork(root, left, identity);
        assert.deepEqual(await readdir(join(left, "pi-session")), files);
        assert.equal(await readFile(join(left, "workspace", "research.md"), "utf8"), "left-only");
        assert.equal(await readFile(join(right, "workspace", "research.md"), "utf8"), "shared evidence");
        await writeFile(join(source, "workspace", "research.md"), "corruption");
        await assert.rejects(initializeFork(root, left, identity), /digest mismatch/);
    } finally { await rm(root, { recursive: true, force: true }); }
});
