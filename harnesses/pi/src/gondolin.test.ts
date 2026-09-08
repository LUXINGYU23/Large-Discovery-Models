import assert from "node:assert/strict";
import test from "node:test";
import { guestPath } from "./gondolin.js";

test("guest paths cannot escape the mounted workspace", () => {
	assert.equal(guestPath("/host/profile", "notes/result.json"), "/workspace/notes/result.json");
	assert.throws(() => guestPath("/host/profile", "../../etc/passwd"));
	assert.throws(() => guestPath("/host/profile", "/workspace/../etc/passwd"));
});
