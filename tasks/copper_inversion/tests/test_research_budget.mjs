import assert from 'node:assert/strict';
import test from 'node:test';
import register from '../resources/harness/tools/copper.mjs';
let guard;
register({on(name, handler) { assert.equal(name, 'tool_call'); guard = handler; }, registerTool() {}});
test('research bash requires a bounded explicit timeout', () => {
  for (const timeout of [undefined, null, 0, -1, 61, Infinity, '60']) {
    assert.equal(guard({toolName:'bash', input:{timeout}}).block, true);
  }
});
test('bounded research commands are allowed', () => {
  for (const timeout of [1, 30, 60]) assert.equal(guard({toolName:'bash', input:{timeout}}), undefined);
});
test('read/write and submission are unaffected', () => {
  for (const toolName of ['read', 'write', 'submit_solver']) assert.equal(guard({toolName,input:{}}), undefined);
});
