import json
import os
import shutil
import subprocess

import pytest

from tasks.atomworld.core.harness import RESOURCE_ROOT


def test_public_history_is_compact_and_fetches_cif_and_rationale_by_id(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the guest-tool protocol test")
    rows = [{"draft_id": f"draft-{i}", "round_idx": i, "generated_output": "data_large_cif",
             "rationale": "public hypothesis " * 40, "output_sha256": str(i),
             "public_validation": {"parseable": True}} for i in range(2)]
    (tmp_path / "public_history.json").write_text(json.dumps({"drafts": rows}))
    script = r'''
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const tools = new Map();
const { default: register } = await import(pathToFileURL(process.argv[1]));
register({registerTool(tool) { tools.set(tool.name, tool); }});
const call = async params => (await tools.get("get_public_history").execute("fixture", params)).details;
const index = await call({});
assert.equal(index.drafts.length, 2);
assert.equal(index.drafts[0].generated_output, undefined);
assert.equal(index.drafts[0].rationale, undefined);
assert.equal(index.drafts[0].rationale_preview.length, 160);
const detail = await call({draft_ids: ["draft-1"], detail: "detailed"});
assert.equal(detail.total, 1);
assert.equal(detail.drafts[0].generated_output, "data_large_cif");
assert.equal(detail.drafts[0].rationale, "public hypothesis ".repeat(40));
await assert.rejects(() => call({detail: "unbounded"}));
'''
    result = subprocess.run([node, "--input-type=module", "-e", script,
        str(RESOURCE_ROOT / "tools/public_context.mjs")],
        env={**os.environ, "LDM_ATOMWORLD_PUBLIC_ROOT": str(tmp_path)},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
