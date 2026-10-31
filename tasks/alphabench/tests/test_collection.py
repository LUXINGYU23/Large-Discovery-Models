import json
import subprocess
import sys

import pytest

from ldm_tts.data.ir import make_complete_design_ir
from tasks.alphabench.core.collection import AcceptedActions


def test_action_export_rejects_corrupt_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(tmp_path / "collection"))
    actions = AcceptedActions(tmp_path / "run")
    ir = make_complete_design_ir(task_id="alphabench", domain="financial factor expressions",
        task_description="Generate a causal factor.", objectives=[{"name": "ic", "direction": "maximize"}],
        design_space_description="Qlib expression", observations=[],
        candidates=[{"expression": "Mean($close,5)"}], request_description="One factor")

    actions.accept("factor-1", ir, {"source": "model"})
    published = actions.export()
    pointer = published.parent / "current.json"
    before = pointer.read_bytes()
    actions.accept("factor-1", ir, {"source": "model"})
    assert pointer.read_bytes() == before
    assert len((published / "ldm_ir.jsonl").read_text().splitlines()) == 1
    assert len((published / "ldm_sft.jsonl").read_text().splitlines()) == 1
    assert json.loads((published / "manifest.json").read_text())["action_ids"] == ["factor-1"]

    (published / "ldm_sft.jsonl").write_text("corrupt\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        actions.export()
    assert pointer.read_bytes() == before


@pytest.mark.parametrize("stage,exit_code,has_generation", [
    ("after_ir", 101, False),
    ("before_pointer", 102, True),
])
def test_export_recovers_pair_after_process_death(tmp_path, stage, exit_code, has_generation):
    script = r'''
import os, sys
from pathlib import Path
from ldm_tts.data.ir import make_complete_design_ir
from tasks.alphabench.core.collection import AcceptedActions

run, stage = Path(sys.argv[1]), sys.argv[2]
os.environ["LDM_DATA_COLLECTION_DIR"] = str(run / "collection")
marker = run / "crashed.marker"
actions = AcceptedActions(run)
ir = make_complete_design_ir(task_id="alphabench", domain="financial factor expressions",
    task_description="Generate a factor.", objectives=[{"name": "ic", "direction": "maximize"}],
    design_space_description="Qlib expression", observations=[],
    candidates=[{"expression": "Mean($close,5)"}], request_description="One factor")
if stage == "after_ir":
    from ldm_tts.data import collection as writer
    original = writer.append_jsonl
    def append(path, row):
        if path.name == "ldm_sft.jsonl" and not marker.exists():
            marker.write_text(stage)
            os._exit(101)
        return original(path, row)
    writer.append_jsonl = append
elif stage == "before_pointer":
    from tasks.alphabench.core import collection as module
    original = module.atomic_json_write
    def write(path, value):
        if path.name == "current.json" and not marker.exists():
            marker.write_text(stage)
            os._exit(102)
        return original(path, value)
    module.atomic_json_write = write
actions.accept("factor-1", ir, {"source": "model"})
actions.export()
'''
    run = tmp_path / "run"
    command = [sys.executable, "-c", script, str(run), stage]
    killed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert killed.returncode == exit_code, killed.stderr
    root = next((run / "collection/alphabench").iterdir())
    assert (run / "crashed.marker").read_text() == stage
    assert not (root / "current.json").exists()
    assert bool(list(root.glob("*/manifest.json"))) is has_generation
    journal = next((run / "accepted_actions").glob("*.json"))
    accepted = journal.read_bytes()
    for _ in range(2):
        resumed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert resumed.returncode == 0, resumed.stderr
    pointer = json.loads((root / "current.json").read_text())
    generation = root / pointer["generation"]
    assert pointer["count"] == 1
    assert journal.read_bytes() == accepted
    assert len(list((run / "accepted_actions").glob("*.json"))) == 1
    assert len(list(root.glob("*/manifest.json"))) == 1
    assert len((generation / "ldm_ir.jsonl").read_text().splitlines()) == 1
    assert len((generation / "ldm_sft.jsonl").read_text().splitlines()) == 1
    assert json.loads((generation / "manifest.json").read_text())["action_ids"] == ["factor-1"]
    baseline = subprocess.run([sys.executable, "-c", script, str(tmp_path / "baseline"), "none"],
                              capture_output=True, text=True, timeout=30)
    assert baseline.returncode == 0, baseline.stderr
    baseline_root = next((tmp_path / "baseline/collection/alphabench").iterdir())
    assert json.loads((baseline_root / "current.json").read_text()) == pointer
    for name in ("ldm_ir.jsonl", "ldm_sft.jsonl", "manifest.json"):
        assert (baseline_root / pointer["generation"] / name).read_bytes() == (generation / name).read_bytes()
