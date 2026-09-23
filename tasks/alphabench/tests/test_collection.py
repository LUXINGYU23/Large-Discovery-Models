import json

import pytest

from ldm_tts.data.ir import make_complete_design_ir
from tasks.alphabench.core.collection import AcceptedActions


def test_action_export_recovers_partial_pair_and_rejects_corrupt_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(tmp_path / "collection"))
    actions = AcceptedActions(tmp_path / "run")
    ir = make_complete_design_ir(task_id="alphabench", domain="financial factor expressions",
        task_description="Generate a causal factor.", objectives=[{"name": "ic", "direction": "maximize"}],
        design_space_description="Qlib expression", observations=[],
        candidates=[{"expression": "Mean($close,5)"}], request_description="One factor")

    from ldm_tts.data import collection as writer
    original = writer.append_jsonl

    def interrupt(path, row):
        if str(path).endswith("ldm_sft.jsonl"):
            raise RuntimeError("interrupted after IR write")
        return original(path, row)

    monkeypatch.setattr(writer, "append_jsonl", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        actions.accept("factor-1", ir, {"source": "model"})
    assert not list((tmp_path / "collection").rglob("current.json"))
    monkeypatch.setattr(writer, "append_jsonl", original)

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


def test_export_recovers_published_generation_before_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(tmp_path / "collection"))
    actions = AcceptedActions(tmp_path / "run")
    ir = make_complete_design_ir(task_id="alphabench", domain="financial factor expressions",
        task_description="Generate a factor.", objectives=[{"name": "ic", "direction": "maximize"}],
        design_space_description="Qlib expression", observations=[],
        candidates=[{"expression": "Mean($close,5)"}], request_description="One factor")
    from tasks.alphabench.core import collection as module
    original = module.atomic_json_write

    def interrupt(path, value):
        if path.name == "current.json":
            raise RuntimeError("interrupted before pointer")
        return original(path, value)

    monkeypatch.setattr(module, "atomic_json_write", interrupt)
    with pytest.raises(RuntimeError, match="before pointer"):
        actions.accept("factor-1", ir, {})
    assert not list((tmp_path / "collection").rglob("current.json"))
    assert len(list((tmp_path / "collection").rglob("manifest.json"))) == 1
    monkeypatch.setattr(module, "atomic_json_write", original)
    published = actions.export()
    assert json.loads((published.parent / "current.json").read_text())["count"] == 1
    assert len((published / "ldm_ir.jsonl").read_text().splitlines()) == 1
    assert len((published / "ldm_sft.jsonl").read_text().splitlines()) == 1
