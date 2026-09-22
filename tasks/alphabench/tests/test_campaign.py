from dataclasses import replace
import json

import pytest

from tasks.alphabench.core.protocol import T3Protocol
from tasks.alphabench.ldm_task.procedure import main


@pytest.mark.parametrize("method", ["llm", "ldm"])
def test_initialized_campaign_resume_has_no_new_calls_or_cost(method, tmp_path, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    monkeypatch.setenv("LDM_DATA_COLLECTION_DIR", str(tmp_path / "collection"))
    run = tmp_path / method
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(replace(T3Protocol(), method=method, cold_seed_count=3).to_dict()))
    arguments = ["--mock", "--protocol-file", str(protocol), "--out-dir", str(run)]
    assert main(arguments) == 0
    checkpoint = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(checkpoint["state"]["observations"]) == 7
    budget = (run / "budget.json").read_bytes()
    model = {p.name: p.read_bytes() for p in (run / "private/model").glob("*.json")}
    selection = (run / "selection_frozen.json").read_bytes()
    published = {}
    for pointer in (tmp_path / "collection").rglob("current.json"):
        current = json.loads(pointer.read_text(encoding="utf-8"))
        generation = pointer.parent / current["generation"]
        assert len((generation / "ldm_ir.jsonl").read_text(encoding="utf-8").splitlines()) == current["count"]
        assert len((generation / "ldm_sft.jsonl").read_text(encoding="utf-8").splitlines()) == current["count"]
        published[pointer] = pointer.read_bytes()
    assert len(published) == 2  # Separate initialization and search collections.
    assert main(arguments + ["--resume-run", str(run)]) == 0
    assert (run / "budget.json").read_bytes() == budget
    assert {p.name: p.read_bytes() for p in (run / "private/model").glob("*.json")} == model
    assert (run / "selection_frozen.json").read_bytes() == selection
    assert all(path.read_bytes() == content for path, content in published.items())
    assert json.loads((run / "result.json").read_text(encoding="utf-8"))["search"]["attempts"] == 4
    assert json.loads((run / "result.json").read_text(encoding="utf-8"))["initialization"]["seed_count"] == 3
    for filename in ("events.jsonl", "checkpoint.json", "summary.json"):
        assert (run / filename).exists()


def test_real_run_requires_data_and_frozen_protocol(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        main(["--out-dir", str(tmp_path)])
