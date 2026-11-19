"""Reproducible risk-decile selection for Q1."""
import json

import pytest

from tasks.copper_inversion.experiments.rsi import q1


def test_selection_strata_repeat_determinism_and_overwrite(tmp_path, monkeypatch):
    runs = [tmp_path / "a", tmp_path / "b"]
    for run in runs:
        run.mkdir()
        (run / "ledger.jsonl").write_text(run.name)
    rows = [{"program_id": f"{i:064x}", "risk": i / 100, "status": "valid"} for i in range(100)]
    monkeypatch.setattr(q1, "read", lambda path: rows[:50] if path.parent.name == "a" else rows[50:])
    a, b = tmp_path / "selection.json", tmp_path / "same.json"
    sha = q1.select(runs, a, 42)
    assert sha == q1.select(reversed(runs), b, 42)
    selection = json.loads(a.read_text())
    assert len(selection["programs"]) == 60
    assert len(selection["repeat_program_ids"]) == 12
    assert set(selection["repeat_program_ids"]) <= {r["program_id"] for r in selection["programs"]}
    for decile in range(1, 11):
        group = [r for r in selection["programs"] if r["decile"] == decile]
        assert len(group) == 6
        assert all((decile - 1) * 10 <= int(r["program_id"], 16) < decile * 10 for r in group)
    with pytest.raises(FileExistsError):
        q1.select(runs, a, 42)
    assert q1.select(runs, tmp_path / "other.json", 43) != sha


def test_selection_requires_sixty_distinct_evaluated_programs(tmp_path, monkeypatch):
    (tmp_path / "ledger.jsonl").write_text("ledger")
    monkeypatch.setattr(q1, "read", lambda _: [{"program_id": "a" * 64, "risk": .2, "status": "valid"}])
    with pytest.raises(ValueError, match="60 distinct"):
        q1.select([tmp_path], tmp_path / "selection.json", 0)
