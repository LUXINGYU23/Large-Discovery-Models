"""Controller semantics with synthetic collaborators; no scientific claims."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.encoding import encoding_spec
from tasks.copper_inversion.rsi.controller import Config, Controller, make_message
from tasks.copper_inversion.rsi.audit import scan_blind, verify_briefs
from tasks.copper_inversion.rsi.exposure import risk_by_case, validate_feedback
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.rsi.render import render, stripped_source, write_sources
from tasks.copper_inversion.rsi.scripted import ScriptedAgent, ScriptedEvaluator
from tasks.copper_inversion.rsi.selection import reserve_and_rank, select
from tasks.copper_inversion.rsi.service import RiskOnlyService
from ldm_tts.optimization.records import SurrogateVector


def scripted_select(pool, history, count):
    predictions = [{"program_id": p["program_id"], "branch": p["branch"],
                    "mean": .5, "std": .1,
                    "acquisition": int(p["program_id"][:8], 16) / 0xffffffff} for p in pool]
    return {"predictions": predictions, "selected": reserve_and_rank(predictions, count)}


def make_controller(root, arm="ldm4geo", reset=2, evaluator=None, agent=None):
    return Controller(root, Config(arm, 7, rounds=3, proposals_per_branch=2, batch=4,
                                  reset_period=reset, token_cap_input=100),
                      agent or ScriptedAgent(), evaluator or ScriptedEvaluator(), scripted_select,
                      "Fixed task brief.", {"synthetic_test": True})


def test_projection_keeps_parent_macro_and_rejects_extra_fields():
    rows = [{"parent_id": "private-a", "case_id": "public-a", "status": "succeeded",
             "metrics": {"confirmation_loss_total": 3., "iou": .999}},
            {"parent_id": "private-a", "case_id": "public-a", "status": "failed", "metrics": {}},
            {"parent_id": "private-b", "case_id": "public-b", "status": "succeeded",
             "metrics": {"confirmation_loss_total": 0., "iou": .123}}]
    assert risk_by_case(rows, ["public-a", "public-b"]) == {"public-a": .875, "public-b": 0.}
    value = {"source_sha256": "a" * 64, "status": "valid", "risk": .4375,
             "risk_by_case": {"public-a": .875, "public-b": 0.}, "legal_rate": 2/3,
             "failed_solves": 1, "seconds_per_solve": 1., "message": ""}
    assert validate_feedback(value) == value
    with pytest.raises(ValueError, match="unexpected"):
        validate_feedback({**value, "mean_iou": .999})


@pytest.mark.parametrize("outcome", ["succeeded", "invalid_interface", "all_failed"])
def test_server_only_returns_allowlisted_feedback(tmp_path, outcome):
    service = RiskOnlyService.__new__(RiskOnlyService)
    service.root = tmp_path
    service.identity = {"test": True}
    service.public_contract = "c" * 64
    from tasks.copper_inversion.rsi.integrity import shared_identity
    service.shared_implementation = shared_identity()
    service.audit = {"sentinel": "SECRET-SENTINEL", "private_parent_ids": ["private-parent"], "case_ids": ["case"]}
    service.evaluator = SimpleNamespace(evaluation_attempt_usage_key=lambda _: "key",
                                        manifest={"cases": [{"case_id": "case", "parent_id": "private-parent"}],
                                                  "seeds": [47, 48]})
    sha = "d" * 64
    failed = outcome in {"invalid_interface", "all_failed"}
    metrics = {"mean_iou": .987, "sentinel": "SECRET-SENTINEL",
               "loss_total": 1. if failed else .5, "legal_rate": 0. if failed else 1.,
               "failed_solves": 2.0 if failed else 0.0, "mean_solver_wall_seconds": 0. if failed else 3.}
    status = "invalid_interface" if outcome == "invalid_interface" else "succeeded"
    result = {"status": status, "source_sha256": sha, "metrics": metrics}
    if status == "succeeded":
        path = tmp_path / "evaluations/key/private-case-results.json"
        path.parent.mkdir(parents=True)
        rows = [{"parent_id": "private-parent", "case_id": "case", "seed": seed,
                 "status": "failed" if failed else "succeeded",
                 "metrics": {} if failed else {"confirmation_loss_total": 1., "iou": .987}}
                for seed in (47, 48)]
        path.write_text(json.dumps(rows))
    with patch("tasks.copper_inversion.v2.service.EvaluatorService.dispatch", return_value=result):
        reply = service.dispatch({"operation": "evaluate", "source": "source", "contract_digest": "c" * 64})
    validate_feedback(reply)
    assert set(reply["risk_by_case"]) == {"case"}
    assert reply["status"] == ("rejected" if outcome == "invalid_interface" else "failed" if failed else "valid")
    assert "SECRET-SENTINEL" not in json.dumps(reply)
    assert "private-parent" not in json.dumps(reply)
    assert "iou" not in json.dumps(reply).lower()
    assert "SECRET-SENTINEL" in (tmp_path / "rsi-feedback" / (sha + ".json")).read_text()


def test_renderer_removes_comments_and_docstrings_only():
    source = ('"""module note"""\n# comment\nclass C:\n    """class note"""\n'
              '    def f(self):\n        """function note"""\n        return "data"\n')
    shown = stripped_source(source)
    assert "note" not in shown and "comment" not in shown
    assert "data" in shown
    namespace = {}
    exec(compile(shown, "display.py", "exec"), namespace)
    assert namespace["C"]().f() == "data"


@pytest.mark.parametrize("reset", [1, 2])
def test_three_epoch_visibility_and_deterministic_replay(tmp_path, reset):
    root = tmp_path / "run"
    controller = make_controller(root, reset=reset)
    result = controller.run()
    assert result["charged_records"] == 13
    ledger = read(root / "ledger.jsonl")
    assert [r["seq"] for r in ledger] == list(range(13))
    prefix = [r for r in ledger if r["round"] < 3]
    _, view = make_message(controller.config, root, prefix, 0, 3, controller.task_brief, [])
    start = ((3 - 1) // reset) * reset + 1
    assert all(r["round"] < start or r["branch"] == 0 for r in view)
    rendered = render(view)
    assert "private_note" not in rendered and "private_rationale" not in rendered
    source_dir = tmp_path / "view"
    write_sources(source_dir, view, root)
    assert not any("private_note" in p.read_text() for p in source_dir.glob("*.py"))
    snapshot = (root / "ledger.jsonl").read_bytes()
    assert make_controller(root, reset=reset).run(resume=True) == result
    assert (root / "ledger.jsonl").read_bytes() == snapshot
    assert verify_briefs(root)["briefs_checked"] == 12


def test_blind_scanner_detects_positive_and_negative_controls(tmp_path):
    root = tmp_path / "run"
    make_controller(root).run()
    request = root / "harness/epoch_00/branch_0/turn.request.bin"
    request.write_bytes(b"allowed public task and risk")
    private = tmp_path / "private"
    private.mkdir()
    (private / "rsi-private-audit.json").write_text(json.dumps({
        "sentinel": "secret-canary-123", "private_parent_ids": ["private-body-123"]}))
    assert scan_blind(root, private)["provider_requests_checked"] == 1
    request.write_bytes(b"request contains secret-canary-123")
    with pytest.raises(ValueError, match="contaminated"):
        scan_blind(root, private)


def test_matching_history_yields_identical_reset_boundary_messages(tmp_path):
    source = tmp_path / "run"
    ctrl = make_controller(source)
    ctrl.run()
    rows = [r for r in read(source / "ledger.jsonl") if r["round"] < 3]
    a = make_message(ctrl.config, source, rows, 0, 3, ctrl.task_brief, [])[0]
    b = make_message(Config("accumulated", 7, rounds=3, proposals_per_branch=2,
                            batch=4, reset_period=2, token_cap_input=100),
                     source, rows, 0, 3, ctrl.task_brief, [])[0]
    assert a == b


def test_planted_note_is_only_in_the_target_round_messages(tmp_path):
    root = tmp_path / "run"
    note = {"round": 2, "target_tag": "fixture_family", "note": "PLANTED-CLAIM-ONLY"}
    ctrl = Controller(root, Config("ldm4geo", 7, rounds=3, proposals_per_branch=1,
                                  batch=4, reset_period=2), ScriptedAgent(), ScriptedEvaluator(),
                      scripted_select, "Fixed task brief.", {"synthetic_test": True}, intervention=note)
    ctrl.run()
    messages = sorted((root / "harness").glob("epoch_*/branch_*/briefs/round_*.json"))
    assert sum("PLANTED-CLAIM-ONLY" in p.read_text() for p in messages) == 4
    assert all(("PLANTED-CLAIM-ONLY" in p.read_text()) == (p.stem == "round_02") for p in messages)
    assert "PLANTED-CLAIM-ONLY" not in (root / "ledger.jsonl").read_text()
    assert verify_briefs(root)["briefs_checked"] == 12


def test_interrupted_evaluation_resumes_without_recharging(tmp_path):
    baseline = tmp_path / "baseline"
    make_controller(baseline).run()
    class Interrupted(ScriptedEvaluator):
        def __init__(self): self.count = 0
        def evaluate_source(self, path):
            self.count += 1
            if self.count == 4:
                raise RuntimeError("synthetic interruption")
            return super().evaluate_source(path)
    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        make_controller(interrupted, evaluator=Interrupted()).run()
    make_controller(interrupted).run(resume=True)
    assert (baseline / "ledger.jsonl").read_bytes() == (interrupted / "ledger.jsonl").read_bytes()


def test_reserved_selection_then_highest_remaining():
    rng = np.random.default_rng(5)
    for _ in range(40):
        values = [{"program_id": f"{branch * 10 + i:064x}", "branch": branch,
                   "acquisition": float(rng.normal())} for branch in range(4) for i in range(5)]
        chosen = reserve_and_rank(values, 12)
        assert len(chosen) == 12
        assert {p["branch"] for p in chosen if p["selection"] == "reserved"} == set(range(4))
        reserved = {p["program_id"] for p in chosen if p["selection"] == "reserved"}
        rest = sorted((p for p in values if p["program_id"] not in reserved),
                      key=lambda p: (-p["acquisition"], p["program_id"]))[:8]
        assert [p["program_id"] for p in chosen[4:]] == [p["program_id"] for p in rest]


def test_existing_fusion_gp_receives_only_code_features():
    version = encoding_spec("fusion").version
    class Encoder:
        def encode(self, candidate):
            vector = np.zeros(1280)
            vector[int(candidate.candidate_id[:4], 16) % len(vector)] = 1.
            return SurrogateVector(tuple(vector), version, source_id=candidate.candidate_id)
    history = [{"program_id": "a" * 64, "risk": .4}]
    pool = [{"program_id": f"{i+1:064x}", "branch": i} for i in range(4)]
    chosen = select(pool, history, Encoder(), 4)
    assert len(chosen["selected"]) == 4
    assert {p["branch"] for p in chosen["selected"]} == set(range(4))
