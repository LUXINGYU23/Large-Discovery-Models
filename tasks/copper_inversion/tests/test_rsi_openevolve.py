"""Unit tests for the OpenEvolve external-arm adapter (mock service, zero API calls)."""

import json
import types
from pathlib import Path

import pytest

from ldm_tts.engine.run_store import BudgetExceededError
from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES
from tasks.copper_inversion.experiments.rsi import freeze, openevolve_arm
from tasks.copper_inversion.rsi import ledger as ledgers
from tasks.copper_inversion.rsi.controller import INITIAL_SOURCE


class FakeRiskEvaluator:
    """In-memory mock of RiskClient enforcing slot cap and aggregate-only responses."""

    def __init__(self, *, slots=5, scores=None, inject_case_risks=False):
        self.slots = slots
        self.scores = dict(scores or {})
        self.inject_case_risks = inject_case_risks
        self.calls = []
        self.seen = {}

    def evaluate_source(self, source_path):
        body = Path(source_path).read_bytes()
        sha = validate_source(body)
        if sha in self.seen:
            return self.seen[sha]
        if len(self.seen) >= self.slots:
            raise BudgetExceededError("risk-only feedback budget exhausted")
        self.calls.append((sha, body.decode("utf-8")))
        spec = self.scores.get(sha, {"status": "valid", "risk": 0.42, "legal_rate": 1.0,
                                     "failed_solves": 0, "seconds_per_solve": 0.5, "message": ""})
        feedback = {
            "source_sha256": sha,
            "status": spec["status"],
            "risk": spec["risk"],
            "risk_by_case": {"case_a": spec["risk"]} if self.inject_case_risks else {},
            "legal_rate": spec["legal_rate"],
            "failed_solves": spec["failed_solves"],
            "seconds_per_solve": spec["seconds_per_solve"],
            "message": spec["message"],
        }
        self.seen[sha] = feedback
        return feedback


def test_prepare_submission_strips_comments_and_enforces_32kb():
    src = b"# secret comment\ndef solve(problem, budget, seed):\n    '''docstring'''\n    return problem\n"
    stripped, sha = openevolve_arm.prepare_submission(src)
    assert b"secret comment" not in stripped
    assert b"docstring" not in stripped
    assert sha == validate_source(stripped)

    # Identical code with different comments hashes identically.
    src2 = b"# another comment\ndef solve(problem, budget, seed):\n    return problem\n"
    stripped2, sha2 = openevolve_arm.prepare_submission(src2)
    assert sha == sha2 and stripped == stripped2

    # Missing solve() or syntax error refused.
    with pytest.raises(ValueError, match="solve"):
        openevolve_arm.prepare_submission(b"x = 1\n")
    with pytest.raises(SyntaxError):
        openevolve_arm.prepare_submission(b"def solve(:\n")

    # Exceeding 32,000 normalized bytes refused.
    big_body = "def solve(problem, budget, seed):\n" + "".join(
        f"    x_{i} = {i}\n" for i in range(4000)
    ) + "    return x_0\n"
    assert len(big_body.encode()) > MAX_NORMALIZED_BYTES
    with pytest.raises(ValueError, match="32000"):
        openevolve_arm.prepare_submission(big_body.encode("utf-8"))


def test_adapter_counts_slots_including_rejections_and_returns_aggregate_only(tmp_path):
    run_dir = tmp_path / "openevolve-s101"
    p_init = tmp_path / "init.py"
    p_init.write_bytes(INITIAL_SOURCE)

    good_src = b"# comment removed\ndef solve(problem, budget, seed):\n    return problem['domain_mask'].astype(float)\n"
    good_stripped, good_sha = openevolve_arm.prepare_submission(good_src)
    p_good = tmp_path / "good.py"
    p_good.write_bytes(good_src)

    rej_src = b"def solve(problem, budget, seed):\n    return 0\n"
    _, rej_sha = openevolve_arm.prepare_submission(rej_src)
    p_rej = tmp_path / "rej.py"
    p_rej.write_bytes(rej_src)

    p_bad_syntax = tmp_path / "syntax.py"
    p_bad_syntax.write_bytes(b"def solve(:\n")

    p_extra = tmp_path / "extra.py"
    p_extra.write_bytes(b"def solve(problem, budget, seed):\n    return 1\n")

    fake = FakeRiskEvaluator(slots=4, scores={
        good_sha: {"status": "valid", "risk": 0.25, "legal_rate": 1.0,
                   "failed_solves": 0, "seconds_per_solve": 0.4, "message": ""},
        rej_sha: {"status": "rejected", "risk": 1.0, "legal_rate": 0.0,
                  "failed_solves": 12, "seconds_per_solve": 0.1, "message": "interface rejected"},
    })
    adapter = openevolve_arm.OpenEvolveAdapter(run_dir, fake, slots=4)

    # Slot 1: initial seed (round 0, branch -1, selection initial).
    m0 = adapter.evaluate_program(p_init)
    assert set(m0) == {"combined_score", "risk"}
    assert m0["risk"] == pytest.approx(0.42)
    assert m0["combined_score"] == pytest.approx(0.58)

    # Slot 2: valid proposal with comments stripped before service submission.
    m1 = adapter.evaluate_program(p_good)
    assert set(m1) == {"combined_score", "risk"}
    assert m1["risk"] == pytest.approx(0.25)
    assert m1["combined_score"] == pytest.approx(0.75)
    assert "# comment removed" not in fake.calls[-1][1]

    # Duplicate submission replays cached aggregate score and does NOT consume a slot.
    p_dup = tmp_path / "dup.py"
    p_dup.write_bytes(b"# different comment\n" + good_stripped)
    m_dup = adapter.evaluate_program(p_dup)
    assert m_dup == m1 and len(fake.calls) == 2

    # Slot 3: service interface rejection IS counted against slots and recorded in ledger.
    m2 = adapter.evaluate_program(p_rej)
    assert m2 == {"combined_score": 0.0, "risk": 1.0}

    # Slot 4: syntax-invalid rejection IS counted locally against the 4-slot budget without calling service.
    m3 = adapter.evaluate_program(p_bad_syntax)
    assert m3 == {"combined_score": 0.0, "risk": 1.0}
    assert len(fake.calls) == 3

    # Slot 5 (> budget 4): refused without calling service.
    m4 = adapter.evaluate_program(p_extra)
    assert m4 == {"combined_score": 0.0, "risk": 1.0}
    assert len(fake.calls) == 3

    # Verify ledger validity and monotonic round order compatible with freeze.py.
    rows = ledgers.read(run_dir / "ledger.jsonl")
    assert len(rows) == 3
    assert rows[0]["round"] == 0 and rows[0]["branch"] == -1 and rows[0]["selection"] == "initial"
    assert [r["status"] for r in rows] == ["valid", "valid", "rejected"]
    assert all(r["risk_by_case"] == {} for r in rows)
    assert all(a["round"] <= b["round"] for a, b in zip(rows, rows[1:]))

    subs = openevolve_arm._read_submissions(run_dir / "openevolve-submissions.jsonl")
    assert sum(1 for s in subs if s["counted"]) == 4
    assert subs[-1]["status"] == "budget_exhausted"


def test_adapter_refuses_per_case_feedback(tmp_path):
    p_init = tmp_path / "init.py"
    p_init.write_bytes(INITIAL_SOURCE)
    fake = FakeRiskEvaluator(slots=2, inject_case_risks=True)
    adapter = openevolve_arm.OpenEvolveAdapter(tmp_path / "run", fake, slots=2)
    with pytest.raises(ValueError, match="per-case"):
        adapter.evaluate_program(p_init)

    with pytest.raises(ValueError, match="aggregate-only"):
        openevolve_arm.OpenEvolveAdapter(tmp_path / "run2", fake, slots=2, case_feedback="opaque")


def test_system_prompt_and_token_usage_summary(tmp_path):
    prompt = openevolve_arm.build_system_prompt()
    assert "solve(problem, budget, seed)" in prompt
    assert "32,000 bytes" in prompt
    assert "combined_score" in prompt
    assert "selfcheck.py" not in prompt

    trace_dir = tmp_path / "oe"
    trace_dir.mkdir()
    (trace_dir / "evolution_trace.jsonl").write_text(
        json.dumps({"iteration": 1, "metadata": {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}}) + "\n"
        + json.dumps({"iteration": 2, "metadata": {"token_usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}}}) + "\n"
    )
    (trace_dir / "evolution_trace_failures.jsonl").write_text(
        json.dumps({"iteration": 3, "error": "diff failed", "token_usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160}}) + "\n"
    )
    summary = openevolve_arm.summarize_token_usage(trace_dir)
    assert summary == {
        "calls": 3,
        "succeeded_iterations": 2,
        "failed_iterations": 1,
        "prompt_tokens": 420,
        "completion_tokens": 170,
        "total_tokens": 590,
        "missing_usage_calls": 0,
    }


def test_ledger_compatible_with_freeze_finalists_and_validation(tmp_path):
    """A run produced by OpenEvolveAdapter passes freeze.finalists, freeze_run, and validate_run with 5 finalists."""
    from unittest.mock import patch
    from tasks.copper_inversion.tests.test_rsi_service import (
        InProcessExecution, LoopbackClient, make_service, write_suite,
    )

    release = tmp_path / "release"
    train_manifest = write_suite(release, "train", split="train", bodies=2, offset=0)
    write_suite(release, "validation", split="validation", bodies=2, offset=10)
    write_suite(release, "test_id", split="test_id", bodies=2, offset=20)
    write_suite(release, "test_ood", split="test_ood", bodies=2, offset=30)

    proto = json.loads((Path(__file__).resolve().parents[1] / "resources/rsi/protocol.r2.example.json").read_text())
    proto["tiers"]["S"]["train_manifest"] = "train/manifest.json"
    pid = "rsi-r2-testfreeze"
    run_id = "openevolve-s101"
    service_dir = tmp_path / "private" / pid / "services" / "M0" / run_id
    iso = proto["case_isolation"]
    service = make_service(tmp_path, train_manifest, name=str(service_dir.relative_to(tmp_path)),
                           finalists=5, case_isolation=iso)

    run_dir = tmp_path / "runs" / pid / "M0" / run_id
    adapter = openevolve_arm.OpenEvolveAdapter(run_dir, LoopbackClient(run_dir, service), slots=10)
    # Submit A0 + 5 distinct valid variants.
    for i in range(6):
        p = tmp_path / f"cand_{i}.py"
        if i == 0:
            p.write_bytes(INITIAL_SOURCE)
        else:
            p.write_text(
                f"from tasks.copper_inversion.v2.baselines import initialize\n"
                f"def solve(problem, budget, seed):\n"
                f"    _k = {i}\n"
                f"    return initialize(problem)\n"
            )
        metrics = adapter.evaluate_program(p)
        assert set(metrics) == {"combined_score", "risk"}

    sealed = service.seal_training()
    assert sealed["used_slots"] == 6
    assert len(sealed["source_sha256s"]) == 5

    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "experiment": "M0", "protocol_id": pid,
        "arm": "openevolve", "tier": "S", "judge": "standard", "seed": 101,
    }))
    (run_dir / "result.json").write_text(json.dumps({
        "status": "completed", "charged_records": 6, "training_finalists": sealed["source_sha256s"],
    }))

    chosen = freeze.finalists(run_dir, service_dir, round_index=None, count=5)
    assert chosen["source_sha256s"] == sealed["source_sha256s"]
    assert len(chosen["source_sha256s"]) == 5
    assert chosen["records"] == 6

    class FullInProcessExecution(InProcessExecution):
        def run(self, identity, *, phase, **kwargs):
            return super().run(identity, phase="solver" if phase == "interface" else phase, **kwargs)

    with patch("tasks.copper_inversion.v2.selection.Execution", FullInProcessExecution):
        rec = freeze.freeze_run(tmp_path, pid, "M0", run_id, round_index=None, release=release, protocol=proto)
        assert rec["prefix"] == "full"
        assert len(rec["source_sha256s"]) == 5
        assert rec["source_sha256s"] == sealed["source_sha256s"]
        sel = freeze.validate_run(tmp_path, pid, "M0", run_id, round_index=None)
        assert sel["selected_sha256"] in set(sealed["source_sha256s"])
