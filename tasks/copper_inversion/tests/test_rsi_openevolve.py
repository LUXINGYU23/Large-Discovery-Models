"""Unit tests for the shared BudgetGate, OpenEvolve adapter, launcher, and REFOE freeze/unseal/export."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ldm_tts.engine.run_store import BudgetExceededError
from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES
from tasks.copper_inversion.experiments.rsi import (
    budget_gate,
    export_outcomes,
    freeze,
    openevolve_arm,
    openevolve_launch,
    unseal,
)
from tasks.copper_inversion.rsi import ledger as ledgers
from tasks.copper_inversion.rsi.controller import INITIAL_SOURCE


class FakeRiskEvaluator:
    """Thread-safe mock of RiskClient enforcing slot cap and aggregate-only responses."""

    def __init__(self, *, slots=5, scores=None, inject_case_risks=False, delay=0.0):
        self.slots = slots
        self.scores = dict(scores or {})
        self.inject_case_risks = inject_case_risks
        self.delay = delay
        self.calls = []
        self.seen = {}
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def evaluate_source(self, source_path):
        body = Path(source_path).read_bytes()
        sha = validate_source(body)
        with self._lock:
            if sha in self.seen:
                return self.seen[sha]
            if len(self.seen) >= self.slots:
                raise BudgetExceededError("risk-only feedback budget exhausted")
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            with self._lock:
                if sha in self.seen:
                    return self.seen[sha]
                if len(self.seen) >= self.slots:
                    raise BudgetExceededError("risk-only feedback budget exhausted")
                self.calls.append((sha, body.decode("utf-8")))
                spec = self.scores.get(
                    sha,
                    {"status": "valid", "risk": 0.42, "legal_rate": 1.0,
                     "failed_solves": 0, "seconds_per_solve": 0.5, "message": ""},
                )
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
        finally:
            with self._lock:
                self.active -= 1


def test_prepare_submission_strips_comments_enforces_32kb_and_case_tokens():
    src = b"# secret comment\ndef solve(problem, budget, seed):\n    '''docstring'''\n    return problem\n"
    stripped, sha = openevolve_arm.prepare_submission(src)
    assert b"secret comment" not in stripped
    assert b"docstring" not in stripped
    assert sha == validate_source(stripped)

    src2 = b"# another comment\ndef solve(problem, budget, seed):\n    return problem\n"
    stripped2, sha2 = openevolve_arm.prepare_submission(src2)
    assert sha == sha2 and stripped == stripped2

    with pytest.raises(ValueError, match="solve"):
        openevolve_arm.prepare_submission(b"x = 1\n")
    with pytest.raises(SyntaxError):
        openevolve_arm.prepare_submission(b"def solve(:\n")

    with pytest.raises(ValueError):
        budget_gate.prepare_submission(
            b"def solve(problem, budget, seed):\n    return 'case_0001'\n",
            forbidden_case_tokens=frozenset({"case_0001"}),
        )

    big_body = "def solve(problem, budget, seed):\n" + "".join(
        f"    x_{i} = {i}\n" for i in range(4000)
    ) + "    return x_0\n"
    assert len(big_body.encode()) > MAX_NORMALIZED_BYTES
    with pytest.raises(ValueError, match="32000"):
        openevolve_arm.prepare_submission(big_body.encode("utf-8"))


def test_adapter_counts_service_evaluations_not_syntax_failures_and_caps_exact_slots(tmp_path):
    run_dir = tmp_path / "openevolve-s401"
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

    p_valid_4 = tmp_path / "valid4.py"
    p_valid_4.write_bytes(b"def solve(problem, budget, seed):\n    return 2\n")

    p_extra = tmp_path / "extra.py"
    p_extra.write_bytes(b"def solve(problem, budget, seed):\n    return 3\n")

    fake = FakeRiskEvaluator(slots=4, scores={
        good_sha: {"status": "valid", "risk": 0.25, "legal_rate": 1.0,
                   "failed_solves": 0, "seconds_per_solve": 0.4, "message": ""},
        rej_sha: {"status": "rejected", "risk": 1.0, "legal_rate": 0.0,
                  "failed_solves": 12, "seconds_per_solve": 0.1, "message": "interface rejected"},
    })
    adapter = openevolve_arm.OpenEvolveAdapter(run_dir, fake, slots=4)

    # Evaluation 1: initial seed A0.
    m0 = adapter.evaluate_program(p_init)
    assert set(m0) == {"combined_score", "risk"}
    assert m0["risk"] == pytest.approx(0.42)
    assert m0["combined_score"] == pytest.approx(0.58)

    # Evaluation 2: valid proposal with comments stripped.
    m1 = adapter.evaluate_program(p_good)
    assert m1["risk"] == pytest.approx(0.25)
    assert "# comment removed" not in fake.calls[-1][1]

    # Duplicate submission replays cached aggregate score and costs 0 evaluations.
    p_dup = tmp_path / "dup.py"
    p_dup.write_bytes(b"# different comment\n" + good_stripped)
    assert adapter.evaluate_program(p_dup) == m1
    assert len(fake.calls) == 2

    # Unparseable/syntax-broken proposal does NOT consume a service evaluation (frameworks-common.md).
    m_bad = adapter.evaluate_program(p_bad_syntax)
    assert m_bad == {"combined_score": 0.0, "risk": 1.0}
    assert len(fake.calls) == 2

    # Evaluation 3: service interface rejection DOES consume 1 evaluation and enters ledger.jsonl.
    m2 = adapter.evaluate_program(p_rej)
    assert m2 == {"combined_score": 0.0, "risk": 1.0}
    assert len(fake.calls) == 3

    # Evaluation 4: fills exact 4-evaluation budget and marks gate.exhausted.
    m3 = adapter.evaluate_program(p_valid_4)
    assert m3["risk"] == pytest.approx(0.42)
    assert adapter.gate.exhausted
    assert (run_dir / "gate.exhausted").is_file()

    # Evaluation 5 (> budget 4): refused without calling service.
    m4 = adapter.evaluate_program(p_extra)
    assert m4 == {"combined_score": 0.0, "risk": 1.0}
    assert len(fake.calls) == 4

    rows = ledgers.read(run_dir / "ledger.jsonl")
    assert len(rows) == 4 == len(fake.calls)
    assert rows[0]["round"] == 0 and rows[0]["branch"] == -1 and rows[0]["selection"] == "initial"
    assert [r["status"] for r in rows] == ["valid", "valid", "rejected", "valid"]
    assert all(r["risk_by_case"] == {} for r in rows)

    subs = budget_gate.read_submissions(run_dir / "openevolve-submissions.jsonl")
    assert sum(1 for s in subs if s["counted"]) == 4
    assert subs[-1]["status"] == "budget_exhausted"


def test_budget_gate_concurrent_evaluations_respect_parallelism_and_exact_cap(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    run_dir = tmp_path / "concurrent-run"
    fake = FakeRiskEvaluator(slots=6, delay=0.05)
    gate = budget_gate.BudgetGate(run_dir, fake, slots=6)

    programs = []
    for i in range(10):
        p = tmp_path / f"prog_{i}.py"
        # Include two identical programs (i=1 and i=2) to exercise concurrent same-SHA deduplication.
        val = 1 if i in (1, 2) else i
        p.write_text(f"def solve(problem, budget, seed):\n    return {val}\n")
        programs.append(p)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(gate.submit, programs))

    assert len(results) == 10
    assert fake.max_active >= 2  # Genuinely concurrent evaluations outside metadata lock
    rows = ledgers.read(run_dir / "ledger.jsonl")
    assert len(rows) == 6 == len(fake.calls)
    assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4, 5]
    assert gate.exhausted


def test_adapter_refuses_per_case_feedback(tmp_path):
    p_init = tmp_path / "init.py"
    p_init.write_bytes(INITIAL_SOURCE)
    fake = FakeRiskEvaluator(slots=2, inject_case_risks=True)
    adapter = openevolve_arm.OpenEvolveAdapter(tmp_path / "run", fake, slots=2)
    with pytest.raises(ValueError, match="per-case"):
        adapter.evaluate_program(p_init)

    with pytest.raises(ValueError, match="aggregate-only"):
        openevolve_arm.OpenEvolveAdapter(tmp_path / "run2", fake, slots=2, case_feedback="opaque")


def test_system_prompt_offpeak_guard_and_token_usage_summary(tmp_path):
    prompt = openevolve_arm.build_system_prompt()
    assert "solve(problem, budget, seed)" in prompt
    assert "32,000 bytes" in prompt
    assert "combined_score" in prompt
    assert "hard_volume_bounds" in prompt
    assert "selfcheck.py" not in prompt

    # Off-peak guard: Saturday 10:00 CST (02:00 UTC) is off-peak; Monday 10:00 CST (02:00 UTC) is peak.
    sat_cst = datetime(2026, 11, 28, 2, 0, tzinfo=timezone.utc)
    mon_peak_cst = datetime(2026, 11, 30, 2, 0, tzinfo=timezone.utc)
    mon_night_cst = datetime(2026, 11, 30, 12, 0, tzinfo=timezone.utc)  # 20:00 CST
    assert openevolve_launch.is_china_offpeak(sat_cst) is True
    assert openevolve_launch.is_china_offpeak(mon_peak_cst) is False
    assert openevolve_launch.is_china_offpeak(mon_night_cst) is True
    with pytest.raises(openevolve_launch.PlanError, match="peak hours"):
        openevolve_launch.require_offpeak(mon_peak_cst)
    openevolve_launch.require_offpeak(mon_peak_cst, allow_peak=True)

    trace_dir = tmp_path / "oe"
    trace_dir.mkdir()
    (trace_dir / "evolution_trace.jsonl").write_text(
        json.dumps({"iteration": 1, "metadata": {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "finish_reason": "stop"}}}) + "\n"
        + json.dumps({"iteration": 2, "metadata": {"token_usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}}}) + "\n"
    )
    (trace_dir / "evolution_trace_failures.jsonl").write_text(
        json.dumps({"iteration": 3, "error": "diff failed", "token_usage": {"prompt_tokens": 120, "completion_tokens": 65536, "total_tokens": 65656, "finish_reason": "length"}}) + "\n"
    )
    summary = openevolve_arm.summarize_token_usage(trace_dir)
    assert summary == {
        "calls": 3,
        "succeeded_iterations": 2,
        "failed_iterations": 1,
        "length_truncated_calls": 1,
        "prompt_tokens": 420,
        "completion_tokens": 65666,
        "total_tokens": 66086,
        "missing_usage_calls": 0,
    }


def test_top5_freeze_validation_and_one_shot_refoe_unseal_export(tmp_path):
    """End-to-end check: top-5 training finalists -> freeze_run -> validate_run -> one-shot REFOE unseal & per-seed export."""
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
    pid = "rsi-r2-testrefoe"
    iso = proto["case_isolation"]

    class FullInProcessExecution(InProcessExecution):
        def run(self, identity, *, phase, **kwargs):
            return super().run(identity, phase="solver" if phase == "interface" else phase, **kwargs)

    refs = {}
    selections = {}
    for seed in (401, 402):
        run_id = f"openevolve-s{seed}"
        service_dir = tmp_path / "private" / pid / "services" / "M0" / run_id
        service = make_service(tmp_path, train_manifest, name=str(service_dir.relative_to(tmp_path)),
                               finalists=5, case_isolation=iso)
        run_dir = tmp_path / "runs" / pid / "M0" / run_id
        adapter = openevolve_arm.OpenEvolveAdapter(run_dir, LoopbackClient(run_dir, service), slots=6)
        for i in range(6):
            p = tmp_path / f"cand_{seed}_{i}.py"
            if i == 0:
                p.write_bytes(INITIAL_SOURCE)
            else:
                p.write_text(
                    f"from tasks.copper_inversion.v2.baselines import initialize\n"
                    f"def solve(problem, budget, seed):\n"
                    f"    _k = {seed * 10 + i}\n"
                    f"    return initialize(problem)\n"
                )
            adapter.evaluate_program(p)
        sealed = service.seal_training()
        assert sealed["used_slots"] == 6 and len(sealed["source_sha256s"]) == 5
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "experiment": "M0", "protocol_id": pid,
            "arm": "openevolve", "tier": "S", "judge": "standard", "seed": seed,
        }))
        (run_dir / "result.json").write_text(json.dumps({
            "status": "completed", "charged_records": 6, "training_finalists": sealed["source_sha256s"],
        }))
        with patch("tasks.copper_inversion.v2.selection.Execution", FullInProcessExecution):
            rec = freeze.freeze_run(tmp_path, pid, "M0", run_id, round_index=None, release=release, protocol=proto)
            assert len(rec["source_sha256s"]) == 5
            sel = freeze.validate_run(tmp_path, pid, "M0", run_id, round_index=None)
            selected_sha = sel["selected_sha256"]
            assert selected_sha in set(sealed["source_sha256s"])
        sel_target = tmp_path / "private" / pid / "selection" / "M0" / run_id / "full"
        src_file = run_dir / "candidates" / selected_sha / "solver.py"
        refs[run_id] = {"source": str(src_file), "sha256": selected_sha, "seed": seed}
        selections[run_id] = str(sel_target)

    refs_path = tmp_path / "private" / pid / "refoe-inputs" / "references.json"
    sels_path = tmp_path / "private" / pid / "refoe-inputs" / "selections.json"
    refs_path.parent.mkdir(parents=True, exist_ok=True)
    refs_path.write_text(json.dumps(refs))
    sels_path.write_text(json.dumps(selections))

    with patch("tasks.copper_inversion.v2.selection.Execution", FullInProcessExecution):
        res = unseal.unseal_references(tmp_path, pid, refs_path, sels_path, operator="tester", experiment="REFOE")
        assert res == {"status": "unsealed", "experiment": "REFOE", "references": 2}
        # One-shot guard refuses second REFOE unseal (already tested + logged).
        with pytest.raises(ValueError, match="untested|already been unsealed"):
            unseal.unseal_references(tmp_path, pid, refs_path, sels_path, operator="tester", experiment="REFOE")

    table = export_outcomes.export_references(tmp_path, pid, references=selections, experiment="REFOE")
    assert table["experiment"] == "REFOE"
    assert len(table["rows"]) == 2
    assert [r["run_id"] for r in table["rows"]] == ["openevolve-s401", "openevolve-s402"]
    assert [r["arm"] for r in table["rows"]] == ["openevolve", "openevolve"]
    assert [r["seed"] for r in table["rows"]] == [401, 402]
    assert [r["judge"] for r in table["rows"]] == ["standard", "standard"]
    assert all(len(r["finalist_sha256s"].split()) == 5 for r in table["rows"])
    assert all(r["test_id_iou"] is not None and r["test_ood_iou"] is not None for r in table["rows"])
