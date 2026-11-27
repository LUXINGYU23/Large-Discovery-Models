"""Unit tests for the shared BudgetGate and LLM4AD external-arm adapter (zero live API calls)."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ldm_tts.engine.run_store import BudgetExceededError
from tasks.copper_inversion.core.candidate import validate_source
from tasks.copper_inversion.core.encoding import MAX_NORMALIZED_BYTES
from tasks.copper_inversion.experiments.rsi import (
    budget_gate,
    export_outcomes,
    freeze,
    llm4ad_arm,
    unseal,
)
from tasks.copper_inversion.experiments.rsi.analysis.outcomes import Outcomes
from tasks.copper_inversion.rsi import audit, ledger as ledgers
from tasks.copper_inversion.rsi.controller import INITIAL_SOURCE


class FakeRiskEvaluator:
    """Thread-safe mock of RiskClient enforcing slot cap and aggregate-only responses."""

    def __init__(self, *, slots=10, scores=None, inject_case_risks=False, delay=0.0):
        self.slots = slots
        self.scores = dict(scores or {})
        self.inject_case_risks = inject_case_risks
        self.delay = float(delay)
        self._lock = threading.Lock()
        self.calls = []
        self.seen = {}

    def evaluate_source(self, source_path):
        body = Path(source_path).read_bytes()
        sha = validate_source(body)
        if self.delay > 0:
            time.sleep(self.delay)
        with self._lock:
            if sha in self.seen:
                return self.seen[sha]
            if len(self.seen) >= self.slots:
                raise BudgetExceededError("risk-only feedback budget exhausted")
            idx = len(self.seen)
            self.calls.append((sha, body.decode("utf-8")))
            spec = self.scores.get(sha, {
                "status": "valid",
                "risk": round(max(0.05, 0.48 - 0.02 * idx), 4),
                "legal_rate": 1.0,
                "failed_solves": 0,
                "seconds_per_solve": 0.4,
                "message": "",
            })
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


def test_template_program_strips_to_exact_a0_and_prompts_have_no_iou_substring():
    tpl = llm4ad_arm.build_template_program()
    desc = llm4ad_arm.build_task_description()
    a0_bytes, a0_sha = budget_gate.prepare_submission(INITIAL_SOURCE)
    tpl_bytes, tpl_sha = budget_gate.prepare_submission(tpl)
    assert tpl_sha == a0_sha
    assert tpl_bytes == a0_bytes
    assert "iou" not in tpl.lower()
    assert "iou" not in desc.lower()
    assert "define helper functions inside `solve`" in desc.lower()
    assert "put all `import` statements inside `solve`" in desc.lower()


def test_normalize_llm_reply_handles_fences_thoughts_completions_and_imports():
    # 1. EoH multiline {thought} + fenced function with top-level import above def.
    raw_eoh = (
        "{\nUse stretched nearest observation\nwith anisotropic z-weight.\n}\n"
        "```python\n"
        "import numpy as np\n\n"
        "def solve(problem, budget, seed):\n"
        "    return initialize(problem)\n"
        "```\n"
        "Extra trailing prose."
    )
    norm_eoh = llm4ad_arm.normalize_llm_reply(raw_eoh)
    assert norm_eoh.startswith("{Use stretched nearest observation with anisotropic z-weight.}\n")
    assert "```" not in norm_eoh
    assert "    import numpy as np" in norm_eoh
    _, _ = budget_gate.prepare_submission(
        "from tasks.copper_inversion.v2.baselines import initialize\n"
        + norm_eoh.split("\n", 1)[1]
    )

    # 2. FunSearch / HillClimb completion-only indented body inside markdown fence.
    raw_comp = "```python\n    x = initialize(problem)\n    return x\n```"
    norm_comp = llm4ad_arm.normalize_llm_reply(raw_comp)
    assert norm_comp.startswith("    x = initialize(problem)")
    assert "```" not in norm_comp

    # 3. ReEvo plain-text short-term reflection stays untouched.
    raw_ref = "Combine vertical anisotropy with hole-grouped smoothing near contacts."
    assert llm4ad_arm.normalize_llm_reply(raw_ref) == raw_ref


def test_budget_gate_counts_evaluations_not_malformed_proposals_and_deduplicates(tmp_path):
    run_dir = tmp_path / "run"
    good_src = "def solve(problem, budget, seed):\n    # comment\n    return problem['domain_mask'].astype(float)\n"
    _, good_sha = budget_gate.prepare_submission(good_src)
    rej_src = "def solve(problem, budget, seed):\n    return 0\n"
    _, rej_sha = budget_gate.prepare_submission(rej_src)
    fail_src = "def solve(problem, budget, seed):\n    raise RuntimeError('boom')\n"
    _, fail_sha = budget_gate.prepare_submission(fail_src)
    extra_src = "def solve(problem, budget, seed):\n    return 1\n"

    exhausted_calls = []
    fake = FakeRiskEvaluator(slots=3, scores={
        good_sha: {"status": "valid", "risk": 0.31, "legal_rate": 1.0,
                   "failed_solves": 0, "seconds_per_solve": 0.5, "message": ""},
        rej_sha: {"status": "rejected", "risk": 1.0, "legal_rate": 0.0,
                  "failed_solves": 12, "seconds_per_solve": 0.1, "message": "interface rejected"},
        fail_sha: {"status": "failed", "risk": 1.0, "legal_rate": 0.0,
                   "failed_solves": 12, "seconds_per_solve": 0.2, "message": "all planned solves failed"},
    })
    gate = budget_gate.BudgetGate(
        run_dir, fake, slots=3, on_exhausted=lambda: exhausted_calls.append(True)
    )

    # Unparseable and oversized proposals do NOT consume evaluation slots and never raise.
    assert gate.submit("def solve(:\n") == (None, "invalid_source")
    big = "def solve(problem, budget, seed):\n" + "".join(
        f"    v_{i} = {i}\n" for i in range(4000)
    ) + "    return v_0\n"
    assert len(big.encode()) > MAX_NORMALIZED_BYTES
    assert gate.submit(big) == (None, "oversized")
    assert gate.used_slots == 0

    # Slot 1: initial A0.
    r0, st0 = gate.submit(INITIAL_SOURCE)
    assert st0 == "valid" and r0 == pytest.approx(0.48)

    # Slot 2: valid candidate (comments stripped).
    r1, st1 = gate.submit(good_src)
    assert (r1, st1) == (pytest.approx(0.31), "valid")
    assert "# comment" not in fake.calls[-1][1]

    # Duplicate of Slot 2 (with different comments) costs 0 slots.
    r_dup, st_dup = gate.submit("# another comment\n" + good_src)
    assert (r_dup, st_dup) == (pytest.approx(0.31), "duplicate")
    assert len(fake.calls) == 2

    # Slot 3: service interface rejection DOES consume 1 slot and returns (None, 'rejected').
    r2, st2 = gate.submit(rej_src)
    assert (r2, st2) == (None, "rejected")
    assert gate.exhausted and (run_dir / "gate-exhausted").is_file()
    assert len(exhausted_calls) >= 1

    # Duplicate of rejected candidate returns (None, 'duplicate_rejected') with 0 extra calls.
    assert gate.submit(rej_src) == (None, "duplicate_rejected")

    # Slot 4 (> 3): refused at gate without calling service.
    assert gate.submit(extra_src) == (None, "budget_exhausted")
    assert len(fake.calls) == 3

    summary = gate.summary()
    assert summary["used_slots"] == 3
    assert summary["counted_evaluations"] == 3
    assert summary["valid_evaluations"] == 2
    assert summary["rejected_evaluations"] == 1
    assert summary["invalid_source_submissions"] == 1
    assert summary["oversized_submissions"] == 1
    assert summary["duplicate_submissions"] == 2
    assert summary["budget_exhausted_submissions"] == 1


def test_budget_gate_concurrent_threads_deduplicate_and_enforce_exact_cap(tmp_path):
    fake = FakeRiskEvaluator(slots=5, delay=0.03)
    gate = budget_gate.BudgetGate(tmp_path / "conc", fake, slots=5)
    gate.submit(INITIAL_SOURCE)

    # 8 threads submit 6 candidates (including duplicates) against a 5-slot cap.
    candidates = [
        f"from tasks.copper_inversion.v2.baselines import initialize\n"
        f"def solve(problem, budget, seed):\n    _tag = {i % 6}\n    return initialize(problem)\n"
        for i in range(10)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(gate.submit, candidates))
    assert gate.used_slots == 5
    assert len(fake.calls) == 5
    assert any(status == "budget_exhausted" for _, status in outcomes)


def test_budget_gate_refuses_per_case_leak(tmp_path):
    fake = FakeRiskEvaluator(slots=2, inject_case_risks=True)
    gate = budget_gate.BudgetGate(tmp_path / "leak", fake, slots=2)
    with pytest.raises(ValueError, match="per-case"):
        gate.submit(INITIAL_SOURCE)


def test_llm4ad_freeze_validation_unseal_and_outcomes_export_pipeline(tmp_path):
    """End-to-end 4-split test of LLM4AD experiment label through freeze, validation, unseal, and export."""
    from tasks.copper_inversion.tests.test_rsi_service import (
        InProcessExecution,
        LoopbackClient,
        make_service,
        write_suite,
    )

    release = tmp_path / "release"
    train_manifest = write_suite(release, "train", split="train", bodies=2, offset=0)
    write_suite(release, "validation", split="validation", bodies=2, offset=10)
    write_suite(release, "test_id", split="test_id", bodies=2, offset=20)
    write_suite(release, "test_ood", split="test_ood", bodies=2, offset=30)

    proto = json.loads((Path(__file__).resolve().parents[1] / "resources/rsi/protocol.r2.example.json").read_text())
    proto["tiers"]["S"]["train_manifest"] = "train/manifest.json"
    pid = "rsi-r2-testllm4ad"
    exp = "LLM4AD"
    iso = proto["case_isolation"]

    run_ids = []
    for method, seed in (("eoh", 401), ("funsearch", 402)):
        run_id = llm4ad_arm.run_id_for(method, seed)
        run_ids.append(run_id)
        service_dir = tmp_path / "private" / pid / "services" / exp / run_id
        service = make_service(
            tmp_path, train_manifest,
            name=str(service_dir.relative_to(tmp_path)),
            finalists=5, case_isolation=iso,
        )
        run_dir = tmp_path / "runs" / pid / exp / run_id
        gate = budget_gate.BudgetGate(run_dir, LoopbackClient(run_dir, service), slots=6)
        gate.submit(llm4ad_arm.build_template_program())
        for i in range(1, 6):
            gate.submit(
                "from tasks.copper_inversion.v2.baselines import initialize\n"
                f"def solve(problem, budget, seed):\n    _variant = {seed * 10 + i}\n    return initialize(problem)\n"
            )
        sealed = service.seal_training()
        assert sealed["used_slots"] == 6
        assert len(sealed["source_sha256s"]) == 5

        (run_dir / "task_brief.txt").write_text(llm4ad_arm.TASK_BRIEF_PATH.read_text(encoding="utf-8"))
        req_dir = run_dir / "harness" / "requests"
        req_dir.mkdir(parents=True, exist_ok=True)
        (req_dir / "0001.request.bin").write_bytes(b'{"model":"deepseek-flash","messages":[]}')
        (run_dir / "manifest.json").write_text(json.dumps({
            "schema": "rsi-run-manifest-v1",
            "run_id": run_id,
            "experiment": exp,
            "protocol_id": pid,
            "arm": llm4ad_arm.METHOD_ARMS[method],
            "tier": "S",
            "judge": "standard",
            "seed": seed,
        }))
        (run_dir / "result.json").write_text(json.dumps({
            "status": "completed",
            "stop_reason": "completed",
            "charged_records": 6,
            "training_finalists": sealed["source_sha256s"],
        }))
        assert audit.reconcile_budget(run_dir, service_dir)["charged_sources"] == 6
        blind = audit.scan_blind(run_dir, service_dir, require_requests=True)
        assert blind["leaks"] == 0 and blind["status"] == "passed"

    class FullInProcessExecution(InProcessExecution):
        def run(self, identity, *, phase, **kwargs):
            return super().run(identity, phase="solver" if phase == "interface" else phase, **kwargs)

    with patch("tasks.copper_inversion.v2.selection.Execution", FullInProcessExecution):
        for run_id in run_ids:
            rec = freeze.freeze_run(tmp_path, pid, exp, run_id, round_index=None, release=release, protocol=proto)
            assert len(rec["source_sha256s"]) == 5
            sel = freeze.validate_run(tmp_path, pid, exp, run_id, round_index=None)
            assert sel["selected_sha256"] in set(rec["source_sha256s"])

        listing = {
            "experiment": exp,
            "protocol_id": pid,
            "solvers": [{"run_id": r, "prefix": "full"} for r in run_ids],
        }
        unseal_res = unseal.unseal(tmp_path, listing, workers=2)
        assert unseal_res["status"] == "unsealed" and unseal_res["solvers"] == 2

        out_json = tmp_path / "private" / pid / f"outcomes-{exp}.json"
        export_outcomes.main([
            "--root", str(tmp_path),
            "--protocol-id", pid,
            "--experiment", exp,
            "--output", str(out_json),
        ])
        loaded = Outcomes.load(out_json)
        assert len(loaded.tables[0]["rows"]) == 2
        by_arm = {r["arm"]: r for r in loaded.tables[0]["rows"]}
        assert set(by_arm) == {"llm4ad_eoh", "llm4ad_funsearch"}
        assert by_arm["llm4ad_eoh"]["seed"] == 401
        assert by_arm["llm4ad_funsearch"]["seed"] == 402
        assert by_arm["llm4ad_eoh"]["test_id_iou"] is not None


@pytest.mark.parametrize("method_name", ("eoh", "reevo", "funsearch", "eps"))
def test_all_four_llm4ad_methods_execute_and_stop_at_exact_slot_gate(tmp_path, method_name):
    pytest.importorskip("llm4ad")
    counter = {"n": 0}
    lock = threading.Lock()

    class MockCompletions:
        def create(self, **kwargs):
            # Verify high reasoning effort and 65536 max_tokens are passed on every call.
            assert kwargs["model"] == "deepseek-flash"
            assert kwargs["max_tokens"] == 65536
            assert kwargs["reasoning_effort"] == "high"
            assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
            prompt_text = kwargs["messages"][-1]["content"]
            with lock:
                counter["n"] += 1
                idx = counter["n"]
            if "using less than" in prompt_text and "words" in prompt_text:
                content = f"Weight vertical distance stronger and smooth boreholes (hint {idx})."
            else:
                content = (
                    f"{{Heuristic variant {idx} adjusting spatial scaling.}}\n"
                    "```python\n"
                    "import numpy as np\n\n"
                    f"def solve_v2(problem, budget, seed):\n"
                    f"    _step = {idx}\n"
                    "    arr = initialize(problem)\n"
                    "    return np.asarray(arr, dtype=float)\n"
                    "```"
                )
            return SimpleNamespace(
                id=f"mock-{idx}",
                model="deepseek-flash",
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=content),
                )],
                usage=SimpleNamespace(
                    prompt_tokens=300,
                    completion_tokens=150,
                    total_tokens=450,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=90),
                    prompt_tokens_details=SimpleNamespace(cached_tokens=100),
                ),
            )

    def mock_factory(**kwargs):
        # Ensure API key is never written to disk while passed to client.
        assert kwargs["api_key"] == "sk-test-secret-never-persist"
        return SimpleNamespace(
            chat=SimpleNamespace(completions=MockCompletions()),
            close=lambda: None,
        )

    out_dir = tmp_path / f"run-{method_name}"
    fake = FakeRiskEvaluator(slots=6)
    res = llm4ad_arm.execute_llm4ad_search(
        method_name=method_name,
        seed=401,
        out_dir=out_dir,
        evaluator=fake,
        api_key="sk-test-secret-never-persist",
        slots=6,
        client_factory=mock_factory,
    )

    rows = ledgers.read(out_dir / "ledger.jsonl")
    assert len(rows) == 6
    assert len(fake.calls) == 6
    assert rows[0]["round"] == 0 and rows[0]["branch"] == -1 and rows[0]["selection"] == "initial"
    assert res["gate_summary"]["used_slots"] == 6
    assert res["gate_summary"]["exhausted"] is True
    assert res["usage"]["calls"] >= 5
    assert res["usage"]["reasoning_tokens"] > 0

    # Secret API key must never appear anywhere in run_dir.
    for path in out_dir.rglob("*"):
        if path.is_file():
            assert b"sk-test-secret-never-persist" not in path.read_bytes()
