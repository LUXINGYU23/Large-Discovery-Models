"""Unit tests for BudgetGate and the ShinkaEvolve external-arm integration (zero live API calls)."""

from datetime import datetime, timezone
import json
import sqlite3
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
    shinka_arm,
    unseal,
)
from tasks.copper_inversion.experiments.rsi.analysis.outcomes import Outcomes
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
        spec = self.scores.get(
            sha,
            {
                "status": "valid",
                "risk": 0.42,
                "legal_rate": 1.0,
                "failed_solves": 0,
                "seconds_per_solve": 0.5,
                "message": "",
            },
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


def test_wrap_initial_source_preserves_a0_sha_and_strips_evolve_markers():
    wrapped = shinka_arm.wrap_initial_source(INITIAL_SOURCE)
    assert wrapped.startswith("# EVOLVE-BLOCK-START\n")
    assert wrapped.endswith("# EVOLVE-BLOCK-END\n")
    stripped_a0, sha_a0 = budget_gate.prepare_submission(INITIAL_SOURCE)
    stripped_wrapped, sha_wrapped = budget_gate.prepare_submission(wrapped.encode("utf-8"))
    assert sha_wrapped == sha_a0
    assert stripped_wrapped == stripped_a0
    assert b"EVOLVE-BLOCK" not in stripped_wrapped


def test_budget_gate_counts_service_evaluations_not_preeval_failures(tmp_path):
    run_dir = tmp_path / "shinka-s401"
    good_src = (
        b"# EVOLVE-BLOCK-START\n"
        b"def solve(problem, budget, seed):\n"
        b"    '''doc'''\n"
        b"    return problem['domain_mask'].astype(float)\n"
        b"# EVOLVE-BLOCK-END\n"
    )
    good_stripped, good_sha = budget_gate.prepare_submission(good_src)

    rej_src = b"def solve(problem, budget, seed):\n    return 0\n"
    _, rej_sha = budget_gate.prepare_submission(rej_src)

    big_src = (
        "def solve(problem, budget, seed):\n"
        + "".join(f"    x_{i} = {i}\n" for i in range(4000))
        + "    return x_0\n"
    ).encode("utf-8")
    assert len(big_src) > MAX_NORMALIZED_BYTES

    fake = FakeRiskEvaluator(
        slots=3,
        scores={
            good_sha: {
                "status": "valid",
                "risk": 0.25,
                "legal_rate": 1.0,
                "failed_solves": 0,
                "seconds_per_solve": 0.4,
                "message": "",
            },
            rej_sha: {
                "status": "rejected",
                "risk": 1.0,
                "legal_rate": 0.0,
                "failed_solves": 12,
                "seconds_per_solve": 0.1,
                "message": "interface rejected",
            },
        },
    )
    gate = budget_gate.BudgetGate(run_dir, fake, slots=3)

    # Pre-evaluation syntax/oversized errors never raise and do NOT consume service slots.
    r_bad, st_bad = budget_gate.submit(b"def solve(:\n", gate=gate)
    assert (r_bad, st_bad) == (None, "invalid_source")
    r_big, st_big = gate.submit(big_src)
    assert (r_big, st_big) == (None, "oversized")
    assert gate.used_slots == 0 and not gate.exhausted

    # Slot 1: wrapped A0.
    r0, st0 = gate.submit(shinka_arm.wrap_initial_source().encode("utf-8"))
    assert st0 == "valid" and r0 == pytest.approx(0.42)
    assert gate.used_slots == 1

    # Slot 2: valid candidate (markers/comments stripped before service call).
    d1 = gate.submit_detailed(good_src)
    assert d1["accepted"] is True
    assert d1["risk"] == pytest.approx(0.25)
    assert d1["combined_score"] == pytest.approx(0.75)
    assert "EVOLVE-BLOCK" not in fake.calls[-1][1]
    assert gate.used_slots == 2

    # Duplicate candidate (even with different comments) costs 0 slots.
    d_dup = gate.submit_detailed(b"# extra comment\n" + good_stripped)
    assert d_dup["status"] == "duplicate"
    assert d_dup["counted"] is False
    assert d_dup["combined_score"] == pytest.approx(0.75)
    assert gate.used_slots == 2 and len(fake.calls) == 2

    # Slot 3: service interface rejection DOES consume 1 slot and exhausts the 3-slot gate.
    d2 = gate.submit_detailed(rej_src)
    assert d2["accepted"] is False
    assert d2["combined_score"] == 0.0
    assert d2["counted"] is True
    assert gate.used_slots == 3
    assert gate.exhausted is True
    assert (run_dir / "gate-exhausted").is_file()

    # Beyond 3 slots: refused without calling service.
    d3 = gate.submit_detailed(b"def solve(problem, budget, seed):\n    return 2\n")
    assert d3["status"] == "budget_exhausted"
    assert d3["counted"] is False
    assert len(fake.calls) == 3

    rows = ledgers.read(run_dir / "ledger.jsonl")
    assert len(rows) == 3 == len(fake.seen)


def test_shinka_evaluate_cli_and_usage_summary(tmp_path):
    from tasks.copper_inversion.tests.test_rsi_service import (
        LoopbackClient,
        make_service,
        write_suite,
    )

    release = tmp_path / "release"
    train_manifest = write_suite(release, "train", split="train", bodies=2, offset=0)
    proto = json.loads(
        (Path(__file__).resolve().parents[1] / "resources/rsi/protocol.r2.example.json").read_text()
    )
    service = make_service(
        tmp_path,
        train_manifest,
        name="svc",
        finalists=5,
        case_isolation=proto["case_isolation"],
    )
    run_dir = tmp_path / "run"
    loop_client = LoopbackClient(run_dir, service)

    cand_path = tmp_path / "cand.py"
    cand_path.write_text(shinka_arm.wrap_initial_source(), encoding="utf-8")
    res_dir = tmp_path / "eval_out"
    gate_cfg = tmp_path / "gate.json"
    gate_cfg.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "evaluation_socket": "/unused.sock",
                "evaluation_contract": service.public_contract,
                "evaluation_service_uid": 0,
                "slots": 4,
                "case_feedback": "none",
            }
        ),
        encoding="utf-8",
    )
    with patch.object(shinka_arm, "RiskClient", return_value=loop_client):
        assert (
            shinka_arm.main(
                [
                    "--program_path",
                    str(cand_path),
                    "--results_dir",
                    str(res_dir),
                    "--gate_config",
                    str(gate_cfg),
                ]
            )
            == 0
        )

    metrics = json.loads((res_dir / "metrics.json").read_text())
    correct = json.loads((res_dir / "correct.json").read_text())
    assert set(metrics) == {"combined_score", "public", "private"}
    assert metrics["public"] == {} and metrics["private"] == {}
    assert 0.0 < metrics["combined_score"] <= 1.0
    assert correct == {"correct": True}

    # Verify DB + JSONL usage summary.
    shinka_dir = tmp_path / "shinka"
    shinka_dir.mkdir()
    db_path = shinka_dir / "programs.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE programs (generation INTEGER, metadata TEXT)")
        conn.execute("CREATE TABLE attempt_log (generation INTEGER, status TEXT, details TEXT)")
        conn.execute("INSERT INTO programs VALUES (0, ?)", (json.dumps({}),))
        conn.execute("INSERT INTO programs VALUES (0, ?)", (json.dumps({"_is_island_copy": True}),))
        conn.execute(
            "INSERT INTO attempt_log VALUES (1, 'failed', ?)",
            (json.dumps({"node_kind": "failed_proposal"}),),
        )
    shinka_arm._append_jsonl_locked(
        shinka_dir / shinka_arm.LLM_CALLS_FILENAME,
        {
            "status": "ok",
            "input_tokens": 900,
            "output_tokens": 300,
            "thinking_tokens": 4200,
            "completion_tokens": 4500,
            "total_tokens": 5400,
            "cost": 0.0028,
            "incomplete_reason": None,
        },
    )
    summary = shinka_arm.summarize_token_usage(shinka_dir)
    assert summary["calls"] == 1
    assert summary["prompt_tokens"] == 900
    assert summary["thinking_tokens"] == 4200
    assert summary["completion_tokens"] == 4500
    assert summary["total_tokens"] == 5400
    assert summary["persisted_generations"] == 1
    assert summary["failed_proposal_generations"] == 1


def test_offpeak_window_check():
    # Wednesday 10:30 CST = 02:30 UTC -> peak
    ok_peak, _ = shinka_arm.is_deepseek_offpeak(datetime(2026, 11, 25, 2, 30, tzinfo=timezone.utc))
    assert ok_peak is False
    # Wednesday 13:00 CST = 05:00 UTC -> lunch off-peak
    ok_lunch, _ = shinka_arm.is_deepseek_offpeak(datetime(2026, 11, 25, 5, 0, tzinfo=timezone.utc))
    assert ok_lunch is True
    # Saturday 10:30 CST = 02:30 UTC -> weekend off-peak
    ok_wknd, _ = shinka_arm.is_deepseek_offpeak(datetime(2026, 11, 28, 2, 30, tzinfo=timezone.utc))
    assert ok_wknd is True


def test_shinka_experiment_freeze_unseal_and_export_outcomes(tmp_path):
    """Full pipeline test: SHINKA runs -> top-5 freeze + validation -> one-shot unseal -> export_outcomes."""
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

    proto = json.loads(
        (Path(__file__).resolve().parents[1] / "resources/rsi/protocol.r2.example.json").read_text()
    )
    proto["tiers"]["S"]["train_manifest"] = "train/manifest.json"
    pid = "rsi-r2-testshinka"

    class FullInProcessExecution(InProcessExecution):
        def run(self, identity, *, phase, **kwargs):
            return super().run(identity, phase="solver" if phase == "interface" else phase, **kwargs)

    for seed in (401, 402):
        run_id = f"shinka-s{seed}"
        service_dir = tmp_path / "private" / pid / "services" / "SHINKA" / run_id
        service = make_service(
            tmp_path,
            train_manifest,
            name=str(service_dir.relative_to(tmp_path)),
            finalists=5,
            case_isolation=proto["case_isolation"],
        )
        run_dir = tmp_path / "runs" / pid / "SHINKA" / run_id
        gate = budget_gate.BudgetGate(run_dir, LoopbackClient(run_dir, service), slots=6)
        for i in range(6):
            if i == 0:
                src = shinka_arm.wrap_initial_source().encode("utf-8")
            else:
                src = (
                    f"# EVOLVE-BLOCK-START\n"
                    f"from tasks.copper_inversion.v2.baselines import initialize\n"
                    f"def solve(problem, budget, seed):\n"
                    f"    _variant = {seed * 10 + i}\n"
                    f"    return initialize(problem)\n"
                    f"# EVOLVE-BLOCK-END\n"
                ).encode("utf-8")
            detail = gate.submit_detailed(src)
            assert detail["accepted"] is True

        assert gate.exhausted is True
        sealed = service.seal_training()
        assert sealed["used_slots"] == 6
        assert len(sealed["source_sha256s"]) == 5

        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "experiment": "SHINKA",
                    "protocol_id": pid,
                    "arm": "shinka",
                    "tier": "S",
                    "judge": "standard",
                    "seed": seed,
                }
            )
        )
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "charged_records": 6,
                    "training_finalists": sealed["source_sha256s"],
                }
            )
        )
        target = freeze.freeze_run(
            tmp_path,
            pid,
            "SHINKA",
            run_id,
            None,
            release=release,
            protocol=proto,
        )
        with patch("tasks.copper_inversion.v2.selection.Execution", FullInProcessExecution):
            report = freeze.validate_run(tmp_path, pid, "SHINKA", run_id, None)
        assert len(report["candidates"]) == 5

    # Verify one-shot REFSHINKA unseal + export_outcomes path (matching REFOE / REF*).
    refs_dir = tmp_path / "refshinka-inputs"
    refs_dir.mkdir()
    refs_map, sels_map = {}, {}
    for seed in (401, 402):
        rid = f"shinka-s{seed}"
        sel_target = tmp_path / "private" / pid / "selection" / "SHINKA" / rid / "full"
        sel_report = json.loads((sel_target / "selection.json").read_text())
        chosen_sha = sel_report["selected_sha256"]
        cand_file = tmp_path / "runs" / pid / "SHINKA" / rid / "candidates" / chosen_sha / "solver.py"
        refs_map[rid] = {"source": str(cand_file.resolve()), "sha256": chosen_sha, "seed": seed}
        sels_map[rid] = str(sel_target.resolve())
    refs_path = refs_dir / "references.json"
    sels_path = refs_dir / "selections.json"
    refs_path.write_text(json.dumps(refs_map))
    sels_path.write_text(json.dumps(sels_map))

    with patch("tasks.copper_inversion.v2.selection.Execution", FullInProcessExecution):
        ref_unsealed = unseal.unseal_references(
            tmp_path, pid, refs_path, sels_path, experiment="REFSHINKA", workers=1
        )
        assert ref_unsealed == {"experiment": "REFSHINKA", "references": 2, "status": "unsealed"}
        with pytest.raises(ValueError, match="untested|already been unsealed"):
            unseal.unseal_references(
                tmp_path, pid, refs_path, sels_path, experiment="REFSHINKA", workers=1
            )

    ref_out_json = tmp_path / "outcomes-REFSHINKA.json"
    assert (
        export_outcomes.main(
            [
                "--root",
                str(tmp_path),
                "--protocol-id",
                pid,
                "--experiment",
                "REFSHINKA",
                "--references",
                str(sels_path),
                "--output",
                str(ref_out_json),
            ]
        )
        == 0
    )
    assert ref_out_json.with_suffix(".csv").is_file()
    loaded_ref = Outcomes.load(ref_out_json)
    assert len(loaded_ref.tables[0]["rows"]) == 2
    assert {r["seed"] for r in loaded_ref.tables[0]["rows"]} == {401, 402}
    assert {r["arm"] for r in loaded_ref.tables[0]["rows"]} == {"shinka"}
    assert loaded_ref.freeze("REFSHINKA", "shinka-s401", "full")["sealed"]["test_id"]["iou"] > 0.0
    assert loaded_ref.freeze("REFSHINKA", "shinka-s401", "full")["sealed"]["test_ood"]["iou"] > 0.0
    assert all(len(r["finalist_sha256s"].split()) == 5 for r in loaded_ref.tables[0]["rows"])


def test_shinka_patches_config_and_llm_call_cap(tmp_path):
    pytest.importorskip("shinka")
    from shinka.llm.providers import deepseek as shinka_deepseek
    from shinka.llm.providers.errors import NonRetryableLLMError

    info = shinka_arm.verify_shinka_patches("deepseek-flash", "high")
    assert info["resolved_provider"] == "deepseek"
    assert info["sampled_kwargs"]["reasoning_effort"] == "high"
    assert info["sampled_kwargs"]["max_tokens"] == 65536
    assert "temperature" not in info["sampled_kwargs"]

    evo_cfg, db_cfg, job_cfg = shinka_arm.build_shinka_configs(
        shinka_dir=tmp_path,
        eval_script_path=tmp_path / "eval.py",
        gate_config_path=tmp_path / "gate.json",
        num_generations=10,
    )
    assert evo_cfg.llm_models == ["deepseek-flash"]
    assert evo_cfg.llm_dynamic_selection is None
    assert evo_cfg.llm_kwargs == {"max_tokens": 65536, "reasoning_efforts": "high"}
    assert evo_cfg.embedding_model is None
    assert evo_cfg.novelty_llm_models is None
    assert evo_cfg.use_text_feedback is False
    assert evo_cfg.meta_rec_interval == 10
    assert evo_cfg.meta_llm_models == ["deepseek-flash"]
    assert db_cfg.num_islands == 2
    assert job_cfg.numeric_threads_per_job == 1

    calls_log = tmp_path / shinka_arm.LLM_CALLS_FILENAME
    shinka_arm._append_jsonl_locked(calls_log, {"status": "ok", "input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0})
    shinka_arm._append_jsonl_locked(calls_log, {"status": "ok", "input_tokens": 1, "output_tokens": 1, "thinking_tokens": 0, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0})
    shinka_arm.install_deepseek_telemetry(calls_log, max_llm_calls=2)
    with pytest.raises(NonRetryableLLMError, match="LLM call cap"):
        shinka_deepseek.query_deepseek(None, "deepseek-flash", "hi", "sys", [], None)

