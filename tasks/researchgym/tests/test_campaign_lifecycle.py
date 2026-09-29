"""Review acceptance behaviors exercised through the procedure interface.

Covers replenishment instead of silent shrinking, Harness pause/resume without
re-running committed sessions or double charging, the compiled-policy campaign
and its fallback, measurement deltas between research turns, and the wall clock.
"""

import json
import sys
from pathlib import Path

import numpy as np

from tasks.researchgym.core import workflow
from tasks.researchgym.ldm_task import procedure

from .test_selection_and_policy import AuthoredFixtureExecutor

FIXTURE = Path(__file__).with_name("fixtures") / "harness_sidecar.py"


def fixture(mode):
    return json.dumps([sys.executable, "-u", str(FIXTURE), mode])


def harness_args(tmp_path, mode, *extra, name="run", method="ldm_harness"):
    return ["--mock", "--case", "continual_learning", "--search-method", method, "--harness-sessions", "2",
            "--harness-command-json", fixture(mode), "--out-dir", str(tmp_path), "--run-name", name, *extra]


def events(run_dir, kind):
    rows = map(json.loads, (run_dir / "events.jsonl").read_text().splitlines())
    return [e for e in rows if e["event_type"] == kind]


def budget(run_dir):
    return json.loads((run_dir / "budget.json").read_text())["counters"]


def test_agreeing_sessions_are_replenished_instead_of_shrinking_the_batch(tmp_path):
    code = procedure.main(harness_args(tmp_path, "agreement", "--iterations", "2", "--proposal-samples", "4",
                                       "--bo-pool-size", "3", "--evaluations-per-round", "3"))
    run_dir = tmp_path / "run"
    assert code == 0
    assert budget(run_dir)["external_evaluations"] == 6
    diagnostics = json.loads((run_dir / "proposal_diagnostics/round-000000.json").read_text())
    assert diagnostics["replenishment_collections"] >= 1 and diagnostics["distinct_programs"] >= 3
    turns = [t for frame in map(json.loads, (run_dir / "harness/fixture_requests.jsonl").read_text().splitlines())
             for t in frame["turns"]]
    refill = [t for t in turns if "_f01_" in t["turnId"]]
    assert refill and "replenishment" in json.loads(refill[0]["message"].split("\n", 1)[1])
    first = events(run_dir, "candidates_selected")[0]["payload"]["metadata"]
    assert first["valid_proposal_occurrences"] > 4  # refill occurrences also enter q0


def test_second_turn_receives_new_measurements_with_research_notes(tmp_path):
    assert procedure.main(harness_args(tmp_path, "repair", "--iterations", "2", "--proposal-samples", "4",
                                       "--bo-pool-size", "2")) == 0
    run_dir = tmp_path / "run"
    frames = [json.loads(line) for line in (run_dir / "harness/fixture_requests.jsonl").read_text().splitlines()]
    payload = json.loads(frames[-1]["turns"][0]["message"].split("\n", 1)[1])
    assert payload["message_type"] == "history_delta" and len(payload["new_measured_candidates"]) == 1
    measured = json.loads((run_dir / "harness/measured_history.json").read_text())["observations"]
    notes = measured[0]["research_annotations"]
    assert notes and notes[0]["rationale"].startswith("fixture") and notes[0]["session_id"].startswith("fixture-session")


def test_harness_failure_pauses_and_resume_replays_the_committed_session(tmp_path):
    args = harness_args(tmp_path, "fail_once", "--iterations", "2", "--proposal-samples", "4", "--bo-pool-size", "2")
    assert procedure.main(args) == 2
    run_dir = tmp_path / "run"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "paused_endpoint_unavailable"
    paused = budget(run_dir)
    assert paused["harness_turns"] == 2 and paused["external_evaluations"] == 0
    resume = [a for a in args if a not in ("--run-name", "run")] + ["--resume-from", str(run_dir)]
    assert procedure.main(resume) == 0
    frames = [json.loads(line) for line in (run_dir / "harness/fixture_requests.jsonl").read_text().splitlines()]
    assert [t["turnId"] for t in frames[0]["turns"]] == [t["turnId"] for t in frames[1]["turns"]]
    resumed = budget(run_dir)
    assert resumed["harness_turns"] == 4 and resumed["recovery_attempts"] == 1
    # The first session committed before the failure: its usage is charged once.
    assert resumed["harness_provider_calls"] == 8 and resumed["external_evaluations"] == 2


def compiled(tmp_path, mode, monkeypatch, name):
    monkeypatch.setattr(workflow, "DockerPolicyExecutor", lambda *args: AuthoredFixtureExecutor())
    code = procedure.main(harness_args(tmp_path, mode, "--iterations", "3", "--proposal-samples", "4",
                                       "--bo-pool-size", "3", "--initialization-mode", "shared_start",
                                       method="ldm_harness_compiled", name=name))
    return code, tmp_path / name


def test_compiled_policy_campaign_applies_mean_and_weights(tmp_path, monkeypatch):
    code, run_dir = compiled(tmp_path, "repair", monkeypatch, "compiled")
    assert code == 0
    policies = [e["payload"]["metadata"]["compiled_policy"] for e in events(run_dir, "candidates_selected")
                if "compiled_policy" in e["payload"]["metadata"]]
    assert len(policies) == 2
    assert all(p["source"] == "artifact" and p["alpha"] == 0.5 and p["eta"] == 1.5 for p in policies)
    assert all(p["prediction_mean_abs_change"] > 0 for p in policies)
    counters = budget(run_dir)
    assert counters["policy_harness_turns"] >= 1 and counters["harness_turns"] == 4
    # Policy-session tokens come from its own provider captures and count toward the API cost.
    assert counters["policy_input_tokens"] == 1000 * counters["policy_harness_turns"]
    assert json.loads((run_dir / "harness_provenance.json").read_text())["pools"]["policy_harness"]


def test_rejected_policy_falls_back_to_defaults_and_marks_degraded(tmp_path, monkeypatch):
    code, run_dir = compiled(tmp_path, "policy_reject", monkeypatch, "fallback")
    assert code == 0
    policies = [e["payload"]["metadata"]["compiled_policy"] for e in events(run_dir, "candidates_selected")
                if "compiled_policy" in e["payload"]["metadata"]]
    assert policies and all(p["source"] == "default" and p["degraded"] for p in policies)
    assert all(p["alpha"] == 1.0 and p["eta"] == 1.0 and np.isclose(p["prediction_mean_abs_change"], 0) for p in policies)


def test_exhausted_wall_clock_stops_cleanly_and_resume_does_not_regain_time(tmp_path):
    args = ["--mock", "--iterations", "2", "--proposal-samples", "2", "--bo-pool-size", "2",
            "--campaign-hours", "0.0000001", "--out-dir", str(tmp_path), "--run-name", "clock"]
    assert procedure.main(args) == 0
    run_dir = tmp_path / "clock"
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "stopped_wall_clock" and budget(run_dir)["llm_requests"] == 0
    elapsed = result["benchmark_clock"]["elapsed_seconds"]
    assert procedure.main([a for a in args if a not in ("--run-name", "clock")] + ["--resume-from", str(run_dir)]) == 0
    resumed = json.loads((run_dir / "result.json").read_text())
    assert resumed["status"] == "stopped_wall_clock" and resumed["benchmark_clock"]["elapsed_seconds"] >= elapsed
