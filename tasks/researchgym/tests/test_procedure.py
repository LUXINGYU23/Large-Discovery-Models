"""The shared-runner procedure interface in mock mode, end to end."""

import json
import sys
from pathlib import Path

import pytest

from ldm_tts.data import validate_ir_record
from tasks.researchgym.core import workflow
from tasks.researchgym.ldm_task import procedure

FIXTURE = Path(__file__).with_name("fixtures") / "harness_sidecar.py"


def run(tmp_path, *extra, name="run"):
    code = procedure.main(["--mock", "--out-dir", str(tmp_path), "--run-name", name, *extra])
    return code, tmp_path / name


def events(run_dir, kind):
    return [e for e in map(json.loads, (run_dir / "events.jsonl").read_text().splitlines()) if e["event_type"] == kind]


def test_mock_ldm_runs_through_shared_campaign_with_exact_topology(tmp_path, monkeypatch):
    monkeypatch.setenv("LDM_DATA_COLLECTION_ENABLED", "1")
    code, run_dir = run(tmp_path, "--case", "continual_learning", "--iterations", "3", "--proposal-samples", "6",
                        "--proposal-batch-size", "3", "--bo-pool-size", "4", "--evaluations-per-round", "2")
    assert code == 0
    for name in ("events.jsonl", "checkpoint.json", "summary.json", "status.json", "budget.json", "ldm_task_spec.json",
                 "experiment_contract.json", "result.json", "trajectory.csv", "evaluations.csv", "best_program.py"):
        assert (run_dir / name).exists(), name
    budget = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert budget["outer_iterations"] == 3 and budget["external_evaluations"] == 6
    # The synthetic client is a local source: parsed and repaired, never charged as model requests.
    assert budget["llm_requests"] == budget["proposal_attempts"] == 0
    diagnostics = json.loads((run_dir / "proposal_diagnostics/round-000000.json").read_text())
    assert diagnostics["collections"][0]["minibatches"][0]["requests"] == 2  # one in-exchange repair
    config = json.loads((run_dir / "config.json").read_text())
    assert config["proposal_counting"] == "bounded_minibatches" and config["proposal_candidates_per_request"] == 3
    assert budget["api_cost_usd"] == 0 and budget["llm_failed_requests"] == 0
    for selection in events(run_dir, "candidates_selected"):
        metadata = selection["payload"]["metadata"]
        assert metadata["valid_proposal_occurrences"] == 6 and metadata["bo_pool_size"] <= 4
        assert abs(sum(row["q0"] for row in metadata["distribution"]) - 1) < 1e-9
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "completed" and result["protocol"] is None and result["comparability"] == "synthetic mock fixture"
    rows = [json.loads(line) for line in (run_dir / "ldm_data/ldm_ir.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        validate_ir_record(row)
    sft = (run_dir / "ldm_data/ldm_sft.jsonl").read_text()
    assert "synthetic_fixture" not in sft and "round_idx\": 0" not in sft


def test_resume_of_completed_campaign_replays_without_new_requests_or_evaluations(tmp_path):
    code, run_dir = run(tmp_path, "--iterations", "2", "--proposal-samples", "4", "--bo-pool-size", "2")
    assert code == 0
    before = json.loads((run_dir / "budget.json").read_text())["counters"]
    code = procedure.main(["--mock", "--iterations", "2", "--proposal-samples", "4", "--bo-pool-size", "2",
                           "--resume-from", str(run_dir)])
    after = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert code == 0
    for counter in ("llm_requests", "external_evaluations", "outer_iterations", "benchmark_jobs"):
        assert after[counter] == before[counter]


def test_resume_rejects_changed_scientific_configuration(tmp_path):
    _, run_dir = run(tmp_path, "--iterations", "1", "--proposal-samples", "2", "--bo-pool-size", "2")
    with pytest.raises(ValueError, match="scientific configuration"):
        procedure.main(["--mock", "--iterations", "1", "--proposal-samples", "2", "--bo-pool-size", "2",
                        "--acquisition-eta", "2", "--resume-from", str(run_dir)])


def test_llm_method_evaluates_the_direct_batch_in_order(tmp_path):
    code, run_dir = run(tmp_path, "--search-method", "llm", "--iterations", "2", "--evaluations-per-round", "2")
    assert code == 0
    selections = events(run_dir, "candidates_selected")
    assert all(s["payload"]["metadata"]["mode"] == "reservoir_order" for s in selections)
    spec = json.loads((run_dir / "ldm_task_spec.json").read_text())
    assert spec["surrogate"]["kind"] == "none"


def test_shared_start_evaluates_the_released_seed_first(tmp_path):
    code, run_dir = run(tmp_path, "--initialization-mode", "shared_start", "--iterations", "3",
                        "--proposal-samples", "4", "--bo-pool-size", "2")
    assert code == 0
    first = events(run_dir, "candidate_evaluated")[0]["payload"]["candidate"]
    assert first["source"] == "campaign_initialization"
    assert json.loads((run_dir / "budget.json").read_text())["counters"]["external_evaluations"] == 3


def test_mock_ldm_harness_uses_protocol_fixture(tmp_path):
    command = json.dumps([sys.executable, "-u", str(FIXTURE), "repair"])
    code, run_dir = run(tmp_path, "--case", "cross_modal_retrieval", "--search-method", "ldm_harness",
                        "--iterations", "2", "--proposal-samples", "4", "--bo-pool-size", "3",
                        "--harness-sessions", "2", "--harness-command-json", command)
    assert code == 0
    budget = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert budget["harness_turns"] == 4 and budget["harness_validation_submissions"] == 8
    assert budget["harness_input_tokens"] == 4000 and budget["llm_requests"] == 0
    assert json.loads((run_dir / "harness_provenance.json").read_text())["pools"] == {"harness": ["harness/manifest.json"]}


def test_unsupported_and_inconsistent_methods_are_rejected():
    with pytest.raises(SystemExit):
        workflow.parse_args(["--search-method", "bo"])
    with pytest.raises(SystemExit):
        workflow.parse_args(["--search-method", "harness", "--harness-sessions", "2"])
    with pytest.raises(SystemExit):
        workflow.parse_args(["--proposal-mode", "mock"])
    with pytest.raises(SystemExit):
        workflow.parse_args(["--api-budget-usd", "5"])


def test_dry_run_describes_runtime_faithful_spec(capsys):
    assert procedure.main(["--dry-run", "--case", "improving_replay_buffers", "--search-method", "ldm_harness_compiled",
                           "--mock", "--harness-command-json", "[\"x\"]"]) == 0
    spec = json.loads(capsys.readouterr().out)["ldm_task_spec"]
    assert spec["acquisition"]["name"] == "ldm_rbf_gp_ucb"
    assert spec["metadata"]["policy_capabilities"] == ["prior_mean@1", "ldm_weights@1"]
    assert spec["candidate_domain"]["constraints"]["case"] == "improving_replay_buffers"


@pytest.mark.parametrize("case_id", ["time_series_explanation", "continual_learning", "cross_modal_retrieval",
                                     "improving_replay_buffers"])
def test_every_case_runs_the_mock_pipeline_with_its_own_slot(tmp_path, case_id):
    code, run_dir = run(tmp_path, "--case", case_id, "--initialization-mode", "shared_start", "--iterations", "2",
                        "--proposal-samples", "4", "--bo-pool-size", "2", name=case_id)
    assert code == 0
    evaluated = events(run_dir, "candidate_evaluated")
    assert len(evaluated) == 2 and evaluated[0]["payload"]["candidate"]["source"] == "campaign_initialization"
    entry = workflow.load_case(case_id).entry["symbol"]
    assert all(entry in e["payload"]["candidate"]["payload"]["program"] for e in evaluated)
