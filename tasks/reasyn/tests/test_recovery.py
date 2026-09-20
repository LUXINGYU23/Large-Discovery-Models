import json
from pathlib import Path
import pytest
from ldm_tts.transport import ProposalResponse
from ldm_tts.transport.openai import EndpointRequestError
from tasks.reasyn.core import workflow


def test_endpoint_failure_resumes_original_round_and_scientific_budget(tmp_path, monkeypatch):
    calls = []
    class Client:
        def __init__(self, **kwargs): pass
        def preflight(self): return {"ok": True}
        def propose(self, request):
            calls.append(request.metadata["round_idx"])
            if len(calls) == 1:
                raise EndpointRequestError("temporary fixture interruption")
            return ProposalResponse(text=json.dumps({"candidates": [{"target_smiles": "CCO"}]}))
    monkeypatch.setattr(workflow, "OpenAICompatibleProposalClient", Client)
    root = tmp_path / "run"
    argv = ["--mock", "--proposal-mode", "openai", "--llm-url", "http://fixture.invalid/v1",
        "--llm-model", "fixture", "--iterations", "2", "--reservoir-size", "1", "--out-dir", str(root)]
    assert workflow.main(argv) == 2
    assert workflow.main(argv + ["--resume-from", str(root)]) == 0
    budget = json.loads((root / "budget.json").read_text())
    assert calls == [0, 0, 1]
    assert budget["counters"]["outer_iterations"] == 2
    assert budget["limits"]["outer_iterations"] == 2
    assert budget["counters"]["llm_requests"] == 3
    assert budget["counters"]["recovery_attempts"] == 1
    assert budget["counters"]["successful_evaluations"] == 2
    assert budget["limits"]["expensive_evaluation_attempts"] == 2
    assert json.loads((root / "status.json").read_text())["status"] == "completed"


def test_recovery_allowance_is_bounded(tmp_path, monkeypatch):
    class Client:
        def __init__(self, **kwargs): pass
        def preflight(self): return {"ok": True}
        def propose(self, request): raise EndpointRequestError("fixture interruption")
    monkeypatch.setattr(workflow, "OpenAICompatibleProposalClient", Client)
    root = tmp_path / "run"
    argv = ["--mock", "--proposal-mode", "openai", "--llm-url", "http://fixture.invalid/v1",
        "--llm-model", "fixture", "--iterations", "2", "--reservoir-size", "1",
        "--recovery-attempts", "1", "--out-dir", str(root)]
    assert workflow.main(argv) == 2
    assert workflow.main(argv + ["--resume-from", str(root)]) == 2
    assert workflow.main(argv + ["--resume-from", str(root)]) == 1
    budget = json.loads((root / "budget.json").read_text())
    assert budget["counters"]["llm_requests"] == 2
    assert budget["counters"]["recovery_attempts"] == 1
    assert budget["counters"]["successful_evaluations"] == 0


def test_completed_resume_does_not_recontact_an_offline_provider(tmp_path, monkeypatch):
    class Client:
        preflights = 0

        def __init__(self, **kwargs): pass

        def preflight(self):
            Client.preflights += 1
            if Client.preflights > 1:
                raise EndpointRequestError("endpoint is offline after completion")
            return {"ok": True}

        def propose(self, request):
            return ProposalResponse(text=json.dumps({"candidates": [{"target_smiles": "CCO"}]}))

    monkeypatch.setattr(workflow, "OpenAICompatibleProposalClient", Client)
    root = tmp_path / "completed"
    argv = ["--mock", "--proposal-mode", "openai", "--llm-url", "http://fixture.invalid/v1",
        "--llm-model", "fixture", "--iterations", "1", "--reservoir-size", "1", "--out-dir", str(root)]
    assert workflow.main(argv) == 0
    assert workflow.main(argv + ["--resume-from", str(root)]) == 0
    assert Client.preflights == 1
    assert json.loads((root / "status.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("benchmark", ["reconstruction", "tdc"])
def test_paired_initialization_and_pilot_aliases(tmp_path, benchmark):
    initial_candidates = []
    for method in ("ldm", "bo", "llm"):
        argv = ["--mock", "--benchmark", benchmark, "--search-method", method,
            "--proposal-mode", "none" if method == "bo" else "callable",
            "--campaign-index", "1", "--initialization-mode", "shared_start", "--iterations", "2",
            "--reservoir-size", "2" if method == "llm" else "4",
            "--evaluations-per-round", "1", "--out-dir", str(tmp_path), "--run-name", method]
        assert workflow.main(argv) == 0
        checkpoint = json.loads((tmp_path / method / "checkpoint.json").read_text())
        observations = checkpoint["state"]["observations"]
        assert len(observations) == 2
        initial_candidates.append(observations[0]["candidate"]["candidate_id"])
        with pytest.raises(FileExistsError):
            workflow.main(argv)
    assert len(set(initial_candidates)) == 1


def test_original_baseline_does_not_need_a_bo_pool(tmp_path):
    assert workflow.main(["--mock", "--proposal-mode", "baseline", "--iterations", "2",
        "--reservoir-size", "1", "--out-dir", str(tmp_path / "baseline")]) == 0


def test_tdc_projection_interruption_resumes_only_pending_occurrences(tmp_path, monkeypatch):
    from tasks.reasyn.core.projector import Projector, ProjectionInterruptedError
    from tasks.reasyn.core.projection_worker import write_progress

    class InterruptingProjector(Projector):
        interrupted = False

        def project(self, targets, **kwargs):
            rows, artifact = super().project(targets, **kwargs)
            if not InterruptingProjector.interrupted:
                InterruptingProjector.interrupted = True
                output = self.run_dir / artifact
                request = json.loads((output.parent / "request.json").read_text())
                write_progress(output, request, rows[:1], {0}, mock=True)
                raise ProjectionInterruptedError(
                    "fixture worker stopped after one target",
                    completed_targets=1, target_count=len(targets), artifact=artifact,
                )
            return rows, artifact

    monkeypatch.setattr(workflow, "Projector", InterruptingProjector)
    root = tmp_path / "projection-resume"
    argv = ["--mock", "--benchmark", "tdc", "--iterations", "1",
        "--reservoir-size", "2", "--evaluations-per-round", "2", "--out-dir", str(root)]
    assert workflow.main(argv) == 2
    assert json.loads((root / "status.json").read_text())["status"] == "paused_projection_interrupted"
    assert workflow.main(argv + ["--resume-from", str(root)]) == 0
    budget = json.loads((root / "budget.json").read_text())["counters"]
    assert budget["projection_targets"] == 3
    assert budget["successful_evaluations"] == budget["oracle_calls"] == 2
    assert budget["outer_iterations"] == 1


def test_reconstruction_preparation_resumes_same_selection_without_spending_trials(tmp_path, monkeypatch):
    from tasks.reasyn.core.projector import Projector, ProjectionInterruptedError

    class Client:
        requests = 0

        def __init__(self, **kwargs): pass
        def preflight(self): return {"ok": True}

        def propose(self, request):
            Client.requests += 1
            return ProposalResponse(text=json.dumps({"candidates": [
                {"target_smiles": "CCO"}, {"target_smiles": "CCN"},
            ]}))

    class InterruptingProjector(Projector):
        preparation_calls = 0
        failed = False

        def project(self, targets, **kwargs):
            InterruptingProjector.preparation_calls += 1
            if InterruptingProjector.preparation_calls == 2 and not InterruptingProjector.failed:
                InterruptingProjector.failed = True
                self.before_project(len(targets))
                raise ProjectionInterruptedError(
                    "fixture stopped while preparing the second selected query",
                    completed_targets=0, target_count=1, artifact="unwritten-result.json",
                )
            return super().project(targets, **kwargs)

    monkeypatch.setattr(workflow, "OpenAICompatibleProposalClient", Client)
    monkeypatch.setattr(workflow, "Projector", InterruptingProjector)
    root = tmp_path / "reconstruction-resume"
    argv = ["--mock", "--proposal-mode", "openai", "--llm-url", "http://fixture.invalid/v1",
        "--llm-model", "fixture", "--iterations", "1", "--reservoir-size", "2",
        "--evaluations-per-round", "2", "--out-dir", str(root)]
    assert workflow.main(argv) == 2
    before = json.loads((root / "budget.json").read_text())["counters"]
    assert before["expensive_evaluation_attempts"] == 0
    assert before["successful_evaluations"] == 0
    assert before["projection_targets"] == 2
    assert len(list((root / "projections").glob("*/result.json"))) == 1
    assert workflow.main(argv + ["--resume-from", str(root)]) == 0
    after = json.loads((root / "budget.json").read_text())["counters"]
    assert after["expensive_evaluation_attempts"] == after["successful_evaluations"] == 2
    assert after["projection_targets"] == 3
    assert after["valid_search_candidates"] == 2
    assert after["llm_requests"] == after["proposal_attempts"] == Client.requests == 1
    assert after["outer_iterations"] == after["recovery_attempts"] == 1
    selections = [event["payload"]["selected_candidate_ids"]
        for event in map(json.loads, (root / "events.jsonl").read_text().splitlines())
        if event["event_type"] == "candidates_selected"]
    assert len(selections) == 2 and selections[0] == selections[1]


@pytest.mark.parametrize("interrupt_round", [0, 1])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("window,error_type", [
    ("after_evaluation", KeyboardInterrupt), ("before_checkpoint", KeyboardInterrupt),
    ("before_checkpoint", RuntimeError),
])
def test_completed_reconstruction_trials_resume_without_duplicate_scientific_charges(
    tmp_path, monkeypatch, interrupt_round, batch_size, window, error_type
):
    from ldm_tts.engine.runtime import LDMEngine
    from tasks.reasyn.core.evaluator import ReconstructionEvaluator
    from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY

    interrupted = False
    evaluate = ReconstructionEvaluator.evaluate
    checkpoint = LDMEngine._checkpoint
    evaluations = 0
    def after_evaluation(self, candidate):
        nonlocal interrupted, evaluations
        result = evaluate(self, candidate)
        evaluations += 1
        if window == "after_evaluation" and evaluations == (interrupt_round + 1) * batch_size and not interrupted:
            interrupted = True
            raise error_type("completed reconstruction trial interrupted")
        return result
    def before_checkpoint(self, state):
        nonlocal interrupted
        if window == "before_checkpoint" and state.next_round == interrupt_round + 1 and not interrupted:
            interrupted = True
            raise error_type("completed reconstruction trial interrupted")
        return checkpoint(self, state)
    monkeypatch.setattr(ReconstructionEvaluator, "evaluate", after_evaluation)
    monkeypatch.setattr(LDMEngine, "_checkpoint", before_checkpoint)
    root = tmp_path / "reconstruction"
    argv = ["--mock", "--benchmark", "reconstruction", "--iterations", "2",
            "--reservoir-size", str(batch_size), "--evaluations-per-round", str(batch_size), "--out-dir", str(root)]
    with pytest.raises(error_type):
        workflow.main(argv)
    before = json.loads((root / "budget.json").read_text())
    assert before["counters"]["external_evaluations"] == (interrupt_round + 1) * batch_size
    assert workflow.main(argv + ["--resume-from", str(root)]) == 0
    after = json.loads((root / "budget.json").read_text())
    assert after["limits"] == before["limits"]
    for name in ("projection_targets", "selected_candidates", "external_evaluations",
                 "expensive_evaluation_attempts", "successful_evaluations", "benchmark_jobs"):
        assert after["counters"][name] == 2 * batch_size
    state = json.loads((root / "checkpoint.json").read_text())["state"]
    assert len(state["observations"]) == 2 * batch_size
    assert state["next_round"] == 2
    keys = [row["evaluation"]["metadata"][EVALUATION_ATTEMPT_RECEIPT_KEY] for row in state["observations"]]
    assert len(set(keys)) == 2 * batch_size
    assert json.loads((root / "status.json").read_text())["status"] == "completed"
    assert json.loads((root / "result.json").read_text())["successful_evaluations"] == 2 * batch_size


def test_mock_pilot_matrix_completes_with_shared_initialization_and_no_model_cost(tmp_path, monkeypatch):
    from ldm_tts.cli.runner import load_config
    from ldm_tts.pilot_evaluation.config import load_pilot_evaluation_spec
    from ldm_tts.pilot_evaluation.execution import run_evaluation

    # The synthetic matrix must also run from a source archive without .git.
    # Repository provenance enforcement is covered by the shared pilot tests.
    monkeypatch.setattr("ldm_tts.pilot_evaluation.execution._repository_state", lambda: {
        "commit": "synthetic-task-test", "dirty": False, "source": "test_fixture",
    })
    # Plotting is independently covered by shared pilot tests; this task-level
    # integration test keeps the declared NumPy/PyYAML/pytest environment sufficient.
    monkeypatch.setattr("ldm_tts.pilot_evaluation.reporting._plot", lambda *args: None)
    repo = Path(__file__).resolve().parents[3]
    base = load_config(repo / "config/reasyn/pilot_evaluation_base.yaml")
    base["mode"] = "mock"
    base["args"].update({"mock": True, "proposal-samples": 4,
        "proposal-batch-size": 2, "bo-pool-size": 4, "evaluations-per-round": 2})
    base["args"].pop("bo-targets-file", None)
    base["env"] = {"LDM_DATA_COLLECTION_ENABLED": "0"}
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    spec = load_config(repo / "config/pilot_evaluation/reasyn.yaml")
    spec.update({"base_config": str(base_path), "methods": ["ldm", "bo", "llm"],
        "method_overrides": {name: spec["method_overrides"][name] for name in ("ldm", "bo", "llm")},
        "optimization_rounds": 2, "output_root": str(tmp_path / "evaluation")})
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    assert run_evaluation(load_pilot_evaluation_spec(spec_path), resume=False, dry_run=False) == 0
    manifest = json.loads((tmp_path / "evaluation/evaluation_manifest.json").read_text())
    assert manifest["state"] == "completed"
    assert manifest["integrity"] == {"errors": [], "valid": True}
    assert len(manifest["runs"]) == 9
    assert (tmp_path / "evaluation/summary.json").is_file()
    for run in manifest["runs"].values():
        root = tmp_path / "evaluation" / run["run_dir"]
        budget = json.loads((root / "budget.json").read_text())["counters"]
        assert budget["llm_requests"] == budget["proposal_attempts"] == 0
        assert budget["successful_evaluations"] == budget["oracle_calls"] == 6
