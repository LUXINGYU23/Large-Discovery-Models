"""Real workflow wiring with deterministic provider and oracle boundaries."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ldm_tts.engine.reporting import load_successful_observations
from ldm_tts.harness import HarnessError
from tasks.nucleobench.core import harness_runtime, workflow
from tasks.nucleobench.core.candidate import MutationContext
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.oracles.official import OfficialAPI, PreparedCase, RecordingSequenceModel
from tasks.nucleobench.tests.test_harness import FakeHarnessClient


@pytest.fixture
def real_workflow(tmp_path, monkeypatch):
    case = get_case("malinois_k562")
    context = MutationContext(case, "a" * 64, 0, "A" * case.sequence_length, tuple(range(case.sequence_length)))
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"deterministic fixture")
    prepared = PreparedCase(context, artifact, {})
    state = SimpleNamespace(interrupt=False, model_loads=0, driver_calls=0, client_events=[])
    monkeypatch.setattr(workflow, "load_prepared_case", lambda *args, **kwargs: prepared)
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "fixture")
    monkeypatch.setenv("LLM_API_KEY", "fixture-key")

    def run_loop(*, model, opt, all_args, ignore_errors):
        state.driver_calls += 1
        assert not ignore_errors
        output = Path(all_args.main_args.output_path)
        output.mkdir(parents=True)
        (output / "START.txt").write_text("START")
        for _ in range(all_args.main_args.max_number_of_rounds):
            opt.run(1)
            model(opt.get_samples(1))
            if state.interrupt:
                state.interrupt = False
                raise RuntimeError("injected driver interruption")
        (output / "results.parquet").write_bytes(b"fixture result")
        (output / "SUCCESS.txt").write_text("SUCCESS")

    def load_official(_source, _prepared, runtime, *, oracle_socket=None):
        state.model_loads += 1
        model = RecordingSequenceModel(
            lambda sequences: [-sum(index + 1 for index, base in enumerate(sequence) if base != "A")
                               for sequence in sequences], runtime,
        )
        return OfficialAPI(model, run_loop, SimpleNamespace)

    monkeypatch.setattr(workflow, "load_official_case", load_official)
    state.argv = [
        "--case-id", case.case_id, "--iterations", "3",
        "--source-dir", str(tmp_path / "source"), "--prepared-dir", str(tmp_path / "prepared"),
        "--expected-start-set-sha256", " " + context.start_set_digest + " ",
        "--out-dir", str(tmp_path),
    ]
    return state


def _tree_client(args, runtime, provider, prepared, profiles, mcp, **kwargs):
    root = kwargs["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "taskId": "nucleobench", "campaignId": runtime.run_id,
        "profiles": [{"profileId": profile.profile_id} for profile in profiles],
    }))

    def save_session(profile, round_index):
        session = root / "sessions" / profile
        (session / "workspace").mkdir(parents=True, exist_ok=True)
        (session / "workspace/research.md").write_text(f"research round {round_index}")
        (session / "session.jsonl").write_text(f"native session round {round_index}")
        return f"sessions/{profile}/session.jsonl"

    if profiles[0].profile_id == "main_research":
        class Main:
            config = SimpleNamespace(limits=SimpleNamespace(wall_time_seconds=120))

            def __enter__(self): return self
            def __exit__(self, *exc): pass

            def run_turn(self, turns, **options):
                turn, = turns
                session = save_session("main_research", turn.round_index)
                return (SimpleNamespace(
                    usage={"providerCalls": 3}, submission_status="accepted",
                    artifacts={"session": session}, submission={"brief": "measured evidence"},
                ),)

        return Main()

    class Branch(FakeHarnessClient):
        def __enter__(self): return self
        def __exit__(self, *exc): pass

        def run_turn(self, turns, **options):
            return tuple(replace(result, artifacts={**result.artifacts,
                         "session": save_session(result.profile_id, result.round_index)})
                         for result in super().run_turn(turns, **options))

    round_index = int(root.name.removeprefix("round_"))
    return Branch(root, {profile.profile_id: [[{"mutations": [{"position": round_index * 4 + index, "base": "C"}]}]]
                         for index, profile in enumerate(profiles)})


@pytest.mark.parametrize("method", ["bo", "ldm_harness"])
def test_real_workflow_resume_preserves_search_budget_and_completed_results(
    tmp_path, monkeypatch, capsys, real_workflow, method,
):
    state = real_workflow
    monkeypatch.setattr(harness_runtime, "create_client", _tree_client)
    argv = [*state.argv, "--search-method", method]
    assert workflow.main([*argv, "--run-name", "uninterrupted"]) == 0
    reference = Path(json.loads(capsys.readouterr().out)["run_dir"])

    state.interrupt = True
    with pytest.raises(RuntimeError, match="injected driver interruption"):
        workflow.main([*argv, "--run-name", "resumed"])
    run_dir = tmp_path / "resumed"
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    assert checkpoint["state"]["next_round"] == 2
    assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"
    assert workflow.main([*argv, "--resume-from", str(run_dir)]) == 0
    capsys.readouterr()

    def observed(path):
        return [(row["candidate"]["candidate_id"], row["evaluation"]["metrics"])
                for row in load_successful_observations(path / "checkpoint.json")]

    assert observed(run_dir) == observed(reference)
    budget = json.loads((run_dir / "budget.json").read_text())["counters"]
    assert budget["initialization_evaluations"] == 1
    assert budget["outer_iterations"] == 3
    assert budget["external_evaluations"] == 2
    if method == "ldm_harness":
        assert budget["main_research_turns"] == 2
        assert budget["harness_turns"] == 10
        assert budget["llm_requests"] == 22

    protected = {name: (run_dir / name).read_bytes() for name in (
        "checkpoint.json", "budget.json", "summary.json", "result.json", "trajectory.csv",
    )}
    loads, calls = state.model_loads, state.driver_calls
    monkeypatch.setattr(harness_runtime, "create_client", lambda *a, **kw: pytest.fail("completed run opened a client"))
    assert workflow.main([*argv, "--resume-from", str(run_dir)]) == 0
    assert (state.model_loads, state.driver_calls) == (loads, calls)
    assert all((run_dir / name).read_bytes() == value for name, value in protected.items())


@pytest.mark.parametrize("error_type", [ValueError, HarnessError])
def test_partial_harness_setup_always_closes_open_clients(
    tmp_path, monkeypatch, capsys, real_workflow, error_type,
):
    state = real_workflow

    class ProposalClient:
        def __enter__(self):
            state.client_events.append("opened")
            return self

        def __exit__(self, *exc):
            state.client_events.append("closed")

    def create_client(*args, **kwargs):
        if kwargs.get("policy"):
            raise error_type("policy setup failed")
        return ProposalClient()

    monkeypatch.setattr(harness_runtime, "create_client", create_client)
    argv = [*state.argv, "--run-name", "setup-failure", "--search-method", "ldm_harness_compiled",
            "--harness-research-strategy-config", "config/research_strategies/independent.json"]
    if error_type is HarnessError:
        assert workflow.main(argv) == 2
        capsys.readouterr()
        status = "paused_endpoint_unavailable"
    else:
        with pytest.raises(error_type, match="policy setup failed"):
            workflow.main(argv)
        status = "failed"
    assert state.client_events == ["opened", "closed"]
    assert state.model_loads == 0
    assert json.loads((tmp_path / "setup-failure/status.json").read_text())["status"] == status
