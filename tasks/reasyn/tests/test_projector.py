"""Serial projection budgets and durable occurrence-level resume behavior."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.reasyn.core.projector import Projector, ProjectionInterruptedError
from tasks.reasyn.core.projection_worker import load_progress, write_progress
from tasks.reasyn.core.workflow import parse_args


def _row(request, index):
    return {
        "target": request["targets"][index],
        "target_index": index,
        "sampling_seed": request["sampling_seed"] + index,
        "smiles": request["targets"][index],
        "synthesis": request["targets"][index],
        "num_steps": 0,
        "pathway_verified": True,
    }


def test_batch_timeout_scales_with_pending_serial_targets_and_preserves_progress(
    tmp_path, monkeypatch
):
    args = parse_args(["--mock"])
    args.mock = False
    projector = Projector(args, tmp_path)
    # Chemistry is orthogonal to testing the subprocess boundary here.
    monkeypatch.setattr("tasks.reasyn.core.chemistry.canonicalize", lambda value, **kwargs: value)
    charged = []
    projector.before_project = charged.append
    timeouts = []

    def run(command, **kwargs):
        request = json.loads(Path(command[command.index("--request") + 1]).read_text())
        output = Path(command[command.index("--output") + 1])
        timeouts.append(kwargs["timeout"])
        if len(timeouts) == 1:
            # One product and one completed empty search survive a timeout on
            # the third occurrence, even though its query repeats the first.
            write_progress(output, request, [_row(request, 0)], {0, 1})
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        partial = load_progress(output, request)
        assert partial["completed_target_indices"] == [0, 1]
        write_progress(output, request, partial["rows"] + [_row(request, 2)], {0, 1, 2})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tasks.reasyn.core.projector.subprocess.run", run)
    with pytest.raises(ProjectionInterruptedError, match="2/3 targets checkpointed") as error:
        projector.project(["CCO", "CCN", "CCO"], sampling_seed=20, identity="round-0")
    assert error.value.completed_targets == 2
    rows, artifact = projector.project(["CCO", "CCN", "CCO"], sampling_seed=20, identity="round-0")
    assert [(row["target_index"], row["sampling_seed"]) for row in rows] == [(0, 20), (2, 22)]
    assert timeouts == [5600, 3600]
    assert charged == [3, 1]
    assert (tmp_path / artifact).exists()
    assert projector.project(["CCO", "CCN", "CCO"], sampling_seed=20, identity="round-0")[0] == rows
    assert charged == [3, 1]


def test_interrupted_request_cannot_be_overwritten_with_another_seed(tmp_path, monkeypatch):
    args = parse_args(["--mock"])
    args.mock = False
    projector = Projector(args, tmp_path)
    monkeypatch.setattr(
        "tasks.reasyn.core.projector.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    with pytest.raises(ProjectionInterruptedError, match="0/1 targets checkpointed"):
        projector.project(["CCO"], sampling_seed=2, identity="query")
    with pytest.raises(ValueError, match="resume request mismatch"):
        projector.project(["CCO"], sampling_seed=3, identity="query")


def test_mock_projects_repeated_query_occurrences_independently(tmp_path):
    projector = Projector(parse_args(["--mock"]), tmp_path)
    rows, _ = projector.project(["CCO", "CCO"], sampling_seed=10, identity="repeated")
    assert [(row["target_index"], row["sampling_seed"]) for row in rows] == [(0, 10), (1, 11)]


def test_incomplete_success_is_resumable_and_never_returns_a_short_batch(tmp_path, monkeypatch):
    args = parse_args(["--mock"])
    args.mock = False
    projector = Projector(args, tmp_path)

    def run(command, **kwargs):
        request = json.loads(Path(command[command.index("--request") + 1]).read_text())
        output = Path(command[command.index("--output") + 1])
        write_progress(output, request, [], {0})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tasks.reasyn.core.projector.subprocess.run", run)
    with pytest.raises(ProjectionInterruptedError, match="1/2 targets checkpointed"):
        projector.project(["CCO", "CCN"], sampling_seed=0, identity="short")


def test_reconstruction_receipt_binds_target_query_seed_configuration_and_completed_output(tmp_path):
    from ldm_tts.contracts import RawProposal
    from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY
    from tasks.reasyn.core.candidate import ReaSynDomain
    from tasks.reasyn.core.evaluator import ReconstructionEvaluator

    args = parse_args(["--mock"])
    projector = Projector(args, tmp_path)
    charged = []
    projector.before_project = charged.append
    domain = ReaSynDomain("reconstruction", mock=True)
    trials = [domain.admit(RawProposal({"target_smiles": query, "sampling_seed": seed}, "fixture"))
              for query, seed in (("CCO", 1), ("CCO", 2), ("CCN", 1))]
    evaluator = ReconstructionEvaluator(args, projector, "CCO")
    with pytest.raises(ValueError, match="receipt request mismatch"):
        evaluator.evaluation_attempt_usage_key(trials[0])
    assert charged == []  # A key lookup must never start a projection.
    evaluator.prepare_evaluations(trials)
    keys = [evaluator.evaluation_attempt_usage_key(c) for c in trials]
    assert len(set(keys)) == 3
    assert ReconstructionEvaluator(args, projector, "CCN").evaluation_attempt_usage_key(trials[0]) != keys[0]
    first = evaluator.evaluate(trials[0])
    assert first.metadata[EVALUATION_ATTEMPT_RECEIPT_KEY] == keys[0]
    assert evaluator.evaluate(trials[0]) == first
    assert charged == [1, 1, 1]
    args.num_cycles += 1
    with pytest.raises(ValueError, match="receipt request mismatch"):
        evaluator.evaluation_attempt_usage_key(trials[0])
    args.num_cycles -= 1
    args.asset_digests = {"checkpoint": "changed"}
    with pytest.raises(ValueError, match="receipt request mismatch"):
        evaluator.evaluation_attempt_usage_key(trials[0])
    args.asset_digests = {}
    output = tmp_path / first.artifacts["projection"]
    request = json.loads((output.parent / "request.json").read_text())
    write_progress(output, request, [], set(), mock=True)
    with pytest.raises(ValueError, match="requires a completed result"):
        evaluator.evaluation_attempt_usage_key(trials[0])
    write_progress(output, request, [], {0}, mock=True)
    empty_key = evaluator.evaluation_attempt_usage_key(trials[0])
    assert empty_key != keys[0]  # Empty-but-complete projections are also identifiable trials.
    empty = evaluator.evaluate(trials[0])
    assert empty.metrics["output_count"] == empty.metrics["similarity"] == 0
    assert empty.metadata[EVALUATION_ATTEMPT_RECEIPT_KEY] == empty_key
    assert charged == [1, 1, 1]
