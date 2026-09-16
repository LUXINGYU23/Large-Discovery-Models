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
