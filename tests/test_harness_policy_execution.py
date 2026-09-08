from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ldm_tts.harness.policy_execution import (
    DockerPolicyExecutor,
    PolicyExecutionError,
)


def _inputs(tmp_path: Path, objective_shape=()) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "optimization_policy.py"
    artifact.write_text("# generated policy\n", encoding="utf-8")
    round_input = tmp_path / "round"
    round_input.mkdir()
    (round_input / "contract.json").write_text("{}", encoding="utf-8")
    (round_input / "input.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(
        round_input / "arrays.npz",
        history_features=np.zeros((2, 3)),
        history_utilities=np.zeros((2, *objective_shape)),
        query_features=np.zeros((4, 3)),
    )
    return artifact, round_input, tmp_path / "output"


@pytest.mark.parametrize("objective_shape", [(), (2,)])
def test_docker_policy_executor_uses_hardened_container(monkeypatch, tmp_path: Path, objective_shape) -> None:
    artifact, round_input, output = _inputs(tmp_path, objective_shape)
    captured: dict[str, object] = {}

    def run(command, **kwargs):
        if "rm" in command:
            captured["cleanup"] = command
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        captured["command"] = command
        captured["kwargs"] = kwargs
        output.mkdir(exist_ok=True)
        np.savez_compressed(
            output / "arrays.npz",
            history_prior_mean=np.full((2, *objective_shape), 0.1),
            query_prior_mean=np.full((4, *objective_shape), 0.3),
        )
        (output / "result.json").write_text(
            json.dumps({
                "status": "ok",
                "stage": "explore",
                "alpha": 0.7,
                "eta": 1.3,
                "prior_clip_count": 2,
            }),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="{}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)

    result = DockerPolicyExecutor("ldm-pi:test", timeout_seconds=10).execute(
        artifact,
        round_input,
        output,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert ["--network", "none"] == command[command.index("--network"):command.index("--network") + 2]
    for expected in ("--read-only", "--cap-drop", "--pids-limit", "--memory", "--cpus"):
        assert expected in command
    assert "no-new-privileges=true" in command
    assert "OPENBLAS_NUM_THREADS=1" in command
    assert captured["cleanup"] == ["docker", "rm", "--force", command[command.index("--name") + 1]]
    assert captured["kwargs"]["timeout"] == 10
    assert result.stage == "explore"
    assert result.query_prior_mean.shape == (4, *objective_shape)
    assert result.prior_clip_count == 2


def test_docker_policy_executor_returns_runner_errors(monkeypatch, tmp_path: Path) -> None:
    artifact, round_input, output = _inputs(tmp_path)

    def run(command, **_kwargs):
        if "rm" in command:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        payload = {
            "status": "error",
            "errors": [{
                "path": "/exports/compute_prior_mean",
                "code": "non_finite_output",
                "message": "prior contains NaN",
                "hint": "check normalization",
            }],
        }
        return subprocess.CompletedProcess(
            command,
            2,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(PolicyExecutionError) as caught:
        DockerPolicyExecutor("ldm-pi:test").execute(artifact, round_input, output)

    assert caught.value.errors[0].code == "non_finite_output"
    assert caught.value.errors[0].path == "/exports/compute_prior_mean"


def test_docker_timeout_removes_container_on_the_configured_daemon(monkeypatch, tmp_path):
    artifact, round_input, output = _inputs(tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "run" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(PolicyExecutionError) as caught:
        DockerPolicyExecutor("fixture", docker_host="unix:///fixture.sock").execute(
            artifact, round_input, output,
        )
    assert caught.value.errors[0].code == "runner_timeout"
    name = commands[0][commands[0].index("--name") + 1]
    assert commands[1] == ["docker", "--host", "unix:///fixture.sock", "rm", "--force", name]


def test_docker_launch_failure_preserves_the_structured_error(monkeypatch, tmp_path):
    artifact, round_input, output = _inputs(tmp_path)
    calls = []
    def unavailable(command, **kwargs):
        calls.append(command)
        raise FileNotFoundError("docker")
    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(PolicyExecutionError) as caught:
        DockerPolicyExecutor("fixture").execute(artifact, round_input, output)
    assert caught.value.errors[0].code == "runner_unavailable"
    assert len(calls) == 1
