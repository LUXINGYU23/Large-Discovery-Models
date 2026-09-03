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


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "optimization_policy.py"
    artifact.write_text("# generated policy\n", encoding="utf-8")
    round_input = tmp_path / "round"
    round_input.mkdir()
    (round_input / "contract.json").write_text("{}", encoding="utf-8")
    (round_input / "input.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(
        round_input / "arrays.npz",
        history_features=np.zeros((2, 3)),
        history_utilities=np.zeros(2),
        query_features=np.zeros((4, 3)),
    )
    return artifact, round_input, tmp_path / "output"


def test_docker_policy_executor_uses_hardened_container(monkeypatch, tmp_path: Path) -> None:
    artifact, round_input, output = _inputs(tmp_path)
    captured: dict[str, object] = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output.mkdir(exist_ok=True)
        np.savez_compressed(
            output / "arrays.npz",
            history_prior_mean=np.asarray([0.1, 0.2]),
            query_prior_mean=np.asarray([0.3, 0.4, 0.5, 0.6]),
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
    assert captured["kwargs"]["timeout"] == 10
    assert result.stage == "explore"
    assert result.query_prior_mean.shape == (4,)
    assert result.prior_clip_count == 2


def test_docker_policy_executor_returns_runner_errors(monkeypatch, tmp_path: Path) -> None:
    artifact, round_input, output = _inputs(tmp_path)

    def run(command, **_kwargs):
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
