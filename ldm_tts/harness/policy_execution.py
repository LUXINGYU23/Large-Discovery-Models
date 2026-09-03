"""Hardened container execution for generated optimization-policy artifacts."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ldm_tts.harness.protocol import HarnessSubmissionError


@dataclass(frozen=True)
class PolicyExecutionResult:
    history_prior_mean: np.ndarray
    query_prior_mean: np.ndarray
    stage: str
    alpha: float
    eta: float
    prior_clip_count: int


class PolicyExecutionError(RuntimeError):
    def __init__(self, errors: tuple[HarnessSubmissionError, ...]) -> None:
        if not errors:
            raise ValueError("policy execution errors must not be empty")
        self.errors = errors
        super().__init__("; ".join(error.message for error in errors))


class PolicyExecutor(Protocol):
    def execute(
        self,
        artifact_path: Path,
        input_directory: Path,
        output_directory: Path,
    ) -> PolicyExecutionResult: ...


@dataclass(frozen=True)
class DockerPolicyExecutor:
    image: str
    docker_host: str = ""
    container_user: str = ""
    timeout_seconds: int = 10

    def __post_init__(self) -> None:
        if not self.image.strip():
            raise ValueError("policy runner image must not be empty")
        if self.timeout_seconds < 1:
            raise ValueError("policy runner timeout must be positive")

    def execute(
        self,
        artifact_path: Path,
        input_directory: Path,
        output_directory: Path,
    ) -> PolicyExecutionResult:
        artifact = artifact_path.resolve()
        round_input = input_directory.resolve()
        output = output_directory.resolve()
        if not artifact.is_file():
            raise PolicyExecutionError((_error(
                "/artifact_path",
                "artifact_not_found",
                f"Policy artifact does not exist: {artifact.name}",
                "Write optimization_policy.py and submit it again.",
            ),))
        if not (round_input / "contract.json").is_file() or not (
            round_input / "arrays.npz"
        ).is_file():
            raise ValueError("policy round input is incomplete")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("arrays.npz", "result.json"):
            (output / name).unlink(missing_ok=True)
        command = ["docker"]
        if self.docker_host:
            command.extend(("--host", self.docker_host))
        command.extend(("run", "--rm", "--network", "none", "--read-only"))
        if self.container_user:
            command.extend(("--user", self.container_user))
        command.extend((
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges=true",
            "--pids-limit", "64",
            "--memory", "1g",
            "--cpus", "1",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--mount", f"type=bind,src={artifact.parent},dst=/policy,readonly",
            "--mount", f"type=bind,src={round_input},dst=/input,readonly",
            "--mount", f"type=bind,src={output},dst=/output",
            "--entrypoint", "python3",
            self.image,
            "/app/policy_runner.py", "execute",
            "--artifact", f"/policy/{artifact.name}",
            "--input", "/input",
            "--output", "/output",
        ))
        environment = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        os.chmod(output, 0o777)
        try:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=environment,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise PolicyExecutionError((_error(
                    "/artifact_path",
                    "runner_timeout",
                    f"Policy execution exceeded the {self.timeout_seconds}-second limit.",
                    "Simplify the policy and remove expensive loops or matrix operations.",
                ),)) from exc
            except OSError as exc:
                raise PolicyExecutionError((_error(
                    "/artifact_path",
                    "runner_unavailable",
                    f"Policy runner could not start: {exc}",
                    "Verify Docker and the configured Pi image before retrying.",
                ),)) from exc
        finally:
            os.chmod(output, 0o755)
        if completed.returncode != 0:
            raise PolicyExecutionError(_parse_errors(completed.stdout, completed.stderr))
        try:
            metadata = json.loads((output / "result.json").read_text(encoding="utf-8"))
            with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
                history_prior = np.asarray(arrays["history_prior_mean"], dtype=float).copy()
                query_prior = np.asarray(arrays["query_prior_mean"], dtype=float).copy()
            with np.load(round_input / "arrays.npz", allow_pickle=False) as arrays:
                history_size = len(arrays["history_features"])
                query_size = len(arrays["query_features"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise PolicyExecutionError((_error(
                "/artifact_path",
                "invalid_runner_output",
                f"Policy runner output is incomplete or invalid: {exc}",
                "Run draft validation and repair the artifact before resubmitting.",
            ),)) from exc
        if (
            history_prior.shape != (history_size,)
            or query_prior.shape != (query_size,)
            or not np.isfinite(history_prior).all()
            or not np.isfinite(query_prior).all()
        ):
            raise PolicyExecutionError((_error(
                "/outputs/prior_mean",
                "invalid_runner_output",
                "Policy runner returned invalid prior-mean arrays.",
                "Return one finite scalar for every requested feature row.",
            ),))
        try:
            if metadata.get("status") != "ok":
                raise ValueError("runner status is not ok")
            stage = metadata["stage"]
            if not isinstance(stage, str):
                raise TypeError("runner stage is not a string")
            alpha = float(metadata["alpha"])
            eta = float(metadata["eta"])
            raw_clip_count = metadata["prior_clip_count"]
            if isinstance(raw_clip_count, bool) or not isinstance(raw_clip_count, int):
                raise TypeError("runner prior_clip_count is not an integer")
            clip_count = raw_clip_count
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyExecutionError((_error(
                "/outputs",
                "invalid_runner_output",
                "Policy runner returned invalid policy metadata.",
                "Return a valid stage, alpha, eta, and prior mean.",
            ),)) from exc
        if (
            not stage.strip()
            or len(stage) > 64
            or not np.isfinite(alpha)
            or not np.isfinite(eta)
            or alpha < 0
            or eta < 0
            or clip_count < 0
        ):
            raise PolicyExecutionError((_error(
                "/outputs",
                "invalid_runner_output",
                "Policy runner returned non-finite or out-of-range policy metadata.",
                "Return a non-empty stage and finite non-negative alpha and eta.",
            ),))
        return PolicyExecutionResult(
            history_prior_mean=history_prior,
            query_prior_mean=query_prior,
            stage=stage,
            alpha=alpha,
            eta=eta,
            prior_clip_count=clip_count,
        )


def _parse_errors(stdout: str, stderr: str) -> tuple[HarnessSubmissionError, ...]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_errors = payload.get("errors") if isinstance(payload, dict) else None
        if not isinstance(raw_errors, list):
            continue
        parsed: list[HarnessSubmissionError] = []
        for value in raw_errors:
            if not isinstance(value, dict):
                continue
            try:
                parsed.append(HarnessSubmissionError(
                    path=str(value["path"]),
                    code=str(value["code"]),
                    message=str(value["message"]),
                    hint=str(value.get("hint", "")),
                ))
            except (KeyError, ValueError):
                continue
        if parsed:
            return tuple(parsed)
    detail = stderr.strip() or stdout.strip() or "unknown runner failure"
    return (_error(
        "/artifact_path",
        "runner_failed",
        f"Policy runner failed: {detail[-1000:]}",
        "Inspect the artifact with the policy validation tool and retry.",
    ),)


def _error(path: str, code: str, message: str, hint: str) -> HarnessSubmissionError:
    return HarnessSubmissionError(path=path, code=code, message=message, hint=hint)


__all__ = [
    "DockerPolicyExecutor",
    "PolicyExecutionError",
    "PolicyExecutionResult",
    "PolicyExecutor",
]
