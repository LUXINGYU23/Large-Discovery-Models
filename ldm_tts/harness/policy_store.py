"""Durable inputs, immutable policy epochs, and compiled round results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness.policy_contract import (
    POLICY_ARTIFACT_NAME,
    CompiledOptimizationPolicy,
    PolicyCapabilityContract,
    PolicyRoundInput,
    json_mapping,
)
from ldm_tts.harness.policy_execution import PolicyExecutionError, PolicyExecutionResult
from ldm_tts.harness.protocol import (
    HarnessSubmissionError,
    HarnessSubmittedArtifact,
    HarnessTurnResult,
    canonical_sha256,
    file_sha256,
)


class PolicyStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def record_predictions(self, round_index: int, predictions: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise ValueError("prediction round_index must be a non-negative integer")
        payload = {
            "round_index": round_index,
            "predictions": [json_mapping(row, "prediction snapshot") for row in predictions],
        }
        path = self.root / "rounds" / f"round_{round_index:03d}" / "predictions.json"
        if path.is_file() and _read_json(path) != payload:
            raise ValueError("recorded pre-measurement predictions cannot be replaced")
        atomic_json_write(path, payload)

    def predictions_before(self, round_index: int) -> list[dict[str, Any]]:
        return [
            _read_json(path) for path in sorted((self.root / "rounds").glob("round_*/predictions.json"))
            if int(path.parent.name.rsplit("_", 1)[1]) < round_index
        ]

    def has_results(self) -> bool:
        return any(path.is_file() for path in (self.root / "rounds").glob("round_*/result.json"))

    def materialize_round(
        self,
        round_input: PolicyRoundInput,
        contract: PolicyCapabilityContract,
    ) -> tuple[Path, str]:
        directory = self.root / "rounds" / f"round_{round_input.round_index:03d}"
        array_digests = {
            "history_features": _array_sha256(round_input.history_features),
            "history_utilities": _array_sha256(round_input.history_utilities),
            "query_features": _array_sha256(round_input.query_features),
            **{name: _array_sha256(value) for name, value in round_input.diagnostic_arrays.items()},
        }
        input_data = {
            "round_index": round_input.round_index,
            "execution_context": dict(round_input.execution_context),
        }
        research = {
            **round_input.research_snapshot,
            "history_candidate_ids": list(round_input.history_candidate_ids),
            "history_rounds": list(round_input.history_rounds),
            "measured_observations": list(round_input.measured_observations),
        }
        input_digest = canonical_sha256({
            "contract_sha256": contract.sha256,
            "arrays": array_digests,
            "input": input_data,
            "research_snapshot": research,
        })
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if manifest.get("input_sha256") != input_digest:
                raise ValueError(
                    f"policy round {round_input.round_index} already exists with different input"
                )
            for name in ("contract.json", "input.json", "research_snapshot.json", "arrays.npz"):
                if file_sha256(directory / name) != manifest.get("files", {}).get(name):
                    raise ValueError(f"persisted policy round input digest mismatch: {name}")
            return directory, input_digest

        directory.mkdir(parents=True, exist_ok=True)
        atomic_json_write(directory / "contract.json", contract.to_dict())
        atomic_json_write(directory / "input.json", input_data)
        atomic_json_write(
            directory / "research_snapshot.json",
            research,
        )
        _atomic_npz(
            directory / "arrays.npz",
            history_features=round_input.history_features,
            history_utilities=round_input.history_utilities,
            query_features=round_input.query_features,
            **round_input.diagnostic_arrays,
        )
        atomic_json_write(
            manifest_path,
            {
                "round_index": round_input.round_index,
                "input_sha256": input_digest,
                "contract_sha256": contract.sha256,
                "array_sha256": array_digests,
                "files": {
                    name: file_sha256(directory / name)
                    for name in (
                        "contract.json",
                        "input.json",
                        "research_snapshot.json",
                        "arrays.npz",
                    )
                },
            },
        )
        return directory, input_digest

    def activate_round(
        self,
        directory: Path,
        input_digest: str,
        contract: PolicyCapabilityContract,
        active: Mapping[str, Any] | None,
    ) -> None:
        atomic_json_write(
            self.root / "active_round.json",
            {
                "round_index": int(directory.name.rsplit("_", 1)[1]),
                "round_path": _relative_path(self.root, directory),
                "input_sha256": input_digest,
                "contract_sha256": contract.sha256,
                "active_policy": active_policy_summary(active),
            },
        )

    def round_snapshot(self, directory: Path, input_digest: str) -> dict[str, str]:
        return {
            "contract_path": _relative_path(self.root, directory / "contract.json"),
            "input_path": _relative_path(self.root, directory / "arrays.npz"),
            "research_path": _relative_path(self.root, directory / "research_snapshot.json"),
            "sha256": input_digest,
        }

    def disable_policy(self) -> None:
        atomic_json_write(
            self.root / "active_policy.json",
            {"epoch_id": None, "artifact_sha256": None, "artifact_path": None},
        )

    def commit_epoch(
        self,
        round_index: int,
        input_digest: str,
        result: HarnessTurnResult,
        descriptor: HarnessSubmittedArtifact,
        execution: PolicyExecutionResult,
        *,
        contract: PolicyCapabilityContract,
    ) -> dict[str, Any]:
        epoch_id = f"epoch_{round_index:03d}"
        directory = self.root / "epochs" / epoch_id
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if (
                manifest.get("artifact_sha256") != descriptor.sha256
                or manifest.get("submission_sha256") != result.submission_digest
            ):
                raise ValueError(f"policy epoch collision: {epoch_id}")
            atomic_json_write(self.root / "active_policy.json", _active_pointer(manifest))
            return manifest

        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / POLICY_ARTIFACT_NAME
        shutil.copyfile(self.submitted_artifact(descriptor), artifact)
        if file_sha256(artifact) != descriptor.sha256:
            raise ValueError("committed policy artifact digest changed during copy")
        os.chmod(artifact, 0o444)
        validation = {
            "status": "accepted",
            "input_sha256": input_digest,
            "submission_sha256": result.submission_digest,
            "artifact_sha256": descriptor.sha256,
            "validation_errors": [],
        }
        evaluation = {
            "input_sha256": input_digest,
            "stage": execution.stage,
            "alpha": execution.alpha,
            "eta": execution.eta,
            "prior_clip_count": execution.prior_clip_count,
        }
        atomic_json_write(directory / "validation.json", validation)
        atomic_json_write(directory / "evaluation.json", evaluation)
        manifest = {
            "epoch_id": epoch_id,
            "round_index": round_index,
            "turn_id": result.turn_id,
            "session_id": result.session_id,
            "input_sha256": input_digest,
            "submission_sha256": result.submission_digest,
            "artifact_sha256": descriptor.sha256,
            "artifact_path": _relative_path(self.root, artifact),
            "api_version": contract.api_version,
            "enabled_capabilities": list(contract.enabled_capabilities),
            "stage": execution.stage,
            "alpha": execution.alpha,
            "eta": execution.eta,
            "prior_clip_count": execution.prior_clip_count,
            "files": {
                name: file_sha256(directory / name)
                for name in (
                    POLICY_ARTIFACT_NAME,
                    "validation.json",
                    "evaluation.json",
                )
            },
        }
        atomic_json_write(manifest_path, manifest)
        atomic_json_write(self.root / "active_policy.json", _active_pointer(manifest))
        return manifest

    def submitted_artifact(self, descriptor: HarnessSubmittedArtifact) -> Path:
        if (
            descriptor.path_pointer != "/artifact_path"
            or descriptor.relative_path != POLICY_ARTIFACT_NAME
        ):
            raise PolicyExecutionError((HarnessSubmissionError(
                "/artifact_path",
                "invalid_artifact_descriptor",
                "The submitted artifact descriptor does not match optimization_policy.py.",
                "Submit the policy through the declared artifact_path field.",
            ),))
        path = (self.root / descriptor.snapshot_path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise PolicyExecutionError((HarnessSubmissionError(
                "/artifact_path",
                "artifact_snapshot_unavailable",
                "The immutable policy artifact snapshot is unavailable.",
                "Rewrite optimization_policy.py and submit it again.",
            ),))
        if path.stat().st_size != descriptor.size_bytes or file_sha256(path) != descriptor.sha256:
            raise PolicyExecutionError((HarnessSubmissionError(
                "/artifact_path",
                "artifact_digest_mismatch",
                "The policy artifact no longer matches its submitted digest.",
                "Rewrite optimization_policy.py and submit a fresh snapshot.",
            ),))
        return path

    def active_policy(self) -> dict[str, Any] | None:
        path = self.root / "active_policy.json"
        if not path.is_file():
            return None
        value = _read_json(path)
        if value.get("epoch_id") is None:
            return None
        required = {"epoch_id", "artifact_sha256", "artifact_path"}
        if set(value) != required or any(not isinstance(value[key], str) for key in required):
            raise ValueError("active policy pointer is invalid")
        artifact = self.active_artifact(value)
        if file_sha256(artifact) != value["artifact_sha256"]:
            raise ValueError("active policy artifact digest mismatch")
        manifest = _read_json(self.root / "epochs" / value["epoch_id"] / "manifest.json")
        if manifest.get("artifact_sha256") != value["artifact_sha256"]:
            raise ValueError("active policy epoch manifest mismatch")
        return value

    def active_artifact(self, active: Mapping[str, Any]) -> Path:
        relative = active.get("artifact_path")
        if not isinstance(relative, str):
            raise ValueError("active policy artifact path is invalid")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("active policy artifact is unavailable")
        return path

    def previous_history_size(self, round_index: int) -> int:
        sizes = []
        for path in (self.root / "rounds").glob("round_*/result.json"):
            value = _read_json(path)
            saved_round = value.get("round_index")
            saved_size = value.get("history_to_seq")
            if (
                isinstance(saved_round, int)
                and saved_round < round_index
                and isinstance(saved_size, int)
                and saved_size >= 0
                and value.get("harness_turn") is not None
            ):
                sizes.append(saved_size)
        return max(sizes, default=0)

    def persist_result(
        self,
        directory: Path,
        input_digest: str,
        policy: CompiledOptimizationPolicy,
    ) -> None:
        _atomic_npz(
            directory / "compiled.npz",
            history_prior_mean=policy.history_prior_mean,
            query_prior_mean=policy.query_prior_mean,
        )
        payload: dict[str, Any] = {
            **dict(policy.metadata),
            "round_index": int(directory.name.rsplit("_", 1)[1]),
            "input_sha256": input_digest,
            "epoch_id": policy.epoch_id,
            "artifact_sha256": policy.artifact_digest,
            "source": policy.source,
            "degraded": policy.degraded,
            "stage": policy.stage,
            "alpha": policy.alpha,
            "eta": policy.eta,
            "history_to_seq": len(policy.history_prior_mean),
            "query_size": len(policy.query_prior_mean),
            "compiled_arrays_sha256": file_sha256(directory / "compiled.npz"),
        }
        atomic_json_write(directory / "result.json", payload)

    def load_result(
        self,
        directory: Path,
        input_digest: str,
    ) -> CompiledOptimizationPolicy | None:
        if not (directory / "result.json").is_file():
            return None
        value = _read_json(directory / "result.json")
        if value.get("input_sha256") != input_digest:
            raise ValueError("persisted policy result input digest mismatch")
        if file_sha256(directory / "compiled.npz") != value.get("compiled_arrays_sha256"):
            raise ValueError("persisted policy result array digest mismatch")
        try:
            with np.load(directory / "compiled.npz", allow_pickle=False) as arrays:
                history = np.asarray(arrays["history_prior_mean"], dtype=float).copy()
                query = np.asarray(arrays["query_prior_mean"], dtype=float).copy()
            return CompiledOptimizationPolicy(
                epoch_id=value["epoch_id"],
                artifact_digest=value["artifact_sha256"],
                history_prior_mean=history,
                query_prior_mean=query,
                stage=str(value["stage"]),
                alpha=float(value["alpha"]),
                eta=float(value["eta"]),
                source=value["source"],
                degraded=value["degraded"],
                metadata={
                    key: value[key]
                    for key in (
                        "action",
                        "status",
                        "submission_sha256",
                        "validation_errors",
                        "harness_turn",
                        "failed_harness_usage",
                        "prior_clip_count",
                    )
                    if key in value
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("persisted policy result is invalid") from exc


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read policy state {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"policy state {path.name} must contain a JSON object")
    return value


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def active_policy_summary(active: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if active is None:
        return None
    return {
        "epoch_id": active["epoch_id"],
        "artifact_sha256": active["artifact_sha256"],
    }


def _active_pointer(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        "epoch_id": str(manifest["epoch_id"]),
        "artifact_sha256": str(manifest["artifact_sha256"]),
        "artifact_path": str(manifest["artifact_path"]),
    }
