"""Persistent research control for compiled LDM optimization policies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

import numpy as np

from ldm_tts.harness.client import HarnessClient, HarnessError
from ldm_tts.harness.policy_diagnostics import optimization_progress, prediction_feedback
from ldm_tts.harness.policy_execution import (
    PolicyExecutionError,
    PolicyExecutionResult,
    PolicyExecutor,
)
from ldm_tts.harness.protocol import (
    HarnessArtifactRule,
    HarnessMcpServer,
    HarnessMcpValue,
    HarnessSubmissionContract,
    HarnessSubmissionError,
    HarnessSubmissionRequest,
    HarnessSubmissionValidation,
    HarnessSubmittedArtifact,
    HarnessTurn,
    HarnessTurnResult,
    canonical_sha256,
    file_sha256,
)

POLICY_API_VERSION = 1
POLICY_CAPABILITIES = frozenset({"prior_mean@1", "ldm_weights@1"})
POLICY_ARTIFACT_NAME = "optimization_policy.py"
POLICY_ARTIFACT_MAX_BYTES = 128 * 1024


@dataclass(frozen=True)
class PolicyCapabilityContract:
    task_id: str
    api_version: int
    enabled_capabilities: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_groups: Mapping[str, tuple[int, int]]
    mean_clip: float
    default_alpha: float
    default_eta: float

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("policy task_id must not be empty")
        if self.api_version != POLICY_API_VERSION:
            raise ValueError(f"policy api_version must equal {POLICY_API_VERSION}")
        capabilities = tuple(sorted(self.enabled_capabilities))
        if (
            not capabilities
            or len(set(capabilities)) != len(capabilities)
            or any(value not in POLICY_CAPABILITIES for value in capabilities)
        ):
            raise ValueError("policy enabled_capabilities are invalid")
        names = tuple(self.feature_names)
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("policy feature_names must be non-empty and unique")
        groups: dict[str, tuple[int, int]] = {}
        for name, raw_range in self.feature_groups.items():
            if not name or len(raw_range) != 2:
                raise ValueError("policy feature_groups require named half-open ranges")
            start, end = raw_range
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > len(names)
            ):
                raise ValueError("policy feature_groups contain an invalid range")
            groups[name] = (start, end)
        numeric: dict[str, float] = {}
        for value, label, positive in (
            (self.mean_clip, "mean_clip", True),
            (self.default_alpha, "default_alpha", False),
            (self.default_eta, "default_eta", False),
        ):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"policy {label} is out of range") from exc
            if isinstance(value, bool) or not np.isfinite(number) or (
                number <= 0 if positive else number < 0
            ):
                raise ValueError(f"policy {label} is out of range")
            numeric[label] = number
        object.__setattr__(self, "enabled_capabilities", capabilities)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_groups", dict(sorted(groups.items())))
        for label, number in numeric.items():
            object.__setattr__(self, label, number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "api_version": self.api_version,
            "enabled_capabilities": list(self.enabled_capabilities),
            "feature_names": list(self.feature_names),
            "feature_groups": {
                name: list(value) for name, value in self.feature_groups.items()
            },
            "mean_clip": float(self.mean_clip),
            "default_alpha": float(self.default_alpha),
            "default_eta": float(self.default_eta),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PolicyRoundInput:
    round_index: int
    history_features: np.ndarray
    history_utilities: np.ndarray
    query_features: np.ndarray
    research_snapshot: Mapping[str, Any]
    execution_context: Mapping[str, Any]
    diagnostic_arrays: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.round_index, bool) or self.round_index < 0:
            raise ValueError("policy round_index must be non-negative")
        history = _readonly_array(self.history_features, dimensions=2, name="history_features")
        utilities = _readonly_array(
            self.history_utilities,
            dimensions=1,
            name="history_utilities",
        )
        query = _readonly_array(self.query_features, dimensions=2, name="query_features")
        if len(history) != len(utilities):
            raise ValueError("policy history features and utilities must have equal length")
        if history.shape[1] != query.shape[1]:
            raise ValueError("policy history and query features must have equal width")
        if len(query) == 0:
            raise ValueError("policy query_features must contain at least one row")
        research = _json_mapping(self.research_snapshot, "research_snapshot")
        context = _json_mapping(self.execution_context, "execution_context")
        if not isinstance(context.get("mean_context", {}), dict) or not isinstance(
            context.get("weight_context", {}), dict
        ):
            raise ValueError("policy execution_context requires object mean/weight contexts")
        object.__setattr__(self, "history_features", history)
        object.__setattr__(self, "history_utilities", utilities)
        object.__setattr__(self, "query_features", query)
        object.__setattr__(self, "research_snapshot", research)
        object.__setattr__(self, "execution_context", context)
        diagnostic_arrays = {}
        for name, value in self.diagnostic_arrays.items():
            if not name.startswith("diagnostic_") or not name.isidentifier():
                raise ValueError("invalid policy diagnostic array name")
            array = np.asarray(value, dtype=float).copy()
            if array.ndim not in (1, 2) or not np.isfinite(array).all():
                raise ValueError("policy diagnostic arrays must be finite vectors or matrices")
            array.setflags(write=False)
            diagnostic_arrays[name] = array
        object.__setattr__(self, "diagnostic_arrays", diagnostic_arrays)


@dataclass(frozen=True)
class CompiledOptimizationPolicy:
    epoch_id: str | None
    artifact_digest: str | None
    history_prior_mean: np.ndarray
    query_prior_mean: np.ndarray
    stage: str
    alpha: float
    eta: float
    source: Literal["artifact", "previous", "default"]
    degraded: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in {"artifact", "previous", "default"}:
            raise ValueError("unsupported compiled policy source")
        if (self.epoch_id is None) != (self.artifact_digest is None):
            raise ValueError("compiled policy epoch and artifact digest must be paired")
        if not self.stage.strip() or len(self.stage) > 64:
            raise ValueError("compiled policy stage is invalid")
        if any(
            isinstance(value, bool) or not np.isfinite(value) or value < 0
            for value in (self.alpha, self.eta)
        ):
            raise ValueError("compiled policy weights must be finite and non-negative")
        object.__setattr__(
            self,
            "history_prior_mean",
            _readonly_array(self.history_prior_mean, dimensions=1, name="history_prior_mean"),
        )
        object.__setattr__(
            self,
            "query_prior_mean",
            _readonly_array(self.query_prior_mean, dimensions=1, name="query_prior_mean"),
        )
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))


class OptimizationPolicyAdapter(Protocol):
    def capability_contract(self) -> PolicyCapabilityContract: ...

    def validate_task_execution(
        self,
        execution: PolicyExecutionResult,
        round_input: PolicyRoundInput,
    ) -> Sequence[HarnessSubmissionError]: ...


def policy_submission_contract(
    max_validation_attempts: int = 3,
) -> HarnessSubmissionContract:
    return HarnessSubmissionContract(
        contract_id="optimization_policy",
        tool_name="submit_optimization_policy",
        payload_schema={
            "type": "object",
            "description": (
                "Use exactly one action shape: replace with artifact_path, "
                "or keep/disable with action only."
            ),
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["replace", "keep", "disable"],
                },
                "artifact_path": {
                    "type": "string",
                    "enum": [POLICY_ARTIFACT_NAME],
                    "description": "Include only when action is replace.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
            "allOf": [{
                "if": {
                    "properties": {"action": {"const": "replace"}},
                    "required": ["action"],
                },
                "then": {"required": ["artifact_path"]},
                "else": {"not": {"required": ["artifact_path"]}},
            }],
        },
        artifact_rules=(HarnessArtifactRule(
            path_pointer="/artifact_path",
            allowed_suffixes=(".py",),
            max_bytes=POLICY_ARTIFACT_MAX_BYTES,
        ),),
        max_validation_attempts=max_validation_attempts,
    )


def policy_mcp_server(
    artifact_root: str = "/artifacts",
    profile_id: str = "policy_architect",
) -> HarnessMcpServer:
    root = PurePosixPath(artifact_root)
    if not root.is_absolute():
        raise ValueError("built-in policy MCP artifact root must be absolute")
    workspace = root / "sessions" / profile_id / "workspace"
    fields = {
        "server_id": "ldm_policy",
        "transport": "stdio",
        "tools": [
            "inspect_policy_contract",
            "validate_policy_draft",
            "evaluate_policy_draft",
        ],
        "command": "node",
        "args": ["/app/dist/policy-mcp.js", "stdio"],
        "env": {
            "LDM_POLICY_ROOT": str(root),
            "LDM_POLICY_WORKSPACE": str(workspace),
        },
    }
    return HarnessMcpServer(
        server_id=fields["server_id"],
        transport=fields["transport"],
        tools=tuple(fields["tools"]),
        config_sha256=canonical_sha256(fields),
        command=fields["command"],
        args=tuple(fields["args"]),
        env=tuple(
            (name, HarnessMcpValue(value=value))
            for name, value in sorted(fields["env"].items())
        ),
    )


class PolicyResearchController:
    def __init__(
        self,
        *,
        client: HarnessClient,
        adapter: OptimizationPolicyAdapter,
        executor: PolicyExecutor,
        root: Path,
        profile_id: str = "policy_architect",
        account: Callable[[Mapping[str, int | float]], None] | None = None,
    ) -> None:
        if not profile_id:
            raise ValueError("policy profile_id must not be empty")
        self.client = client
        self.adapter = adapter
        self.executor = executor
        self.root = Path(root).resolve()
        self.profile_id = profile_id
        self.account = account
        self.contract = adapter.capability_contract()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, round_input: PolicyRoundInput) -> CompiledOptimizationPolicy:
        round_input = self._with_feedback(round_input)
        self._validate_round_input(round_input)
        round_directory, input_digest = self._materialize_round(round_input)
        result_path = round_directory / "result.json"
        if result_path.is_file():
            return self._load_result(round_directory, input_digest)

        active = self._active_policy()
        _atomic_json(
            self.root / "active_round.json",
            {
                "round_index": round_input.round_index,
                "round_path": _relative_path(self.root, round_directory),
                "input_sha256": input_digest,
                "contract_sha256": self.contract.sha256,
                "active_policy": _active_summary(active),
            },
        )
        turn = self._turn(round_input, round_directory, input_digest, active)
        validated: dict[
            str,
            tuple[str, PolicyExecutionResult | None, HarnessSubmittedArtifact | None],
        ] = {}

        def validate(request: HarnessSubmissionRequest) -> HarnessSubmissionValidation:
            return self._validate_submission(
                request,
                round_input,
                round_directory,
                active,
                validated,
            )

        try:
            started = time.perf_counter()
            results = self.client.run_turn((turn,), submission_validator=validate)
            elapsed = time.perf_counter() - started
            if len(results) != 1:
                raise HarnessError("policy harness must commit exactly one turn")
            result = results[0]
            self._account(result, elapsed)
            if result.submission_status != "accepted":
                policy = _annotate_policy(
                    self._fallback(round_input, round_directory, active),
                    action="fallback",
                    status="submission_rejected",
                    submission_digest=result.submission_digest,
                    validation_errors=result.validation_errors,
                    turn=result,
                )
                self._persist_result(round_directory, input_digest, policy)
                return policy
            policy = _annotate_policy(
                self._apply_submission(
                    result,
                    round_input,
                    round_directory,
                    input_digest,
                    active,
                    validated,
                ),
                action=str(result.submission["action"]),
                status="accepted",
                submission_digest=result.submission_digest,
                validation_errors=(),
                turn=result,
            )
            self._persist_result(round_directory, input_digest, policy)
            return policy
        except (HarnessError, PolicyExecutionError, OSError, ValueError) as exc:
            errors = (
                exc.errors
                if isinstance(exc, PolicyExecutionError)
                else (HarnessSubmissionError(
                    "",
                    "policy_round_failed",
                    f"Policy research round failed: {exc}",
                    "Inspect the policy Harness trace and retry this campaign round.",
                ),)
            )
            policy = _annotate_policy(
                self._fallback(round_input, round_directory, active),
                action="fallback",
                status="runtime_fallback",
                submission_digest=None,
                validation_errors=errors,
                turn=None,
            )
            self._persist_result(round_directory, input_digest, policy)
            return policy

    def _validate_round_input(self, round_input: PolicyRoundInput) -> None:
        if round_input.history_features.shape[1] != len(self.contract.feature_names):
            raise ValueError("policy round feature width does not match the capability contract")

    def record_predictions(
        self, round_index, baseline, active, q0, *, alpha, eta, normalize_acquisition,
    ) -> None:
        signals = {
            "q0": np.asarray(q0, dtype=float),
            "baseline_acquisition": np.asarray([item.acquisition_score for item in baseline], dtype=float),
            "active_acquisition": np.asarray([item.acquisition_score for item in active], dtype=float),
        }
        ranks = {
            name: 1 + np.searchsorted(np.sort(-values), -values, side="left")
            for name, values in signals.items()
        }
        normalized = normalize_acquisition(signals["active_acquisition"])
        logits = alpha * np.log(signals["q0"] + 1e-12) + eta * normalized
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        max_q0 = signals["q0"].max()
        rows = []
        for index, (first, second, mass) in enumerate(zip(baseline, active, q0, strict=True)):
            if first.candidate_id != second.candidate_id:
                raise ValueError("policy feedback predictions must be aligned")
            rows.append({
                "candidate_id": first.candidate_id,
                "q0": float(mass),
                "q0_relative_to_max": float(mass / max_q0),
                "pool_size": len(q0),
                **{name + "_rank": int(values[index]) for name, values in ranks.items()},
                "baseline_acquisition": first.acquisition_score,
                "active_acquisition": second.acquisition_score,
                "normalized_acquisition": float(normalized[index]),
                "first_draw_probability": float(probabilities[index]),
                "alpha": float(alpha), "eta": float(eta),
                "baseline_mean": first.mean[0], "baseline_std": first.std[0],
                "active_mean": second.mean[0], "active_std": second.std[0],
            })
        payload = {"round_index": round_index, "predictions": rows}
        path = self.root / "rounds" / f"round_{round_index:03d}" / "predictions.json"
        if path.is_file() and _read_json(path) != payload:
            raise ValueError("recorded pre-measurement predictions cannot be replaced")
        _atomic_json(path, payload)

    def _with_feedback(self, round_input: PolicyRoundInput) -> PolicyRoundInput:
        research = dict(round_input.research_snapshot)
        ids = research.get("history_candidate_ids", [])
        rounds = research.get("history_rounds", [])
        if not ids:
            return round_input
        records = [
            _read_json(path) for path in sorted((self.root / "rounds").glob("round_*/predictions.json"))
            if int(path.parent.name.rsplit("_", 1)[1]) < round_input.round_index
        ]
        context = dict(round_input.execution_context)
        context["weight_context"] = {
            **context["weight_context"],
            "prediction_feedback": prediction_feedback(ids, rounds, round_input.history_utilities, records),
            "optimization_progress": optimization_progress(rounds, round_input.history_utilities),
        }
        return replace(round_input, execution_context=context)

    def _materialize_round(
        self,
        round_input: PolicyRoundInput,
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
        input_digest = canonical_sha256({
            "contract_sha256": self.contract.sha256,
            "arrays": array_digests,
            "input": input_data,
            "research_snapshot": dict(round_input.research_snapshot),
        })
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if manifest.get("input_sha256") != input_digest:
                raise ValueError(
                    f"policy round {round_input.round_index} already exists with different input"
                )
            return directory, input_digest

        directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(directory / "contract.json", self.contract.to_dict())
        _atomic_json(directory / "input.json", input_data)
        _atomic_json(
            directory / "research_snapshot.json",
            dict(round_input.research_snapshot),
        )
        _atomic_npz(
            directory / "arrays.npz",
            history_features=round_input.history_features,
            history_utilities=round_input.history_utilities,
            query_features=round_input.query_features,
            **round_input.diagnostic_arrays,
        )
        _atomic_json(
            manifest_path,
            {
                "round_index": round_input.round_index,
                "input_sha256": input_digest,
                "contract_sha256": self.contract.sha256,
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

    def _turn(
        self,
        round_input: PolicyRoundInput,
        round_directory: Path,
        input_digest: str,
        active: Mapping[str, Any] | None,
    ) -> HarnessTurn:
        history_to = len(round_input.history_features)
        history_from = self._previous_history_size(round_input.round_index)
        if history_to < history_from:
            raise ValueError("policy history size cannot decrease across rounds")
        measured = round_input.research_snapshot.get("measured_observations", [])
        if not isinstance(measured, list) or len(measured) != history_to:
            raise ValueError("measured_observations must align with the numeric history")
        new_observations = measured[history_from:history_to]
        forbidden_terms = round_input.research_snapshot.get("forbidden_query_terms", [])
        if not isinstance(forbidden_terms, list) or any(
            not isinstance(value, str) for value in forbidden_terms
        ):
            raise ValueError("forbidden_query_terms must be an array of strings")
        first_turn = not any(
            path.is_file()
            for path in (self.root / "rounds").glob("round_*/result.json")
        )
        message: dict[str, Any] = {
            "message_type": "optimization_policy_round",
            "round_index": round_input.round_index,
            "new_measured_observations": new_observations,
            "authoritative_snapshot": {
                "contract_path": _relative_path(self.root, round_directory / "contract.json"),
                "input_path": _relative_path(self.root, round_directory / "arrays.npz"),
                "research_path": _relative_path(
                    self.root,
                    round_directory / "research_snapshot.json",
                ),
                "sha256": input_digest,
            },
            "active_policy": _active_summary(active),
            "snapshot_access": (
                "Call inspect_policy_contract to obtain the read-only guest snapshot paths, "
                "execution contexts, and prediction feedback. Load its arrays.npz with NumPy "
                "for research; do not reconstruct numerical features by hand."
            ),
            "required_action": "Research, validate, then submit exactly one terminal action.",
            "terminal_actions": {
                "replace": {
                    "action": "replace",
                    "artifact_path": POLICY_ARTIFACT_NAME,
                },
                "keep": {"action": "keep"},
                "disable": {"action": "disable"},
            },
        }
        if first_turn:
            message["policy_contract"] = self.contract.to_dict()
            message["responsibility"] = (
                "Design a standardized conditional prior mean and LDM alpha/eta only. "
                "History utilities are raw task values; normalize them with the supplied "
                "target location and scale. Do not select candidates, call the oracle, "
                "or modify fixed task GP components."
            )
        turn_id = (
            f"{self.contract.task_id}-{self.profile_id}-"
            f"{round_input.round_index:03d}-{input_digest[:12]}"
        )
        return HarnessTurn(
            profile_id=self.profile_id,
            turn_id=turn_id,
            round_index=round_input.round_index,
            history_from_seq=history_from,
            history_to_seq=history_to,
            history_digest=canonical_sha256(new_observations),
            message=json.dumps(message, indent=2, sort_keys=True),
            forbidden_query_terms=tuple(forbidden_terms),
        )

    def _validate_submission(
        self,
        request: HarnessSubmissionRequest,
        round_input: PolicyRoundInput,
        round_directory: Path,
        active: Mapping[str, Any] | None,
        validated: dict[
            str,
            tuple[str, PolicyExecutionResult | None, HarnessSubmittedArtifact | None],
        ],
    ) -> HarnessSubmissionValidation:
        submission = dict(request.submission)
        action = submission.get("action")
        allowed_keys = {"action", "artifact_path"} if action == "replace" else {"action"}
        if action not in {"replace", "keep", "disable"}:
            return _retry(HarnessSubmissionError(
                "/action",
                "invalid_action",
                "action must be replace, keep, or disable.",
                "Choose one supported policy action.",
            ))
        if set(submission) != allowed_keys:
            return _retry(HarnessSubmissionError(
                "",
                "invalid_submission_shape",
                f"{action} requires exactly these fields: {sorted(allowed_keys)}.",
                "Remove unrelated fields and include artifact_path only for replace.",
            ))
        if action == "replace":
            if submission.get("artifact_path") != POLICY_ARTIFACT_NAME:
                return _retry(HarnessSubmissionError(
                    "/artifact_path",
                    "invalid_artifact_path",
                    f"replace must reference {POLICY_ARTIFACT_NAME}.",
                    f"Write the complete policy to {POLICY_ARTIFACT_NAME} and resubmit.",
                ))
            if len(request.artifacts) != 1:
                return _retry(HarnessSubmissionError(
                    "/artifact_path",
                    "missing_artifact_snapshot",
                    "replace requires exactly one immutable artifact snapshot.",
                    "Ensure optimization_policy.py exists in the session workspace.",
                ))
            descriptor = request.artifacts[0]
            try:
                artifact = self._submitted_artifact(descriptor)
                execution = self.executor.execute(
                    artifact,
                    round_directory,
                    round_directory / "validations" / request.digest,
                )
                errors = self._task_errors(execution, round_input)
            except PolicyExecutionError as exc:
                return HarnessSubmissionValidation("retry", exc.errors)
            if errors:
                return HarnessSubmissionValidation("retry", errors)
            validated[request.digest] = (action, execution, descriptor)
            return HarnessSubmissionValidation()

        if request.artifacts:
            return _retry(HarnessSubmissionError(
                "/artifact_path",
                "unexpected_artifact",
                f"{action} must not submit an artifact.",
                "Remove artifact_path or use action=replace.",
            ))
        if action == "keep":
            if active is None:
                return _retry(HarnessSubmissionError(
                    "/action",
                    "no_active_policy",
                    "keep is unavailable because no custom policy is active.",
                    "Submit action=replace with a valid policy, or action=disable.",
                ))
            try:
                execution = self.executor.execute(
                    self._active_artifact(active),
                    round_directory,
                    round_directory / "validations" / request.digest,
                )
                errors = self._task_errors(execution, round_input)
            except PolicyExecutionError as exc:
                return HarnessSubmissionValidation("retry", exc.errors)
            if errors:
                return HarnessSubmissionValidation("retry", errors)
            validated[request.digest] = (action, execution, None)
        else:
            validated[request.digest] = (action, None, None)
        return HarnessSubmissionValidation()

    def _apply_submission(
        self,
        result: HarnessTurnResult,
        round_input: PolicyRoundInput,
        round_directory: Path,
        input_digest: str,
        active: Mapping[str, Any] | None,
        validated: dict[
            str,
            tuple[str, PolicyExecutionResult | None, HarnessSubmittedArtifact | None],
        ],
    ) -> CompiledOptimizationPolicy:
        cached = validated.get(result.submission_digest)
        if cached is None:
            request = HarnessSubmissionRequest(
                profile_id=result.profile_id,
                turn_id=result.turn_id,
                attempt_index=1,
                submission=result.submission,
                artifacts=result.submitted_artifacts,
            )
            validation = self._validate_submission(
                request,
                round_input,
                round_directory,
                active,
                validated,
            )
            if validation.decision != "accept":
                raise PolicyExecutionError(validation.errors)
            cached = validated[result.submission_digest]
        action, execution, descriptor = cached
        if action == "disable":
            _atomic_json(
                self.root / "active_policy.json",
                {"epoch_id": None, "artifact_sha256": None, "artifact_path": None},
            )
            return self._default_policy(round_input, degraded=False)
        if execution is None:
            raise ValueError("accepted policy submission is missing execution output")
        if action == "keep":
            if active is None:
                raise ValueError("accepted keep submission has no active policy")
            return _compiled_policy(
                execution,
                epoch_id=str(active["epoch_id"]),
                artifact_digest=str(active["artifact_sha256"]),
                source="previous",
                degraded=False,
            )
        if descriptor is None:
            raise ValueError("accepted replace submission is missing its artifact")
        epoch = self._commit_epoch(
            round_input.round_index,
            input_digest,
            result,
            descriptor,
            execution,
        )
        return _compiled_policy(
            execution,
            epoch_id=str(epoch["epoch_id"]),
            artifact_digest=str(epoch["artifact_sha256"]),
            source="artifact",
            degraded=False,
        )

    def _commit_epoch(
        self,
        round_index: int,
        input_digest: str,
        result: HarnessTurnResult,
        descriptor: HarnessSubmittedArtifact,
        execution: PolicyExecutionResult,
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
            _atomic_json(self.root / "active_policy.json", _active_pointer(manifest))
            return manifest

        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / POLICY_ARTIFACT_NAME
        shutil.copyfile(self._submitted_artifact(descriptor), artifact)
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
        _atomic_json(directory / "validation.json", validation)
        _atomic_json(directory / "evaluation.json", evaluation)
        manifest = {
            "epoch_id": epoch_id,
            "round_index": round_index,
            "turn_id": result.turn_id,
            "session_id": result.session_id,
            "input_sha256": input_digest,
            "submission_sha256": result.submission_digest,
            "artifact_sha256": descriptor.sha256,
            "artifact_path": _relative_path(self.root, artifact),
            "api_version": self.contract.api_version,
            "enabled_capabilities": list(self.contract.enabled_capabilities),
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
        _atomic_json(manifest_path, manifest)
        _atomic_json(self.root / "active_policy.json", _active_pointer(manifest))
        return manifest

    def _fallback(
        self,
        round_input: PolicyRoundInput,
        round_directory: Path,
        active: Mapping[str, Any] | None,
    ) -> CompiledOptimizationPolicy:
        if active is not None:
            try:
                execution = self.executor.execute(
                    self._active_artifact(active),
                    round_directory,
                    round_directory / "fallback",
                )
                if not self._task_errors(execution, round_input):
                    return _compiled_policy(
                        execution,
                        epoch_id=str(active["epoch_id"]),
                        artifact_digest=str(active["artifact_sha256"]),
                        source="previous",
                        degraded=True,
                    )
            except (PolicyExecutionError, OSError, ValueError):
                pass
        return self._default_policy(round_input, degraded=True)

    def _default_policy(
        self,
        round_input: PolicyRoundInput,
        *,
        degraded: bool,
    ) -> CompiledOptimizationPolicy:
        return CompiledOptimizationPolicy(
            epoch_id=None,
            artifact_digest=None,
            history_prior_mean=np.zeros(len(round_input.history_features)),
            query_prior_mean=np.zeros(len(round_input.query_features)),
            stage="default",
            alpha=float(self.contract.default_alpha),
            eta=float(self.contract.default_eta),
            source="default",
            degraded=degraded,
            metadata={"prior_clip_count": 0},
        )

    def _task_errors(
        self,
        execution: PolicyExecutionResult,
        round_input: PolicyRoundInput,
    ) -> tuple[HarnessSubmissionError, ...]:
        try:
            errors = tuple(self.adapter.validate_task_execution(execution, round_input))
        except Exception as exc:
            return (HarnessSubmissionError(
                "/artifact_path",
                "task_validation_exception",
                f"Task-level policy validation failed: {type(exc).__name__}: {exc}",
                "Inspect task diagnostics, repair the policy, and validate it again.",
            ),)
        if any(not isinstance(error, HarnessSubmissionError) for error in errors):
            raise TypeError("policy adapter must return HarnessSubmissionError values")
        return errors

    def _submitted_artifact(self, descriptor: HarnessSubmittedArtifact) -> Path:
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

    def _active_policy(self) -> dict[str, Any] | None:
        path = self.root / "active_policy.json"
        if not path.is_file():
            return None
        value = _read_json(path)
        if value.get("epoch_id") is None:
            return None
        required = {"epoch_id", "artifact_sha256", "artifact_path"}
        if set(value) != required or any(not isinstance(value[key], str) for key in required):
            raise ValueError("active policy pointer is invalid")
        artifact = self._active_artifact(value)
        if file_sha256(artifact) != value["artifact_sha256"]:
            raise ValueError("active policy artifact digest mismatch")
        manifest = _read_json(self.root / "epochs" / value["epoch_id"] / "manifest.json")
        if manifest.get("artifact_sha256") != value["artifact_sha256"]:
            raise ValueError("active policy epoch manifest mismatch")
        return value

    def _active_artifact(self, active: Mapping[str, Any]) -> Path:
        relative = active.get("artifact_path")
        if not isinstance(relative, str):
            raise ValueError("active policy artifact path is invalid")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("active policy artifact is unavailable")
        return path

    def _previous_history_size(self, round_index: int) -> int:
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

    def _persist_result(
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
        _atomic_json(directory / "result.json", payload)

    def _load_result(
        self,
        directory: Path,
        input_digest: str,
    ) -> CompiledOptimizationPolicy:
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
                        "prior_clip_count",
                    )
                    if key in value
                },
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("persisted policy result is invalid") from exc

    def _account(self, result: HarnessTurnResult, wall_time_seconds: float) -> None:
        if self.account is None or result.replayed:
            return
        self.account({
            "policy_harness_turns": 1,
            "policy_provider_requests": int(result.usage["providerCalls"]),
            "policy_tool_calls": sum(result.usage["toolCalls"].values()),
            "policy_validation_submissions": int(
                result.usage.get("validationSubmissions", 0)
            ),
            "policy_artifact_bytes": int(result.usage["artifactBytes"]),
            "policy_wall_time_seconds": wall_time_seconds,
        })


def _readonly_array(value: Any, *, dimensions: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.ndim != dimensions or not np.isfinite(array).all():
        raise ValueError(f"policy {name} must be a finite {dimensions}-dimensional array")
    array.setflags(write=False)
    return array


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"policy {name} must be a mapping")
    try:
        copied = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy {name} must be JSON serializable") from exc
    return copied


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


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


def _active_summary(active: Mapping[str, Any] | None) -> dict[str, Any] | None:
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


def _compiled_policy(
    execution: PolicyExecutionResult,
    *,
    epoch_id: str,
    artifact_digest: str,
    source: Literal["artifact", "previous"],
    degraded: bool,
) -> CompiledOptimizationPolicy:
    return CompiledOptimizationPolicy(
        epoch_id=epoch_id,
        artifact_digest=artifact_digest,
        history_prior_mean=execution.history_prior_mean,
        query_prior_mean=execution.query_prior_mean,
        stage=execution.stage,
        alpha=execution.alpha,
        eta=execution.eta,
        source=source,
        degraded=degraded,
        metadata={"prior_clip_count": execution.prior_clip_count},
    )


def _annotate_policy(
    policy: CompiledOptimizationPolicy,
    *,
    action: str,
    status: str,
    submission_digest: str | None,
    validation_errors: Sequence[HarnessSubmissionError],
    turn: HarnessTurnResult | None,
) -> CompiledOptimizationPolicy:
    metadata: dict[str, Any] = {
        **dict(policy.metadata),
        "action": action,
        "status": status,
        "submission_sha256": submission_digest,
        "validation_errors": [error.to_dict() for error in validation_errors],
    }
    if turn is not None:
        metadata["harness_turn"] = {
            "profile_id": turn.profile_id,
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "replayed": turn.replayed,
            "submission_id": turn.submission_id,
            "usage": turn.usage,
            "tool_budget": turn.tool_budget,
            "artifacts": turn.artifacts,
        }
    return replace(policy, metadata=metadata)


def _retry(error: HarnessSubmissionError) -> HarnessSubmissionValidation:
    return HarnessSubmissionValidation("retry", (error,))


__all__ = [
    "POLICY_API_VERSION",
    "POLICY_ARTIFACT_MAX_BYTES",
    "POLICY_ARTIFACT_NAME",
    "POLICY_CAPABILITIES",
    "CompiledOptimizationPolicy",
    "OptimizationPolicyAdapter",
    "PolicyCapabilityContract",
    "PolicyResearchController",
    "PolicyRoundInput",
    "policy_mcp_server",
    "policy_submission_contract",
]
