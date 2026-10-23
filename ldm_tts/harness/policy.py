"""Execution orchestration for compiled LDM optimization policies."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ldm_tts.harness.client import HarnessClient
from ldm_tts.harness.policy_contract import (
    POLICY_ARTIFACT_NAME,
    CompiledOptimizationPolicy,
    OptimizationPolicyAdapter,
    PolicyRoundInput,
)
from ldm_tts.harness.policy_execution import (
    PolicyExecutionError,
    PolicyExecutionResult,
    PolicyExecutor,
)
from ldm_tts.harness.policy_store import PolicyStore, active_policy_summary
from ldm_tts.harness.protocol import (
    HarnessError,
    HarnessProviderAuthorizationRequest,
    HarnessSubmissionError,
    HarnessSubmissionRequest,
    HarnessSubmissionValidation,
    HarnessSubmittedArtifact,
    HarnessTurn,
    HarnessTurnResult,
    canonical_sha256,
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
        recovery_budget: Callable[[], float] | None = None,
        provider_authorizer: Callable[[HarnessProviderAuthorizationRequest], bool] | None = None,
    ) -> None:
        if not profile_id:
            raise ValueError("policy profile_id must not be empty")
        self.client = client
        self.adapter = adapter
        self.executor = executor
        self.root = Path(root).resolve()
        self.profile_id = profile_id
        self.account = account
        self.recovery_budget = recovery_budget
        self.provider_authorizer = provider_authorizer
        self.contract = adapter.capability_contract()
        self.store = PolicyStore(self.root)

    def resolve(self, round_input: PolicyRoundInput) -> CompiledOptimizationPolicy:
        round_input = self.adapter.with_feedback(
            round_input, self.store.predictions_before(round_input.round_index),
        )
        if round_input.history_features.shape[1] != len(self.contract.feature_names):
            raise ValueError("policy round feature width does not match the capability contract")
        round_directory, input_digest = self.store.materialize_round(round_input, self.contract)
        saved = self.store.load_result(round_directory, input_digest)
        if saved is not None:
            return saved

        active = self.store.active_policy()
        self.store.activate_round(round_directory, input_digest, self.contract, active)
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

        started = time.perf_counter()
        result = None
        failed_usage: dict[str, Any] = {}
        try:
            results = self.client.run_turn(
                (turn,), submission_validator=validate,
                **({"provider_authorizer": self.provider_authorizer} if self.provider_authorizer else {}),
                recovery_timeout_seconds=(0 if self.recovery_budget is None else self.recovery_budget()),
            )
            elapsed = time.perf_counter() - started
            if len(results) != 1:
                raise HarnessError("policy harness must commit exactly one turn")
            result = results[0]
            if result.submission_status != "accepted":
                policy = _annotate_policy(
                    self._fallback(round_input, round_directory, active),
                    action="fallback",
                    status="submission_rejected",
                    submission_digest=result.submission_digest,
                    validation_errors=result.validation_errors,
                    turn=result,
                )
            else:
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
        except (HarnessError, PolicyExecutionError, OSError, ValueError) as exc:
            if result is None:
                elapsed = time.perf_counter() - started
            if isinstance(exc, HarnessError):
                failed_usage = dict(exc.turn_usage.get(turn.profile_id, {}))
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
                submission_digest=result.submission_digest if result is not None else None,
                validation_errors=errors,
                turn=result,
                failed_usage=failed_usage if result is None else None,
            )
        if result is None or not result.replayed:
            self._account(
                result.usage if result is not None else failed_usage,
                elapsed,
            )
        self.store.persist_result(round_directory, input_digest, policy)
        return policy

    def record_predictions(self, round_index: int, predictions: Sequence[Mapping[str, Any]]) -> None:
        self.store.record_predictions(round_index, predictions)

    def _turn(
        self,
        round_input: PolicyRoundInput,
        round_directory: Path,
        input_digest: str,
        active: Mapping[str, Any] | None,
    ) -> HarnessTurn:
        history_to = len(round_input.history_features)
        history_from = self.store.previous_history_size(round_input.round_index)
        if history_to < history_from:
            raise ValueError("policy history size cannot decrease across rounds")
        new_observations = list(round_input.measured_observations[history_from:history_to])
        forbidden_terms = round_input.research_snapshot.get("forbidden_query_terms", [])
        if not isinstance(forbidden_terms, list) or any(
            not isinstance(value, str) for value in forbidden_terms
        ):
            raise ValueError("forbidden_query_terms must be an array of strings")
        message: dict[str, Any] = {
            "message_type": "optimization_policy_round",
            "round_index": round_input.round_index,
            "new_measured_observations": new_observations,
            "authoritative_snapshot": self.store.round_snapshot(round_directory, input_digest),
            "active_policy": active_policy_summary(active),
            "snapshot_access": (
                "Use the task-provided research snapshot and registered tools to access "
                "execution contexts, numerical arrays, and prediction feedback. "
                "Do not reconstruct numerical features by hand."
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
        if not self.store.has_results():
            message["policy_contract"] = self.contract.to_dict()
            message["responsibility"] = (
                "Implement only the enabled capabilities in policy_contract. "
                "Follow the task instructions and execution contexts for objective "
                "order, direction, units, transformations, and scientific constraints. "
                "Do not infer a target transformation or scalarization. Do not select "
                "candidates, call the oracle, or modify components outside the contract."
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
            message=json.dumps(message, separators=(",", ":"), sort_keys=True),
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
                artifact = self.store.submitted_artifact(descriptor)
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
                    self.store.active_artifact(active),
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
            self.store.disable_policy()
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
        epoch = self.store.commit_epoch(
            round_input.round_index,
            input_digest,
            result,
            descriptor,
            execution,
            contract=self.contract,
        )
        return _compiled_policy(
            execution,
            epoch_id=str(epoch["epoch_id"]),
            artifact_digest=str(epoch["artifact_sha256"]),
            source="artifact",
            degraded=False,
        )

    def _fallback(
        self,
        round_input: PolicyRoundInput,
        round_directory: Path,
        active: Mapping[str, Any] | None,
    ) -> CompiledOptimizationPolicy:
        if active is not None:
            try:
                execution = self.executor.execute(
                    self.store.active_artifact(active),
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
            history_prior_mean=np.zeros_like(round_input.history_utilities),
            query_prior_mean=np.zeros((len(round_input.query_features), *round_input.history_utilities.shape[1:])),
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
        for name, values, shape in (
            ("history_prior_mean", execution.history_prior_mean, round_input.history_utilities.shape),
            ("query_prior_mean", execution.query_prior_mean,
             (len(round_input.query_features), *round_input.history_utilities.shape[1:])),
        ):
            if values.shape != shape or not np.isfinite(values).all():
                return (HarnessSubmissionError(
                    f"/outputs/{name}", "invalid_prior_shape",
                    f"{name} must have finite values and shape {shape}.",
                    "Preserve the objective dimensions of history_utilities for each requested row.",
                ),)
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

    def _account(self, usage: Mapping[str, Any], wall_time_seconds: float) -> None:
        if self.account is None:
            return
        cost = {
            "policy_harness_turns": 1,
            "policy_wall_time_seconds": wall_time_seconds,
        }
        for key, counter in (
            ("providerCalls", "policy_provider_requests"),
            ("validationSubmissions", "policy_validation_submissions"),
            ("artifactBytes", "policy_artifact_bytes"),
        ):
            if key in usage:
                cost[counter] = int(usage[key])
        if "toolCalls" in usage:
            cost["policy_tool_calls"] = sum(usage["toolCalls"].values())
        self.account(cost)


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
    failed_usage: Mapping[str, Any] | None = None,
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
    elif failed_usage is not None:
        metadata["failed_harness_usage"] = dict(failed_usage)
    return replace(policy, metadata=metadata)


def _retry(error: HarnessSubmissionError) -> HarnessSubmissionValidation:
    return HarnessSubmissionValidation("retry", (error,))


__all__ = ["PolicyResearchController"]
