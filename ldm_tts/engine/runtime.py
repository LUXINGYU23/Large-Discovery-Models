"""Task-neutral Large Discovery Model campaign engine."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from ldm_tts.contracts import (
    BatchCandidateEvaluator,
    Candidate,
    CandidateDomainAdapter,
    CandidateEvaluationPreparer,
    CandidateEvaluator,
    EvaluationResult,
    LDMTaskSpec,
    ObjectiveSet,
    Observation,
    ReservoirBuilder,
)
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY, EvaluationPaused
from ldm_tts.engine.expansion import (
    PROPOSAL_ATTEMPT_RECEIPT_KEY,
    ExpansionRequest,
    ReservoirExpander,
)
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOSelectionResult,
    SurrogateEncoder,
)


@dataclass(frozen=True)
class LDMEngineConfig:
    """Task-neutral lifecycle policy for one engine campaign."""

    iterations: int
    reservoir_size: int
    evaluations_per_round: int = 1
    max_empty_reservoir_rounds: int = 3
    target_observations: int | None = None
    target_successful_evaluations: int | None = None
    max_evaluation_attempts: int | None = None
    max_evaluation_attempts_per_round: int | None = None
    replace_failed_evaluations: bool = False

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("engine iterations must be non-negative")
        if self.reservoir_size < 1:
            raise ValueError("engine reservoir_size must be positive")
        if self.evaluations_per_round < 1:
            raise ValueError("engine evaluations_per_round must be positive")
        if self.max_empty_reservoir_rounds < 1:
            raise ValueError("engine max_empty_reservoir_rounds must be positive")
        if self.target_observations is not None and self.target_observations < 0:
            raise ValueError("engine target_observations must be non-negative")
        if (
            self.target_successful_evaluations is not None
            and self.target_successful_evaluations < 0
        ):
            raise ValueError(
                "engine target_successful_evaluations must be non-negative"
            )
        if (
            self.target_observations is not None
            and self.target_successful_evaluations is not None
        ):
            raise ValueError(
                "engine must target observations or successful evaluations, not both"
            )
        if self.max_evaluation_attempts is not None and self.max_evaluation_attempts < 0:
            raise ValueError("engine max_evaluation_attempts must be non-negative")
        if (
            self.max_evaluation_attempts_per_round is not None
            and self.max_evaluation_attempts_per_round < 1
        ):
            raise ValueError(
                "engine max_evaluation_attempts_per_round must be positive"
            )
        if self.replace_failed_evaluations and self.target_successful_evaluations is None:
            raise ValueError(
                "replace_failed_evaluations requires target_successful_evaluations"
            )


@dataclass
class LDMEngineState:
    """Caller-resumable in-memory engine state."""

    observations: list[Observation] = field(default_factory=list)
    expansion_schema: dict[str, Any] = field(default_factory=dict)
    next_round: int = 0
    empty_reservoir_rounds: int = 0

    def __post_init__(self) -> None:
        if self.next_round < 0:
            raise ValueError("engine next_round must be non-negative")
        if self.empty_reservoir_rounds < 0:
            raise ValueError("engine empty_reservoir_rounds must be non-negative")

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "next_round": self.next_round,
            "empty_reservoir_rounds": self.empty_reservoir_rounds,
            "expansion_schema": _jsonable(self.expansion_schema),
            "observations": [_jsonable(item.to_dict()) for item in self.observations],
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "LDMEngineState":
        observations: list[Observation] = []
        raw_observations = payload.get("observations", [])
        if not isinstance(raw_observations, list):
            raise ValueError("engine checkpoint observations must be a list")
        for raw in raw_observations:
            if not isinstance(raw, Mapping):
                raise ValueError("engine checkpoint observation must be an object")
            candidate_payload = raw.get("candidate")
            evaluation_payload = raw.get("evaluation")
            if not isinstance(candidate_payload, Mapping) or not isinstance(
                evaluation_payload, Mapping
            ):
                raise ValueError("engine checkpoint observation records are malformed")
            candidate = Candidate(
                candidate_id=str(candidate_payload.get("candidate_id", "")),
                payload=candidate_payload.get("payload"),
                canonical_key=str(candidate_payload.get("canonical_key", "")),
                source=str(candidate_payload.get("source", "")),
                metadata=dict(candidate_payload.get("metadata", {})),
            )
            evaluation = EvaluationResult(
                candidate_id=str(evaluation_payload.get("candidate_id", "")),
                status=str(evaluation_payload.get("status", "failed")),  # type: ignore[arg-type]
                metrics=dict(evaluation_payload.get("metrics", {})),
                artifacts=dict(evaluation_payload.get("artifacts", {})),
                resource_usage=dict(evaluation_payload.get("resource_usage", {})),
                error=str(evaluation_payload.get("error", "")),
                metadata=dict(evaluation_payload.get("metadata", {})),
            )
            round_idx = raw.get("round_idx")
            observations.append(
                Observation(
                    candidate=candidate,
                    evaluation=evaluation,
                    surrogate=None,
                    round_idx=None if round_idx is None else int(round_idx),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
        expansion_schema = payload.get("expansion_schema", {})
        if not isinstance(expansion_schema, Mapping):
            raise ValueError("engine checkpoint expansion_schema must be an object")
        return cls(
            observations=observations,
            expansion_schema=dict(expansion_schema),
            next_round=int(payload.get("next_round", 0)),
            empty_reservoir_rounds=int(payload.get("empty_reservoir_rounds", 0)),
        )


@dataclass(frozen=True)
class LDMEngineResult:
    """Final engine state and publication-facing summary."""

    state: LDMEngineState
    rounds_run: int
    stop_reason: str
    summary: dict[str, Any]


ParentSelector = Callable[[Sequence[Observation], ObjectiveSet], Optional[Candidate]]


class LDMEngine:
    """Coordinate expansion, admission, selection, evaluation, and persistence."""

    def __init__(
        self,
        *,
        task_spec: LDMTaskSpec,
        expander: ReservoirExpander,
        candidate_domain: CandidateDomainAdapter,
        evaluator: CandidateEvaluator,
        runtime: CampaignRuntime,
        selector: AcquisitionSelector | None = None,
        surrogate_encoder: SurrogateEncoder | None = None,
        parent_selector: ParentSelector | None = None,
    ) -> None:
        if selector is None and surrogate_encoder is not None:
            raise ValueError("surrogate_encoder requires a selector")
        if runtime.task != task_spec.task:
            raise ValueError(
                f"campaign runtime task {runtime.task!r} does not match "
                f"task spec {task_spec.task!r}"
            )
        self.task_spec = task_spec
        self.expander = expander
        self.reservoir_builder = ReservoirBuilder(candidate_domain)
        self.evaluator = evaluator
        self.runtime = runtime
        self.selector = selector
        self.surrogate_encoder = surrogate_encoder
        self.parent_selector = parent_selector or _default_parent
        self.objectives = ObjectiveSet.from_specs(task_spec.objectives)
        if selector is not None:
            selector_spec = selector.describe()
            if tuple(selector_spec.objective_names) != self.objectives.names:
                raise ValueError(
                    "selector objectives do not match the task objective declaration"
                )
        if surrogate_encoder is not None:
            encoder_spec = surrogate_encoder.describe()
            if task_spec.surrogate.kind == "none":
                raise ValueError("task spec disables the surrogate used by the selector")
            comparable_fields = ("kind", "dimension_policy", "dimension", "version")
            mismatches = [
                name
                for name in comparable_fields
                if getattr(encoder_spec, name) != getattr(task_spec.surrogate, name)
            ]
            if mismatches:
                raise ValueError(
                    "surrogate encoder description does not match task spec field(s): "
                    + ", ".join(mismatches)
                )

    def run(
        self,
        config: LDMEngineConfig,
        *,
        state: LDMEngineState | None = None,
        context: Mapping[str, Any] | None = None,
        finalize_runtime: bool = True,
    ) -> LDMEngineResult:
        checkpoint = self.runtime.load_checkpoint() if state is None else None
        active = state or (LDMEngineState.from_checkpoint(checkpoint) if checkpoint else LDMEngineState())
        rounds_run = 0
        stop_reason = "iteration_budget"
        try:
            completed = _completion_reason(active, config)
            if completed is not None:
                stop_reason = completed
            for round_idx in range(active.next_round, config.iterations):
                pending = self.runtime.budget.metadata.get("engine_active_selection")
                if pending is not None and pending["round_idx"] == round_idx:
                    active.expansion_schema.update(pending["expansion_schema"])
                    exhausted = self._execute_selection(active, pending, config, recovering=True)
                    active.next_round = round_idx + 1
                    rounds_run += 1
                    self._checkpoint(active)
                    self.runtime.budget.metadata.pop("engine_active_selection", None)
                    self.runtime.budget.write()
                    if exhausted:
                        stop_reason = _evaluation_budget_stop_reason(self.runtime, active, config) or "evaluation_attempt_budget"
                        break
                    continue
                completed = _completion_reason(active, config)
                if completed is not None:
                    stop_reason = completed
                    break
                remaining_attempts = _remaining_fresh_attempts(self.runtime, active, config)
                exhausted = _evaluation_budget_stop_reason(self.runtime, active, config)
                if exhausted is not None:
                    stop_reason = exhausted
                    break
                self.runtime.consume_many(
                    {"outer_iterations": 1}, usage_key=f"engine:round:{round_idx}"
                )
                self.runtime.status.update(
                    "running",
                    phase="reservoir_expansion",
                    iteration=round_idx,
                    budget=self.runtime.budget,
                )
                parent = self.parent_selector(active.observations, self.objectives)
                desired = _desired_round_results(active, config)
                effective = min(desired, config.max_evaluation_attempts_per_round or desired)
                if remaining_attempts is not None:
                    effective = min(effective, remaining_attempts)
                expansion = self.expander.expand(
                    ExpansionRequest(
                        round_idx=round_idx,
                        reservoir_size=config.reservoir_size,
                        observations=tuple(active.observations),
                        parent=parent,
                        expansion_schema=dict(active.expansion_schema),
                        context={**dict(context or {}), "evaluation_budget": {
                            "requested": config.evaluations_per_round,
                            "effective": effective,
                            "remaining_new_attempts": remaining_attempts,
                        }},
                    )
                )
                if expansion.schema_update is not None:
                    active.expansion_schema.update(expansion.schema_update)
                self.runtime.record(
                    "reservoir_expanded",
                    {
                        "proposal_count": len(expansion.proposals),
                        "attempts": [item.to_dict() for item in expansion.attempts],
                        "schema_update": expansion.schema_update,
                        "metadata": expansion.metadata,
                    },
                    iteration=round_idx,
                )
                if expansion.attempts:
                    consume_proposal_attempts(self.runtime, expansion.attempts)

                reservoir_limit = config.reservoir_size
                if self.task_spec.reservoir.max_size is not None:
                    reservoir_limit = min(reservoir_limit, self.task_spec.reservoir.max_size)
                reservoir = self.reservoir_builder.build(
                    expansion.proposals,
                    evaluated_keys=(item.canonical_key for item in active.observations),
                    max_size=reservoir_limit,
                    metadata={"round_idx": round_idx},
                )
                self.runtime.record(
                    "reservoir_built",
                    {
                        "candidate_ids": [item.candidate_id for item in reservoir.candidates],
                        "drop_counts": reservoir.drop_counts,
                        "rejections": [item.to_dict() for item in reservoir.rejections],
                    },
                    iteration=round_idx,
                )
                if reservoir.candidates:
                    self.runtime.consume_many(
                        {"valid_search_candidates": len(reservoir.candidates)},
                        usage_key=f"engine:reservoir:{round_idx}",
                    )

                if not reservoir.candidates:
                    schema_only = expansion.schema_update is not None and not expansion.proposals
                    active.empty_reservoir_rounds = (
                        0 if schema_only else active.empty_reservoir_rounds + 1
                    )
                    active.next_round = round_idx + 1
                    rounds_run += 1
                    self._checkpoint(active)
                    if (
                        not schema_only
                        and active.empty_reservoir_rounds
                        >= config.max_empty_reservoir_rounds
                    ):
                        stop_reason = "empty_reservoir_limit"
                        break
                    continue

                active.empty_reservoir_rounds = 0
                selection_count = effective
                if config.replace_failed_evaluations:
                    selection_count = (
                        config.max_evaluation_attempts_per_round
                        or len(reservoir.candidates)
                    )
                selection_count = min(selection_count, len(reservoir.candidates))
                if remaining_attempts is not None:
                    selection_count = min(selection_count, remaining_attempts)
                selection = self._select(
                    active.observations,
                    reservoir.candidates,
                    selection_count,
                    round_idx=round_idx,
                    use_reservoir_order=expansion.selection_mode == "reservoir_order",
                )
                selected = self._resolve_selection(reservoir.candidates, selection)
                if not selected:
                    stop_reason = "empty_selection"
                    active.next_round = round_idx + 1
                    rounds_run += 1
                    self._checkpoint(active)
                    break

                pending = {
                    "round_idx": round_idx, "desired": desired,
                    "expansion_schema": _jsonable(active.expansion_schema),
                    "selection": _jsonable(selection.to_dict()),
                    "entries": [{"candidate": _jsonable(candidate.to_dict()),
                                 "receipt": self._evaluation_attempt_usage_key(candidate),
                                 "dispatched": False, "result": None}
                                for candidate in selected],
                }
                self.runtime.budget.metadata["engine_active_selection"] = pending
                self.runtime.budget.write()
                self._checkpoint(active)
                budget_exhausted = self._execute_selection(active, pending, config, recovering=False)
                active.next_round = round_idx + 1
                rounds_run += 1
                self._checkpoint(active)
                self.runtime.budget.metadata.pop("engine_active_selection", None)
                self.runtime.budget.write()
                if budget_exhausted:
                    stop_reason = _evaluation_budget_stop_reason(self.runtime, active, config) or "evaluation_attempt_budget"
                    break
                completed = _completion_reason(active, config)
                if completed is not None:
                    stop_reason = completed
                    break

            stop_reason = _completion_reason(active, config) or stop_reason
            if stop_reason == "iteration_budget" and active.next_round < config.iterations:
                stop_reason = _evaluation_budget_stop_reason(self.runtime, active, config) or stop_reason
            summary = self._summary(active, rounds_run, stop_reason)
            terminal = (
                "completed"
                if stop_reason
                in {
                    "iteration_budget",
                    "observation_target",
                    "successful_evaluation_target",
                }
                else "stopped"
            )
            if finalize_runtime:
                self.runtime.finish(summary, status=terminal)
            else:
                self.runtime.status.update(
                    "running",
                    phase="awaiting_external_driver",
                    iteration=active.next_round,
                    budget=self.runtime.budget,
                    details=summary,
                )
            return LDMEngineResult(active, rounds_run, stop_reason, summary)
        except EvaluationPaused as exc:
            self._checkpoint(active)
            self.runtime.pause(exc.status, phase="evaluation_recovery", message=str(exc))
            raise
        except Exception as exc:
            self.runtime.fail(exc)
            raise

    def _select(
        self,
        observations: Sequence[Observation],
        candidates: Sequence[Candidate],
        count: int,
        *,
        round_idx: int,
        use_reservoir_order: bool = False,
    ) -> BOSelectionResult:
        if use_reservoir_order or self.selector is None:
            return BOSelectionResult(
                selected_candidate_ids=tuple(item.candidate_id for item in candidates[:count]),
                metadata={
                    "mode": "reservoir_order",
                    "selection_source": "expander" if use_reservoir_order else "engine",
                },
            )
        history = [
            BOObservation.from_observation(
                observation,
                objective_names=self.objectives.names,
                feature=(
                    observation.surrogate
                    if observation.surrogate is not None
                    else (
                        self.surrogate_encoder.encode(observation.candidate)
                        if self.surrogate_encoder is not None
                        else None
                    )
                ),
                metadata={"round_idx": observation.round_idx},
            )
            for observation in observations
            if observation.evaluation.succeeded
        ]
        self.selector.fit(history)
        representations = (
            {
                candidate.candidate_id: self.surrogate_encoder.encode(candidate)
                for candidate in candidates
            }
            if self.surrogate_encoder is not None
            else {}
        )
        return self.selector.select(
            candidates, representations, count=count, round_idx=round_idx
        )

    def _execute_selection(
        self, state: LDMEngineState, pending: dict[str, Any], config: LDMEngineConfig,
        *, recovering: bool,
    ) -> bool:
        round_idx = pending["round_idx"]
        entries = pending["entries"]
        self.runtime.record("candidates_selected", pending["selection"], iteration=round_idx,
                            event_key=f"engine:selection:{round_idx}")
        observed = {item.candidate_id for item in state.observations if item.round_idx == round_idx}
        self.runtime.status.update("running", phase="evaluation_recovery" if recovering else "evaluation",
                                   iteration=round_idx, budget=self.runtime.budget)

        def reserve(index, entry):
            candidate = Candidate(**entry["candidate"])
            receipt = entry["receipt"]
            if recovering and entry["dispatched"]:
                if receipt is None or self._evaluation_attempt_usage_key(candidate) != receipt:
                    raise EvaluationPaused("Dispatched evaluation has no matching durable replay receipt")
            key = receipt if receipt is not None else f"round:{round_idx}:slot:{index}"
            usage_key = f"engine:evaluation_attempt:{key}"
            paid = self.runtime.budget.metadata.get("cumulative_usage", {}).get(usage_key, {})
            if not paid.get("external_evaluations") and _remaining_fresh_attempts(self.runtime, state, config) == 0:
                return None
            self.runtime.consume_many({"selected_candidates": 1, "external_evaluations": 1,
                                       "expensive_evaluation_attempts": 1}, usage_key=usage_key)
            entry["dispatched"] = True
            self.runtime.budget.write()
            return candidate

        def commit(index, entry, result):
            candidate = Candidate(**entry["candidate"])
            expected = entry["receipt"]
            actual = result.metadata.get(EVALUATION_ATTEMPT_RECEIPT_KEY)
            if actual is not None and expected is not None and actual != expected:
                raise ValueError("evaluation result receipt does not match reservation")
            result = EvaluationResult(**{**result.to_dict(), "metadata": {
                **result.metadata, EVALUATION_ATTEMPT_RECEIPT_KEY:
                    expected if expected is not None else f"round:{round_idx}:slot:{index}",
            }})
            entry["result"] = _jsonable(result.to_dict())
            self.runtime.budget.write()
            self._record_evaluation(state, candidate, result, round_idx)
            self._checkpoint(state)

        # Results persist before events/checkpoints; re-publishing cannot call the oracle.
        for index, entry in enumerate(entries):
            if entry["candidate"]["candidate_id"] not in observed and entry["result"] is not None:
                commit(index, entry, EvaluationResult(**entry["result"]))
                observed.add(entry["candidate"]["candidate_id"])
        remaining = [(i, entry) for i, entry in enumerate(entries)
                     if entry["candidate"]["candidate_id"] not in observed]
        if isinstance(self.evaluator, CandidateEvaluationPreparer) and remaining:
            self.evaluator.prepare_evaluations([Candidate(**entry["candidate"]) for _, entry in remaining])
        batched = isinstance(self.evaluator, BatchCandidateEvaluator) and not config.replace_failed_evaluations
        if batched:
            dispatched = []
            for index, entry in remaining:
                candidate = reserve(index, entry)
                if candidate is None:
                    break
                dispatched.append((index, entry, candidate))
            if dispatched:
                results = self._evaluate_batch([item[2] for item in dispatched])
                # Save the entire returned batch before committing its first observation.
                for (_, entry, _), result in zip(dispatched, results, strict=True):
                    entry["result"] = _jsonable(result.to_dict())
                self.runtime.budget.write()
                for (index, entry, _), result in zip(dispatched, results, strict=True):
                    commit(index, entry, result)
            return len(dispatched) < len(remaining)
        for index, entry in remaining:
            completed = [item for item in state.observations if item.round_idx == round_idx]
            count = sum(item.evaluation.succeeded for item in completed) if config.target_successful_evaluations is not None else len(completed)
            if count >= pending["desired"]:
                break
            candidate = reserve(index, entry)
            if candidate is None:
                return True
            commit(index, entry, self._evaluate(candidate))
        return False

    def _evaluation_attempt_usage_key(self, candidate: Candidate) -> str | None:
        """Task hook promises durable replay, or refusal of an ambiguous retry.

        Fresh external calls must use fresh keys (or None), never a reused
        candidate key without a durable evaluation receipt behind it.
        """
        keyer = getattr(self.evaluator, "evaluation_attempt_usage_key", None)
        if keyer is None:
            return None
        value = keyer(candidate)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("evaluation attempt usage key must be a nonempty string")
        return value.strip()

    @staticmethod
    def _evaluation_result_usage_key(evaluation: EvaluationResult) -> str | None:
        receipt = evaluation.metadata.get(EVALUATION_ATTEMPT_RECEIPT_KEY)
        if receipt is None:
            return None
        if not isinstance(receipt, str) or not receipt.strip():
            raise ValueError("evaluation attempt receipt must be a nonempty string")
        return receipt.strip()

    def _resolve_selection(
        self,
        candidates: Sequence[Candidate],
        selection: BOSelectionResult,
    ) -> list[Candidate]:
        by_id = {item.candidate_id: item for item in candidates}
        unknown = [item for item in selection.selected_candidate_ids if item not in by_id]
        if unknown:
            raise ValueError(
                "selector returned candidate ids outside the active reservoir: "
                + ", ".join(unknown)
            )
        return [by_id[item] for item in selection.selected_candidate_ids]

    def _evaluate(self, candidate: Candidate) -> EvaluationResult:
        try:
            result = self.evaluator.evaluate(candidate)
            if result.candidate_id != candidate.candidate_id:
                raise ValueError("evaluator returned a mismatched candidate_id")
            return self.objectives.validate_result(result)
        except EvaluationPaused:
            raise
        except TimeoutError as exc:
            return EvaluationResult(candidate.candidate_id, "timed_out", error=str(exc))
        except Exception as exc:
            return EvaluationResult(candidate.candidate_id, "failed", error=str(exc))

    def _evaluate_batch(
        self, candidates: Sequence[Candidate]
    ) -> tuple[EvaluationResult, ...]:
        evaluator = self.evaluator
        if not isinstance(evaluator, BatchCandidateEvaluator):
            raise TypeError("batch evaluation requires BatchCandidateEvaluator")
        try:
            results = tuple(evaluator.evaluate_batch(candidates))
        except EvaluationPaused:
            raise
        except TimeoutError as exc:
            return tuple(
                EvaluationResult(candidate.candidate_id, "timed_out", error=str(exc))
                for candidate in candidates
            )
        except Exception as exc:
            return tuple(
                EvaluationResult(candidate.candidate_id, "failed", error=str(exc))
                for candidate in candidates
            )
        if len(results) != len(candidates):
            raise ValueError(
                "batch evaluator result count does not match the candidate count"
            )
        validated: list[EvaluationResult] = []
        for candidate, result in zip(candidates, results, strict=True):
            if not isinstance(result, EvaluationResult):
                raise TypeError("batch evaluator must return EvaluationResult objects")
            if result.candidate_id != candidate.candidate_id:
                raise ValueError(
                    "batch evaluator results must preserve candidate order and ids"
                )
            try:
                validated.append(self.objectives.validate_result(result))
            except Exception as exc:
                validated.append(
                    EvaluationResult(candidate.candidate_id, "failed", error=str(exc))
                )
        return tuple(validated)

    def _record_evaluation(
        self,
        state: LDMEngineState,
        candidate: Candidate,
        evaluation: EvaluationResult,
        round_idx: int,
    ) -> int:
        benchmark_jobs = evaluation.resource_usage.get("benchmark_jobs", 0)
        amounts = {}
        if benchmark_jobs:
            amounts["benchmark_jobs"] = benchmark_jobs
        if evaluation.succeeded:
            amounts["successful_evaluations"] = 1
        if amounts:
            usage_key = self._evaluation_result_usage_key(evaluation)
            self.runtime.consume_many(
                amounts,
                usage_key=(
                    f"engine:evaluation_result:{usage_key}"
                    if usage_key is not None
                    else None
                ),
            )
        representation = (
            self.surrogate_encoder.encode(candidate)
            if self.surrogate_encoder is not None and evaluation.succeeded
            else None
        )
        observation = Observation(
            candidate=candidate,
            evaluation=evaluation,
            surrogate=representation,
            round_idx=round_idx,
        )
        state.observations.append(observation)
        self.runtime.record(
            "candidate_evaluated",
            observation.to_dict(),
            iteration=round_idx,
            candidate_id=candidate.candidate_id,
            event_key=f"engine:evaluated:{round_idx}:{candidate.candidate_id}",
        )
        return int(evaluation.succeeded)

    def _checkpoint(self, state: LDMEngineState) -> None:
        self.runtime.checkpoint(state.to_checkpoint())

    def _summary(
        self,
        state: LDMEngineState,
        rounds_run: int,
        stop_reason: str,
    ) -> dict[str, Any]:
        successful = [item for item in state.observations if item.evaluation.succeeded]
        summary: dict[str, Any] = {
            "task": self.task_spec.task,
            "rounds_run": rounds_run,
            "next_round": state.next_round,
            "observation_count": len(state.observations),
            "successful_evaluation_count": len(successful),
            "failed_evaluation_count": len(state.observations) - len(successful),
            "stop_reason": stop_reason,
            "expansion_schema": _jsonable(state.expansion_schema),
        }
        if len(self.objectives.specs) == 1:
            incumbent = self.objectives.incumbent(state.observations)
            summary["incumbent"] = None if incumbent is None else _jsonable(incumbent.to_dict())
        else:
            summary["pareto_candidate_ids"] = [
                item.candidate_id for item in self.objectives.pareto_front(state.observations)
            ]
        return summary


def _default_parent(
    observations: Sequence[Observation], objectives: ObjectiveSet
) -> Candidate | None:
    if not observations:
        return None
    if len(objectives.specs) == 1:
        incumbent = objectives.incumbent(observations)
        return None if incumbent is None else incumbent.candidate
    front = objectives.pareto_front(observations)
    return front[0].candidate if front else None


def consume_proposal_attempts(
    runtime: CampaignRuntime,
    attempts: Sequence[Any],
) -> None:
    fresh_count = 0
    receipts: set[str] = set()
    for attempt in attempts:
        metadata = getattr(attempt, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("proposal attempt metadata must be an object")
        receipt = metadata.get(PROPOSAL_ATTEMPT_RECEIPT_KEY)
        if receipt is None:
            fresh_count += 1
            continue
        if not isinstance(receipt, str) or not receipt.strip():
            raise ValueError("proposal attempt receipt must be a nonempty string")
        receipt = receipt.strip()
        if receipt in receipts:
            raise ValueError("duplicate proposal attempt receipt in expansion")
        receipts.add(receipt)
    for receipt in sorted(receipts):
        runtime.consume_many(
            {"proposal_attempts": 1},
            usage_key=f"engine:proposal_attempt:{receipt}",
        )
    if fresh_count:
        runtime.consume_many({"proposal_attempts": fresh_count})


def _successful_evaluation_count(state: LDMEngineState) -> int:
    return sum(item.evaluation.succeeded for item in state.observations)


def _completion_reason(
    state: LDMEngineState,
    config: LDMEngineConfig,
) -> str | None:
    if (
        config.target_observations is not None
        and len(state.observations) >= config.target_observations
    ):
        return "observation_target"
    if (
        config.target_successful_evaluations is not None
        and _successful_evaluation_count(state)
        >= config.target_successful_evaluations
    ):
        return "successful_evaluation_target"
    return None


def _desired_round_results(
    state: LDMEngineState,
    config: LDMEngineConfig,
) -> int:
    if config.target_observations is not None:
        remaining = max(0, config.target_observations - len(state.observations))
        return min(config.evaluations_per_round, remaining)
    if config.target_successful_evaluations is not None:
        remaining = max(
            0,
            config.target_successful_evaluations
            - _successful_evaluation_count(state),
        )
        return min(config.evaluations_per_round, remaining)
    return config.evaluations_per_round


def _remaining_evaluation_attempts(
    runtime: CampaignRuntime,
    state: LDMEngineState,
    config: LDMEngineConfig,
) -> int | None:
    if config.max_evaluation_attempts is None:
        return None
    consumed = int(
        runtime.budget.counters.get(
            "external_evaluations",
            len(state.observations),
        )
    )
    return max(0, config.max_evaluation_attempts - consumed)


def _evaluation_budget_stop_reason(
    runtime: CampaignRuntime, state: LDMEngineState, config: LDMEngineConfig
) -> str | None:
    if _remaining_evaluation_attempts(runtime, state, config) == 0:
        return "evaluation_attempt_budget"
    for name in ("selected_candidates", "external_evaluations", "expensive_evaluation_attempts"):
        remaining = runtime.budget.remaining(name)
        if remaining is not None and remaining < 1:
            return "external_evaluation_budget"
    return None


def _remaining_fresh_attempts(runtime, state, config):
    limits = [_remaining_evaluation_attempts(runtime, state, config)]
    limits.extend(runtime.budget.remaining(name) for name in
                  ("selected_candidates", "external_evaluations", "expensive_evaluation_attempts"))
    finite = [max(0, int(value)) for value in limits if value is not None]
    return min(finite) if finite else None


def _jsonable(value: Any) -> Any:
    """Normalize task metadata for durable engine artifacts."""

    return json.loads(json.dumps(value, default=str))


__all__ = [
    "LDMEngine",
    "LDMEngineConfig",
    "LDMEngineResult",
    "LDMEngineState",
    "ParentSelector",
    "consume_proposal_attempts",
]
