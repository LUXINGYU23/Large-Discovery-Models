"""Task-neutral Large Discovery Model campaign engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from ldm_tts.contracts import (
    Candidate,
    CandidateDomainAdapter,
    CandidateEvaluator,
    LDMTaskSpec,
    ObjectiveSet,
    Observation,
    ReservoirBuilder,
)
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.evaluation import (
    EvaluationExecutor, evaluation_budget_stop_reason, remaining_fresh_attempts,
)
from ldm_tts.engine.expansion import (
    PROPOSAL_ATTEMPT_RECEIPT_KEY,
    ExpansionRequest,
    ReservoirExpander,
)
from ldm_tts.engine.state import (
    LDMEngineConfig, LDMEngineState, LDMEngineResult,
    jsonable, completion_reason, desired_round_results,
)
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOSelectionResult,
    SurrogateEncoder,
)


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
        evaluations = EvaluationExecutor(self.runtime, self.evaluator, self.objectives, self.surrogate_encoder)
        rounds_run = 0
        stop_reason = "iteration_budget"
        try:
            completed = completion_reason(active, config)
            if completed is not None:
                stop_reason = completed
            for round_idx in range(active.next_round, config.iterations):
                pending = evaluations.pending_for(round_idx)
                if pending is not None:
                    active.expansion_schema.update(pending["expansion_schema"])
                    exhausted = evaluations.execute(active, pending, config, recovering=True)
                    active.next_round = round_idx + 1
                    rounds_run += 1
                    evaluations.complete(active)
                    if exhausted:
                        stop_reason = evaluation_budget_stop_reason(self.runtime, active, config) or "evaluation_attempt_budget"
                        break
                    continue
                completed = completion_reason(active, config)
                if completed is not None:
                    stop_reason = completed
                    break
                remaining_attempts = remaining_fresh_attempts(self.runtime, active, config)
                exhausted = evaluation_budget_stop_reason(self.runtime, active, config)
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
                desired = desired_round_results(active, config)
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

                pending = evaluations.prepare(active, selected, selection, desired, round_idx)
                budget_exhausted = evaluations.execute(active, pending, config, recovering=False)
                active.next_round = round_idx + 1
                rounds_run += 1
                evaluations.complete(active)
                if budget_exhausted:
                    stop_reason = evaluation_budget_stop_reason(self.runtime, active, config) or "evaluation_attempt_budget"
                    break
                completed = completion_reason(active, config)
                if completed is not None:
                    stop_reason = completed
                    break

            stop_reason = completion_reason(active, config) or stop_reason
            if stop_reason == "iteration_budget" and active.next_round < config.iterations:
                stop_reason = evaluation_budget_stop_reason(self.runtime, active, config) or stop_reason
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
        except BudgetExceededError as exc:
            self._checkpoint(active)
            self.runtime.pause("paused_budget", phase="budget_recovery", message=str(exc))
            raise EvaluationPaused(str(exc), status="paused_budget") from exc
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
            "expansion_schema": jsonable(state.expansion_schema),
        }
        if len(self.objectives.specs) == 1:
            incumbent = self.objectives.incumbent(state.observations)
            summary["incumbent"] = None if incumbent is None else jsonable(incumbent.to_dict())
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


__all__ = [
    "LDMEngine",
    "ParentSelector",
    "consume_proposal_attempts",
]
