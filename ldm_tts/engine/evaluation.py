"""Durable evaluation reservation, execution and observation publication."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ldm_tts.contracts import (BatchCandidateEvaluator, Candidate, CandidateEvaluationPreparer,
    CandidateEvaluator, EvaluationResult, ObjectiveSet, Observation)
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY, EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, CampaignRuntime
from ldm_tts.engine.state import LDMEngineConfig, LDMEngineState, jsonable
from ldm_tts.optimization.records import BOSelectionResult, SurrogateEncoder


class EvaluationExecutor:
    def __init__(self, runtime: CampaignRuntime, evaluator: CandidateEvaluator,
                 objectives: ObjectiveSet, surrogate_encoder: SurrogateEncoder | None):
        self.runtime = runtime
        self.evaluator = evaluator
        self.objectives = objectives
        self.surrogate_encoder = surrogate_encoder

    def pending_for(self, round_idx: int) -> dict[str, Any] | None:
        pending = self.runtime.budget.metadata.get("engine_active_selection")
        return pending if pending is not None and pending["round_idx"] == round_idx else None

    def prepare(self, state: LDMEngineState, selected: Sequence[Candidate],
                selection: BOSelectionResult, desired: int, round_idx: int) -> dict[str, Any]:
        pending = {
            "round_idx": round_idx, "desired": desired,
            "expansion_schema": jsonable(state.expansion_schema),
            "selection": jsonable(selection.to_dict()),
            "entries": [{"candidate": jsonable(candidate.to_dict()),
                         "receipt": None,
                         "dispatched": False, "result": None}
                        for candidate in selected],
        }
        self.runtime.budget.metadata["engine_active_selection"] = pending
        self.runtime.budget.write()
        self.runtime.checkpoint(state.to_checkpoint())
        return pending

    def complete(self, state: LDMEngineState) -> None:
        self.runtime.checkpoint(state.to_checkpoint())
        self.runtime.budget.metadata.pop("engine_active_selection", None)
        self.runtime.budget.write()

    def execute(
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
            actual_receipt = self._evaluation_attempt_usage_key(candidate)
            receipt = entry["receipt"]
            if receipt is None and entry["dispatched"]:
                raise EvaluationPaused("Dispatched evaluation has no durable replay receipt")
            if receipt is not None and receipt != actual_receipt:
                raise EvaluationPaused("Evaluation receipt changed after selection preparation")
            if receipt is None:
                receipt = actual_receipt
                entry["receipt"] = receipt
                self.runtime.budget.write()
            key = receipt if receipt is not None else f"round:{round_idx}:slot:{index}"
            usage_key = f"engine:evaluation_attempt:{key}"
            paid = self.runtime.budget.metadata.get("cumulative_usage", {}).get(usage_key, {})
            if not paid.get("external_evaluations") and remaining_fresh_attempts(self.runtime, state, config) == 0:
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
            entry["result"] = jsonable(result.to_dict())
            self.runtime.budget.write()
            self.record(state, candidate, result, round_idx)
            self.runtime.checkpoint(state.to_checkpoint())

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
                    entry["result"] = jsonable(result.to_dict())
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

    def _evaluate(self, candidate: Candidate) -> EvaluationResult:
        try:
            result = self.evaluator.evaluate(candidate)
            if result.candidate_id != candidate.candidate_id:
                raise ValueError("evaluator returned a mismatched candidate_id")
            return self.objectives.validate_result(result)
        except (EvaluationPaused, BudgetExceededError):
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
        except (EvaluationPaused, BudgetExceededError):
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

    def record(
        self,
        state: LDMEngineState,
        candidate: Candidate,
        evaluation: EvaluationResult,
        round_idx: int,
    ) -> None:
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


def evaluation_budget_stop_reason(
    runtime: CampaignRuntime, state: LDMEngineState, config: LDMEngineConfig
) -> str | None:
    if _remaining_evaluation_attempts(runtime, state, config) == 0:
        return "evaluation_attempt_budget"
    for name in ("selected_candidates", "external_evaluations", "expensive_evaluation_attempts"):
        remaining = runtime.budget.remaining(name)
        if remaining is not None and remaining < 1:
            return "external_evaluation_budget"
    return None


def remaining_fresh_attempts(runtime, state, config):
    limits = [_remaining_evaluation_attempts(runtime, state, config)]
    limits.extend(runtime.budget.remaining(name) for name in
                  ("selected_candidates", "external_evaluations", "expensive_evaluation_attempts"))
    finite = [max(0, int(value)) for value in limits if value is not None]
    return min(finite) if finite else None
