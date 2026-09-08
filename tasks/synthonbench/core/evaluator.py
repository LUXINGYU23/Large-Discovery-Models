"""Evaluator adapter that charges exactly one official unique oracle call."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ldm_tts.contracts import Candidate, EvaluationResult, Observation
from tasks.synthonbench.core.constants import OBJECTIVE_NAME


@dataclass
class OfficialSynthonEvaluator:
    """Delegate all new evaluations to an injected official GlobalSynthonTask."""

    task: object

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        reaction_id, synthon_ids = _candidate_tuple(candidate)
        if self.task.seen((reaction_id, synthon_ids)):
            return EvaluationResult(candidate.candidate_id, "invalid", error="duplicate official oracle query")
        utility = float(self.task.score((reaction_id, synthon_ids)))
        trace = self.task.trace[-1]
        return EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={OBJECTIVE_NAME: utility},
            resource_usage={"benchmark_jobs": 1},
            metadata={
                "product_id": candidate.canonical_key,
                "raw_score": float(trace["raw_score"]),
                "utility": utility,
                "official_calls": int(self.task.calls),
                "direction": self.task.direction,
            },
        )

    def restore_observations(self, observations: Sequence[Observation]) -> None:
        """Reconstruct the official task ledger after validating frozen oracle values."""

        if self.task.calls or self.task.observed_ids:
            raise ValueError("official task restore requires a fresh task instance")
        for observation in observations:
            self._restore_observation(observation)

    def _restore_observation(self, observation: Observation) -> None:
        if not observation.evaluation.succeeded:
            return
        reaction_id, synthon_ids = _candidate_tuple(observation.candidate)
        product_id = observation.canonical_key
        if product_id in self.task.observed_ids:
            raise ValueError("checkpoint contains duplicate SynthonBench product IDs")
        raw_score = float(self.task.oracle(reaction_id, synthon_ids))
        utility = self.task.raw_to_utility(raw_score)
        recorded = observation.evaluation.metrics.get(OBJECTIVE_NAME)
        if recorded is None or not math.isclose(utility, recorded, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("checkpoint utility does not match the pinned official oracle")
        self._append_restored_call(product_id, reaction_id, synthon_ids, raw_score, utility)

    def _append_restored_call(self, product_id: str, reaction_id: str, synthon_ids: tuple[int, ...],
                              raw_score: float, utility: float) -> None:
        self.task.calls += 1
        self.task.observed_ids.add(product_id)
        self.task._scores[product_id] = utility
        if utility > self.task.best_utility:
            self.task.best_utility = utility
            self.task.best_raw_score = raw_score
            self.task.best_candidate = (reaction_id, synthon_ids)
        self.task.history_best.append(self.task.best_utility)
        self.task.trace.append({
            "call_idx": self.task.calls,
            "product_id": product_id,
            "reaction_id": reaction_id,
            "synthon_ids": list(synthon_ids),
            "raw_score": raw_score,
            "utility": utility,
        })


def _candidate_tuple(candidate: Candidate) -> tuple[str, tuple[int, ...]]:
    payload = candidate.payload
    if not isinstance(payload, dict):
        raise TypeError("SynthonBench evaluator requires a mapping payload")
    reaction_id = payload.get("reaction_id")
    synthon_ids = payload.get("synthon_ids")
    if not isinstance(reaction_id, str) or not isinstance(synthon_ids, list):
        raise TypeError("SynthonBench evaluator payload is malformed")
    return reaction_id, tuple(int(item) for item in synthon_ids)


__all__ = ["OfficialSynthonEvaluator"]
