"""Official SequenceOptimizer-compatible adapter over the shared LDM engine."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ldm_tts.contracts import Observation
from ldm_tts.engine import LDMEngine, LDMEngineConfig, LDMEngineState
from ldm_tts.engine.run_store import CampaignRuntime
from tasks.nucleobench.core.candidate import (
    MutationContext,
    make_start_candidate,
    prepare_candidate_payload,
    rebuild_sequence,
)
from tasks.nucleobench.core.evaluator import NucleoBenchEvaluator


class NucleoBenchDesigner:
    """Expose one start-bound LDM campaign through NucleoBench's duck-typed API."""

    def __init__(
        self,
        *,
        engine: LDMEngine,
        state: LDMEngineState,
        context: MutationContext,
        reservoir_size: int,
        evaluations_per_step: int,
        max_empty_reservoir_rounds: int = 3,
    ) -> None:
        if reservoir_size < 1 or evaluations_per_step < 1:
            raise ValueError(
                "designer reservoir and evaluation counts must be positive"
            )
        self.engine = engine
        self.state = state
        self.context = context
        self.reservoir_size = reservoir_size
        self.evaluations_per_step = evaluations_per_step
        self.max_empty_reservoir_rounds = max_empty_reservoir_rounds
        self._finished = False
        self._last_stop_reason = "initialized"

    @property
    def active_steps(self) -> int:
        return max(0, self.state.next_round - 1)

    def run(self, n_steps: int) -> None:
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if n_steps == 0:
            return
        result = self.engine.run(
            LDMEngineConfig(
                iterations=self.state.next_round + n_steps,
                reservoir_size=self.reservoir_size,
                evaluations_per_round=self.evaluations_per_step,
                max_empty_reservoir_rounds=self.max_empty_reservoir_rounds,
            ),
            state=self.state,
            context={
                "case_id": self.context.case.case_id,
                "start_index": self.context.start_index,
                "start_set_digest": self.context.start_set_digest,
            },
            finalize_runtime=False,
        )
        self.state = result.state
        self._last_stop_reason = result.stop_reason
        if result.rounds_run == n_steps:
            return
        if result.stop_reason in {"empty_reservoir_limit", "empty_selection"}:
            self._finished = True
            return
        error = RuntimeError(
            f"LDM engine advanced {result.rounds_run} of {n_steps} requested steps "
            f"before {result.stop_reason}"
        )
        self.engine.runtime.fail(error)
        raise error

    def get_samples(self, n_samples: int) -> list[str]:
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, int)
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer")
        ranked = sorted(
            (
                observation
                for observation in self.state.observations
                if observation.evaluation.succeeded
            ),
            key=lambda observation: (
                -observation.evaluation.metrics["utility"],
                observation.canonical_key,
            ),
        )
        samples: list[str] = []
        seen: set[str] = set()
        for observation in ranked:
            prepared = prepare_candidate_payload(
                observation.candidate.payload,
                self.context,
                allow_empty=True,
            )
            if prepared.sequence_sha256 in seen:
                continue
            seen.add(prepared.sequence_sha256)
            samples.append(
                rebuild_sequence(
                    self.context.start_sequence,
                    prepared.payload["mutations"],
                )
            )
            if len(samples) == n_samples:
                break
        return samples

    def measured_sample_energies(self, sequences: Sequence[str]) -> list[float]:
        """Export existing measurements without spending another oracle query."""
        measured = {
            item.evaluation.metadata["sequence_sha256"]: item.metrics["energy"]
            for item in self.state.observations if item.evaluation.succeeded
        }
        digests = [hashlib.sha256(sequence.encode()).hexdigest() for sequence in sequences]
        if any(digest not in measured for digest in digests):
            raise ValueError("official export requested a sequence without a measurement")
        self.engine.runtime.record("official_export_scores_reused", {
            "sequence_sha256": digests, "count": len(digests),
        })
        return [float(measured[digest]) for digest in digests]

    def is_finished(self) -> bool:
        return self._finished

    def summary(self) -> dict[str, object]:
        successful = [
            item for item in self.state.observations if item.evaluation.succeeded
        ]
        return {
            "task": self.engine.task_spec.task,
            "rounds_run": self.active_steps,
            "next_round": self.state.next_round,
            "observation_count": len(self.state.observations),
            "successful_evaluation_count": len(successful),
            "failed_evaluation_count": len(self.state.observations) - len(successful),
            "stop_reason": self._last_stop_reason,
        }


def initialize_designer_state(
    context: MutationContext,
    evaluator: NucleoBenchEvaluator,
    runtime: CampaignRuntime,
) -> LDMEngineState:
    """Evaluate the paired start once outside the active-search budget."""

    candidate = make_start_candidate(context)
    evaluation = evaluator.evaluate(candidate)
    if not evaluation.succeeded:
        raise RuntimeError(f"failed to evaluate the paired start: {evaluation.error}")
    runtime.consume("initialization_evaluations")
    observation = Observation(
        candidate=candidate,
        evaluation=evaluation,
        round_idx=0,
        metadata={"phase": "shared_initialization"},
    )
    state = LDMEngineState(observations=[observation], next_round=1)
    runtime.record(
        "baseline_evaluated",
        observation.to_dict(),
        iteration=0,
        candidate_id=candidate.candidate_id,
    )
    runtime.checkpoint(state.to_checkpoint())
    return state


__all__ = ["NucleoBenchDesigner", "initialize_designer_state"]
