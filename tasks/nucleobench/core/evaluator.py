"""Sequence reconstruction and batch energy evaluation for NucleoBench."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ldm_tts.contracts import Candidate, EvaluationResult
from tasks.nucleobench.core.candidate import (
    MutationContext,
    prepare_candidate_payload,
    rebuild_sequence,
)


@dataclass(frozen=True)
class NucleoBenchEvaluator:
    """Evaluate admitted patches through one task-owned batch energy callable."""

    context: MutationContext
    score_sequences: Callable[[Sequence[str]], Sequence[float]]

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        return self.evaluate_batch((candidate,))[0]

    def evaluate_batch(
        self, candidates: Sequence[Candidate]
    ) -> tuple[EvaluationResult, ...]:
        prepared = []
        sequences = []
        for candidate in candidates:
            item = prepare_candidate_payload(
                candidate.payload,
                self.context,
                allow_empty=True,
            )
            if item.canonical_key != candidate.canonical_key:
                raise ValueError("candidate identity does not match its mutation patch")
            prepared.append(item)
            sequences.append(
                rebuild_sequence(self.context.start_sequence, item.payload["mutations"])
            )

        energies = tuple(self.score_sequences(sequences))
        if len(energies) != len(candidates):
            raise ValueError("batch scorer returned the wrong number of energies")

        results = []
        for candidate, item, raw_energy in zip(
            candidates, prepared, energies, strict=True
        ):
            if isinstance(raw_energy, (bool, str, bytes)):
                raise TypeError("batch scorer energies must be numeric")
            try:
                energy = float(raw_energy)
            except (TypeError, ValueError) as exc:
                raise ValueError("batch scorer energies must be numeric") from exc
            if not math.isfinite(energy):
                raise ValueError("batch scorer energies must be finite")
            results.append(
                EvaluationResult(
                    candidate_id=candidate.candidate_id,
                    status="succeeded",
                    metrics={
                        "energy": energy,
                        "utility": -energy,
                        "hamming_distance": float(item.hamming_distance),
                    },
                    resource_usage={"oracle_calls": 1.0},
                    metadata={"sequence_sha256": item.sequence_sha256},
                )
            )
        return tuple(results)


__all__ = ["NucleoBenchEvaluator"]
