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
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("oracle batch size must be a positive integer")

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

        batch_size = self.batch_size or max(1, len(sequences))
        energies = []
        for offset in range(0, len(sequences), batch_size):
            batch = sequences[offset:offset + batch_size]
            batch_energies = tuple(self.score_sequences(batch))
            if len(batch_energies) != len(batch):
                raise ValueError("batch scorer returned the wrong number of energies")
            energies.extend(batch_energies)

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
