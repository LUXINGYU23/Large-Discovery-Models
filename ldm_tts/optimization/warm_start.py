"""Acquisition-selector decorator for explicit fixed prior observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOSelectionResult,
    SurrogateVector,
)


class WarmStartAcquisitionSelector:
    """Fit a selector on immutable priors followed by campaign history."""

    def __init__(
        self,
        delegate: AcquisitionSelector,
        priors: Sequence[BOObservation],
    ) -> None:
        self.delegate = delegate
        self.priors = tuple(priors)

    def describe(self) -> AcquisitionSpec:
        return self.delegate.describe()

    def fit(self, history: Sequence[BOObservation]) -> None:
        observations = self.priors + tuple(history)
        _require_unique_candidate_ids(observations)
        self.delegate.fit(observations)

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        return self.delegate.select(candidates, representations, count=count)


def _require_unique_candidate_ids(observations: Sequence[BOObservation]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for observation in observations:
        if observation.candidate_id in seen:
            duplicates.add(observation.candidate_id)
        seen.add(observation.candidate_id)
    if duplicates:
        raise ValueError(
            "duplicate candidate_id across warm-start priors and campaign history: "
            + ", ".join(sorted(duplicates))
        )


__all__ = ["WarmStartAcquisitionSelector"]
