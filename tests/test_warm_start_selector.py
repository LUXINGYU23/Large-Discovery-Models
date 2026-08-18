from __future__ import annotations

import pytest

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.optimization import (
    BOObservation,
    BOSelectionResult,
    SurrogateVector,
    WarmStartAcquisitionSelector,
)


class RecordingSelector:
    def __init__(self) -> None:
        self.spec = AcquisitionSpec(
            name="recording",
            objective_names=("score",),
            score_direction="maximize",
            selection_rule="recorded",
        )
        self.fit_history: tuple[BOObservation, ...] = ()
        self.select_result = BOSelectionResult(("candidate-2",))
        self.select_arguments: tuple[object, ...] = ()

    def describe(self) -> AcquisitionSpec:
        return self.spec

    def fit(self, history: tuple[BOObservation, ...]) -> None:
        self.fit_history = tuple(history)

    def select(
        self,
        candidates: tuple[Candidate, ...],
        representations: dict[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        self.select_arguments = (candidates, representations, count)
        return self.select_result


def observation(candidate_id: str, score: float) -> BOObservation:
    return BOObservation.scalar(candidate_id, score, (score,), feature_version="test-v1")


def test_warm_start_prepends_immutable_priors_without_mutating_history() -> None:
    prior = observation("prior-1", 1.0)
    priors = [prior]
    history = [observation("campaign-1", 2.0)]
    delegate = RecordingSelector()
    selector = WarmStartAcquisitionSelector(delegate, priors)
    priors.append(observation("later-prior", 3.0))

    selector.fit(history)

    assert delegate.fit_history == (prior, history[0])
    assert history == [observation("campaign-1", 2.0)]


def test_warm_start_delegates_describe_and_select_without_rewriting_result() -> None:
    delegate = RecordingSelector()
    selector = WarmStartAcquisitionSelector(delegate, ())
    candidate = Candidate("candidate-2", 2, "2")
    representations = {
        "candidate-2": SurrogateVector((2.0,), "test-v1", source_id="candidate-2")
    }

    result = selector.select((candidate,), representations, count=1)

    assert selector.describe() is delegate.spec
    assert result is delegate.select_result
    assert delegate.select_arguments == ((candidate,), representations, 1)


def test_warm_start_rejects_duplicate_candidate_id_across_priors_and_history() -> None:
    selector = WarmStartAcquisitionSelector(RecordingSelector(), (observation("shared", 1.0),))

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        selector.fit((observation("shared", 2.0),))
