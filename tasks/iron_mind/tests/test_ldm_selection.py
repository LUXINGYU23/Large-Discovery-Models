"""Iron Mind acquisition-tilted selection semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.optimization import (
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    SurrogateVector,
)
from tasks.iron_mind.core.candidate import IRON_MIND_Q0_METADATA_KEY
from tasks.iron_mind.core.ldm_policy import robust_z, tilted_probabilities
from tasks.iron_mind.core.ldm_selector import AcquisitionTiltedSelector


class FixedAcquisitionSelector:
    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = dict(scores)
        self.history: tuple[BOObservation, ...] = ()
        self.last_candidate_ids: tuple[str, ...] = ()

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ucb",
            objective_names=("score",),
            score_direction="maximize",
            selection_rule="highest fixed UCB",
            parameters={"beta": 1.0},
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        self.history = tuple(history)

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        del representations
        self.last_candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        predictions = tuple(
            BOPrediction.scalar(
                candidate.candidate_id,
                mean=self.scores[candidate.candidate_id],
                std=0.0,
                acquisition_score=self.scores[candidate.candidate_id],
            )
            for candidate in candidates
        )
        ranked = sorted(predictions, key=lambda item: item.acquisition_score or 0.0, reverse=True)
        return BOSelectionResult(
            tuple(item.candidate_id for item in ranked[:count]),
            predictions,
            metadata={"base": "fixed"},
        )


def test_tilted_probabilities_match_the_ldm_formula() -> None:
    q0 = np.asarray((0.75, 0.25))
    acquisition = np.asarray((0.0, 2.0))

    base_only = tilted_probabilities(q0, acquisition, alpha=1.0, eta=0.0)
    acquisition_only = tilted_probabilities(q0, acquisition, alpha=0.0, eta=3.0)

    np.testing.assert_allclose(base_only, q0)
    assert acquisition_only[1] > acquisition_only[0]
    np.testing.assert_allclose(robust_z(np.ones(3)), np.zeros(3))


def test_selector_uses_duplicate_frequency_as_q0_and_records_tilt_terms() -> None:
    candidates = (_candidate("a", 3, 4), _candidate("b", 1, 4))
    selector = AcquisitionTiltedSelector(
        FixedAcquisitionSelector({"a": 0.0, "b": 2.0}),
        alpha=1.0,
        eta=0.0,
        seed=17,
    )
    selector.fit(())

    result = selector.select(candidates, _vectors(candidates))
    metadata = {item.candidate_id: item.metadata for item in result.predictions}

    assert metadata["a"]["q0_base_mass"] == pytest.approx(0.75)
    assert metadata["b"]["q0_base_mass"] == pytest.approx(0.25)
    assert metadata["a"]["selection_probability"] == pytest.approx(0.75)
    assert metadata["b"]["selection_probability"] == pytest.approx(0.25)
    assert result.metadata["selection_mode"] == "acquisition_tilted_sampling"
    assert result.metadata["base_measure"] == "empirical_proposal_frequency"
    assert result.metadata["alpha_base_measure"] == 1.0
    assert result.metadata["eta_acquisition_tilt"] == 0.0
    assert result.metadata["proposal_base_measure"] == [
        {"candidate_id": "a", "occurrence_count": 3, "proposal_q0_base_mass": 0.75},
        {"candidate_id": "b", "occurrence_count": 1, "proposal_q0_base_mass": 0.25},
    ]


def test_selector_is_seeded_order_invariant_and_samples_without_replacement() -> None:
    candidates = tuple(_candidate(value, 1, 3) for value in ("a", "b", "c"))

    def select(candidate_order):
        selector = AcquisitionTiltedSelector(
            FixedAcquisitionSelector({"a": 0.0, "b": 1.0, "c": 2.0}),
            alpha=1.0,
            eta=3.0,
            seed=23,
        )
        selector.fit(())
        return selector.select(candidate_order, _vectors(candidate_order), count=2)

    forward = select(candidates)
    reversed_result = select(tuple(reversed(candidates)))

    assert forward.selected_candidate_ids == reversed_result.selected_candidate_ids
    assert len(set(forward.selected_candidate_ids)) == 2
    assert forward.metadata["selection_seed"] == reversed_result.metadata["selection_seed"]


def test_selector_maintains_a_smaller_q0_sampled_bo_pool_before_acquisition() -> None:
    candidates = tuple(_candidate(value, 1, 4) for value in ("a", "b", "c", "d"))
    base = FixedAcquisitionSelector({value: float(index) for index, value in enumerate("abcd")})
    selector = AcquisitionTiltedSelector(
        base,
        alpha=1.0,
        eta=3.0,
        seed=29,
        pool_size=2,
        proposal_sample_count=4,
    )
    selector.fit(())

    result = selector.select(candidates, _vectors(candidates))

    assert len(base.last_candidate_ids) == 2
    assert len(result.predictions) == 2
    assert result.metadata["proposal_samples_requested"] == 4
    assert result.metadata["valid_proposal_occurrences"] == 4
    assert result.metadata["unique_candidates_admitted"] == 4
    assert result.metadata["bo_pool_size"] == 2
    assert result.metadata["configured_bo_pool_size"] == 2
    assert result.metadata["pool_maintenance"] == "q0_gumbel_top_k_without_replacement"
    assert set(result.metadata["bo_pool_candidate_ids"]) == set(base.last_candidate_ids)


def test_selector_requires_proposal_oversampling_when_both_sizes_are_configured() -> None:
    with pytest.raises(ValueError, match="proposal_sample_count must exceed pool_size"):
        AcquisitionTiltedSelector(
            FixedAcquisitionSelector({}),
            pool_size=4,
            proposal_sample_count=4,
        )


def _vectors(candidates: Sequence[Candidate]) -> dict[str, SurrogateVector]:
    return {
        candidate.candidate_id: SurrogateVector((float(index),), "test", candidate.candidate_id)
        for index, candidate in enumerate(candidates)
    }


def _candidate(value: str, occurrence_count: int, valid_occurrence_count: int) -> Candidate:
    return Candidate(
        value,
        value,
        value,
        metadata={
            IRON_MIND_Q0_METADATA_KEY: {
                "occurrence_count": occurrence_count,
                "valid_occurrence_count": valid_occurrence_count,
                "probability": occurrence_count / valid_occurrence_count,
            }
        },
    )
