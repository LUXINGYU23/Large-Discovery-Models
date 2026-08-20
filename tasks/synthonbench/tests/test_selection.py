"""Task-local LDM pool and acquisition-tilt semantics."""

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
from tasks.synthonbench.core.constants import Q0_METADATA_KEY
from tasks.synthonbench.core.ldm_policy import (
    robust_z,
    softmax_probabilities,
    tilted_logits,
)
from tasks.synthonbench.core.ldm_selector import AcquisitionTiltedSelector
from tasks.synthonbench.core.tanimoto_gp import TanimotoGPUCBConfig


class FixedSelector:
    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = dict(scores)
        self.last_candidates: tuple[str, ...] = ()

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec("ucb", ("synthon_utility",), "maximize", "fixed")

    def fit(self, _history: Sequence[BOObservation]) -> None:
        return None

    def select(
        self,
        candidates: Sequence[Candidate],
        _representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        self.last_candidates = tuple(item.candidate_id for item in candidates)
        predictions = tuple(
            BOPrediction.scalar(
                item.candidate_id,
                mean=self.scores[item.candidate_id],
                std=0.0,
                acquisition_score=self.scores[item.candidate_id],
            )
            for item in candidates
        )
        ranked = sorted(predictions, key=lambda item: item.acquisition_score or 0.0, reverse=True)
        return BOSelectionResult(tuple(item.candidate_id for item in ranked[:count]), predictions)


def test_tilted_logits_match_the_declared_ldm_formula() -> None:
    q0 = np.asarray((0.75, 0.25))
    acquisition = np.asarray((0.0, 2.0))
    config = _selector_config(alpha=1.0, eta=0.0)

    probabilities = softmax_probabilities(tilted_logits(q0, acquisition, config=config.config))

    np.testing.assert_allclose(probabilities, q0)
    np.testing.assert_allclose(robust_z(np.ones(3)), np.zeros(3))


def test_selector_maintains_a_smaller_q0_pool_before_scoring() -> None:
    candidates = tuple(_candidate(name, 1, 4) for name in "abcd")
    base = FixedSelector({name: float(index) for index, name in enumerate("abcd")})
    selector = AcquisitionTiltedSelector(base, alpha=1.0, eta=3.0, z_clip=5.0,
                                         seed=17, pool_size=2, proposal_sample_count=4)

    selector.fit(())
    result = selector.select(candidates, _vectors(candidates))

    assert len(base.last_candidates) == 2
    assert len(result.predictions) == 2
    assert result.metadata["proposal_samples_requested"] == 4
    assert result.metadata["valid_proposal_occurrences"] == 4
    assert result.metadata["unique_candidates_admitted"] == 4
    assert result.metadata["bo_pool_size"] == 2
    assert result.metadata["pool_maintenance"] == "q0_gumbel_top_k_without_replacement"


def test_selector_preserves_duplicate_frequency_in_the_empirical_base_measure() -> None:
    candidates = (_candidate("a", 3, 4), _candidate("b", 1, 4))
    selector = AcquisitionTiltedSelector(FixedSelector({"a": 0.0, "b": 1.0}), alpha=1.0,
                                         eta=0.0, z_clip=5.0, seed=0,
                                         pool_size=2, proposal_sample_count=4)

    selector.fit(())
    result = selector.select(candidates, _vectors(candidates))
    metadata = {item.candidate_id: item.metadata for item in result.predictions}

    assert metadata["a"]["q0_base_mass"] == pytest.approx(0.75)
    assert metadata["b"]["q0_base_mass"] == pytest.approx(0.25)
    assert metadata["a"]["selection_probability"] == pytest.approx(0.75)
    assert metadata["b"]["selection_probability"] == pytest.approx(0.25)


def test_tanimoto_gp_rejects_an_invalid_confidence_probability() -> None:
    with pytest.raises(ValueError, match="smaller than one"):
        TanimotoGPUCBConfig(confidence_delta=1.0)


def _selector_config(alpha: float, eta: float) -> AcquisitionTiltedSelector:
    return AcquisitionTiltedSelector(FixedSelector({}), alpha=alpha, eta=eta, z_clip=5.0,
                                     seed=0, pool_size=2, proposal_sample_count=4)


def _candidate(name: str, occurrences: int, total: int) -> Candidate:
    return Candidate(
        candidate_id=name,
        payload={"reaction_id": "r1", "synthon_ids": [1, 2]},
        canonical_key=name,
        source="test",
        metadata={
            Q0_METADATA_KEY: {
                "occurrence_count": occurrences,
                "valid_occurrence_count": total,
                "probability": occurrences / total,
            }
        },
    )


def _vectors(candidates: Sequence[Candidate]) -> dict[str, SurrogateVector]:
    return {
        item.candidate_id: SurrogateVector((float(index),), "test", item.candidate_id)
        for index, item in enumerate(candidates)
    }
