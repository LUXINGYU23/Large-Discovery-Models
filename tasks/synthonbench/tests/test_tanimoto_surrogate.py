"""Task-local chemistry representation and sparse-GP contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ldm_tts.contracts import Candidate
from ldm_tts.optimization import BOObservation, SurrogateVector
from tasks.synthonbench.core.nystrom_encoder import SynthonNystromEncoder
from tasks.synthonbench.core.product_proxy import SynthonProductProxy
from tasks.synthonbench.core.tanimoto_gp import (
    SynthonTanimotoGPUCBSelector,
    TanimotoGPUCBConfig,
)
from tasks.synthonbench.core.tanimoto_kernel import count_tanimoto


@dataclass(frozen=True)
class _Synthon:
    synthon_id: int
    position: int
    reaction_id: str
    smiles: str


class _Space:
    def __init__(self) -> None:
        self._synthons = (
            _Synthon(1, 1, "r1", "C[U]"),
            _Synthon(2, 1, "r1", "CC[U]"),
            _Synthon(11, 2, "r1", "[U]N"),
            _Synthon(12, 2, "r1", "[U]O"),
            _Synthon(21, 1, "r2", "c1cc([Np])ccc1"),
            _Synthon(22, 1, "r2", "CC[Np]"),
        )
        self._by_key = {
            (item.reaction_id, item.position, item.synthon_id): item.smiles
            for item in self._synthons
        }

    def positions(self, reaction_id: str) -> tuple[int, ...]:
        return tuple(sorted({item.position for item in self._synthons if item.reaction_id == reaction_id}))

    def synthon_ids(self, reaction_id: str, position: int) -> tuple[int, ...]:
        return tuple(item.synthon_id for item in self._synthons
                     if item.reaction_id == reaction_id and item.position == position)

    def synthon_smiles(self, reaction_id: str, position: int, synthon_id: int) -> str:
        return self._by_key[(reaction_id, position, synthon_id)]

    def product_count_estimate(self, reaction_id: str) -> int:
        count = 1
        for position in self.positions(reaction_id):
            count *= len(self.synthon_ids(reaction_id, position))
        return count


def test_product_proxy_sums_raw_connector_count_fingerprints() -> None:
    proxy = SynthonProductProxy(_Space(), ("r1", "r2"), fingerprint_bits=128)

    first = proxy.synthon_counts("r1", 1, 1)
    second = proxy.synthon_counts("r1", 2, 11)
    combined = proxy.tuple_counts("r1", (1, 11))

    assert np.array_equal(combined, first + second)
    assert first.sum() > 0
    assert second.sum() > 0


def test_count_tanimoto_uses_min_max_similarity() -> None:
    left = np.asarray((2.0, 1.0, 0.0))
    right = np.asarray((1.0, 3.0, 0.0))

    assert count_tanimoto(left, right) == pytest.approx(2.0 / 5.0)


def test_nystrom_encoder_is_deterministic_and_preserves_prior_diagonal() -> None:
    first = SynthonNystromEncoder(_Space(), ("r1", "r2"), landmark_count=3, seed=7,
                                  fingerprint_bits=128)
    second = SynthonNystromEncoder(_Space(), ("r1", "r2"), landmark_count=3, seed=7,
                                   fingerprint_bits=128)
    candidate = _candidate("r1", (1, 11))

    first_vector = first.encode(candidate)
    second_vector = second.encode(candidate)

    assert first.version == second.version
    assert first_vector == second_vector
    assert len(first_vector.values) == 4
    phi = np.asarray(first_vector.values[:-1])
    residual = first_vector.values[-1]
    assert residual >= 0.0
    assert phi @ phi + residual == pytest.approx(1.0, abs=1.0e-8)


def test_prior_prediction_includes_fitc_residual() -> None:
    encoder = SynthonNystromEncoder(_Space(), ("r1", "r2"), landmark_count=3, seed=4,
                                    fingerprint_bits=128)
    candidate = _candidate("r2", (21,))
    vector = encoder.encode(candidate)
    selector = SynthonTanimotoGPUCBSelector(
        objective_name="utility",
        feature_dimension=encoder.dimension,
        feature_version=encoder.version,
        config=TanimotoGPUCBConfig(mean_std=0.0, signal_std=1.0, observation_noise_std=0.5),
    )

    selector.fit(())
    prediction = selector.select((candidate,), {candidate.candidate_id: vector}).predictions[0]

    assert prediction.scalar_mean == pytest.approx(0.0)
    assert prediction.scalar_std == pytest.approx(1.0, abs=1.0e-8)


def test_online_posterior_matches_closed_form_bayesian_linear_update() -> None:
    version = "test-tanimoto-fitc"
    selector = SynthonTanimotoGPUCBSelector(
        objective_name="utility",
        feature_dimension=3,
        feature_version=version,
        config=TanimotoGPUCBConfig(mean_std=0.0, signal_std=1.0, observation_noise_std=0.5),
    )
    history = (
        BOObservation.scalar("a", 1.0, (0.2, 0.4, 0.0), feature_version=version),
        BOObservation.scalar("b", -0.5, (0.6, -0.1, 0.0), feature_version=version),
    )
    candidate = Candidate("query", {"reaction_id": "r1", "synthon_ids": [1, 11]}, "query", "test")
    query = SurrogateVector((0.3, 0.5, 0.0), version, candidate.candidate_id)

    selector.fit(history)
    prediction = selector.select((candidate,), {candidate.candidate_id: query}).predictions[0]

    design = np.asarray(((0.2, 0.4), (0.6, -0.1)))
    target = np.asarray((1.0, -0.5))
    noise_variance = 0.25
    precision = np.eye(2) + design.T @ design / noise_variance
    covariance = np.linalg.inv(precision)
    mean = covariance @ design.T @ target / noise_variance
    query_feature = np.asarray((0.3, 0.5))

    assert prediction.scalar_mean == pytest.approx(float(query_feature @ mean))
    assert prediction.scalar_std == pytest.approx(float(np.sqrt(query_feature @ covariance @ query_feature)))


def test_selector_rejects_negative_observation_noise() -> None:
    with pytest.raises(ValueError, match="observation_noise_std"):
        TanimotoGPUCBConfig(observation_noise_std=-1.0)


def _candidate(reaction_id: str, synthon_ids: tuple[int, ...]) -> Candidate:
    key = f"{reaction_id}|{'_'.join(str(item) for item in synthon_ids)}"
    return Candidate(
        candidate_id=f"candidate:{key}",
        payload={"reaction_id": reaction_id, "synthon_ids": list(synthon_ids)},
        canonical_key=key,
        source="test",
    )
