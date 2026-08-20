"""Iron Mind selector adapter for acquisition-tilted finite-reservoir sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.optimization.records import (
    AcquisitionSelector,
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    SurrogateVector,
)
from tasks.iron_mind.core.ldm_policy import (
    AcquisitionTiltConfig,
    DEFAULT_Z_CLIP,
    effective_sample_size,
    gumbel_top_k,
    probability_entropy,
    robust_z,
    softmax_probabilities,
    tilted_logits,
)
from tasks.iron_mind.core.proposal_pool import (
    EmpiricalPool,
    candidate_set_seed,
    empirical_base_masses,
    maintain_empirical_pool,
    proposal_base_measure_records,
)


@dataclass(frozen=True)
class _TiltState:
    base_result: BOSelectionResult
    predictions: tuple[BOPrediction, ...]
    q0: np.ndarray
    normalized_acquisition: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray


class AcquisitionTiltedSelector:
    """Sample from q0^alpha exp(eta * normalized acquisition)."""

    def __init__(
        self,
        base_selector: AcquisitionSelector,
        *,
        alpha: float = 1.0,
        eta: float = 3.0,
        z_clip: float = DEFAULT_Z_CLIP,
        seed: int = 0,
        pool_size: int | None = None,
        proposal_sample_count: int | None = None,
    ) -> None:
        self.base_selector = base_selector
        self.config = AcquisitionTiltConfig(
            alpha=alpha,
            eta=eta,
            z_clip=z_clip,
            seed=seed,
            pool_size=pool_size,
            proposal_sample_count=proposal_sample_count,
        )
        self.history_size = 0

    def describe(self) -> AcquisitionSpec:
        base = self.base_selector.describe()
        return AcquisitionSpec(
            name=f"{base.name}_tilted",
            objective_names=base.objective_names,
            score_direction="sample",
            selection_rule=(
                "sample without replacement from empirical proposal mass tilted by "
                "robust-z acquisition scores"
            ),
            parameters={
                "base_acquisition": base.name,
                "base_acquisition_parameters": dict(base.parameters),
                "base_measure": "empirical_proposal_frequency",
                "alpha_base_measure": self.config.alpha,
                "eta_acquisition_tilt": self.config.eta,
                "normalization": "robust_z",
                "z_clip": self.config.z_clip,
                "sampling": "gumbel_top_k_without_replacement",
                "seed": self.config.seed,
                "pool_size": self.config.pool_size,
                "proposal_sample_count": self.config.proposal_sample_count,
            },
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        self.history_size = len(history)
        self.base_selector.fit(history)

    def select(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        if count < 1:
            raise ValueError("selection count must be positive")
        proposal_reservoir = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        if not proposal_reservoir:
            return BOSelectionResult(())
        pool = maintain_empirical_pool(
            proposal_reservoir,
            self.config,
            self.history_size,
        )
        state = self._score_pool(pool, representations)
        selection_seed = candidate_set_seed(
            self.config.seed,
            self.history_size,
            pool.candidates,
            phase="tilted_selection",
        )
        indices = gumbel_top_k(
            state.probabilities,
            min(count, len(pool.candidates)),
            np.random.default_rng(selection_seed),
        )
        return BOSelectionResult(
            selected_candidate_ids=tuple(pool.candidates[index].candidate_id for index in indices),
            predictions=_annotate_predictions(state),
            fallback_reason=state.base_result.fallback_reason,
            metadata=_selection_metadata(
                state,
                pool,
                self.config,
                selection_seed=selection_seed,
            ),
        )

    def _score_pool(
        self,
        pool: EmpiricalPool,
        representations: Mapping[str, SurrogateVector],
    ) -> _TiltState:
        base = self.base_selector.select(
            pool.candidates,
            representations,
            count=len(pool.candidates),
        )
        predictions = _ordered_predictions(pool.candidates, base.predictions)
        q0 = empirical_base_masses(pool.candidates)
        acquisition = _acquisition_scores(predictions)
        normalized = robust_z(acquisition, clip=self.config.z_clip)
        logits = tilted_logits(q0, acquisition, config=self.config)
        return _TiltState(
            base,
            predictions,
            q0,
            normalized,
            logits,
            softmax_probabilities(logits),
        )


def _ordered_predictions(candidates, predictions) -> tuple[BOPrediction, ...]:
    by_id = {item.candidate_id: item for item in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("base selector returned duplicate candidate predictions")
    missing = [item.candidate_id for item in candidates if item.candidate_id not in by_id]
    if missing:
        raise ValueError("base selector omitted candidate predictions: " + ", ".join(missing))
    return tuple(by_id[item.candidate_id] for item in candidates)


def _acquisition_scores(predictions: Sequence[BOPrediction]) -> np.ndarray:
    values = []
    for prediction in predictions:
        if prediction.acquisition_score is None:
            raise ValueError("base selector prediction has no acquisition score")
        values.append(prediction.acquisition_score)
    acquisition = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(acquisition)):
        raise ValueError("acquisition scores must be finite")
    return acquisition


def _annotate_predictions(state: _TiltState) -> tuple[BOPrediction, ...]:
    return tuple(
        replace(
            prediction,
            metadata={
                **prediction.metadata,
                "q0_base_mass": float(base_mass),
                "normalized_acquisition": float(acquisition_value),
                "tilt_log_weight": float(logit),
                "selection_probability": float(probability),
            },
        )
        for prediction, base_mass, acquisition_value, logit, probability in zip(
            state.predictions,
            state.q0,
            state.normalized_acquisition,
            state.logits,
            state.probabilities,
            strict=True,
        )
    )


def _selection_metadata(
    state: _TiltState,
    pool: EmpiricalPool,
    config: AcquisitionTiltConfig,
    *,
    selection_seed: int,
) -> dict[str, object]:
    return {
        "selection_mode": "acquisition_tilted_sampling",
        "base_measure": "empirical_proposal_frequency",
        "alpha_base_measure": config.alpha,
        "eta_acquisition_tilt": config.eta,
        "normalization": "robust_z",
        "z_clip": config.z_clip,
        "sampling": "gumbel_top_k_without_replacement",
        "seed": config.seed,
        "selection_seed": selection_seed,
        "proposal_samples_requested": config.proposal_sample_count,
        "valid_proposal_occurrences": pool.valid_proposal_occurrences,
        "unique_candidates_admitted": len(pool.proposal_reservoir),
        "proposal_base_measure": proposal_base_measure_records(pool.proposal_reservoir),
        "bo_pool_size": len(pool.candidates),
        "configured_bo_pool_size": config.pool_size,
        "bo_pool_candidate_ids": [item.candidate_id for item in pool.candidates],
        "pool_maintenance": pool.maintenance_method,
        "pool_seed": pool.maintenance_seed,
        "selection_q0_scope": "conditioned_on_maintained_bo_pool",
        "probability_entropy": probability_entropy(state.probabilities),
        "probability_effective_sample_size": effective_sample_size(state.probabilities),
        "base_selection": dict(state.base_result.metadata),
    }


__all__ = ["AcquisitionTiltedSelector", "empirical_base_masses"]
