"""Empirical-q0 pool maintenance and acquisition-tilted LDM selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.harness import CompiledOptimizationPolicy, PolicyResearchController
from ldm_tts.harness.policy_diagnostics import prepare_policy_research
from ldm_tts.optimization import (
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    SurrogateVector,
)
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
from tasks.nucleobench.core.hamming_gp import HammingGPUCBSelector
from tasks.nucleobench.core.optimization_policy import NucleoOptimizationPolicyAdapter

EPSILON = 1.0e-12
MAD_SCALE = 1.4826


@dataclass(frozen=True)
class _TiltConfig:
    alpha: float
    eta: float
    z_clip: float
    seed: int
    pool_size: int
    proposal_sample_count: int

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("eta", self.eta)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.z_clip) or self.z_clip <= 0.0:
            raise ValueError("z_clip must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.pool_size < 1 or self.proposal_sample_count < 1:
            raise ValueError("pool and proposal sample counts must be positive")
        if self.proposal_sample_count <= self.pool_size:
            raise ValueError("proposal_sample_count must exceed pool_size")


@dataclass(frozen=True)
class _EmpiricalPool:
    proposal_reservoir: tuple[Candidate, ...]
    candidates: tuple[Candidate, ...]
    valid_occurrences: int
    seed: int | None
    method: str


class AcquisitionTiltedSelector:
    """Sample q0(x)^alpha exp(eta z(UCB(x))) without replacement."""

    def __init__(
        self,
        base_selector,
        *,
        alpha: float,
        eta: float,
        z_clip: float,
        seed: int,
        pool_size: int,
        proposal_sample_count: int,
        policy_controller: PolicyResearchController | None = None,
        policy_adapter: NucleoOptimizationPolicyAdapter | None = None,
        policy_mode: bool = False,
    ) -> None:
        self.base_selector = base_selector
        self.config = _TiltConfig(
            alpha,
            eta,
            z_clip,
            seed,
            pool_size,
            proposal_sample_count,
        )
        if (policy_controller is None) != (policy_adapter is None):
            raise ValueError("compiled policy controller and adapter must be configured together")
        if policy_controller is not None and not isinstance(base_selector, HammingGPUCBSelector):
            raise TypeError("compiled Nucleo policies require the Hamming GP selector")
        self.policy_controller = policy_controller
        self.policy_adapter = policy_adapter
        self.policy_mode = policy_mode or policy_controller is not None
        self.history: tuple[BOObservation, ...] = ()
        self.history_size = 0

    def describe(self) -> AcquisitionSpec:
        base = self.base_selector.describe()
        parameters = {
            "base_acquisition": base.name,
            "base_acquisition_parameters": dict(base.parameters),
            "base_measure": "empirical_proposal_frequency",
            "alpha_base_measure": self.config.alpha,
            "eta_acquisition_tilt": self.config.eta,
            "normalization": "robust_z",
            "z_clip": self.config.z_clip,
            "pool_size": self.config.pool_size,
            "proposal_sample_count": self.config.proposal_sample_count,
            "sampling": "gumbel_top_k_without_replacement",
            "seed": self.config.seed,
        }
        if self.policy_mode:
            parameters["optimization_policy"] = "persistent_harness_compiled"
        return AcquisitionSpec(
            name=f"{base.name}_tilted",
            objective_names=base.objective_names,
            score_direction="sample",
            selection_rule=(
                "empirical proposal frequency tilted by robust-z GP-UCB and "
                "sampled without replacement"
            ),
            parameters=parameters,
        )

    def fit(self, history: Sequence[BOObservation]) -> None:
        self.history = tuple(history)
        self.history_size = len(history)
        if self.policy_controller is None:
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
        reservoir = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        if not reservoir:
            return BOSelectionResult(())
        pool = _maintain_pool(reservoir, self.config, self.history_size)
        q0 = _base_masses(pool.candidates)
        policy: CompiledOptimizationPolicy | None = None
        if self.policy_controller is not None and self.history:
            assert isinstance(self.base_selector, HammingGPUCBSelector)
            assert self.policy_adapter is not None
            self.base_selector.fit(self.history)
            baseline = self.base_selector.select(
                pool.candidates, representations, count=len(pool.candidates)
            )
            ordered_baseline = _ordered_predictions(
                pool.candidates, baseline.predictions
            )
            round_input = self.policy_adapter.build_selection_round(
                history=self.history,
                candidates=pool.candidates,
                representations=representations,
                q0=q0,
                baseline_predictions=ordered_baseline,
                valid_proposal_occurrences=pool.valid_occurrences,
            )
            round_input = prepare_policy_research(
                round_input, history=self.history, candidates=pool.candidates,
                representations=representations, baseline=ordered_baseline,
                q0=q0, selector=self.base_selector, z_clip=self.config.z_clip,
                normalize_acquisition=lambda values: _robust_z(values, self.config.z_clip),
            )
            policy = self.policy_controller.resolve(round_input)
            self.base_selector.fit(
                self.history,
                history_prior_mean=policy.history_prior_mean,
                mean_source=f"compiled:{policy.source}",
                artifact_digest=policy.artifact_digest,
            )
            base = self.base_selector.select(
                pool.candidates,
                representations,
                count=len(pool.candidates),
                query_prior_mean=policy.query_prior_mean,
            )
            compiled_predictions = _ordered_predictions(
                pool.candidates, base.predictions
            )
            self.policy_controller.record_predictions(
                round_input.round_index, ordered_baseline, compiled_predictions, q0,
                alpha=policy.alpha, eta=policy.eta,
                normalize_acquisition=lambda values: _robust_z(values, self.config.z_clip),
            )
            policy = replace(
                policy,
                metadata={
                    **dict(policy.metadata),
                    "baseline_prediction_diagnostics": _prediction_diagnostics(
                        ordered_baseline
                    ),
                    "compiled_prediction_diagnostics": _prediction_diagnostics(
                        compiled_predictions
                    ),
                    "baseline_compiled_top10_overlap": _ranking_overlap(
                        ordered_baseline,
                        compiled_predictions,
                        10,
                    ),
                },
            )
            alpha, eta = policy.alpha, policy.eta
        else:
            if self.policy_controller is not None:
                self.base_selector.fit(self.history)
            base = self.base_selector.select(
                pool.candidates,
                representations,
                count=len(pool.candidates),
            )
            alpha, eta = self.config.alpha, self.config.eta
        predictions = _ordered_predictions(pool.candidates, base.predictions)
        acquisition = np.asarray(
            [_finite_acquisition(prediction) for prediction in predictions],
            dtype=float,
        )
        normalized = _robust_z(acquisition, self.config.z_clip)
        logits = alpha * np.log(q0 + EPSILON) + eta * normalized
        probabilities = _softmax(logits)
        selection_seed = _candidate_set_seed(
            self.config.seed,
            self.history_size,
            pool.candidates,
            "tilted_selection",
        )
        indices = _gumbel_top_k(
            probabilities,
            min(count, len(pool.candidates)),
            np.random.default_rng(selection_seed),
        )
        annotated = tuple(
            replace(
                prediction,
                metadata={
                    **prediction.metadata,
                    "q0_base_mass": float(base_mass),
                    "normalized_acquisition": float(z_score),
                    "tilt_log_weight": float(logit),
                    "selection_probability": float(probability),
                },
            )
            for prediction, base_mass, z_score, logit, probability in zip(
                predictions,
                q0,
                normalized,
                logits,
                probabilities,
                strict=True,
            )
        )
        metadata = {
            "selection_mode": "acquisition_tilted_sampling",
            "base_measure": "empirical_proposal_frequency",
            "alpha_base_measure": alpha,
            "eta_acquisition_tilt": eta,
            "z_clip": self.config.z_clip,
            "selection_seed": selection_seed,
            "valid_proposal_occurrences": pool.valid_occurrences,
            "unique_candidates_admitted": len(pool.proposal_reservoir),
            "proposal_base_measure": [
                {
                    "candidate_id": item.candidate_id,
                    "occurrence_count": _occurrence_count(item),
                    "proposal_q0_base_mass": _occurrence_count(item) / _valid_occurrence_count(item),
                }
                for item in pool.proposal_reservoir
            ],
            "bo_pool_size": len(pool.candidates),
            "configured_bo_pool_size": self.config.pool_size,
            "bo_pool_candidate_ids": [item.candidate_id for item in pool.candidates],
            "pool_maintenance": pool.method,
            "pool_seed": pool.seed,
            "base_selection": dict(base.metadata),
        }
        if policy is not None:
            metadata["compiled_policy"] = {
                "epoch_id": policy.epoch_id,
                "artifact_sha256": policy.artifact_digest,
                "source": policy.source,
                "degraded": policy.degraded,
                "stage": policy.stage,
                "alpha": policy.alpha,
                "eta": policy.eta,
                **dict(policy.metadata),
            }
        return BOSelectionResult(
            tuple(pool.candidates[index].candidate_id for index in indices),
            annotated,
            fallback_reason=base.fallback_reason,
            metadata=metadata,
        )


def _maintain_pool(
    candidates: tuple[Candidate, ...],
    config: _TiltConfig,
    history_size: int,
) -> _EmpiricalPool:
    occurrences = sum(_occurrence_count(item) for item in candidates)
    if {_valid_occurrence_count(item) for item in candidates} != {occurrences}:
        raise ValueError("candidate q0 occurrence totals are inconsistent")
    if occurrences > config.proposal_sample_count:
        raise ValueError(
            "valid proposal occurrences exceed configured proposal samples"
        )
    if len(candidates) <= config.pool_size:
        return _EmpiricalPool(
            candidates,
            candidates,
            occurrences,
            None,
            "all_unique_candidates",
        )
    seed = _candidate_set_seed(
        config.seed,
        history_size,
        candidates,
        "bo_pool_maintenance",
    )
    indices = _gumbel_top_k(
        _base_masses(candidates),
        config.pool_size,
        np.random.default_rng(seed),
    )
    maintained = tuple(
        sorted(
            (candidates[index] for index in indices),
            key=lambda item: item.candidate_id,
        )
    )
    return _EmpiricalPool(
        candidates,
        maintained,
        occurrences,
        seed,
        "q0_gumbel_top_k_without_replacement",
    )


def _base_masses(candidates: Sequence[Candidate]) -> np.ndarray:
    values = np.asarray(
        [
            _occurrence_count(item) / _valid_occurrence_count(item)
            for item in candidates
        ],
        dtype=float,
    )
    return values / float(values.sum())


def _occurrence_count(candidate: Candidate) -> int:
    return _positive_integer(
        _q0_record(candidate).get("occurrence_count"), "occurrence_count"
    )


def _valid_occurrence_count(candidate: Candidate) -> int:
    return _positive_integer(
        _q0_record(candidate).get("valid_occurrence_count"),
        "valid_occurrence_count",
    )


def _q0_record(candidate: Candidate) -> Mapping[str, object]:
    record = candidate.metadata.get(NUCLEOBENCH_Q0_METADATA_KEY)
    if not isinstance(record, Mapping):
        raise TypeError(
            f"candidate {candidate.candidate_id!r} has no empirical q0 record"
        )
    return record


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"q0 {name} must be a positive integer")
    return value


def _ordered_predictions(
    candidates: Sequence[Candidate],
    predictions: Sequence[BOPrediction],
) -> tuple[BOPrediction, ...]:
    by_id = {item.candidate_id: item for item in predictions}
    if len(by_id) != len(predictions) or any(
        item.candidate_id not in by_id for item in candidates
    ):
        raise ValueError("base selector predictions must cover the BO pool exactly")
    return tuple(by_id[item.candidate_id] for item in candidates)


def _finite_acquisition(prediction: BOPrediction) -> float:
    value = prediction.acquisition_score
    if value is None or not math.isfinite(value):
        raise ValueError("base selector acquisition scores must be finite")
    return float(value)


def _prediction_diagnostics(
    predictions: Sequence[BOPrediction],
) -> dict[str, float]:
    means = np.asarray([item.scalar_mean for item in predictions], dtype=float)
    acquisition = np.asarray(
        [_finite_acquisition(item) for item in predictions], dtype=float
    )
    return {
        "mean_mean": float(means.mean()),
        "mean_std": float(means.std()),
        "acquisition_mean": float(acquisition.mean()),
        "acquisition_std": float(acquisition.std()),
    }


def _ranking_overlap(
    baseline: Sequence[BOPrediction],
    compiled: Sequence[BOPrediction],
    count: int,
) -> float:
    size = min(count, len(baseline), len(compiled))
    if not size:
        return 1.0
    baseline_ids = {
        item.candidate_id
        for item in sorted(
            baseline,
            key=lambda item: (_finite_acquisition(item), item.candidate_id),
            reverse=True,
        )[:size]
    }
    compiled_ids = {
        item.candidate_id
        for item in sorted(
            compiled,
            key=lambda item: (_finite_acquisition(item), item.candidate_id),
            reverse=True,
        )[:size]
    }
    return len(baseline_ids & compiled_ids) / size


def _robust_z(values: np.ndarray, clip: float) -> np.ndarray:
    median = float(np.median(values))
    scale = MAD_SCALE * float(np.median(np.abs(values - median)))
    if scale <= EPSILON:
        scale = float(np.std(values))
    if scale <= EPSILON:
        return np.zeros_like(values)
    return np.clip((values - median) / (scale + EPSILON), -clip, clip)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - float(np.max(logits)))
    return exponentials / float(exponentials.sum())


def _gumbel_top_k(
    probabilities: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    scores = np.log(probabilities) + rng.gumbel(size=len(probabilities))
    return [int(index) for index in np.argsort(scores)[::-1][:count]]


def _candidate_set_seed(
    seed: int,
    history_size: int,
    candidates: Sequence[Candidate],
    phase: str,
) -> int:
    payload = json.dumps(
        {
            "seed": seed,
            "history_size": history_size,
            "candidate_ids": [item.candidate_id for item in candidates],
            "phase": phase,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


__all__ = ["AcquisitionTiltedSelector"]
