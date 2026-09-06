"""Acquisition-tilted selection over a task-local SynthonBench BO pool."""

from __future__ import annotations

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
from tasks.synthonbench.core.optimization_policy import (
    SynthonOptimizationPolicyAdapter,
)
from tasks.synthonbench.core.ldm_policy import (
    AcquisitionTiltConfig,
    effective_sample_size,
    gumbel_top_k,
    probability_entropy,
    robust_z,
    softmax_probabilities,
    tilted_logits,
)
from tasks.synthonbench.core.proposal_pool import (
    EmpiricalPool,
    candidate_set_seed,
    empirical_base_masses,
    maintain_empirical_pool,
    proposal_base_measure_records,
)
from tasks.synthonbench.core.tanimoto_gp import SynthonTanimotoGPUCBSelector


@dataclass(frozen=True)
class _TiltState:
    base_result: BOSelectionResult
    predictions: tuple[BOPrediction, ...]
    q0: np.ndarray
    normalized_acquisition: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray


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
        policy_adapter: SynthonOptimizationPolicyAdapter | None = None,
        policy_mode: bool = False,
    ) -> None:
        self.base_selector = base_selector
        self.config = AcquisitionTiltConfig(
            alpha,
            eta,
            z_clip,
            seed,
            pool_size,
            proposal_sample_count,
        )
        if (policy_controller is None) != (policy_adapter is None):
            raise ValueError("compiled policy controller and adapter must be configured together")
        if policy_controller is not None and not isinstance(
            base_selector, SynthonTanimotoGPUCBSelector
        ):
            raise TypeError("compiled Synthon policies require the Tanimoto GP selector")
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
            selection_rule="q0-maintained BO pool sampled by robust-z UCB tilted LDM policy",
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
        pool = maintain_empirical_pool(reservoir, self.config, self.history_size)
        policy: CompiledOptimizationPolicy | None = None
        if self.policy_controller is not None and self.history:
            base_result, policy = self._compiled_base_result(pool, representations)
            alpha, eta = policy.alpha, policy.eta
        else:
            if self.policy_controller is not None:
                self.base_selector.fit(self.history)
            base_result = self.base_selector.select(
                pool.candidates,
                representations,
                count=len(pool.candidates),
            )
            alpha, eta = self.config.alpha, self.config.eta
        state = self._score_pool(pool, base_result, alpha=alpha, eta=eta)
        seed = candidate_set_seed(
            self.config.seed,
            self.history_size,
            pool.candidates,
            phase="tilted_selection",
        )
        indices = gumbel_top_k(
            state.probabilities,
            min(count, len(pool.candidates)),
            np.random.default_rng(seed),
        )
        metadata = _selection_metadata(
            state,
            pool,
            replace(self.config, alpha=alpha, eta=eta),
            seed,
            count,
        )
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
            selected_candidate_ids=tuple(pool.candidates[index].candidate_id for index in indices),
            predictions=_annotate_predictions(state),
            fallback_reason=state.base_result.fallback_reason,
            metadata=metadata,
        )

    def _compiled_base_result(
        self,
        pool: EmpiricalPool,
        representations: Mapping[str, SurrogateVector],
    ) -> tuple[BOSelectionResult, CompiledOptimizationPolicy]:
        assert isinstance(self.base_selector, SynthonTanimotoGPUCBSelector)
        assert self.policy_controller is not None and self.policy_adapter is not None
        self.base_selector.fit(self.history)
        baseline = self.base_selector.select(
            pool.candidates,
            representations,
            count=len(pool.candidates),
        )
        ordered_baseline = _ordered_predictions(pool.candidates, baseline.predictions)
        q0 = empirical_base_masses(pool.candidates)
        round_input = self.policy_adapter.build_selection_round(
            history=self.history,
            candidates=pool.candidates,
            baseline_predictions=ordered_baseline,
            valid_proposal_occurrences=pool.valid_proposal_occurrences,
        )
        round_input = prepare_policy_research(
            round_input, history=self.history, candidates=pool.candidates,
            representations=representations, baseline=ordered_baseline,
            q0=q0, selector=self.base_selector,
            z_clip=self.config.z_clip,
            normalize_acquisition=lambda values: robust_z(values, clip=self.config.z_clip),
        )
        policy = self.policy_controller.resolve(round_input)
        self.base_selector.fit(
            self.history,
            history_prior_mean=policy.history_prior_mean,
            mean_source=f"compiled:{policy.source}",
            artifact_digest=policy.artifact_digest,
        )
        result = self.base_selector.select(
            pool.candidates,
            representations,
            count=len(pool.candidates),
            query_prior_mean=policy.query_prior_mean,
        )
        compiled_predictions = _ordered_predictions(pool.candidates, result.predictions)
        self.policy_controller.record_predictions(
            round_input.round_index, ordered_baseline, compiled_predictions,
            q0,
            alpha=policy.alpha, eta=policy.eta,
            normalize_acquisition=lambda values: robust_z(values, clip=self.config.z_clip),
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
        return result, policy

    def _score_pool(
        self,
        pool: EmpiricalPool,
        base: BOSelectionResult,
        *,
        alpha: float,
        eta: float,
    ) -> _TiltState:
        predictions = _ordered_predictions(pool.candidates, base.predictions)
        acquisition = _acquisition_scores(predictions)
        q0 = empirical_base_masses(pool.candidates)
        logits = tilted_logits(
            q0,
            acquisition,
            config=replace(self.config, alpha=alpha, eta=eta),
        )
        return _TiltState(
            base,
            predictions,
            q0,
            robust_z(acquisition, clip=self.config.z_clip),
            logits,
            softmax_probabilities(logits),
        )


def _ordered_predictions(candidates, predictions) -> tuple[BOPrediction, ...]:
    by_id = {item.candidate_id: item for item in predictions}
    missing = [item.candidate_id for item in candidates if item.candidate_id not in by_id]
    if len(by_id) != len(predictions) or missing:
        raise ValueError("base selector predictions do not cover the maintained BO pool exactly")
    return tuple(by_id[item.candidate_id] for item in candidates)


def _acquisition_scores(predictions: Sequence[BOPrediction]) -> np.ndarray:
    values = [item.acquisition_score for item in predictions]
    if any(value is None for value in values):
        raise ValueError("base selector prediction is missing acquisition score")
    scores = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("acquisition scores must be finite")
    return scores


def _prediction_diagnostics(
    predictions: Sequence[BOPrediction],
) -> dict[str, float]:
    means = np.asarray([item.scalar_mean for item in predictions], dtype=float)
    acquisition = _acquisition_scores(predictions)
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
    if size == 0:
        return 0.0
    baseline_top = {
        item.candidate_id
        for item in sorted(
            baseline,
            key=lambda item: float(item.acquisition_score),
            reverse=True,
        )[:size]
    }
    compiled_top = {
        item.candidate_id
        for item in sorted(
            compiled,
            key=lambda item: float(item.acquisition_score),
            reverse=True,
        )[:size]
    }
    return len(baseline_top & compiled_top) / size


def _annotate_predictions(state: _TiltState) -> tuple[BOPrediction, ...]:
    return tuple(
        replace(prediction, metadata={
            **prediction.metadata,
            "q0_base_mass": float(q0),
            "normalized_acquisition": float(acquisition),
            "tilt_log_weight": float(logit),
            "selection_probability": float(probability),
        })
        for prediction, q0, acquisition, logit, probability in zip(
            state.predictions, state.q0, state.normalized_acquisition, state.logits,
            state.probabilities, strict=True
        )
    )


def _selection_metadata(
    state: _TiltState,
    pool: EmpiricalPool,
    config: AcquisitionTiltConfig,
    seed: int,
    top_k_count: int,
) -> dict[str, object]:
    return {
        "selection_mode": "acquisition_tilted_sampling",
        "base_measure": "empirical_proposal_frequency",
        "alpha_base_measure": config.alpha,
        "eta_acquisition_tilt": config.eta,
        "normalization": "robust_z",
        "z_clip": config.z_clip,
        "sampling": "gumbel_top_k_without_replacement",
        "selection_seed": seed,
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
        "base_probability_entropy": probability_entropy(state.q0),
        "probability_entropy": probability_entropy(state.probabilities),
        "probability_effective_sample_size": effective_sample_size(state.probabilities),
        "tilted_kl_from_q0": _probability_kl(state.probabilities, state.q0),
        "q0_tilted_topk_overlap": _distribution_top_k_overlap(
            state.q0,
            state.probabilities,
            top_k_count,
        ),
        "base_selection": dict(state.base_result.metadata),
    }


def _probability_kl(probabilities: np.ndarray, base: np.ndarray) -> float:
    return float(np.sum(probabilities * (np.log(probabilities) - np.log(base))))


def _distribution_top_k_overlap(
    left: np.ndarray,
    right: np.ndarray,
    count: int,
) -> float:
    size = min(max(count, 1), len(left))
    left_top = set(np.argsort(-left, kind="stable")[:size])
    right_top = set(np.argsort(-right, kind="stable")[:size])
    return len(left_top & right_top) / size


__all__ = ["AcquisitionTiltedSelector"]
