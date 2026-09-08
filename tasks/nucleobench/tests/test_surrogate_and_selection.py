from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pytest

from ldm_tts.contracts import AcquisitionSpec, Candidate, RawProposal
from ldm_tts.harness import CompiledOptimizationPolicy
from ldm_tts.optimization import (
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    SurrogateVector,
)
from tasks.nucleobench.core import factory
from tasks.nucleobench.core.benchmark_clock import BenchmarkClock
from tasks.nucleobench.core.candidate import MutationContext, NucleoBenchCandidateDomain
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
from tasks.nucleobench.core.hamming_gp import (
    HammingGPUCBConfig,
    HammingGPUCBSelector,
    NucleotideHammingEncoder,
    normalized_hamming_kernel,
)
from tasks.nucleobench.core.optimization_policy import NucleoOptimizationPolicyAdapter
from tasks.nucleobench.core.policy_features import NucleoPolicyFeatureEncoder
from tasks.nucleobench.core.selection import AcquisitionTiltedSelector


def _context(*, start_sequence: str = "AAAAAAAA") -> MutationContext:
    case = replace(
        get_case("malinois_k562"),
        sequence_length=8,
        editable_position_count=4,
    )
    return MutationContext(
        case=case,
        start_set_digest="a" * 64,
        start_index=7,
        start_sequence=start_sequence,
        editable_positions=(0, 2, 4, 6),
    )


def _candidate(
    context: MutationContext,
    mutations: list[dict[str, object]],
    *,
    occurrence_count: int | None = None,
    valid_occurrence_count: int | None = None,
) -> Candidate:
    metadata = {}
    if occurrence_count is not None and valid_occurrence_count is not None:
        metadata[NUCLEOBENCH_Q0_METADATA_KEY] = {
            "occurrence_count": occurrence_count,
            "valid_occurrence_count": valid_occurrence_count,
            "probability": occurrence_count / valid_occurrence_count,
        }
    admitted = NucleoBenchCandidateDomain(context).admit(
        RawProposal({"mutations": mutations}, "test", metadata)
    )
    assert isinstance(admitted, Candidate)
    return admitted


def test_hamming_kernel_uses_positionwise_categorical_identity() -> None:
    identical = normalized_hamming_kernel((0, 1, 2, 3), (0, 1, 2, 3), 0.5)
    one_mutation = normalized_hamming_kernel((0, 1, 2, 3), (0, 1, 2, 0), 0.5)
    two_mutations = normalized_hamming_kernel((0, 1, 2, 3), (1, 0, 2, 3), 0.5)

    assert identical.item() == pytest.approx(1.0)
    assert 1.0 > one_mutation.item() > two_mutations.item() > 0.0


def test_encoder_ignores_noneditable_background_but_not_position_identity() -> None:
    first_context = _context(start_sequence="AAAAAAAA")
    second_context = _context(start_sequence="ATAAAAAA")
    first = _candidate(
        first_context,
        [{"position": 0, "base": "C"}, {"position": 2, "base": "G"}],
    )
    second = _candidate(
        second_context,
        [{"position": 0, "base": "C"}, {"position": 2, "base": "G"}],
    )
    swapped = _candidate(
        first_context,
        [{"position": 0, "base": "G"}, {"position": 2, "base": "C"}],
    )

    first_vector = NucleotideHammingEncoder(first_context).encode(first)
    second_vector = NucleotideHammingEncoder(second_context).encode(second)
    swapped_vector = NucleotideHammingEncoder(first_context).encode(swapped)

    assert first_vector.values == second_vector.values
    assert first_vector.version == second_vector.version
    assert first_vector.values != swapped_vector.values


def test_gp_working_set_is_capped_and_keeps_the_global_best() -> None:
    context = _context()
    encoder = NucleotideHammingEncoder(context)
    candidates = (
        _candidate(context, [{"position": 0, "base": "C"}]),
        _candidate(context, [{"position": 0, "base": "G"}]),
        _candidate(context, [{"position": 2, "base": "C"}]),
        _candidate(context, [{"position": 2, "base": "G"}]),
        _candidate(context, [{"position": 4, "base": "C"}]),
        _candidate(context, [{"position": 6, "base": "T"}]),
    )
    history = tuple(
        BOObservation.scalar(
            candidate.candidate_id,
            score,
            encoder.encode(candidate).values,
            feature_version=encoder.version,
        )
        for candidate, score in zip(
            candidates, (100.0, 1.0, 2.0, 3.0, 4.0, 5.0), strict=True
        )
    )
    selector = HammingGPUCBSelector(
        objective_name="utility",
        feature_dimension=encoder.dimension,
        feature_version=encoder.version,
        config=HammingGPUCBConfig(
            min_history_for_fit=2,
            max_observations=4,
            global_best_observations=1,
        ),
    )

    selector.fit(history)
    result = selector.select(
        candidates[-2:],
        {item.candidate_id: encoder.encode(item) for item in candidates[-2:]},
    )

    summary = result.metadata["surrogate"]
    assert summary["history_size"] == 6
    assert summary["working_set_size"] == 4
    assert candidates[0].candidate_id in summary["working_set_candidate_ids"]
    assert summary["fit_status"] == "fitted_grid_length_scale"
    assert all(np.isfinite(item.scalar_mean) for item in result.predictions)
    assert all(item.scalar_std > 0.0 for item in result.predictions)
    projection = selector.posterior_projection(history, np.asarray([encoder.encode(item).values for item in candidates[-2:]]))
    location, scale = projection["location_scale"]
    expected = location + scale * (projection["weights"] @ ((np.asarray([item.scalar_score for item in history]) - location) / scale))
    np.testing.assert_allclose(expected, [item.scalar_mean for item in result.predictions])
    np.testing.assert_allclose(scale * projection["std"], [item.scalar_std for item in result.predictions])


class _FixedSelector:
    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = dict(scores)

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec("ucb", ("utility",), "maximize", "fixed")

    def fit(self, _history: Sequence[BOObservation]) -> None:
        return None

    def select(
        self,
        candidates: Sequence[Candidate],
        _representations: Mapping[str, SurrogateVector],
        *,
        count: int = 1,
        round_idx: int = 0,
    ) -> BOSelectionResult:
        predictions = tuple(
            BOPrediction.scalar(
                item.candidate_id,
                mean=self.scores[item.candidate_id],
                std=0.0,
                acquisition_score=self.scores[item.candidate_id],
            )
            for item in candidates
        )
        return BOSelectionResult(
            tuple(item.candidate_id for item in candidates[:count]),
            predictions,
        )


def test_ldm_selector_preserves_q0_and_is_seeded() -> None:
    context = _context()
    candidates = (
        _candidate(
            context,
            [{"position": 0, "base": "C"}],
            occurrence_count=3,
            valid_occurrence_count=4,
        ),
        _candidate(
            context,
            [{"position": 2, "base": "G"}],
            occurrence_count=1,
            valid_occurrence_count=4,
        ),
    )

    def select() -> BOSelectionResult:
        selector = AcquisitionTiltedSelector(
            _FixedSelector(
                {
                    item.candidate_id: float(index)
                    for index, item in enumerate(candidates)
                }
            ),
            alpha=1.0,
            eta=0.0,
            z_clip=5.0,
            seed=23,
            pool_size=2,
            proposal_sample_count=4,
        )
        selector.fit(())
        return selector.select(candidates, {}, count=1)

    first = select()
    second = select()
    metadata = {item.candidate_id: item.metadata for item in first.predictions}

    assert first.selected_candidate_ids == second.selected_candidate_ids
    assert metadata[candidates[0].candidate_id][
        "selection_probability"
    ] == pytest.approx(0.75)
    assert metadata[candidates[1].candidate_id][
        "selection_probability"
    ] == pytest.approx(0.25)


def test_ldm_selector_maintains_the_three_b_pool_before_gp_scoring() -> None:
    context = _context()
    candidates = tuple(
        _candidate(
            context,
            mutations,
            occurrence_count=1,
            valid_occurrence_count=4,
        )
        for mutations in (
            [{"position": 0, "base": "C"}],
            [{"position": 0, "base": "G"}],
            [{"position": 2, "base": "C"}],
            [{"position": 4, "base": "T"}],
        )
    )
    selector = AcquisitionTiltedSelector(
        _FixedSelector({item.candidate_id: 0.0 for item in candidates}),
        alpha=1.0,
        eta=1.0,
        z_clip=5.0,
        seed=29,
        pool_size=3,
        proposal_sample_count=4,
    )

    selector.fit(())
    result = selector.select(candidates, {}, count=2)

    assert result.metadata["bo_pool_size"] == 3
    assert result.metadata["pool_maintenance"] == "q0_gumbel_top_k_without_replacement"
    assert len(result.metadata["proposal_base_measure"]) == 4
    assert len(set(result.selected_candidate_ids)) == 2


def test_direct_methods_do_not_construct_surrogate_components(monkeypatch) -> None:
    class ForbiddenEncoder:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("direct methods must not construct an encoder")

    monkeypatch.setattr(factory, "NucleotideHammingEncoder", ForbiddenEncoder)

    assert factory.build_surrogate_components(
        "llm", _context(), evaluations_per_round=4
    ) == (None, None)
    assert factory.build_surrogate_components(
        "harness", _context(), evaluations_per_round=4
    ) == (None, None)


def test_policy_features_are_stable_and_mean_inputs_exclude_selection_state() -> None:
    context = _context()
    features = NucleoPolicyFeatureEncoder(context)
    encoder = NucleotideHammingEncoder(context)
    observed = _candidate(context, [{"position": 0, "base": "C"}])
    candidates = (
        _candidate(
            context,
            [{"position": 2, "base": "G"}],
            occurrence_count=3,
            valid_occurrence_count=4,
        ),
        _candidate(
            context,
            [{"position": 4, "base": "T"}],
            occurrence_count=1,
            valid_occurrence_count=4,
        ),
    )
    history = (
        BOObservation.scalar(
            observed.candidate_id,
            2.0,
            encoder.encode(observed).values,
            feature_version=encoder.version,
            metadata={"round_idx": 0},
        ),
    )
    predictions = tuple(
        BOPrediction.scalar(
            candidate.candidate_id,
            mean=float(index),
            std=1.0,
            acquisition_score=float(index + 1),
        )
        for index, candidate in enumerate(candidates)
    )
    clock = BenchmarkClock(28_800)
    clock.start()
    adapter = NucleoOptimizationPolicyAdapter(
        features,
        seed=7,
        gp_config=HammingGPUCBConfig(),
        default_alpha=1.0,
        default_eta=1.0,
        evaluations_per_round=128,
        benchmark_clock=clock,
    )

    round_input = adapter.build_selection_round(
        round_index=1,
        history=history,
        candidates=candidates,
        representations={
            candidate.candidate_id: encoder.encode(candidate)
            for candidate in candidates
        },
        q0=np.asarray((0.75, 0.25)),
        baseline_predictions=predictions,
        valid_proposal_occurrences=4,
    )

    assert round_input.round_index == 1
    assert "round_index" not in round_input.execution_context["weight_context"]
    assert round_input.history_features.shape == (1, len(features.feature_names))
    assert round_input.query_features.shape == (2, len(features.feature_names))
    assert np.isfinite(round_input.query_features).all()
    mean_context = str(round_input.execution_context["mean_context"])
    assert all(term not in mean_context for term in ("q0", "acquisition", "candidate_id"))
    assert list(round_input.measured_observations)[0]["utility"] == 2.0
    pool = round_input.research_snapshot["proposal_pool"]
    assert pool["requested_evaluation_batch"] == 128
    assert pool["effective_evaluation_batch"] == 2
    assert pool["evaluated_pool_fraction"] == 1.0
    assert round_input.execution_context["weight_context"]["effective_evaluation_batch"] == 2
    assert 0 < round_input.research_snapshot["benchmark_time"]["remaining_seconds"] <= 28_800


class _StaticPolicyController:
    def record_predictions(self, round_index, predictions):
        assert predictions and all("candidate_id" in row for row in predictions)

    def resolve(self, round_input):
        return CompiledOptimizationPolicy(
            epoch_id="epoch_001",
            artifact_digest="a" * 64,
            history_prior_mean=np.zeros(len(round_input.history_features)),
            query_prior_mean=np.asarray((2.0, -2.0)),
            stage="exploit",
            alpha=0.0,
            eta=0.0,
            source="artifact",
            degraded=False,
            metadata={"action": "replace", "status": "accepted"},
        )


def test_compiled_policy_changes_gp_mean_and_ldm_weights() -> None:
    context = _context()
    encoder = NucleotideHammingEncoder(context)
    observed = _candidate(context, [{"position": 0, "base": "C"}])
    candidates = tuple(
        sorted(
            (
                _candidate(
                    context,
                    [{"position": 2, "base": "G"}],
                    occurrence_count=3,
                    valid_occurrence_count=4,
                ),
                _candidate(
                    context,
                    [{"position": 4, "base": "T"}],
                    occurrence_count=1,
                    valid_occurrence_count=4,
                ),
            ),
            key=lambda item: item.candidate_id,
        )
    )
    history = (
        BOObservation.scalar(
            observed.candidate_id,
            2.0,
            encoder.encode(observed).values,
            feature_version=encoder.version,
            metadata={"round_idx": 0},
        ),
    )
    controller = _StaticPolicyController()
    selector = AcquisitionTiltedSelector(
        HammingGPUCBSelector(
            objective_name="utility",
            feature_dimension=encoder.dimension,
            feature_version=encoder.version,
        ),
        alpha=1.0,
        eta=1.0,
        z_clip=5.0,
        seed=3,
        pool_size=2,
        proposal_sample_count=4,
        policy_controller=controller,  # type: ignore[arg-type]
        policy_adapter=NucleoOptimizationPolicyAdapter(
            NucleoPolicyFeatureEncoder(context),
            seed=3,
            gp_config=HammingGPUCBConfig(),
            default_alpha=1.0,
            default_eta=1.0,
        ),
    )
    selector.fit(history)

    result = selector.select(
        candidates,
        {candidate.candidate_id: encoder.encode(candidate) for candidate in candidates},
        round_idx=1,
    )

    assert {
        prediction.metadata["prior_mean_standardized"]
        for prediction in result.predictions
    } == {2.0, -2.0}
    assert [
        prediction.metadata["selection_probability"]
        for prediction in result.predictions
    ] == pytest.approx([0.5, 0.5])
    assert result.metadata["alpha_base_measure"] == 0.0
    assert result.metadata["eta_acquisition_tilt"] == 0.0
    assert result.metadata["compiled_policy"]["action"] == "replace"
    assert "compiled_prediction_diagnostics" in result.metadata["compiled_policy"]
