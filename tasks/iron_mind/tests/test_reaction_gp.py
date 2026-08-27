"""Tests for the factor-aware Iron Mind GP-UCB selector."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ldm_tts.contracts import Candidate
from ldm_tts.optimization import BOObservation
from tasks.iron_mind.core.reaction_gp import (
    ReactionCategoricalGPUCBSelector,
    ReactionKernelParameters,
    reaction_ard_kernel,
)
from tasks.iron_mind.core.schema import (
    ReactionDatasetSchema,
    ReactionFactor,
    canonical_schema_payload,
    schema_sha256,
)
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder


def _schema(*, discrete: bool = False) -> ReactionDatasetSchema:
    factors = (
        ReactionFactor("temperature", (0.0, 10.0, 20.0), "discrete"),
        ReactionFactor("solvent", ("A", "B")),
    ) if discrete else (
        ReactionFactor("base", ("A", "B")),
        ReactionFactor("solvent", ("X", "Y")),
    )
    payload = canonical_schema_payload(
        dataset_id="reaction_gp_test",
        factors=factors,
        measurements=("yield",),
        objective="reaction_score",
        direction="maximize",
        observation_policy="single_row",
    )
    return ReactionDatasetSchema(
        "reaction_gp_test", factors, ("yield",), "reaction_score", "maximize", "single_row", schema_sha256(payload)
    )


def _candidate(schema: ReactionDatasetSchema, identifier: str, **conditions: object) -> Candidate:
    return Candidate(
        candidate_id=f"candidate-{identifier}",
        canonical_key=f"key-{identifier}",
        payload={"dataset_id": schema.dataset_id, "conditions": conditions},
    )


def test_categorical_kernel_has_one_weight_per_reaction_factor() -> None:
    schema = _schema()
    codes = np.asarray(((0, 0), (0, 1), (1, 0)))
    parameters = ReactionKernelParameters(1.0, (2.0, 0.25))

    kernel = reaction_ard_kernel(codes, codes, schema, parameters)

    assert kernel[0, 1] == pytest.approx(math.exp(-0.25))
    assert kernel[0, 2] == pytest.approx(math.exp(-2.0))


def test_discrete_factor_uses_ordered_option_distance() -> None:
    schema = _schema(discrete=True)
    codes = np.asarray(((0, 0), (1, 0), (2, 0)))
    kernel = reaction_ard_kernel(codes, codes, schema, ReactionKernelParameters(1.0, (1.0, 1.0)))

    assert kernel[0, 1] == pytest.approx(math.exp(-0.25))
    assert kernel[0, 2] == pytest.approx(math.exp(-1.0))


def test_selector_records_calibrated_ucb_and_ard_fit_summary() -> None:
    schema = _schema()
    encoder = ReactionOneHotEncoder(schema)
    candidates = (
        _candidate(schema, "a", base="A", solvent="X"),
        _candidate(schema, "b", base="A", solvent="Y"),
        _candidate(schema, "c", base="B", solvent="X"),
        _candidate(schema, "d", base="B", solvent="Y"),
    )
    history = [
        BOObservation.scalar(item.candidate_id, score, encoder.encode(item).values, feature_version=encoder.version)
        for item, score in zip(candidates[:3], (1.0, 1.0, 9.0), strict=True)
    ]
    selector = ReactionCategoricalGPUCBSelector(
        schema=schema,
        objective_name="reaction_score",
        feature_version=encoder.version,
    )
    selector.fit(history)

    result = selector.select(candidates, {item.candidate_id: encoder.encode(item) for item in candidates})

    summary = result.metadata["surrogate"]
    assert result.metadata["effective_beta"] == result.metadata["base_beta"] == 1.0
    assert selector.describe().parameters["model_mismatch_variance"] == pytest.approx(0.04)
    assert summary["fit_status"] == "fitted_ard_marginal_likelihood"
    assert set(summary["kernel"]["factor_weights"]) == {"base", "solvent"}
    assert all(item.metadata["surrogate"] == "reaction_categorical_ard_gp" for item in result.predictions)


def test_single_observation_shapes_the_next_round_posterior() -> None:
    schema = _schema()
    encoder = ReactionOneHotEncoder(schema)
    observed = _candidate(schema, "observed", base="A", solvent="X")
    distant = _candidate(schema, "distant", base="B", solvent="Y")
    selector = ReactionCategoricalGPUCBSelector(
        schema=schema,
        objective_name="reaction_score",
        feature_version=encoder.version,
    )
    selector.fit(
        [
            BOObservation.scalar(
                observed.candidate_id,
                4.0,
                encoder.encode(observed).values,
                feature_version=encoder.version,
            )
        ]
    )

    result = selector.select(
        (observed, distant),
        {item.candidate_id: encoder.encode(item) for item in (observed, distant)},
    )

    std_by_id = {item.candidate_id: item.scalar_std for item in result.predictions}
    assert result.metadata["surrogate"]["fit_status"] == "fitted_default_hyperparameters"
    assert std_by_id[distant.candidate_id] > std_by_id[observed.candidate_id]
    assert result.selected_candidate_ids == (distant.candidate_id,)


def test_selector_replays_a_fixed_reservoir_deterministically() -> None:
    schema = _schema()
    encoder = ReactionOneHotEncoder(schema)
    candidates = (
        _candidate(schema, "a", base="A", solvent="X"),
        _candidate(schema, "b", base="B", solvent="Y"),
    )
    selector = ReactionCategoricalGPUCBSelector(
        schema=schema,
        objective_name="reaction_score",
        feature_version=encoder.version,
    )
    selector.fit([BOObservation.scalar(candidates[0].candidate_id, 4.0, encoder.encode(candidates[0]).values, feature_version=encoder.version)])
    vectors = {item.candidate_id: encoder.encode(item) for item in candidates}

    first = selector.select(candidates, vectors).to_dict()
    second = selector.select(candidates, vectors).to_dict()

    assert first == second
