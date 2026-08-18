"""Tests for schema-bound Iron Mind surrogate representations."""

from __future__ import annotations

import math
import multiprocessing
from pathlib import Path
from typing import Any

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
)
from ldm_tts.optimization import BOObservation, BOSelectionResult, SurrogateVector, WarmStartAcquisitionSelector
from tasks.iron_mind.core.schema import ReactionDatasetSchema, load_reaction_schemas
from tasks.iron_mind.core.surrogate import ReactionOneHotEncoder, reaction_surrogate_space


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"


class RecordingSelector:
    """A test-only shared-selector delegate that records fit input."""

    def __init__(self) -> None:
        self.fit_history: tuple[BOObservation, ...] = ()

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="recording",
            objective_names=("reaction_score",),
            score_direction="maximize",
            selection_rule="recorded",
        )

    def fit(self, history: tuple[BOObservation, ...]) -> None:
        self.fit_history = tuple(history)

    def select(
        self,
        candidates: tuple[Candidate, ...],
        representations: dict[str, SurrogateVector],
        *,
        count: int = 1,
    ) -> BOSelectionResult:
        del candidates, representations, count
        raise AssertionError("select is not used by the warm-start fit boundary test")


def _schema(dataset_id: str) -> ReactionDatasetSchema:
    return load_reaction_schemas(SCHEMA_PATH)[dataset_id]


def _candidate(schema: ReactionDatasetSchema, *, reverse_conditions: bool = False) -> Candidate:
    conditions = {factor.name: factor.categories[-1] for factor in schema.factors}
    if reverse_conditions:
        conditions = dict(reversed(tuple(conditions.items())))
    return Candidate(
        candidate_id=f"candidate-{schema.dataset_id}",
        payload={"dataset_id": schema.dataset_id, "conditions": conditions},
        canonical_key=f"key-{schema.dataset_id}",
    )


def _expected_values(candidate: Candidate, schema: ReactionDatasetSchema) -> tuple[float, ...]:
    conditions = candidate.payload["conditions"]
    return tuple(
        float(value == conditions[factor.name])
        for factor in schema.factors
        for value in factor.categories
    )


def _encode_in_subprocess(candidate: Candidate) -> SurrogateVector:
    schema = _schema(candidate.payload["dataset_id"])
    return ReactionOneHotEncoder(schema).encode(candidate)


def _task_spec(surrogate: Any) -> LDMTaskSpec:
    return LDMTaskSpec(
        task="iron_mind",
        candidate_domain=CandidateDomainSpec(
            name="reaction_conditions",
            kind="categorical",
            dimension=None,
        ),
        objectives=(ObjectiveSpec("reaction_score", "maximize"),),
        response_spaces=(ResponseSpaceSpec("reaction_json", "json_object"),),
        acquisition=AcquisitionSpec(
            name="ucb",
            objective_names=("reaction_score",),
            score_direction="maximize",
            selection_rule="shared selector",
        ),
        reservoir=ReservoirSpec(
            name="reaction_reservoir",
            expansions=(
                ReservoirExpansionSpec(
                    name="reaction_proposal",
                    action_kind="emit_candidate",
                    response_space="reaction_json",
                    produces_candidates=True,
                ),
            ),
            candidate_validator="IronMindCandidateDomain",
            deduplication_key="canonical_key",
            max_size=4,
        ),
        surrogate=surrogate,
    )


def test_encoder_has_exact_schema_dimensions_and_one_hot_segments() -> None:
    expected_dimensions = {"buchwald_hartwig": 47, "chan_lam_full": 26}

    for dataset_id, dimension in expected_dimensions.items():
        schema = _schema(dataset_id)
        candidate = _candidate(schema)
        vector = ReactionOneHotEncoder(schema).encode(candidate)

        assert len(vector.values) == dimension
        assert vector.values == _expected_values(candidate, schema)
        assert all(math.isfinite(value) for value in vector.values)
        offset = 0
        for factor in schema.factors:
            segment = vector.values[offset : offset + len(factor.categories)]
            assert sum(segment) == 1.0
            offset += len(factor.categories)


def test_encoder_uses_only_schema_category_order_and_is_process_deterministic() -> None:
    schema = _schema("buchwald_hartwig")
    candidate = _candidate(schema, reverse_conditions=True)
    encoder = ReactionOneHotEncoder(schema)
    local = encoder.encode(candidate)

    assert local.values == _expected_values(candidate, schema)
    assert local.version == encoder.version
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=2) as pool:
        encoded = pool.map(_encode_in_subprocess, [candidate, candidate])
    assert encoded == [local, local]


def test_surrogate_space_is_the_single_task_spec_source_of_truth() -> None:
    schema = _schema("buchwald_hartwig")
    encoder = ReactionOneHotEncoder(schema)
    description = encoder.describe()
    task_spec = _task_spec(reaction_surrogate_space(schema))

    assert description.kind == "vector"
    assert description.dimension_policy == "fixed"
    assert description.dimension == schema.one_hot_dimension == 47
    assert description.version == encoder.version
    assert task_spec.surrogate == description


def test_seed_prior_fits_only_through_shared_warm_start_selector() -> None:
    schema = _schema("buchwald_hartwig")
    encoder = ReactionOneHotEncoder(schema)
    seed_candidate = _candidate(schema)
    seed_prior = BOObservation.scalar(
        seed_candidate.candidate_id,
        45.92980056,
        encoder.encode(seed_candidate).values,
        feature_version=encoder.version,
    )
    delegate = RecordingSelector()
    selector = WarmStartAcquisitionSelector(delegate, (seed_prior,))
    engine_state = {"observations": []}
    trajectory_fixture: list[object] = []

    selector.fit(())

    assert delegate.fit_history == (seed_prior,)
    assert engine_state == {"observations": []}
    assert trajectory_fixture == []
