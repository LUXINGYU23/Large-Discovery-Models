"""Schema-bound one-hot reaction representations for shared surrogate selection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ldm_tts.contracts import Candidate, SurrogateSpaceSpec
from ldm_tts.optimization import SurrogateVector

from tasks.iron_mind.core.candidate import CandidatePayloadError, normalize_candidate_payload
from tasks.iron_mind.core.schema import ReactionDatasetSchema, ReactionValue


ENCODER_ALGORITHM = "reaction_one_hot_v1"
ENCODER_PATH = "tasks.iron_mind.core.surrogate:ReactionOneHotEncoder"


class ReactionOneHotEncoder:
    """Encode an admitted reaction candidate in tracked schema order."""

    def __init__(self, schema: ReactionDatasetSchema) -> None:
        self.schema = schema

    @property
    def version(self) -> str:
        """Return the immutable representation identity for this schema."""

        return reaction_encoder_version(self.schema)

    def describe(self) -> SurrogateSpaceSpec:
        """Describe the fixed vector representation used by shared selection."""

        return reaction_surrogate_space(self.schema)

    def encode(self, candidate: Candidate) -> SurrogateVector:
        """Return one finite one-hot vector for an already admitted candidate."""

        conditions = _schema_ordered_conditions(candidate.payload, self.schema)
        values = _one_hot_values(conditions, self.schema)
        if len(values) != self.schema.one_hot_dimension:
            raise ValueError("Reaction one-hot vector dimension does not match the schema.")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Reaction one-hot vector contains a non-finite value.")
        return SurrogateVector(
            values=values,
            version=self.version,
            source_id=candidate.candidate_id,
            metadata={
                "encoder": ENCODER_ALGORITHM,
                "dataset_id": self.schema.dataset_id,
                "schema_sha256": self.schema.schema_sha256,
            },
        )


def reaction_surrogate_space(schema: ReactionDatasetSchema) -> SurrogateSpaceSpec:
    """Return the sole task-spec description for a schema's reaction vector."""

    return SurrogateSpaceSpec(
        kind="vector",
        representation="Schema-ordered categorical reaction-condition one-hot vector.",
        dimension_policy="fixed",
        dimension=schema.one_hot_dimension,
        encoder=ENCODER_PATH,
        version=reaction_encoder_version(schema),
        metadata={
            "algorithm": ENCODER_ALGORITHM,
            "dataset_id": schema.dataset_id,
            "factor_order": list(schema.factor_names),
            "schema_sha256": schema.schema_sha256,
        },
    )


def reaction_encoder_version(schema: ReactionDatasetSchema) -> str:
    """Build the version from every schema property that controls encoding."""

    return "|".join(
        (
            ENCODER_ALGORITHM,
            f"dataset={schema.dataset_id}",
            f"factors={','.join(schema.factor_names)}",
            f"dimension={schema.one_hot_dimension}",
            f"schema_sha256={schema.schema_sha256}",
        )
    )


def _schema_ordered_conditions(
    payload: Any, schema: ReactionDatasetSchema
) -> dict[str, ReactionValue]:
    try:
        normalized = normalize_candidate_payload(payload, schema)
    except CandidatePayloadError as exc:
        raise ValueError(f"Candidate payload cannot be encoded: {exc.reason}.") from exc
    return dict(normalized["conditions"])


def _one_hot_values(
    conditions: Mapping[str, ReactionValue], schema: ReactionDatasetSchema
) -> tuple[float, ...]:
    values = []
    for factor in schema.factors:
        selected = conditions[factor.name]
        values.extend(float(option == selected) for option in factor.options)
    return tuple(values)
