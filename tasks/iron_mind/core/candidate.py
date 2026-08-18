"""Strict candidate admission for source-pinned reaction conditions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, make_complete_design_ir

from tasks.iron_mind.core.data import FrozenReactionTable
from tasks.iron_mind.core.schema import ReactionDatasetSchema, ReactionValue


CANONICAL_JSON_SEPARATORS = (",", ":")
PAYLOAD_FIELDS = ("dataset_id", "conditions")


class CandidatePayloadError(ValueError):
    """A user-facing candidate payload error with a stable rejection reason."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class IronMindCandidateDomain:
    """Admit only exact, source-table-backed reaction-condition candidates."""

    schema: ReactionDatasetSchema
    table: FrozenReactionTable
    sink: DataCollectionSink = field(default_factory=DataCollectionSink.disabled)

    def __post_init__(self) -> None:
        if self.table.schema != self.schema:
            raise ValueError("Candidate domain table schema does not match the supplied schema.")

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        """Return a candidate only after exact schema and table admission checks."""

        try:
            payload = normalize_candidate_payload(proposal.payload, self.schema)
        except CandidatePayloadError as exc:
            return CandidateRejection(exc.reason, str(exc), proposal.source)

        key = canonical_candidate_key(payload)
        if not self.table.rows_for_conditions(payload["conditions"]):
            return CandidateRejection(
                "off_table_conditions",
                "Candidate conditions are not present in the frozen reaction table.",
                proposal.source,
                {"canonical_key": key},
            )

        candidate = Candidate(
            candidate_id=f"iron-mind-{key}",
            payload=payload,
            canonical_key=key,
            source=proposal.source,
            metadata=_candidate_metadata(self.schema, self.table),
        )
        if bool(proposal.metadata.get("collectable")):
            self._collect(candidate, proposal.source)
        return candidate

    def _collect(self, candidate: Candidate, source: str) -> None:
        ir = make_complete_design_ir(
            task_id="iron_mind",
            domain="source-pinned categorical reaction conditions",
            task_description=(
                "Propose an exact finite reaction-condition combination from a "
                "source-pinned Iron Mind reaction table."
            ),
            objectives=[
                {
                    "name": "reaction_score",
                    "direction": "maximize",
                    "description": "Frozen-table reaction score; higher is better.",
                }
            ],
            design_space_description=(
                "Every factor must use one tracked option, and the complete "
                "combination must exist in the frozen reaction table."
            ),
            observations=[],
            candidates=[candidate.payload],
            request_description="Propose one source-valid reaction-condition combination.",
            num_candidates=1,
            allows_new_parameters=False,
            reasoning_available=False,
            active_parameters=_active_parameters(self.schema),
        )
        self.sink.append(
            ir,
            provenance={
                "candidate_id": candidate.candidate_id,
                "canonical_key": candidate.canonical_key,
                "source": source,
                **candidate.metadata,
            },
        )


def normalize_candidate_payload(
    payload: Any, schema: ReactionDatasetSchema
) -> dict[str, Any]:
    """Validate an untrusted payload and return its schema-ordered form."""

    if not isinstance(payload, Mapping):
        raise CandidatePayloadError("invalid_payload", "Candidate payload must be a JSON object.")
    _require_exact_fields(payload, PAYLOAD_FIELDS, "payload")
    if payload["dataset_id"] != schema.dataset_id:
        raise CandidatePayloadError(
            "unknown_dataset", "Candidate dataset_id does not match the configured dataset."
        )
    conditions = payload["conditions"]
    if not isinstance(conditions, Mapping):
        raise CandidatePayloadError("invalid_conditions", "Candidate conditions must be a JSON object.")
    _require_exact_fields(conditions, schema.factor_names, "condition")
    return {
        "dataset_id": schema.dataset_id,
        "conditions": _normalize_conditions(conditions, schema),
    }


def canonical_candidate_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one normalized payload into its identity-defining UTF-8 bytes."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=CANONICAL_JSON_SEPARATORS,
    ).encode("utf-8")


def canonical_candidate_key(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of a normalized candidate payload."""

    return hashlib.sha256(canonical_candidate_bytes(payload)).hexdigest()


def _require_exact_fields(
    payload: Mapping[Any, Any], expected: tuple[str, ...], label: str
) -> None:
    actual = set(payload)
    expected_fields = set(expected)
    unexpected = actual - expected_fields
    if unexpected:
        raise CandidatePayloadError(
            f"unexpected_{label}_fields",
            f"Candidate {label} contains unexpected field(s).",
        )
    missing = expected_fields - actual
    if missing:
        raise CandidatePayloadError(
            f"missing_{label}_fields",
            f"Candidate {label} is missing required field(s).",
        )


def _normalize_conditions(
    conditions: Mapping[str, Any], schema: ReactionDatasetSchema
) -> dict[str, ReactionValue]:
    normalized = {}
    for factor in schema.factors:
        value = conditions[factor.name]
        try:
            normalized[factor.name] = factor.admit(value)
        except ValueError as exc:
            raise CandidatePayloadError(
                "unknown_option",
                f"Candidate condition has an invalid option for factor {factor.name!r}.",
            ) from exc
    return normalized


def _candidate_metadata(
    schema: ReactionDatasetSchema, table: FrozenReactionTable
) -> dict[str, Any]:
    return {
        "dataset_id": schema.dataset_id,
        "schema_sha256": schema.schema_sha256,
        "oracle_row_count": len(table.rows),
    }


def _active_parameters(schema: ReactionDatasetSchema) -> list[dict[str, Any]]:
    return [
        {
            "name": factor.name,
            "type": factor.parameter_type,
            "options": list(factor.options),
        }
        for factor in schema.factors
    ]
