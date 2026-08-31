"""Strict canonical candidate admission for the SynthonBench domain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, make_complete_design_ir
from tasks.synthonbench.core.constants import OBJECTIVE_NAME, Q0_METADATA_KEY, TASK_ID
from tasks.synthonbench.core.space_order import ordered_positions, ordered_synthon_ids


class CandidatePayloadError(ValueError):
    """Raised when a payload is not an exact valid SynthonBench tuple."""


@dataclass(frozen=True)
class PreparedSynthonCandidate:
    """Validated payload and its official product identifier."""

    payload: dict[str, object]
    product_id: str


@dataclass(frozen=True)
class SynthonCandidateDomain:
    """Admit only source-valid reaction and synthon tuples."""

    space: Any
    allowed_reactions: tuple[str, ...]
    target: str
    sink: DataCollectionSink = field(default_factory=DataCollectionSink.disabled)

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            prepared = prepare_candidate_payload(
                proposal.payload, self.space, self.allowed_reactions
            )
        except CandidatePayloadError as exc:
            return CandidateRejection("invalid_candidate", str(exc), proposal.source)
        candidate = _candidate_from_prepared(prepared, proposal)
        self._collect(candidate, proposal.metadata)
        return candidate

    def _collect(self, candidate: Candidate, metadata: Mapping[str, Any]) -> None:
        if not metadata.get("collectable"):
            return
        payload = candidate.payload
        assert isinstance(payload, Mapping)
        ir = make_complete_design_ir(
            task_id=TASK_ID,
            domain="synthon_combinatorial_space",
            task_description="Select one valid reaction and ordered synthon tuple.",
            objectives=[{"name": OBJECTIVE_NAME, "direction": "maximize"}],
            design_space_description="Reaction-specific finite synthon IDs and public SMILES.",
            observations=[],
            candidates=[dict(payload)],
            request_description="Propose one source-valid tuple from the supplied public slate.",
            num_candidates=1,
            round_idx=_integer_or_none(metadata.get("round_idx")),
            raw_context={"target": self.target, "product_id": candidate.metadata["product_id"]},
        )
        self.sink.append(ir, provenance={"candidate_id": candidate.candidate_id})


def prepare_candidate_payload(
    payload: Any, space: Any, allowed_reactions: Sequence[str]
) -> PreparedSynthonCandidate:
    """Validate a public tuple against the exact official synthon space."""

    if not isinstance(payload, Mapping) or set(payload) != {"reaction_id", "synthon_ids"}:
        raise CandidatePayloadError("candidate must contain exactly reaction_id and synthon_ids")
    reaction_id = payload["reaction_id"]
    if not isinstance(reaction_id, str) or not reaction_id.strip():
        raise CandidatePayloadError("reaction_id must be a non-empty string")
    reaction_id = reaction_id.strip()
    if reaction_id not in set(allowed_reactions):
        raise CandidatePayloadError("reaction_id is outside the allowed official reaction set")
    synthon_ids = _normalize_synthon_ids(payload["synthon_ids"], space, reaction_id)
    product_id = _official_product_id(reaction_id, synthon_ids)
    return PreparedSynthonCandidate(
        payload={"reaction_id": reaction_id, "synthon_ids": list(synthon_ids)},
        product_id=product_id,
    )


def _normalize_synthon_ids(raw: Any, space: Any, reaction_id: str) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise CandidatePayloadError("synthon_ids must be a JSON array")
    positions = ordered_positions(space, reaction_id)
    if len(raw) != len(positions):
        raise CandidatePayloadError("synthon_ids length does not match the reaction slot count")
    normalized = tuple(_valid_slot_id(value, space, reaction_id, position)
                       for value, position in zip(raw, positions, strict=True))
    return normalized


def _valid_slot_id(value: Any, space: Any, reaction_id: str, position: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidatePayloadError("synthon IDs must be integer values")
    synthon_id = int(value)
    allowed = set(ordered_synthon_ids(space, reaction_id, position))
    if synthon_id not in allowed:
        raise CandidatePayloadError(
            f"synthon ID {synthon_id} is invalid for reaction {reaction_id!r} slot {position}"
        )
    return synthon_id


def _official_product_id(reaction_id: str, synthon_ids: Sequence[int]) -> str:
    from synthonbench.ids import product_id

    return str(product_id(reaction_id, synthon_ids))


def _candidate_from_prepared(prepared: PreparedSynthonCandidate, proposal: RawProposal) -> Candidate:
    metadata = {
        "product_id": prepared.product_id,
        "reaction_id": prepared.payload["reaction_id"],
        "synthon_ids": list(prepared.payload["synthon_ids"]),
    }
    q0 = proposal.metadata.get(Q0_METADATA_KEY)
    if isinstance(q0, Mapping):
        metadata[Q0_METADATA_KEY] = dict(q0)
    lineage = proposal.metadata.get("harness_lineage")
    if isinstance(lineage, Mapping):
        metadata["harness_lineage"] = dict(lineage)
    return Candidate(
        candidate_id=f"synthonbench:{prepared.product_id}",
        payload=prepared.payload,
        canonical_key=prepared.product_id,
        source=proposal.source,
        metadata=metadata,
    )


def _integer_or_none(value: Any) -> int | None:
    return None if isinstance(value, bool) or not isinstance(value, int) else value


__all__ = [
    "CandidatePayloadError",
    "PreparedSynthonCandidate",
    "SynthonCandidateDomain",
    "prepare_candidate_payload",
]
