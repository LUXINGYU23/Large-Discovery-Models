"""Canonical mutation-patch admission for one official paired start."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, make_complete_design_ir
from tasks.nucleobench.core.cases import NucleoBenchCase
from tasks.nucleobench.core.constants import OFFICIAL_START_COUNT, TASK_ID


DNA_BASES = frozenset("ACGT")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CandidatePayloadError(ValueError):
    """A candidate error with a stable reason for proposal repair."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class MutationContext:
    """The official case and paired start that define candidate identity."""

    case: NucleoBenchCase
    start_set_digest: str
    start_index: int
    start_sequence: str
    editable_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not SHA256_PATTERN.fullmatch(self.start_set_digest):
            raise ValueError("start_set_digest must be a lowercase SHA-256 digest")
        if (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or not 0 <= self.start_index < OFFICIAL_START_COUNT
        ):
            raise ValueError(
                f"start_index must be in [0, {OFFICIAL_START_COUNT - 1}]"
            )
        if len(self.start_sequence) != self.case.sequence_length:
            raise ValueError("start_sequence length does not match the selected case")
        if not set(self.start_sequence).issubset(DNA_BASES):
            raise ValueError("start_sequence must contain only A, C, G, and T")

        positions = tuple(self.editable_positions)
        if any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in positions
        ):
            raise ValueError("editable_positions must contain only integers")
        if len(positions) != len(set(positions)):
            raise ValueError("editable_positions must not contain duplicates")
        if any(position < 0 or position >= self.case.sequence_length for position in positions):
            raise ValueError("editable_positions contains a position outside the sequence")
        if len(positions) != self.case.editable_position_count:
            raise ValueError("editable_positions count does not match the selected case")
        object.__setattr__(self, "editable_positions", tuple(sorted(positions)))


@dataclass(frozen=True)
class PreparedMutationCandidate:
    """One normalized patch and its start-bound identity."""

    payload: dict[str, list[dict[str, Any]]]
    canonical_key: str
    sequence_sha256: str
    hamming_distance: int


@dataclass(frozen=True)
class NucleoBenchCandidateDomain:
    """Admit mutation patches without storing reconstructed full sequences."""

    context: MutationContext
    sink: DataCollectionSink = field(default_factory=DataCollectionSink.disabled)

    def admit(self, proposal: RawProposal) -> Candidate | CandidateRejection:
        try:
            prepared = prepare_candidate_payload(proposal.payload, self.context)
        except CandidatePayloadError as exc:
            return CandidateRejection(
                exc.reason,
                str(exc),
                proposal.source,
                exc.metadata,
            )

        candidate = _make_candidate(self.context, prepared, proposal.source)
        if proposal.metadata.get("collectable"):
            self._collect(candidate, proposal.metadata)
        return candidate

    def _collect(self, candidate: Candidate, metadata: Mapping[str, Any]) -> None:
        ir = make_complete_design_ir(
            task_id=TASK_ID,
            domain="nucleobench_mutation_patch",
            task_description=(
                "Propose a nucleotide mutation patch relative to one configured "
                "NucleoBench paired start."
            ),
            objectives=[{"name": "utility", "direction": "maximize"}],
            design_space_description=(
                "Zero-based editable positions with replacement bases from A, C, G, and T."
            ),
            observations=[],
            candidates=[candidate.payload],
            request_description="Propose one valid non-empty mutation patch.",
            num_candidates=1,
            round_idx=_integer_or_none(metadata.get("round_idx")),
            reasoning_available=False,
            raw_context={
                "case_id": self.context.case.case_id,
                "start_set_digest": self.context.start_set_digest,
                "start_index": self.context.start_index,
                "sequence_sha256": candidate.metadata["sequence_sha256"],
            },
        )
        self.sink.append(
            ir,
            provenance={
                "candidate_id": candidate.candidate_id,
                "canonical_key": candidate.canonical_key,
                "source": candidate.source,
            },
        )


def prepare_candidate_payload(
    payload: Any,
    context: MutationContext,
    *,
    allow_empty: bool = False,
) -> PreparedMutationCandidate:
    """Validate and normalize one start-relative mutation patch."""

    if not isinstance(payload, Mapping):
        raise CandidatePayloadError(
            "invalid_payload", "Candidate payload must be a JSON object."
        )
    _require_exact_fields(payload, {"mutations"}, "payload")
    raw_mutations = payload["mutations"]
    if not isinstance(raw_mutations, list):
        raise CandidatePayloadError(
            "invalid_mutations", "Candidate mutations must be a JSON array."
        )
    if not raw_mutations and not allow_empty:
        raise CandidatePayloadError(
            "invalid_mutations", "Candidate mutations must be a non-empty JSON array."
        )

    editable = frozenset(context.editable_positions)
    normalized: list[dict[str, Any]] = []
    seen_positions: set[int] = set()
    for index, raw in enumerate(raw_mutations):
        if not isinstance(raw, Mapping):
            raise CandidatePayloadError(
                "invalid_mutations",
                "Each mutation must be a JSON object.",
                metadata={"index": index},
            )
        _require_exact_fields(raw, {"position", "base"}, "mutation", index=index)
        position = raw["position"]
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
            or position >= context.case.sequence_length
        ):
            raise CandidatePayloadError(
                "invalid_position",
                "Mutation position must be a zero-based index inside the sequence.",
                metadata={"index": index, "position": position},
            )
        if position in seen_positions:
            raise CandidatePayloadError(
                "duplicate_position",
                "A candidate may mutate each position at most once.",
                metadata={"index": index, "position": position},
            )
        if position not in editable:
            raise CandidatePayloadError(
                "non_mutable_position",
                "Mutation position is outside the official editable mask.",
                metadata={"index": index, "position": position},
            )
        base = raw["base"]
        if not isinstance(base, str) or base not in DNA_BASES or len(base) != 1:
            raise CandidatePayloadError(
                "invalid_base",
                "Mutation base must be exactly one of A, C, G, or T.",
                metadata={"index": index, "position": position},
            )
        if context.start_sequence[position] == base:
            raise CandidatePayloadError(
                "unchanged_base",
                "Mutation base must differ from the paired start at that position.",
                metadata={"index": index, "position": position},
            )
        seen_positions.add(position)
        normalized.append({"position": position, "base": base})

    normalized.sort(key=lambda mutation: mutation["position"])
    sequence_sha256 = _sequence_digest(context.start_sequence, normalized)
    identity = {
        "case_id": context.case.case_id,
        "start_set_digest": context.start_set_digest,
        "start_index": context.start_index,
        "sequence_sha256": sequence_sha256,
    }
    canonical_key = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PreparedMutationCandidate(
        payload={"mutations": normalized},
        canonical_key=canonical_key,
        sequence_sha256=sequence_sha256,
        hamming_distance=len(normalized),
    )


def make_start_candidate(context: MutationContext) -> Candidate:
    """Represent the paired start as an internal baseline observation."""

    prepared = prepare_candidate_payload(
        {"mutations": []},
        context,
        allow_empty=True,
    )
    return _make_candidate(context, prepared, "official_start")


def _make_candidate(
    context: MutationContext,
    prepared: PreparedMutationCandidate,
    source: str,
) -> Candidate:
    return Candidate(
        candidate_id=f"nucleobench:{prepared.canonical_key}",
        payload=prepared.payload,
        canonical_key=prepared.canonical_key,
        source=source,
        metadata={
            "case_id": context.case.case_id,
            "start_set_digest": context.start_set_digest,
            "start_index": context.start_index,
            "sequence_sha256": prepared.sequence_sha256,
            "hamming_distance": prepared.hamming_distance,
        },
    )


def rebuild_sequence(
    start_sequence: str,
    mutations: Sequence[Mapping[str, Any]],
) -> str:
    """Materialize a validated patch at the evaluator boundary."""

    rebuilt = list(start_sequence)
    for mutation in mutations:
        rebuilt[int(mutation["position"])] = str(mutation["base"])
    return "".join(rebuilt)


def _sequence_digest(
    start_sequence: str,
    mutations: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    offset = 0
    for mutation in mutations:
        position = int(mutation["position"])
        digest.update(start_sequence[offset:position].encode("ascii"))
        digest.update(str(mutation["base"]).encode("ascii"))
        offset = position + 1
    digest.update(start_sequence[offset:].encode("ascii"))
    return digest.hexdigest()


def _require_exact_fields(
    value: Mapping[Any, Any],
    expected: set[str],
    label: str,
    *,
    index: int | None = None,
) -> None:
    metadata = {} if index is None else {"index": index}
    unexpected = sorted(str(field) for field in set(value).difference(expected))
    if unexpected:
        raise CandidatePayloadError(
            f"unexpected_{label}_fields",
            f"Candidate {label} contains unexpected field(s): {', '.join(unexpected)}.",
            metadata=metadata,
        )
    missing = sorted(expected.difference(value))
    if missing:
        raise CandidatePayloadError(
            f"missing_{label}_fields",
            f"Candidate {label} is missing required field(s): {', '.join(missing)}.",
            metadata=metadata,
        )


def _integer_or_none(value: Any) -> int | None:
    return None if isinstance(value, bool) or not isinstance(value, int) else value


__all__ = [
    "CandidatePayloadError",
    "MutationContext",
    "NucleoBenchCandidateDomain",
    "PreparedMutationCandidate",
    "make_start_candidate",
    "prepare_candidate_payload",
    "rebuild_sequence",
]
