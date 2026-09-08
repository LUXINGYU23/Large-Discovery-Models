from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, read_jsonl
from tasks.nucleobench.core.candidate import (
    CandidatePayloadError,
    MutationContext,
    NucleoBenchCandidateDomain,
    prepare_candidate_payload,
    rebuild_sequence,
)
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.evaluator import NucleoBenchEvaluator


def _context() -> MutationContext:
    return MutationContext(
        case=get_case("malinois_k562"),
        start_set_digest="a" * 64,
        start_index=7,
        start_sequence="A" * 200,
        editable_positions=tuple(range(200)),
    )


def test_patch_order_does_not_change_candidate_identity() -> None:
    context = _context()
    first = prepare_candidate_payload(
        {"mutations": [{"position": 17, "base": "G"}, {"position": 3, "base": "T"}]},
        context,
    )
    second = prepare_candidate_payload(
        {"mutations": [{"base": "T", "position": 3}, {"base": "G", "position": 17}]},
        context,
    )

    assert first == second
    assert first.payload == {
        "mutations": [{"position": 3, "base": "T"}, {"position": 17, "base": "G"}]
    }
    rebuilt = rebuild_sequence(context.start_sequence, first.payload["mutations"])
    assert first.sequence_sha256 == hashlib.sha256(rebuilt.encode()).hexdigest()
    assert first.hamming_distance == 2


def test_candidate_identity_is_bound_to_the_paired_start() -> None:
    payload = {"mutations": [{"position": 0, "base": "C"}]}
    first = prepare_candidate_payload(payload, _context())
    second = prepare_candidate_payload(
        payload,
        replace(_context(), start_index=8),
    )

    assert first.sequence_sha256 == second.sequence_sha256
    assert first.canonical_key != second.canonical_key


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ([], "invalid_payload"),
        ({}, "missing_payload_fields"),
        ({"mutations": [], "note": "x"}, "unexpected_payload_fields"),
        ({"mutations": "0:C"}, "invalid_mutations"),
        ({"mutations": []}, "invalid_mutations"),
        ({"mutations": [{"position": 0}]}, "missing_mutation_fields"),
        (
            {"mutations": [{"position": 0, "base": "C", "score": 1}]},
            "unexpected_mutation_fields",
        ),
        ({"mutations": [{"position": True, "base": "C"}]}, "invalid_position"),
        ({"mutations": [{"position": 200, "base": "C"}]}, "invalid_position"),
        ({"mutations": [{"position": 0, "base": "c"}]}, "invalid_base"),
        (
            {
                "mutations": [
                    {"position": 0, "base": "C"},
                    {"position": 0, "base": "G"},
                ]
            },
            "duplicate_position",
        ),
        ({"mutations": [{"position": 0, "base": "A"}]}, "unchanged_base"),
    ],
)
def test_invalid_patches_have_stable_rejection_reasons(
    payload: object, reason: str
) -> None:
    with pytest.raises(CandidatePayloadError) as error:
        prepare_candidate_payload(payload, _context())
    assert error.value.reason == reason


def test_patch_may_only_edit_declared_positions() -> None:
    case = replace(_context().case, editable_position_count=2)
    context = replace(_context(), case=case, editable_positions=(0, 1))

    with pytest.raises(CandidatePayloadError) as error:
        prepare_candidate_payload(
            {"mutations": [{"position": 2, "base": "C"}]}, context
        )
    assert error.value.reason == "non_mutable_position"


def test_domain_collects_only_admitted_collectable_patches(tmp_path: Path) -> None:
    sink = DataCollectionSink(tmp_path / "collection", sft_filename=None)
    domain = NucleoBenchCandidateDomain(_context(), sink=sink)

    rejected = domain.admit(
        RawProposal(
            payload={"mutations": [{"position": 0, "base": "A"}]},
            source="test",
            metadata={"collectable": True},
        )
    )
    ignored = domain.admit(
        RawProposal(
            payload={"mutations": [{"position": 0, "base": "C"}]},
            source="test",
        )
    )
    collected = domain.admit(
        RawProposal(
            payload={"mutations": [{"position": 1, "base": "G"}]},
            source="test",
            metadata={"collectable": True, "round_idx": 2},
        )
    )

    assert isinstance(rejected, CandidateRejection)
    assert rejected.reason == "unchanged_base"
    assert isinstance(ignored, Candidate)
    assert isinstance(collected, Candidate)
    assert "sequence" not in collected.metadata
    assert "start_sequence" not in collected.metadata

    rows = read_jsonl(tmp_path / "collection" / "ldm_ir.jsonl")
    assert len(rows) == 1
    assert rows[0]["task"]["domain"] == "nucleobench_mutation_patch"
    assert rows[0]["action"]["payload"]["candidates"] == [collected.payload]
    assert "A" * 200 not in (tmp_path / "collection" / "ldm_ir.jsonl").read_text()


def test_batch_evaluator_rebuilds_sequences_once_at_the_oracle_boundary() -> None:
    domain = NucleoBenchCandidateDomain(_context())
    candidates = [
        domain.admit(
            RawProposal(
                {"mutations": [{"position": position, "base": base}]},
                "test",
            )
        )
        for position, base in ((0, "C"), (1, "G"))
    ]
    assert all(isinstance(candidate, Candidate) for candidate in candidates)
    received: list[str] = []

    def score(sequences):
        received.extend(sequences)
        return [-1.0, -2.0]

    results = NucleoBenchEvaluator(_context(), score).evaluate_batch(candidates)

    assert received == ["C" + "A" * 199, "AG" + "A" * 198]
    assert [result.metrics["utility"] for result in results] == [1.0, 2.0]
    assert all("sequence" not in result.artifacts for result in results)
