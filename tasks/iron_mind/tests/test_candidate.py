"""Tests for strict Iron Mind candidate admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from ldm_tts.contracts import Candidate, CandidateRejection, RawProposal
from ldm_tts.data import DataCollectionSink, read_jsonl
from tasks.iron_mind.core.candidate import (
    IronMindCandidateDomain,
    canonical_candidate_bytes,
    canonical_candidate_key,
)
from tasks.iron_mind.core.data import FrozenReactionTable, ReactionRow
from tasks.iron_mind.core.schema import load_reaction_schemas


TASK_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = TASK_ROOT / "resources" / "reaction_schemas.json"
VALID_CONDITIONS = {
    "base": "P2Et",
    "ligand": "XPhos",
    "aryl_halide": "None",
    "additive": "None",
}


def _payload(conditions: dict[str, str] | None = None) -> dict[str, object]:
    return {"dataset_id": "buchwald_hartwig", "conditions": conditions or dict(VALID_CONDITIONS)}


def test_canonical_identity_ignores_input_mapping_order() -> None:
    domain = _domain()
    reversed_conditions = dict(reversed(tuple(VALID_CONDITIONS.items())))

    first = domain.admit(RawProposal(_payload(), "test"))
    second = domain.admit(RawProposal(_payload(reversed_conditions), "test"))

    assert isinstance(first, Candidate)
    assert isinstance(second, Candidate)
    expected = (
        b'{"dataset_id":"buchwald_hartwig","conditions":'
        b'{"base":"P2Et","ligand":"XPhos","aryl_halide":"None","additive":"None"}}'
    )
    assert canonical_candidate_bytes(first.payload) == expected
    assert first.canonical_key == canonical_candidate_key(second.payload)
    assert first.candidate_id == second.candidate_id == f"iron-mind-{first.canonical_key}"
    assert list(first.payload["conditions"]) == list(VALID_CONDITIONS)
    assert first.metadata == {
        "dataset_id": "buchwald_hartwig",
        "schema_sha256": _schema().schema_sha256,
        "oracle_row_count": 1,
    }


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"dataset_id": "chan_lam_full", "conditions": VALID_CONDITIONS}, "unknown_dataset"),
        ({"dataset_id": "buchwald_hartwig"}, "missing_payload_fields"),
        ({"dataset_id": "buchwald_hartwig", "conditions": VALID_CONDITIONS, "extra": 1}, "unexpected_payload_fields"),
        (_payload({key: value for key, value in VALID_CONDITIONS.items() if key != "additive"}), "missing_condition_fields"),
        (_payload({**VALID_CONDITIONS, "extra": "value"}), "unexpected_condition_fields"),
        (_payload({**VALID_CONDITIONS, "ligand": "xphos"}), "unknown_category"),
        (_payload({**VALID_CONDITIONS, "base": "not-tracked"}), "unknown_category"),
        (_payload({**VALID_CONDITIONS, "base": "BTMG"}), "off_table_conditions"),
    ],
)
def test_invalid_payloads_have_stable_rejection_reasons(
    payload: dict[str, object], reason: str
) -> None:
    admitted = _domain().admit(RawProposal(payload, "test"))

    assert isinstance(admitted, CandidateRejection)
    assert admitted.reason == reason
    assert admitted.message.isascii()


def test_blocked_seed_or_resume_key_is_rejected_before_collection(tmp_path: Path) -> None:
    sink = DataCollectionSink(tmp_path / "collection")
    blocked_key = canonical_candidate_key(_payload())
    domain = _domain(sink=sink, blocked_canonical_keys={blocked_key})

    admitted = domain.admit(RawProposal(_payload(), "test", {"collectable": True}))

    assert isinstance(admitted, CandidateRejection)
    assert admitted.reason == "blocked_canonical_key"
    assert not (tmp_path / "collection" / "ldm_ir.jsonl").exists()


def test_collection_records_only_admitted_canonical_proposals(tmp_path: Path) -> None:
    collection_dir = tmp_path / "collection"
    domain = _domain(sink=DataCollectionSink(collection_dir))

    rejected = domain.admit(
        RawProposal(_payload({**VALID_CONDITIONS, "base": "BTMG"}), "rejected", {"collectable": True})
    )
    admitted = domain.admit(
        RawProposal(_payload(), "collection-only-source", {"collectable": True})
    )

    assert isinstance(rejected, CandidateRejection)
    assert isinstance(admitted, Candidate)
    rows = read_jsonl(collection_dir / "ldm_ir.jsonl")
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "ldm-2.0"
    assert rows[0]["action"] == {
        "type": "propose",
        "reasoning": None,
        "payload": {"candidates": [admitted.payload]},
        "summary": None,
    }
    assert rows[0]["collection"]["provenance"]["source"] == "collection-only-source"
    assert json.dumps(rows[0], ensure_ascii=False).isascii()
    rendered = (collection_dir / "ldm_sft.jsonl").read_text(encoding="utf-8")
    assert "collection-only-source" not in rendered
    assert "oracle_row_count" not in rendered


def _domain(
    *,
    sink: DataCollectionSink | None = None,
    blocked_canonical_keys: set[str] | None = None,
) -> IronMindCandidateDomain:
    schema = _schema()
    return IronMindCandidateDomain(
        schema,
        _table(schema),
        sink or DataCollectionSink.disabled(),
        blocked_canonical_keys=blocked_canonical_keys or set(),
    )


def _schema():
    return load_reaction_schemas(SCHEMA_PATH)["buchwald_hartwig"]


def _table(schema) -> FrozenReactionTable:
    conditions = dict(VALID_CONDITIONS)
    row = ReactionRow(
        row_id=1,
        conditions=MappingProxyType(conditions),
        measurements=MappingProxyType({"yield": 26.8886154}),
        raw_row_sha256=hashlib.sha256(b"mock-row").hexdigest(),
    )
    key = tuple(conditions[name] for name in schema.factor_names)
    return FrozenReactionTable(
        schema=schema,
        rows=(row,),
        rows_by_conditions=MappingProxyType({key: (row,)}),
    )
