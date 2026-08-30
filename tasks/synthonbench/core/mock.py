"""Deterministic proposal provider for the official SynthonBench example task."""

from __future__ import annotations

import json

from ldm_tts.transport import ProposalRequest, ProposalResponse


def mock_proposal_response(request: ProposalRequest) -> ProposalResponse:
    """Choose a slate member deterministically without reading oracle values."""

    metadata = request.metadata
    plans = metadata.get("proposal_plans")
    if isinstance(plans, list):
        candidates = [
            {
                "proposal_index": int(plan["proposal_index"]),
                **_candidate_from_metadata(plan),
            }
            for plan in plans
        ]
        return ProposalResponse(
            text=json.dumps({"candidates": candidates}),
            metadata=dict(metadata),
        )
    return ProposalResponse(
        text=json.dumps(_candidate_from_metadata(metadata)),
        metadata=dict(metadata),
    )


def _candidate_from_metadata(metadata) -> dict[str, object]:
    round_idx = int(metadata["round_idx"])
    proposal_index = int(metadata["proposal_index"])
    complete_options = metadata.get("complete_tuple_options")
    if complete_options:
        tuple_index = (round_idx * 37 + proposal_index * 11) % len(complete_options)
        return {
            "reaction_id": metadata["reaction_id"],
            "synthon_ids": complete_options[tuple_index],
        }
    slots = metadata["slot_synthon_ids"]
    synthon_ids = [_slot_choice(ids, round_idx, proposal_index, index)
                   for index, ids in enumerate(slots)]
    return {"reaction_id": metadata["reaction_id"], "synthon_ids": synthon_ids}


def _slot_choice(raw_ids, round_idx: int, proposal_index: int, slot_index: int) -> int:
    ids = tuple(int(item) for item in raw_ids)
    if not ids:
        raise ValueError("mock slate contains an empty synthon slot")
    index = (round_idx * 37 + proposal_index * 11 + slot_index * 5) % len(ids)
    return ids[index]


__all__ = ["mock_proposal_response"]
