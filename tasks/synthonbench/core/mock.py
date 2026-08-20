"""Deterministic proposal provider for the official SynthonBench example task."""

from __future__ import annotations

import json

from ldm_tts.transport import ProposalRequest, ProposalResponse


def mock_proposal_response(request: ProposalRequest) -> ProposalResponse:
    """Choose a slate member deterministically without reading oracle values."""

    metadata = request.metadata
    round_idx = int(metadata["round_idx"])
    proposal_index = int(metadata["proposal_index"])
    slots = metadata["slot_synthon_ids"]
    synthon_ids = [_slot_choice(ids, round_idx, proposal_index, index)
                   for index, ids in enumerate(slots)]
    payload = {"reaction_id": metadata["reaction_id"], "synthon_ids": synthon_ids}
    return ProposalResponse(text=json.dumps(payload), metadata=dict(metadata))


def _slot_choice(raw_ids, round_idx: int, proposal_index: int, slot_index: int) -> int:
    ids = tuple(int(item) for item in raw_ids)
    if not ids:
        raise ValueError("mock slate contains an empty synthon slot")
    index = (round_idx * 37 + proposal_index * 11 + slot_index * 5) % len(ids)
    return ids[index]


__all__ = ["mock_proposal_response"]
