"""Prompt evidence and proposal-diversity tests for SynthonBench."""

from __future__ import annotations

import json

from synthonbench.space import Synthon, SynthonSpace

from ldm_tts.contracts import Candidate, EvaluationResult, Observation
from ldm_tts.engine.expansion import ExpansionRequest
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.prompting import (
    build_synthon_batch_prompt_messages,
    build_synthon_prompt_messages,
)
from tasks.synthonbench.core.proposal_parsing import parse_synthon_response


def test_unique_anchor_diversity_does_not_restrict_ldm_slot_combinations() -> None:
    catalog = SynthonProposalCatalog(
        _space(),
        allowed_reactions=("r1",),
        slate_size=2,
        seed=7,
        unique_anchors=True,
        proposals_per_round=1,
    )
    plan = catalog.build_plan(round_idx=0, proposal_index=0)
    payload = {
        "reaction_id": plan.reaction_id,
        "synthon_ids": [slot[-1].synthon_id for slot in plan.slot_options],
    }

    assert not plan.complete_tuple_options
    assert parse_synthon_response(json.dumps(payload), plan) == payload


def test_prompt_exposes_structures_for_history_and_complete_options() -> None:
    space = _space()
    catalog = SynthonProposalCatalog(
        space,
        allowed_reactions=("r1",),
        slate_size=2,
        seed=11,
        unique_anchors=True,
        proposals_per_round=1,
        restrict_to_complete_tuples=True,
    )
    plan = catalog.build_plan(round_idx=1, proposal_index=0)
    candidate = Candidate(
        "observed",
        {"reaction_id": "r1", "synthon_ids": [1, 11]},
        "r1|1_11",
    )
    observation = Observation(
        candidate,
        EvaluationResult("observed", "succeeded", metrics={"synthon_utility": -4.0}),
    )
    request = ExpansionRequest(round_idx=1, reservoir_size=1, observations=(observation,))

    messages = build_synthon_prompt_messages(
        request,
        plan,
        target="kif11",
        space=space,
        prompt_policy="direct_v1",
    )

    user = messages[1]["content"]
    assert '"components"' in user
    assert '"reaction_id":"r1"' in user
    assert '"synthon_ids":[' in user
    assert "option_index" not in user
    assert '"smiles":"CC"' in user
    assert '"smiles":"N"' in user


def test_batch_prompt_requests_one_indexed_candidate_per_slot() -> None:
    space = _space()
    catalog = SynthonProposalCatalog(
        space,
        allowed_reactions=("r1",),
        slate_size=2,
        seed=11,
        unique_anchors=True,
        proposals_per_round=2,
    )
    plans = tuple(catalog.build_plan(round_idx=0, proposal_index=index) for index in range(2))

    messages = build_synthon_batch_prompt_messages(
        ExpansionRequest(round_idx=0, reservoir_size=2),
        plans,
        target="kif11",
        space=space,
    )

    user = messages[1]["content"]
    assert '"candidate_count":2' in user
    assert '"proposal_index":0' in user
    assert '"proposal_index":1' in user
    assert "Include every proposal_index exactly once" in user


def _space() -> SynthonSpace:
    return SynthonSpace(
        [
            Synthon(1, 1, "r1", "CC"),
            Synthon(2, 1, "r1", "CO"),
            Synthon(11, 2, "r1", "N"),
            Synthon(12, 2, "r1", "O"),
        ]
    )
