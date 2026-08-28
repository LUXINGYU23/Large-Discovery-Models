"""Persistent research-harness proposal expansion for SynthonBench."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.harness import (
    HarnessClient,
    HarnessProfile,
    HarnessTurn,
    HarnessTurnResult,
    canonical_sha256,
    file_sha256,
    profile_set_sha256,
)
from tasks.synthonbench.core.candidate import (
    CandidatePayloadError,
    SynthonCandidateDomain,
    prepare_candidate_payload,
)
from tasks.synthonbench.core.catalog import (
    ProposalSlotPlan,
    SynthonProposalCatalog,
    validate_payload_against_plan,
)
from tasks.synthonbench.core.constants import OBJECTIVE_NAME, TASK_ID
from tasks.synthonbench.core.prompting import serialize_observations
from tasks.synthonbench.core.proposal_base_measure import attach_empirical_base_measure
from tasks.synthonbench.core.proposals import excluded_anchor_ids

HARNESS_PROFILE_IDS = (
    "target_sar",
    "reaction_feasibility",
    "scaffold_exploration",
    "property_risk",
)
HARNESS_SOURCE = "synthonbench_persistent_research_harness"
HARNESS_ALLOWED_HOSTS = (
    "arxiv.org",
    "biorxiv.org",
    "chemrxiv.org",
    "doi.org",
    "europepmc.org",
    "github.com",
    "nature.com",
    "ncbi.nlm.nih.gov",
    "nih.gov",
    "pubs.acs.org",
    "raw.githubusercontent.com",
    "sciencedirect.com",
    "wiley.com",
)
HARNESS_DENIED_HOSTS = ("huggingface.co", "datasets-server.huggingface.co")
HARNESS_FORBIDDEN_PATTERNS = (
    r"synthon\s*bench",
    r"mireklzicar[/\\]synthonbench",
)
HARNESS_FORBIDDEN_TERMS = (
    "synthonbench",
    "mireklzicar/synthonbench",
    "4e89d72a19ebc5f9e59513bb57771ea8e8db4336",
)
HARNESS_CANDIDATE_SCHEMA = {
    "item_index": "copy the assigned zero-based item_index",
    "option_indices": "one zero-based option_index from every slot, in slot order",
}
_LOCAL_PROFILE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness" / "profiles"


def harness_profiles(
    candidates_per_turn: int,
    *,
    resource_root: Path = Path("/resources/profiles"),
) -> tuple[HarnessProfile, ...]:
    return tuple(
        HarnessProfile(
            profile_id,
            resource_root / profile_id / "AGENTS.md",
            candidates_per_turn,
            agents_sha256=file_sha256(_LOCAL_PROFILE_ROOT / profile_id / "AGENTS.md"),
        )
        for profile_id in HARNESS_PROFILE_IDS
    )


class SynthonHarnessExpander:
    """Collect one validated minibatch from each persistent research session."""

    def __init__(
        self,
        client: HarnessClient,
        domain: SynthonCandidateDomain,
        catalog: SynthonProposalCatalog,
        *,
        target: str,
        profiles: Sequence[HarnessProfile],
        campaign_id: str,
        first_active_round: int,
        account: Callable[[dict[str, int]], None] | None = None,
    ) -> None:
        if not profiles:
            raise ValueError("Synthon harness requires at least one profile")
        if first_active_round < 0:
            raise ValueError("first_active_round must be non-negative")
        self.client = client
        self.domain = domain
        self.catalog = catalog
        self.target = target
        self.profiles = tuple(profiles)
        self.campaign_id = campaign_id
        self.profile_set_sha256 = profile_set_sha256(self.profiles)
        self.first_active_round = first_active_round
        self.account = account

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        expected = sum(profile.candidates_per_turn for profile in self.profiles)
        if request.reservoir_size != expected:
            raise ValueError(
                f"harness reservoir size must equal the configured minibatch total ({expected})"
            )
        plans = self._plans(request)
        turns = self._turns(request, plans)
        if self.account is not None:
            self.account({
                "proposal_attempts": len(turns),
                "harness_turns": len(turns),
            })
        results = self.client.run_turn(turns)
        raw_proposals = self._proposals(request, plans, results)
        proposals = attach_empirical_base_measure(
            raw_proposals, request, self.domain
        )
        if self.account is not None:
            self.account(_usage_counts(results))
        return ExpansionResult(
            proposals=proposals,
            metadata={
                "sampling_mode": "persistent_parallel_research_sessions",
                "round_idx": request.round_idx,
                "session_count": len(self.profiles),
                "candidates_per_session": [
                    profile.candidates_per_turn for profile in self.profiles
                ],
                "proposal_count": len(proposals),
                "candidate_lineage": [
                    _candidate_lineage(item, self.domain) for item in proposals
                ],
                "sessions": [_result_summary(item) for item in results],
            },
        )

    def _plans(self, request: ExpansionRequest) -> tuple[ProposalSlotPlan, ...]:
        excluded = excluded_anchor_ids(request, self.catalog)
        return tuple(
            self.catalog.build_plan(
                round_idx=request.round_idx,
                proposal_index=index,
                excluded_anchor_ids=excluded,
            )
            for index in range(request.reservoir_size)
        )

    def _turns(
        self,
        request: ExpansionRequest,
        plans: Sequence[ProposalSlotPlan],
    ) -> tuple[HarnessTurn, ...]:
        history = _history_delta(request, self.first_active_round)
        serialized_history = serialize_observations(history, self.catalog.space)
        history_to_seq = len(request.observations)
        history_from_seq = history_to_seq - len(history)
        history_digest = canonical_sha256(serialized_history)
        forbidden_query_terms = _forbidden_query_terms(request, plans)
        turns: list[HarnessTurn] = []
        offset = 0
        for profile in self.profiles:
            assigned = plans[offset : offset + profile.candidates_per_turn]
            turns.append(HarnessTurn(
                profile_id=profile.profile_id,
                turn_id=_turn_id(
                    campaign_id=self.campaign_id,
                    profile_id=profile.profile_id,
                    profile_set_digest=self.profile_set_sha256,
                    round_index=request.round_idx,
                    history_from_seq=history_from_seq,
                    history_to_seq=history_to_seq,
                    history_digest=history_digest,
                ),
                round_index=request.round_idx,
                history_from_seq=history_from_seq,
                history_to_seq=history_to_seq,
                history_digest=history_digest,
                message=_turn_message(
                    request,
                    profile,
                    assigned,
                    target=self.target,
                    observations=serialized_history,
                    initial=request.round_idx == self.first_active_round,
                    history_from_seq=history_from_seq,
                    history_to_seq=history_to_seq,
                    history_digest=history_digest,
                ),
                forbidden_query_terms=forbidden_query_terms,
            ))
            offset += profile.candidates_per_turn
        return tuple(turns)

    def _proposals(
        self,
        request: ExpansionRequest,
        plans: Sequence[ProposalSlotPlan],
        results: Sequence[HarnessTurnResult],
    ) -> tuple[RawProposal, ...]:
        by_profile = {result.profile_id: result for result in results}
        if len(by_profile) != len(self.profiles):
            raise ValueError("harness strict barrier requires one result per profile")
        evaluated = {item.canonical_key for item in request.observations}
        proposals: list[RawProposal] = []
        offset = 0
        for profile in self.profiles:
            result = by_profile.get(profile.profile_id)
            if result is None:
                raise ValueError(f"harness profile did not commit: {profile.profile_id}")
            assigned = plans[offset : offset + profile.candidates_per_turn]
            if len(result.candidates) != len(assigned):
                raise ValueError(
                    f"harness profile {profile.profile_id} must submit exactly {len(assigned)} candidates"
                )
            accepted = [
                _validated_candidate(candidate, item_index, plan, self.domain, evaluated)
                for item_index, (candidate, plan) in enumerate(
                    zip(result.candidates, assigned, strict=True)
                )
            ]
            proposals.extend(
                RawProposal(
                    candidate,
                    HARNESS_SOURCE,
                    {
                        "collectable": False,
                        "round_idx": request.round_idx,
                        "sampling_mode": "persistent_parallel_research_sessions",
                        "harness_lineage": {
                            "campaign_id": self.campaign_id,
                            "round_index": request.round_idx,
                            "profile_id": result.profile_id,
                            "session_id": result.session_id,
                            "turn_id": result.turn_id,
                            "submission_id": result.submission_id,
                            "item_index": index,
                        },
                        **plan.metadata(),
                    },
                )
                for index, (candidate, plan) in enumerate(
                    zip(accepted, assigned, strict=True)
                )
            )
            offset += profile.candidates_per_turn
        return tuple(proposals)

def _history_delta(
    request: ExpansionRequest,
    first_active_round: int,
) -> tuple[Any, ...]:
    if request.round_idx == first_active_round:
        return request.observations
    return tuple(
        observation
        for observation in request.observations
        if observation.round_idx == request.round_idx - 1
    )


def _turn_id(
    *,
    campaign_id: str,
    profile_id: str,
    profile_set_digest: str,
    round_index: int,
    history_from_seq: int,
    history_to_seq: int,
    history_digest: str,
) -> str:
    digest = canonical_sha256({
        "campaignId": campaign_id,
        "historyDigest": history_digest,
        "historyFromSeq": history_from_seq,
        "historyToSeq": history_to_seq,
        "profileId": profile_id,
        "profileSetSha256": profile_set_digest,
        "roundIndex": round_index,
    })
    return f"round_{round_index:04d}_{profile_id}_{digest[:16]}"


def _forbidden_query_terms(
    request: ExpansionRequest,
    plans: Sequence[ProposalSlotPlan],
) -> tuple[str, ...]:
    terms = set(HARNESS_FORBIDDEN_TERMS)
    terms.update(plan.reaction_id for plan in plans)
    for observation in request.observations:
        terms.add(observation.candidate_id)
        terms.add(observation.canonical_key)
    return tuple(sorted(term for term in terms if term))


def _turn_message(
    request: ExpansionRequest,
    profile: HarnessProfile,
    plans: Sequence[ProposalSlotPlan],
    *,
    target: str,
    observations: Sequence[dict[str, object]],
    initial: bool,
    history_from_seq: int,
    history_to_seq: int,
    history_digest: str,
) -> str:
    payload = {
        "message_type": "campaign_bootstrap" if initial else "history_delta",
        "task": TASK_ID,
        "round_index": request.round_idx,
        "history_from_seq": history_from_seq,
        "history_to_seq": history_to_seq,
        "history_digest": history_digest,
        "target_label": target,
        "objective": f"maximize measured {OBJECTIVE_NAME}; higher is better",
        "new_measured_observations": list(observations),
        "assigned_items": [_plan_payload(index, plan) for index, plan in enumerate(plans)],
        "submission_contract": {
            "tool": "submit_candidates",
            "candidate_count": profile.candidates_per_turn,
            "order": "candidate j must have item_index j and answer assigned_items[j]",
            "candidate_schema": HARNESS_CANDIDATE_SCHEMA,
        },
        "constraints": [
            "Use only public literature and the supplied slate data.",
            "Do not search for this benchmark, its repository, datasets, or hidden scores.",
            "Use only option_index values listed in each item's corresponding slot.",
            "Do not estimate or claim benchmark scores as measurements.",
            "Treat this as a bounded selection turn, not open-ended research.",
            "All selectable data is in assigned_items; do not inspect the repository, filesystem, or installed software for additional task data.",
            "Either submit immediately or make one batch of optional tool calls and then submit; never start a second research batch.",
            "If a tool fails or gives no decisive evidence, continue from the supplied data and submit.",
            "Submit exactly once after checking every item and preserve item order.",
        ],
    }
    return (
        "Continue your persistent SynthonBench molecular-design research role. "
        "Use your private session history together with only the new measured observations below. "
        "You may research public chemistry or target biology and use the isolated workspace for analysis.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _plan_payload(index: int, plan: ProposalSlotPlan) -> dict[str, object]:
    return {
        "item_index": index,
        "proposal_index": plan.proposal_index,
        "reaction_id": plan.reaction_id,
        "proposal_role": plan.role,
        "slot_options": [
            {
                "position": options[0].position,
                "options": [
                    {"option_index": option_index, "smiles": option.smiles}
                    for option_index, option in enumerate(options)
                ],
            }
            for options in plan.slot_options
        ],
    }


def _validated_candidate(
    candidate: dict[str, Any],
    item_index: int,
    plan: ProposalSlotPlan,
    domain: SynthonCandidateDomain,
    evaluated: set[str],
) -> dict[str, object]:
    if set(candidate) != {"item_index", "option_indices"}:
        raise ValueError("harness candidate must contain exactly item_index and option_indices")
    raw_item_index = candidate["item_index"]
    if (
        isinstance(raw_item_index, bool)
        or not isinstance(raw_item_index, int)
        or raw_item_index != item_index
    ):
        raise ValueError("harness candidate item_index does not match its assigned item")
    raw_indices = candidate["option_indices"]
    if not isinstance(raw_indices, list) or len(raw_indices) != len(plan.slot_options):
        raise ValueError("harness option_indices do not match the assigned reaction arity")
    selected_ids: list[int] = []
    for slot_index, (raw_index, options) in enumerate(
        zip(raw_indices, plan.slot_options, strict=True)
    ):
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
            or raw_index >= len(options)
        ):
            raise ValueError(f"option_indices[{slot_index}] is not present in the supplied slate")
        selected_ids.append(options[raw_index].synthon_id)
    payload = {"reaction_id": plan.reaction_id, "synthon_ids": selected_ids}
    validate_payload_against_plan(payload, plan)
    try:
        prepared = prepare_candidate_payload(payload, domain.space, domain.allowed_reactions)
    except CandidatePayloadError as exc:
        raise ValueError(str(exc)) from exc
    if prepared.product_id in evaluated:
        raise ValueError("harness candidate is already present in measured history")
    return prepared.payload


def _usage_counts(results: Sequence[HarnessTurnResult]) -> dict[str, int]:
    fields = {
        "llm_requests": "providerCalls",
        "harness_web_calls": "webCalls",
        "harness_context7_calls": "context7Calls",
        "harness_artifact_bytes": "artifactBytes",
    }
    return {
        counter: sum(int(result.usage.get(field, 0)) for result in results)
        for counter, field in fields.items()
    }


def _candidate_lineage(
    proposal: RawProposal,
    domain: SynthonCandidateDomain,
) -> dict[str, object]:
    prepared = prepare_candidate_payload(
        proposal.payload, domain.space, domain.allowed_reactions
    )
    lineage = proposal.metadata["harness_lineage"]
    assert isinstance(lineage, dict)
    return {"canonical_key": prepared.product_id, **lineage}


def _result_summary(result: HarnessTurnResult) -> dict[str, object]:
    return {
        "profile_id": result.profile_id,
        "session_id": result.session_id,
        "session_turn_id": result.turn_id,
        "round_index": result.round_index,
        "history_from_seq": result.history_from_seq,
        "history_to_seq": result.history_to_seq,
        "history_digest": result.history_digest,
        "submission_id": result.submission_id,
        "candidate_count": len(result.candidates),
        "usage": result.usage,
        "artifacts": result.artifacts,
    }


__all__ = [
    "HARNESS_ALLOWED_HOSTS",
    "HARNESS_CANDIDATE_SCHEMA",
    "HARNESS_DENIED_HOSTS",
    "HARNESS_FORBIDDEN_PATTERNS",
    "HARNESS_PROFILE_IDS",
    "SynthonHarnessExpander",
    "harness_profiles",
]
