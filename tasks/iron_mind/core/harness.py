"""Persistent research-harness proposal expansion for Iron Mind."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (
    HarnessClient,
    HarnessProfile,
    HarnessSubmissionRejection,
    HarnessSubmissionRequest,
    HarnessSubmissionValidation,
    HarnessToolExtension,
    HarnessTurn,
    HarnessTurnResult,
    canonical_sha256,
    file_sha256,
    profile_set_sha256,
)
from tasks.iron_mind.core.candidate import (
    CandidatePayloadError,
    IronMindCandidateDomain,
    PreparedCandidatePayload,
    prepare_candidate_payload,
)
from tasks.iron_mind.core.constants import OBJECTIVE_NAME, TASK_ID
from tasks.iron_mind.core.proposal_base_measure import attach_empirical_base_measure


HARNESS_PROFILE_IDS = (
    "mechanistic_chemistry",
    "empirical_interactions",
    "literature_evidence",
    "design_space_exploration",
)
HARNESS_SOURCE = "iron_mind_persistent_research_harness"
HARNESS_FORBIDDEN_PATTERNS = (
    r"iron[\s_-]*mind",
    r"gomesgroup[/\\]iron-mind-public",
)
HARNESS_FORBIDDEN_TERMS = (
    "iron mind",
    "iron-mind-public",
    "gomesgroup/iron-mind-public",
    "476c555e45e2556e2ee4b24c726e774c2bfb7762",
)
HARNESS_TOOL_NAMES = (
    "describe_reaction_space",
    "search_reaction_conditions",
    "validate_reaction_candidate",
)
_LOCAL_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"
_LOCAL_PROFILE_ROOT = _LOCAL_RESOURCE_ROOT / "profiles"
_LOCAL_TOOL_PATH = _LOCAL_RESOURCE_ROOT / "tools" / "reaction_space.mjs"


def harness_candidate_schema(schema) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "dataset_id": {
                "type": "string",
                "enum": [schema.dataset_id],
            },
            "conditions": {
                "type": "object",
                "properties": {
                    factor.name: {
                        "type": (
                            "string"
                            if factor.parameter_type == "categorical"
                            else "number"
                        ),
                        "enum": list(factor.options),
                    }
                    for factor in schema.factors
                },
                "required": list(schema.factor_names),
                "additionalProperties": False,
            },
        },
        "required": ["dataset_id", "conditions"],
        "additionalProperties": False,
    }


def write_harness_space_catalog(domain: IronMindCandidateDomain, output_path: Path) -> None:
    schema = domain.schema
    by_key: dict[str, dict[str, Any]] = {}
    for row in domain.table.rows:
        prepared = prepare_candidate_payload(
            {"dataset_id": schema.dataset_id, "conditions": dict(row.conditions)},
            schema,
            domain.table,
        )
        by_key[prepared.canonical_key] = prepared.payload
    atomic_json_write(
        output_path,
        {
            "schema_version": 1,
            "dataset_id": schema.dataset_id,
            "schema_sha256": schema.schema_sha256,
            "factors": [
                {
                    "name": factor.name,
                    "type": factor.parameter_type,
                    "options": list(factor.options),
                }
                for factor in schema.factors
            ],
            "condition_count": len(by_key),
            "candidates": [by_key[key] for key in sorted(by_key)],
        },
    )


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


def harness_tool_extensions(
    *, resource_root: Path = Path("/resources/tools")
) -> tuple[HarnessToolExtension, ...]:
    return (
        HarnessToolExtension(
            resource_root / "reaction_space.mjs",
            file_sha256(_LOCAL_TOOL_PATH),
            HARNESS_TOOL_NAMES,
        ),
    )


class IronMindHarnessExpander:
    """Collect one validated minibatch from each persistent research session."""

    def __init__(
        self,
        client: HarnessClient,
        domain: IronMindCandidateDomain,
        *,
        profiles: Sequence[HarnessProfile],
        campaign_id: str,
        first_active_round: int,
        account: Callable[[dict[str, int]], None] | None = None,
    ) -> None:
        if not profiles:
            raise ValueError("Iron Mind harness requires at least one profile")
        if first_active_round < 0:
            raise ValueError("first_active_round must be non-negative")
        self.client = client
        self.domain = domain
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
        evaluated, evaluated_candidates = _evaluated_history(request, self.domain)
        turns = self._turns(request, evaluated_candidates)
        if self.account is not None:
            self.account({"proposal_attempts": len(turns), "harness_turns": len(turns)})
        results = self.client.run_turn(
            turns,
            submission_validator=lambda submission: _validate_submission(
                submission, self.domain, evaluated
            ),
        )
        if self.account is not None:
            self.account(_usage_counts(results))
        raw_proposals = self._proposals(request, results, evaluated)
        if len(raw_proposals) != expected:
            raise RuntimeError(
                f"validated harness minibatches must contain exactly {expected} occurrences"
            )
        proposals = attach_empirical_base_measure(raw_proposals, request, self.domain)
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
                "submitted_candidate_count": expected,
                "candidate_lineage": [
                    _candidate_lineage(item, self.domain) for item in proposals
                ],
                "sessions": [_result_summary(item) for item in results],
            },
        )

    def _turns(
        self,
        request: ExpansionRequest,
        evaluated_candidates: Sequence[dict[str, object]],
    ) -> tuple[HarnessTurn, ...]:
        history = _history_delta(request, self.first_active_round)
        serialized_history = _serialize_observations(history)
        history_to_seq = len(request.observations)
        history_from_seq = history_to_seq - len(history)
        history_digest = canonical_sha256(serialized_history)
        forbidden_query_terms = _forbidden_query_terms(request)
        return tuple(
            HarnessTurn(
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
                    candidate_schema=harness_candidate_schema(self.domain.schema),
                    observations=serialized_history,
                    evaluated_candidates=evaluated_candidates,
                    initial=request.round_idx == self.first_active_round,
                    history_from_seq=history_from_seq,
                    history_to_seq=history_to_seq,
                    history_digest=history_digest,
                ),
                forbidden_query_terms=forbidden_query_terms,
            )
            for profile in self.profiles
        )

    def _proposals(
        self,
        request: ExpansionRequest,
        results: Sequence[HarnessTurnResult],
        evaluated: set[str],
    ) -> tuple[RawProposal, ...]:
        by_profile = {result.profile_id: result for result in results}
        if len(by_profile) != len(self.profiles):
            raise ValueError("harness strict barrier requires one result per profile")
        proposals: list[RawProposal] = []
        for profile in self.profiles:
            result = by_profile.get(profile.profile_id)
            if result is None:
                raise ValueError(f"harness profile did not commit: {profile.profile_id}")
            if len(result.candidates) != profile.candidates_per_turn:
                raise ValueError(
                    f"harness profile {profile.profile_id} must submit exactly "
                    f"{profile.candidates_per_turn} candidates"
                )
            profile_keys: set[str] = set()
            for index, candidate in enumerate(result.candidates):
                try:
                    prepared = _validated_candidate(candidate, self.domain, evaluated)
                except ValueError as exc:
                    raise RuntimeError(
                        "committed harness candidate failed authoritative validation: "
                        f"{result.profile_id}[{index}]: {exc}"
                    ) from exc
                if prepared.canonical_key in profile_keys:
                    raise RuntimeError(
                        "committed harness profile contains a duplicate occurrence: "
                        f"{result.profile_id}[{index}]"
                    )
                profile_keys.add(prepared.canonical_key)
                proposals.append(
                    RawProposal(
                        prepared.payload,
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
                        },
                    )
                )
        return tuple(proposals)


def _history_delta(request: ExpansionRequest, first_active_round: int) -> tuple[Any, ...]:
    if request.round_idx == first_active_round:
        return request.observations
    return tuple(
        observation
        for observation in request.observations
        if observation.round_idx == request.round_idx - 1
    )


def _serialize_observations(observations: Sequence[Any]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "round_index": observation.round_idx,
            "dataset_id": observation.candidate.payload["dataset_id"],
            "conditions": observation.candidate.payload["conditions"],
            OBJECTIVE_NAME: observation.metrics[OBJECTIVE_NAME],
        }
        for observation in observations
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
    digest = canonical_sha256(
        {
            "campaignId": campaign_id,
            "historyDigest": history_digest,
            "historyFromSeq": history_from_seq,
            "historyToSeq": history_to_seq,
            "profileId": profile_id,
            "profileSetSha256": profile_set_digest,
            "roundIndex": round_index,
        }
    )
    return f"round_{round_index:04d}_{profile_id}_{digest[:16]}"


def _forbidden_query_terms(request: ExpansionRequest) -> tuple[str, ...]:
    terms = set(HARNESS_FORBIDDEN_TERMS)
    for observation in request.observations:
        terms.add(observation.candidate_id)
        terms.add(observation.canonical_key)
    return tuple(sorted(term for term in terms if term))


def _turn_message(
    request: ExpansionRequest,
    profile: HarnessProfile,
    *,
    candidate_schema: dict[str, object],
    observations: Sequence[dict[str, object]],
    evaluated_candidates: Sequence[dict[str, object]],
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
        "dataset_id": request.observations[0].candidate.payload["dataset_id"] if request.observations else None,
        "objective": f"maximize measured {OBJECTIVE_NAME}; higher is better",
        "new_measured_observations": list(observations),
        "evaluated_candidates": list(evaluated_candidates),
        "novelty_contract": {
            "evaluated_candidates_are_forbidden": True,
            "prior_unmeasured_submissions_may_be_reproposed": True,
            "required_not_evaluated_candidate_count": profile.candidates_per_turn,
            "same_round_cross_session_agreement_is_allowed": True,
            "same_session_duplicates_are_forbidden": True,
            "validate_before_submission": True,
        },
        "reaction_space_tools": list(HARNESS_TOOL_NAMES),
        "submission_contract": {
            "tool": "submit_candidates",
            "candidate_count": profile.candidates_per_turn,
            "candidate_schema": candidate_schema,
        },
        "time_budget": {
            "hard_wall_time_minutes": 30,
            "end_open_ended_research_by_minute": 20,
            "first_submission_by_minute": 25,
            "remaining_time_use": "repair_rejected_entries_only",
        },
        "constraints": [
            "Choose reaction-condition hypotheses autonomously from the structured source-pinned condition-space tools.",
            "Do not search for this benchmark, its repository, datasets, evaluation tables, or hidden scores.",
            "Every candidate must exactly match the configured dataset and one legal complete condition combination.",
            "Campaign measurements are the only measured objective values; do not present predictions as measurements.",
            "The isolated sandbox contains no authoritative task data; use the structured tools for the legal condition space and measurements.",
            "Research autonomously when useful: search public literature, inspect public documents, and run scratch analysis code in the sandbox.",
            "Prioritize the distinct research perspective in your AGENTS.md. Cross-session agreement is allowed when your own evidence supports it, but do not collapse into generic ranking by assumption.",
            "End open-ended research by minute 20 and make the first complete validated submission by minute 25; delivering the minibatch takes priority over further research.",
            "Never submit a candidate listed in evaluated_candidates.",
            "A candidate proposed in an earlier turn remains eligible if it is absent from evaluated_candidates; do not maintain a private exclusion set of prior submissions.",
            "Historical repeats, invalid candidates, and duplicates within your own minibatch will be rejected with exact reasons.",
            "If rejected, replace the reported entries and resubmit the complete minibatch without restarting the research phase.",
        ],
    }
    return (
        "Continue your persistent Iron Mind reaction-optimization research role. "
        "Use your private session history, the new measured observations, public evidence, "
        "and the structured source-pinned reaction-space tools. Every submitted candidate "
        "must be absent from evaluated_candidates.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _validated_candidate(
    candidate: dict[str, Any],
    domain: IronMindCandidateDomain,
    evaluated: set[str],
) -> PreparedCandidatePayload:
    prepared = prepare_candidate_payload(candidate, domain.schema, domain.table)
    if prepared.canonical_key in evaluated:
        raise ValueError("harness candidate is already present in measured history")
    return prepared


def _validate_submission(
    submission: HarnessSubmissionRequest,
    domain: IronMindCandidateDomain,
    evaluated: set[str],
) -> HarnessSubmissionValidation:
    rejections: list[HarnessSubmissionRejection] = []
    first_index_by_key: dict[str, int] = {}
    for index, candidate in enumerate(submission.candidates):
        try:
            prepared = prepare_candidate_payload(candidate, domain.schema, domain.table)
        except CandidatePayloadError as exc:
            rejections.append(
                HarnessSubmissionRejection(
                    index,
                    "invalid_candidate",
                    f"Candidate at index {index} is not a legal source-pinned reaction condition "
                    f"({exc.reason}): {exc}",
                )
            )
            continue
        candidate_label = json.dumps(
            prepared.payload, ensure_ascii=False, separators=(",", ":")
        )
        if prepared.canonical_key in evaluated:
            rejections.append(
                HarnessSubmissionRejection(
                    index,
                    "historical_duplicate",
                    f"Candidate at index {index} {candidate_label} was already evaluated in a "
                    "previous round. Replace it with a different unseen condition.",
                )
            )
            continue
        first_index = first_index_by_key.get(prepared.canonical_key)
        if first_index is not None:
            rejections.append(
                HarnessSubmissionRejection(
                    index,
                    "same_session_duplicate",
                    f"Candidate at index {index} {candidate_label} duplicates index {first_index} "
                    "in this submission. Keep the first occurrence and replace this one.",
                )
            )
            continue
        first_index_by_key[prepared.canonical_key] = index
    return HarnessSubmissionValidation(tuple(rejections))


def _evaluated_history(
    request: ExpansionRequest,
    domain: IronMindCandidateDomain,
) -> tuple[set[str], tuple[dict[str, object], ...]]:
    by_key: dict[str, dict[str, object]] = {}
    for observation in request.observations:
        prepared = prepare_candidate_payload(
            observation.candidate.payload, domain.schema, domain.table
        )
        by_key[prepared.canonical_key] = prepared.payload
    return set(by_key), tuple(by_key[key] for key in sorted(by_key))


def _usage_counts(results: Sequence[HarnessTurnResult]) -> dict[str, int]:
    return {
        "llm_requests": sum(int(result.usage["providerCalls"]) for result in results),
        "harness_tool_calls": sum(
            sum(int(count) for count in result.usage["toolCalls"].values())
            for result in results
        ),
        "harness_artifact_bytes": sum(
            int(result.usage["artifactBytes"]) for result in results
        ),
    }


def _candidate_lineage(
    proposal: RawProposal, domain: IronMindCandidateDomain
) -> dict[str, object]:
    prepared = prepare_candidate_payload(proposal.payload, domain.schema, domain.table)
    lineage = proposal.metadata["harness_lineage"]
    assert isinstance(lineage, dict)
    return {"canonical_key": prepared.canonical_key, **lineage}


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
    "HARNESS_FORBIDDEN_PATTERNS",
    "HARNESS_PROFILE_IDS",
    "HARNESS_TOOL_NAMES",
    "IronMindHarnessExpander",
    "harness_candidate_schema",
    "harness_profiles",
    "harness_tool_extensions",
    "write_harness_space_catalog",
]
