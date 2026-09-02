"""Persistent research-harness proposal expansion for NucleoBench."""

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
from tasks.nucleobench.core.candidate import (
    CandidatePayloadError,
    MutationContext,
    NucleoBenchCandidateDomain,
    PreparedMutationCandidate,
    prepare_candidate_payload,
)
from tasks.nucleobench.core.proposals import attach_empirical_base_measure

HARNESS_PROFILE_IDS = (
    "target_biology",
    "regulatory_grammar",
    "sequence_landscape",
    "evidence_critic",
)
DIRECT_HARNESS_PROFILE_ID = "direct_research"
HARNESS_SOURCE = "nucleobench_persistent_research_harness"
HARNESS_FORBIDDEN_PATTERNS = (
    r"nucleo\s*bench",
    r"move37[-_/\s]*labs.*nucleo",
)
HARNESS_FORBIDDEN_TERMS = (
    "nucleobench",
    "move37-labs/nucleobench",
)
HARNESS_DENIED_HOSTS = (
    "api.github.com",
    "github.com",
    "raw.githubusercontent.com",
    "storage.googleapis.com",
)
HARNESS_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "mutations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "integer", "minimum": 0},
                    "base": {"type": "string", "enum": ["A", "C", "G", "T"]},
                },
                "required": ["position", "base"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["mutations"],
    "additionalProperties": False,
}
HARNESS_TOOL_NAMES = (
    "get_task_context",
    "get_sequence_window",
    "list_mutable_regions",
    "validate_mutations",
)
_PRESERVED_REJECTION_CODES = frozenset(
    {"duplicate_position", "non_mutable_position", "unchanged_base"}
)
_LOCAL_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"


def write_harness_sequence_context(context: MutationContext, output_path: Path) -> None:
    """Write the public paired-start context exposed to task-local tools."""

    atomic_json_write(
        output_path,
        {
            "schema_version": 1,
            "case": {
                "target": context.case.target,
                "sequence_length": context.case.sequence_length,
                "editable_position_count": len(context.editable_positions),
            },
            "paired_start": {
                "start_set_digest": context.start_set_digest,
                "start_index": context.start_index,
                "start_sequence": context.start_sequence,
                "editable_positions": list(context.editable_positions),
            },
        },
    )


def harness_profiles(
    candidates_per_turn: int,
    *,
    resource_root: Path = Path("/resources/profiles"),
) -> tuple[HarnessProfile, ...]:
    return tuple(
        _profile(profile_id, candidates_per_turn, resource_root)
        for profile_id in HARNESS_PROFILE_IDS
    )


def direct_harness_profile(
    candidates_per_turn: int,
    *,
    resource_root: Path = Path("/resources/profiles"),
) -> tuple[HarnessProfile, ...]:
    return (_profile(DIRECT_HARNESS_PROFILE_ID, candidates_per_turn, resource_root),)


def _profile(
    profile_id: str,
    candidates_per_turn: int,
    resource_root: Path,
) -> HarnessProfile:
    local_path = _LOCAL_RESOURCE_ROOT / "profiles" / profile_id / "AGENTS.md"
    return HarnessProfile(
        profile_id,
        resource_root / profile_id / "AGENTS.md",
        candidates_per_turn,
        agents_sha256=file_sha256(local_path),
    )


def harness_tool_extensions(
    *,
    resource_root: Path = Path("/resources/tools"),
) -> tuple[HarnessToolExtension, ...]:
    local_path = _LOCAL_RESOURCE_ROOT / "tools" / "sequence_context.mjs"
    return (
        HarnessToolExtension(
            resource_root / local_path.name,
            file_sha256(local_path),
            HARNESS_TOOL_NAMES,
        ),
    )


class NucleoBenchHarnessExpander:
    """Collect one validated mutation minibatch from each persistent session."""

    def __init__(
        self,
        client: HarnessClient,
        domain: NucleoBenchCandidateDomain,
        *,
        profiles: Sequence[HarnessProfile],
        campaign_id: str,
        first_active_round: int,
        attach_empirical_q0: bool,
        account: Callable[[dict[str, int]], None] | None = None,
    ) -> None:
        if not profiles:
            raise ValueError("NucleoBench harness requires at least one profile")
        if not campaign_id.strip():
            raise ValueError("NucleoBench harness campaign_id must not be empty")
        if first_active_round < 0:
            raise ValueError("first_active_round must be non-negative")
        self.client = client
        self.domain = domain
        self.profiles = tuple(profiles)
        self.campaign_id = campaign_id
        self.profile_set_sha256 = profile_set_sha256(self.profiles)
        self.first_active_round = first_active_round
        self.attach_empirical_q0 = attach_empirical_q0
        self.account = account

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        expected = sum(profile.candidates_per_turn for profile in self.profiles)
        if request.reservoir_size != expected:
            raise ValueError(
                "harness reservoir size must equal the configured minibatch total "
                f"({expected})"
            )
        evaluated, evaluated_candidates = _evaluated_history(
            request,
            self.domain.context,
        )
        turns = self._turns(request, evaluated_candidates)
        if self.account is not None:
            self.account(
                {
                    "proposal_attempts": len(turns),
                    "harness_turns": len(turns),
                }
            )
        results = self.client.run_turn(
            turns,
            submission_validator=lambda submission: _validate_submission(
                submission,
                self.domain.context,
                evaluated,
            ),
        )
        if self.account is not None:
            self.account(_usage_counts(results))
        sampling_mode = (
            "persistent_parallel_research_sessions"
            if self.attach_empirical_q0
            else "persistent_direct_research_session"
        )
        raw = self._proposals(request, results, evaluated, sampling_mode)
        proposals = (
            attach_empirical_base_measure(raw, request, self.domain)
            if self.attach_empirical_q0
            else raw
        )
        return ExpansionResult(
            proposals=proposals,
            selection_mode=(
                "acquisition" if self.attach_empirical_q0 else "reservoir_order"
            ),
            metadata={
                "sampling_mode": sampling_mode,
                "round_idx": request.round_idx,
                "session_count": len(self.profiles),
                "candidates_per_session": [
                    profile.candidates_per_turn for profile in self.profiles
                ],
                "proposal_count": len(proposals),
                "candidate_lineage": [
                    _candidate_lineage(proposal, self.domain.context)
                    for proposal in proposals
                ],
                "sessions": [_result_summary(result) for result in results],
            },
        )

    def _turns(
        self,
        request: ExpansionRequest,
        evaluated_candidates: Sequence[dict[str, object]],
    ) -> tuple[HarnessTurn, ...]:
        history = _history_delta(request, self.first_active_round)
        serialized_history = _serialize_observations(history, self.domain.context)
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
                    observations=serialized_history,
                    evaluated_candidates=evaluated_candidates,
                    initial=request.round_idx == self.first_active_round,
                    history_from_seq=history_from_seq,
                    history_to_seq=history_to_seq,
                    history_digest=history_digest,
                    context=self.domain.context,
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
        sampling_mode: str,
    ) -> tuple[RawProposal, ...]:
        by_profile = {result.profile_id: result for result in results}
        if len(by_profile) != len(results) or set(by_profile) != {
            profile.profile_id for profile in self.profiles
        }:
            raise ValueError("harness strict barrier requires one result per profile")
        proposals: list[RawProposal] = []
        for profile in self.profiles:
            result = by_profile[profile.profile_id]
            if len(result.candidates) != profile.candidates_per_turn:
                raise ValueError(
                    f"harness profile {profile.profile_id} must submit exactly "
                    f"{profile.candidates_per_turn} candidates"
                )
            profile_keys: set[str] = set()
            for index, candidate in enumerate(result.candidates):
                try:
                    prepared = _validated_candidate(
                        candidate,
                        self.domain.context,
                        evaluated,
                    )
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
                            "sampling_mode": sampling_mode,
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


def _serialize_observations(
    observations: Sequence[Any],
    context: MutationContext,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "mutations": prepare_candidate_payload(
                observation.candidate.payload,
                context,
                allow_empty=True,
            ).payload["mutations"],
            "utility": observation.metrics.get("utility"),
        }
        for observation in observations
        if observation.evaluation.succeeded
    )


def _evaluated_history(
    request: ExpansionRequest,
    context: MutationContext,
) -> tuple[set[str], tuple[dict[str, object], ...]]:
    evaluated: set[str] = set()
    candidates: list[dict[str, object]] = []
    for observation in request.observations:
        prepared = prepare_candidate_payload(
            observation.candidate.payload,
            context,
            allow_empty=True,
        )
        if prepared.canonical_key in evaluated:
            continue
        evaluated.add(prepared.canonical_key)
        candidates.append(prepared.payload)
    return evaluated, tuple(candidates)


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
    observations: Sequence[dict[str, object]],
    evaluated_candidates: Sequence[dict[str, object]],
    initial: bool,
    history_from_seq: int,
    history_to_seq: int,
    history_digest: str,
    context: MutationContext,
) -> str:
    payload = {
        "message_type": "campaign_bootstrap" if initial else "history_delta",
        "round_index": request.round_idx,
        "history_from_seq": history_from_seq,
        "history_to_seq": history_to_seq,
        "history_digest": history_digest,
        "case": {
            "target": context.case.target,
            "sequence_length": context.case.sequence_length,
            "editable_position_count": len(context.editable_positions),
        },
        "objective": "maximize measured utility; higher numeric values are better",
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
        "sequence_tools": list(HARNESS_TOOL_NAMES),
        "submission_contract": {
            "tool": "submit_candidates",
            "candidate_count": profile.candidates_per_turn,
            "candidate_schema": HARNESS_CANDIDATE_SCHEMA,
        },
        "constraints": [
            "Use zero-based positions and only the exact editable positions exposed by the structured tools.",
            "Every replacement base must differ from the paired start base at that position.",
            "Do not seek task implementations, evaluator models or weights, evaluation tables, hidden scores, or other benchmark-only assets.",
            "Campaign measurements are the only measured objective values; do not present a prediction as a measurement.",
            "Research public target biology and sequence-regulatory evidence, and use scratch code in the isolated sandbox when useful.",
            "Only candidates in evaluated_candidates are forbidden. Earlier proposals absent from that list remain eligible.",
            "Cross-session agreement is allowed, but duplicates within your own submitted minibatch are forbidden.",
            "Validate every candidate with validate_mutations before submission.",
            "If rejected, replace the reported entries using their exact indices, codes, and reasons, then resubmit the complete minibatch.",
        ],
    }
    return (
        "Continue your persistent sequence-design research role. Use your "
        "session history, new measurements, public evidence, scratch analysis, and the "
        "structured paired-start tools. Submit a complete validated minibatch before the "
        "turn deadline.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _validated_candidate(
    candidate: dict[str, Any],
    context: MutationContext,
    evaluated: set[str],
) -> PreparedMutationCandidate:
    prepared = prepare_candidate_payload(candidate, context)
    if prepared.canonical_key in evaluated:
        raise ValueError("harness candidate is already present in measured history")
    return prepared


def _validate_submission(
    submission: HarnessSubmissionRequest,
    context: MutationContext,
    evaluated: set[str],
) -> HarnessSubmissionValidation:
    rejections: list[HarnessSubmissionRejection] = []
    first_index_by_key: dict[str, int] = {}
    for index, candidate in enumerate(submission.candidates):
        try:
            prepared = prepare_candidate_payload(candidate, context)
        except CandidatePayloadError as exc:
            code = (
                exc.reason
                if exc.reason in _PRESERVED_REJECTION_CODES
                else "invalid_candidate"
            )
            rejections.append(
                HarnessSubmissionRejection(
                    index,
                    code,
                    f"Candidate at index {index} is not a legal mutation patch "
                    f"({exc.reason}): {exc}",
                )
            )
            continue
        label = json.dumps(prepared.payload, separators=(",", ":"))
        if prepared.canonical_key in evaluated:
            rejections.append(
                HarnessSubmissionRejection(
                    index,
                    "historical_duplicate",
                    f"Candidate at index {index} {label} was already evaluated in a "
                    "previous round. Replace it with a different unseen patch.",
                )
            )
            continue
        first_index = first_index_by_key.get(prepared.canonical_key)
        if first_index is not None:
            rejections.append(
                HarnessSubmissionRejection(
                    index,
                    "same_session_duplicate",
                    f"Candidate at index {index} {label} duplicates index {first_index} "
                    "in this submission. Keep the first occurrence and replace this one.",
                )
            )
            continue
        first_index_by_key[prepared.canonical_key] = index
    return HarnessSubmissionValidation(tuple(rejections))


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
    proposal: RawProposal,
    context: MutationContext,
) -> dict[str, object]:
    prepared = prepare_candidate_payload(proposal.payload, context)
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
        "tool_budget": result.tool_budget,
        "artifacts": result.artifacts,
    }


__all__ = [
    "DIRECT_HARNESS_PROFILE_ID",
    "HARNESS_CANDIDATE_SCHEMA",
    "HARNESS_FORBIDDEN_PATTERNS",
    "HARNESS_PROFILE_IDS",
    "HARNESS_TOOL_NAMES",
    "NucleoBenchHarnessExpander",
    "direct_harness_profile",
    "harness_profiles",
    "harness_tool_extensions",
    "write_harness_sequence_context",
]
