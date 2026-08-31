"""Persistent research-harness proposal expansion for SynthonBench."""

from __future__ import annotations

import csv
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
from tasks.synthonbench.core.candidate import (
    CandidatePayloadError,
    PreparedSynthonCandidate,
    SynthonCandidateDomain,
    prepare_candidate_payload,
)
from tasks.synthonbench.core.constants import OBJECTIVE_NAME, TASK_ID
from tasks.synthonbench.core.prompting import serialize_observations
from tasks.synthonbench.core.proposal_base_measure import attach_empirical_base_measure
from tasks.synthonbench.core.space_order import ordered_positions, ordered_reactions, ordered_synthon_ids

HARNESS_PROFILE_IDS = (
    "target_sar",
    "reaction_feasibility",
    "scaffold_exploration",
    "property_risk",
)
HARNESS_SOURCE = "synthonbench_persistent_research_harness"
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
    "type": "object",
    "properties": {
        "reaction_id": {"type": "string", "minLength": 1},
        "synthon_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1,
        },
    },
    "required": ["reaction_id", "synthon_ids"],
    "additionalProperties": False,
}
HARNESS_TOOL_NAMES = (
    "list_synthon_reactions",
    "search_synthon_space",
    "validate_synthon_candidate",
)
_LOCAL_PROFILE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness" / "profiles"
_LOCAL_TOOL_PATH = Path(__file__).resolve().parents[1] / "resources" / "harness" / "tools" / "synthon_space.mjs"


def write_harness_space_catalog(
    space: Any,
    allowed_reactions: Sequence[str],
    output_path: Path,
    *,
    reactions_path: Path | None = None,
) -> None:
    metadata = _reaction_metadata(reactions_path)
    reactions = []
    for reaction_id in ordered_reactions(allowed_reactions):
        positions = []
        for position in ordered_positions(space, reaction_id):
            synthons = []
            for synthon_id in ordered_synthon_ids(space, reaction_id, position):
                smiles = space.synthon_smiles(reaction_id, position, synthon_id)
                if not isinstance(smiles, str) or not smiles:
                    raise ValueError(f"synthon {synthon_id} lacks a public SMILES value")
                synthons.append({"synthon_id": synthon_id, "smiles": smiles})
            positions.append({"position": position, "synthons": synthons})
        reactions.append({
            "reaction_id": reaction_id,
            "positions": positions,
            "metadata": metadata.get(reaction_id, {}),
        })
    atomic_json_write(output_path, {"schema_version": 1, "reactions": reactions})


def _reaction_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {
            str(row["reaction_id"]): {
                str(key): str(value)
                for key, value in row.items()
                if key != "reaction_id" and value not in (None, "", "-")
            }
            for row in rows
        }


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
            resource_root / "synthon_space.mjs",
            file_sha256(_LOCAL_TOOL_PATH),
            HARNESS_TOOL_NAMES,
        ),
    )


class SynthonHarnessExpander:
    """Collect one validated minibatch from each persistent research session."""

    def __init__(
        self,
        client: HarnessClient,
        domain: SynthonCandidateDomain,
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
        evaluated, evaluated_candidates = _evaluated_history(request, self.domain)
        turns = self._turns(request, evaluated_candidates)
        if self.account is not None:
            self.account({
                "proposal_attempts": len(turns),
                "harness_turns": len(turns),
            })
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
        proposals = attach_empirical_base_measure(
            raw_proposals, request, self.domain
        )
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
        serialized_history = serialize_observations(history, self.domain.space)
        history_to_seq = len(request.observations)
        history_from_seq = history_to_seq - len(history)
        history_digest = canonical_sha256(serialized_history)
        forbidden_query_terms = _forbidden_query_terms(request)
        turns: list[HarnessTurn] = []
        for profile in self.profiles:
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
                    target=self.target,
                    observations=serialized_history,
                    evaluated_candidates=evaluated_candidates,
                    initial=request.round_idx == self.first_active_round,
                    history_from_seq=history_from_seq,
                    history_to_seq=history_to_seq,
                    history_digest=history_digest,
                ),
                forbidden_query_terms=forbidden_query_terms,
            ))
        return tuple(turns)

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
                    f"harness profile {profile.profile_id} must submit exactly {profile.candidates_per_turn} candidates"
                )
            profile_keys: set[str] = set()
            for index, candidate in enumerate(result.candidates):
                try:
                    prepared = _validated_candidate(candidate, self.domain, evaluated)
                except ValueError as exc:
                    raise RuntimeError(
                        f"committed harness candidate failed authoritative validation: "
                        f"{result.profile_id}[{index}]: {exc}"
                    ) from exc
                if prepared.product_id in profile_keys:
                    raise RuntimeError(
                        f"committed harness profile contains a duplicate occurrence: "
                        f"{result.profile_id}[{index}]"
                    )
                profile_keys.add(prepared.product_id)
                proposals.append(RawProposal(
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
                ))
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
    target: str,
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
        "target_label": target,
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
        "synthon_space_tools": list(HARNESS_TOOL_NAMES),
        "submission_contract": {
            "tool": "submit_candidates",
            "candidate_count": profile.candidates_per_turn,
            "candidate_schema": HARNESS_CANDIDATE_SCHEMA,
        },
        "constraints": [
            "Choose reaction types and search directions autonomously using only the structured official SynthonSpace tools.",
            "Do not search for this benchmark, its repository, datasets, or hidden scores.",
            "Every candidate must be an exact legal reaction_id plus ordered synthon_ids tuple returned by the structured tools.",
            "Do not estimate or claim benchmark scores as measurements.",
            "The isolated sandbox contains no authoritative task data; use the structured tools for the official space and supplied measurements.",
            "Research autonomously when it can improve the choices: search public literature, inspect public documents, and write or run scratch analysis code in the sandbox.",
            "Prioritize the distinct research perspective in your AGENTS.md. Cross-session agreement is allowed when your own evidence supports it, but do not collapse into generic ranking by assumption.",
            "Use the reaction summary to narrow the turn to roughly two to six evidence-backed reaction directions; do not enumerate the entire SynthonSpace without a specific reason.",
            "Use tools iteratively when useful, but leave enough of the 30-minute turn window to validate and submit the complete minibatch.",
            "If a tool fails or gives no decisive evidence, continue from the supplied data rather than withholding a candidate.",
            "Never submit an exact reaction_id plus ordered synthon_ids tuple listed in evaluated_candidates.",
            "A tuple proposed in an earlier turn remains eligible if it is absent from evaluated_candidates; do not maintain a private exclusion set of prior submissions.",
            "Historical repeats, invalid tuples, and duplicates within your own minibatch will be rejected with exact reasons.",
            "If submission is rejected, replace the reported entries and resubmit the complete minibatch; do not repeat the research phase.",
        ],
    }
    return (
        "Continue your persistent SynthonBench molecular-design research role. "
        "Use your private session history, the new measured observations, and the structured official SynthonSpace tools. "
        "Choose the reaction types and exact synthon tuples yourself; use public research and the isolated workspace when useful. "
        "Every submitted tuple must be absent from evaluated_candidates.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _validated_candidate(
    candidate: dict[str, Any],
    domain: SynthonCandidateDomain,
    evaluated: set[str],
) -> PreparedSynthonCandidate:
    prepared = prepare_candidate_payload(candidate, domain.space, domain.allowed_reactions)
    if prepared.product_id in evaluated:
        raise ValueError("harness candidate is already present in measured history")
    return prepared


def _validate_submission(
    submission: HarnessSubmissionRequest,
    domain: SynthonCandidateDomain,
    evaluated: set[str],
) -> HarnessSubmissionValidation:
    rejections: list[HarnessSubmissionRejection] = []
    first_index_by_key: dict[str, int] = {}
    for index, candidate in enumerate(submission.candidates):
        try:
            prepared = prepare_candidate_payload(
                candidate, domain.space, domain.allowed_reactions
            )
        except CandidatePayloadError as exc:
            rejections.append(HarnessSubmissionRejection(
                index,
                "invalid_candidate",
                f"Candidate at index {index} is not a legal official SynthonSpace tuple: {exc}",
            ))
            continue
        candidate_label = json.dumps(
            prepared.payload, ensure_ascii=False, separators=(",", ":")
        )
        if prepared.product_id in evaluated:
            rejections.append(HarnessSubmissionRejection(
                index,
                "historical_duplicate",
                f"Candidate at index {index} {candidate_label} was already evaluated in a previous round. "
                "Replace it with a different unseen tuple.",
            ))
            continue
        first_index = first_index_by_key.get(prepared.product_id)
        if first_index is not None:
            rejections.append(HarnessSubmissionRejection(
                index,
                "same_session_duplicate",
                f"Candidate at index {index} {candidate_label} duplicates index {first_index} in this submission. "
                "Keep the first occurrence and replace this one with a different unseen tuple.",
            ))
            continue
        first_index_by_key[prepared.product_id] = index
    return HarnessSubmissionValidation(tuple(rejections))


def _evaluated_history(
    request: ExpansionRequest,
    domain: SynthonCandidateDomain,
) -> tuple[set[str], tuple[dict[str, object], ...]]:
    by_key: dict[str, dict[str, object]] = {}
    for observation in request.observations:
        prepared = prepare_candidate_payload(
            observation.candidate.payload,
            domain.space,
            domain.allowed_reactions,
        )
        by_key[prepared.product_id] = prepared.payload
    return set(by_key), tuple(by_key[key] for key in sorted(by_key))


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
    "HARNESS_CANDIDATE_SCHEMA",
    "HARNESS_FORBIDDEN_PATTERNS",
    "HARNESS_PROFILE_IDS",
    "HARNESS_TOOL_NAMES",
    "SynthonHarnessExpander",
    "harness_profiles",
    "harness_tool_extensions",
    "write_harness_space_catalog",
]
