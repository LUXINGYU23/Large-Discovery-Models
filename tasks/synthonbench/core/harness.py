"""Task-local persistent research proposals, validation, and measured feedback."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (
    HarnessArtifactRule,
    HarnessClient,
    HarnessError,
    HarnessProfile,
    HarnessSubmissionContract,
    HarnessSubmissionError,
    HarnessSubmissionRequest,
    HarnessSubmissionValidation,
    HarnessToolExtension,
    HarnessTurn,
    HarnessTurnResult,
    canonical_sha256,
    directory_sha256,
    file_sha256,
    profile_set_sha256,
)
from ldm_tts.harness.pi import PiGuestRuntime, load_pi_guest_runtime
from tasks.synthonbench.core.candidate import (
    SynthonCandidateDomain,
    prepare_candidate_payload,
)
from tasks.synthonbench.core.constants import (
    FORBIDDEN_QUERY_TERMS,
    OBJECTIVE_NAME,
    TASK_ID,
)
from tasks.synthonbench.core.proposal_base_measure import attach_empirical_base_measure
from tasks.synthonbench.core.research import (
    read_candidate_file,
    serialize_measured_observations,
    write_measured_history,
)
from tasks.synthonbench.core.space_order import (
    ordered_positions,
    ordered_reactions,
    ordered_synthon_ids,
)

HARNESS_PROFILE_IDS = tuple(
    f"comprehensive_research_{index:02d}" for index in range(1, 5)
)
DIRECT_HARNESS_PROFILE_ID = "direct_research"
HARNESS_SKILL_IDS = (
    "rdkit",
    "experimental-design",
    "scientific-critical-thinking",
    "statsmodels",
)
HARNESS_SOURCE = "synthonbench_persistent_research_harness"
HARNESS_TOOL_NAMES = (
    "list_synthon_reactions",
    "search_synthon_space",
    "validate_synthon_candidate",
    "get_measured_history",
)
_LOCAL_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"
_LOCAL_PROFILE_PATH = (
    _LOCAL_RESOURCE_ROOT / "profiles" / "comprehensive_research" / "AGENTS.md"
)
_LOCAL_TOOL_PATH = _LOCAL_RESOURCE_ROOT / "tools" / "synthon_space.mjs"
_LOCAL_IMAGE_ROOT = _LOCAL_RESOURCE_ROOT / "image"
_CANDIDATE_FIELDS = ("reaction_id", "synthon_ids")


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
                    raise ValueError(
                        f"synthon {synthon_id} lacks a public SMILES value"
                    )
                synthons.append({"synthon_id": synthon_id, "smiles": smiles})
            positions.append({"position": position, "synthons": synthons})
        reactions.append(
            {
                "reaction_id": reaction_id,
                "positions": positions,
                "metadata": metadata.get(reaction_id, {}),
            }
        )
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


def harness_submission_contract(candidate_count: int) -> HarnessSubmissionContract:
    if candidate_count < 1:
        raise ValueError("Harness candidate count must be positive")
    return HarnessSubmissionContract(
        contract_id="synthonbench_candidate_batch",
        tool_name="submit_candidates",
        payload_schema={
            "type": "object",
            "properties": {
                "artifact_path": {
                    "type": "string",
                    "const": "candidates.json",
                    "description": (
                        f"Workspace-relative UTF-8 JSON file containing only a candidates array of exactly {candidate_count} "
                        "objects with reaction_id, synthon_ids, change_summary, and rationale. "
                        "Write the file with code; submit its path, not its contents."
                    ),
                },
            },
            "required": ["artifact_path"],
            "additionalProperties": False,
        },
        artifact_rules=(
            HarnessArtifactRule("/artifact_path", (".json",), candidate_count * 65536),
        ),
    )


def harness_profiles() -> tuple[HarnessProfile, ...]:
    return tuple(_profile(profile_id) for profile_id in HARNESS_PROFILE_IDS)


def direct_harness_profile() -> tuple[HarnessProfile, ...]:
    return (_profile(DIRECT_HARNESS_PROFILE_ID),)


def _profile(profile_id: str) -> HarnessProfile:
    return HarnessProfile(
        profile_id,
        Path("/resources/profiles/comprehensive_research/AGENTS.md"),
        agents_sha256=file_sha256(_LOCAL_PROFILE_PATH),
        skill_dirs=tuple(
            Path("/resources/skills") / name for name in HARNESS_SKILL_IDS
        ),
        skill_dir_sha256=tuple(
            directory_sha256(_LOCAL_RESOURCE_ROOT / "skills" / name)
            for name in HARNESS_SKILL_IDS
        ),
    )


def harness_tool_extensions() -> tuple[HarnessToolExtension, ...]:
    return (
        HarnessToolExtension(
            Path("/resources/tools/synthon_space.mjs"),
            file_sha256(_LOCAL_TOOL_PATH),
            HARNESS_TOOL_NAMES,
        ),
    )


def harness_guest_runtime() -> PiGuestRuntime:
    return load_pi_guest_runtime(TASK_ID, _LOCAL_IMAGE_ROOT)


class SynthonHarnessExpander:
    def __init__(
        self,
        client: HarnessClient,
        domain: SynthonCandidateDomain,
        *,
        target: str,
        profiles: Sequence[HarnessProfile],
        candidates_per_profile: int,
        campaign_id: str,
        first_active_round: int,
        attach_empirical_q0: bool,
        artifact_root: Path,
        wall_time_seconds: int = 1800,
        account: Callable[..., Any] | None = None,
    ) -> None:
        if not profiles or len({item.profile_id for item in profiles}) != len(profiles):
            raise ValueError("Harness requires distinct session identities")
        if (
            first_active_round < 0
            or candidates_per_profile < 1
            or wall_time_seconds < 1
        ):
            raise ValueError(
                "Harness round, candidate count, and wall time are invalid"
            )
        self.client = client
        self.domain = domain
        self.target = target
        self.profiles = tuple(profiles)
        self.candidates_per_profile = candidates_per_profile
        self.campaign_id = campaign_id
        self.profile_set_sha256 = profile_set_sha256(self.profiles)
        self.first_active_round = first_active_round
        self.attach_empirical_q0 = attach_empirical_q0
        self.artifact_root = artifact_root
        self.wall_time_seconds = wall_time_seconds
        self.account = account

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        expected = len(self.profiles) * self.candidates_per_profile
        if request.reservoir_size != expected:
            raise ValueError(
                f"harness reservoir size must equal the configured minibatch total ({expected})"
            )
        evaluated = {
            prepare_candidate_payload(
                observation.candidate.payload,
                self.domain.space,
                self.domain.allowed_reactions,
            ).product_id
            for observation in request.observations
        }
        write_measured_history(
            self.artifact_root, request.observations, self.domain.space
        )
        turns = self._turns(request)
        if self.account is not None:
            self.account(
                {"proposal_attempts": len(turns), "harness_turns": len(turns)},
                usage_key=f"harness:round:{request.round_idx}",
            )
        started = time.perf_counter()
        usage_by_profile: dict[str, Mapping[str, Any]] = {}
        try:
            results = self.client.run_turn(
                turns,
                submission_validator=lambda submission: _validate_submission(
                    submission,
                    self.domain,
                    evaluated,
                    artifact_root=self.artifact_root,
                    candidate_count=self.candidates_per_profile,
                ),
                recovery_timeout_seconds=2.0 * self.wall_time_seconds,
            )
            usage_by_profile = {result.profile_id: result.usage for result in results}
        except HarnessError as exc:
            usage_by_profile = exc.turn_usage
            raise
        finally:
            if self.account is not None:
                for turn in turns:
                    self.account(
                        _usage_counts(usage_by_profile.get(turn.profile_id, {})),
                        usage_key=f"harness:{turn.turn_id}",
                    )
                self.account({"harness_wall_time_seconds": time.perf_counter() - started})
        sampling_mode = (
            "persistent_parallel_research_sessions"
            if self.attach_empirical_q0
            else "persistent_direct_research_session"
        )
        raw_proposals = self._proposals(request, results, evaluated, sampling_mode)
        proposals = (
            attach_empirical_base_measure(raw_proposals, request, self.domain)
            if self.attach_empirical_q0
            else raw_proposals
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
                "candidates_per_session": [self.candidates_per_profile]
                * len(self.profiles),
                "proposal_count": len(proposals),
                "submitted_candidate_count": expected,
                "candidate_lineage": [
                    {
                        "canonical_key": prepare_candidate_payload(
                            item.payload,
                            self.domain.space,
                            self.domain.allowed_reactions,
                        ).product_id,
                        **item.metadata["harness_lineage"],
                    }
                    for item in proposals
                ],
                "sessions": [
                    _result_summary(item, self.candidates_per_profile)
                    for item in results
                ],
            },
        )

    def _turns(self, request: ExpansionRequest) -> tuple[HarnessTurn, ...]:
        history = (
            request.observations
            if request.round_idx == self.first_active_round
            else tuple(
                item
                for item in request.observations
                if item.round_idx == request.round_idx - 1
            )
        )
        serialized_history = serialize_measured_observations(history, self.domain.space)
        history_to_seq = len(request.observations)
        history_from_seq = history_to_seq - len(history)
        history_digest = canonical_sha256(serialized_history)
        payload = {
            "message_type": (
                "campaign_bootstrap"
                if request.round_idx == self.first_active_round
                else "history_delta"
            ),
            "task": TASK_ID,
            "round_index": request.round_idx,
            "target_label": self.target,
            "objective": f"maximize measured {OBJECTIVE_NAME}; higher is better",
            "history_from_seq": history_from_seq,
            "history_to_seq": history_to_seq,
            "history_digest": history_digest,
            "new_measured_observations": tuple(
                {key: row[key] for key in ("candidate_id", "round_index", OBJECTIVE_NAME)}
                for row in serialized_history
            ),
            "evaluated_candidates": {
                "tool": "get_measured_history",
                "count": len(request.observations),
                "membership_tool": "validate_synthon_candidate",
                "membership_field": "already_evaluated",
            },
            "measurement_feedback": (
                "Only selected unique candidates are measured; selection is not evidence of quality."
                if self.attach_empirical_q0
                else "Every accepted distinct candidate is directly measured."
            ),
            "novelty_contract": {
                "evaluated_candidates_are_forbidden": True,
                "prior_unmeasured_submissions_may_be_reproposed": True,
                "same_round_cross_session_agreement_is_allowed": True,
                "same_session_repeated_occurrences_are_allowed": False,
                "repeated_occurrences_contribute_to_empirical_q0": self.attach_empirical_q0,
            },
            "submission_contract": {
                "tool": "submit_candidates",
                "artifact_path": "candidates.json",
                "candidate_count": self.candidates_per_profile,
                "candidate_fields": [*_CANDIDATE_FIELDS, "change_summary", "rationale"],
            },
            "time_budget": {
                "hard_wall_time_seconds": self.wall_time_seconds,
                "end_open_ended_research_by_seconds": self.wall_time_seconds * 2 // 3,
                "first_submission_by_seconds": self.wall_time_seconds * 5 // 6,
            },
        }
        message = (
            "Continue your independent research. Query measured IDs for exact candidates and original research_annotations, "
            "including failed controls and competing hypotheses. Only already evaluated candidates are forbidden; "
            "previously proposed but unmeasured candidates remain eligible. Write and submit the complete candidate file "
            "before the deadline; repair the indexed errors in the same session.\n\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        forbidden_terms = set(FORBIDDEN_QUERY_TERMS)
        for observation in request.observations:
            forbidden_terms.update(
                (observation.candidate_id, observation.canonical_key)
            )
        turns = []
        for profile in self.profiles:
            digest = canonical_sha256(
                {
                    "campaignId": self.campaign_id,
                    "profileId": profile.profile_id,
                    "profileSetSha256": self.profile_set_sha256,
                    "roundIndex": request.round_idx,
                    "historyFromSeq": history_from_seq,
                    "historyToSeq": history_to_seq,
                    "historyDigest": history_digest,
                }
            )
            turns.append(
                HarnessTurn(
                    profile_id=profile.profile_id,
                    turn_id=f"round_{request.round_idx:04d}_{profile.profile_id}_{digest[:16]}",
                    round_index=request.round_idx,
                    history_from_seq=history_from_seq,
                    history_to_seq=history_to_seq,
                    history_digest=history_digest,
                    message=message,
                    forbidden_query_terms=tuple(
                        sorted(term for term in forbidden_terms if term)
                    ),
                )
            )
        return tuple(turns)

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
        proposals = []
        annotations_by_key: dict[str, list[dict[str, Any]]] = {}
        for profile in self.profiles:
            result = by_profile[profile.profile_id]
            if result.submission_status != "accepted":
                raise RuntimeError(
                    f"harness profile submission was rejected: {profile.profile_id}"
                )
            candidates = read_candidate_file(
                result.submission,
                result.submitted_artifacts,
                self.artifact_root,
                self.candidates_per_profile,
            )
            profile_keys: set[str] = set()
            for index, candidate in enumerate(candidates):
                prepared, annotation = _submission_candidate(candidate, self.domain)
                key = prepared.product_id
                if key in evaluated or key in profile_keys:
                    raise RuntimeError(
                        f"committed harness candidate repeats measured history or an earlier entry: {profile.profile_id}[{index}]"
                    )
                profile_keys.add(key)
                # Equal occurrences share all hypotheses before canonical reservoir admission.
                annotations = annotations_by_key.setdefault(key, [])
                annotations.append(
                    {
                        "profile_id": result.profile_id,
                        "round_index": request.round_idx,
                        "submission_id": result.submission_id,
                        "item_index": index,
                        **annotation,
                    }
                )
                proposals.append(
                    RawProposal(
                        prepared.payload,
                        HARNESS_SOURCE,
                        {
                            "collectable": False,
                            "round_idx": request.round_idx,
                            "sampling_mode": sampling_mode,
                            "research_annotations": annotations,
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


def _submission_candidate(candidate: Any, domain: SynthonCandidateDomain):
    fields = {*_CANDIDATE_FIELDS, "change_summary", "rationale"}
    if not isinstance(candidate, dict) or set(candidate) != fields:
        raise ValueError(f"Each candidate must contain exactly {sorted(fields)}.")
    annotation = {}
    for name in ("change_summary", "rationale"):
        value = candidate[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty short English research note.")
        annotation[name] = value.strip()
    payload = {name: candidate[name] for name in _CANDIDATE_FIELDS}
    return (
        prepare_candidate_payload(payload, domain.space, domain.allowed_reactions),
        annotation,
    )


def _validate_submission(
    submission: HarnessSubmissionRequest,
    domain: SynthonCandidateDomain,
    evaluated: set[str],
    *,
    artifact_root: Path,
    candidate_count: int,
) -> HarnessSubmissionValidation:
    try:
        candidates = read_candidate_file(
            submission.submission,
            submission.artifacts,
            artifact_root,
            candidate_count,
        )
    except (ValueError, OSError) as exc:
        return HarnessSubmissionValidation(
            "retry",
            (
                HarnessSubmissionError(
                    "/artifact_path",
                    "invalid_candidate_file",
                    str(exc),
                    "Repair candidates.json in the workspace and submit its path again.",
                ),
            ),
        )
    errors = []
    first_index_by_key: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        path = f"/candidates/{index}"
        try:
            prepared, _ = _submission_candidate(candidate, domain)
        except ValueError as exc:
            annotation_field = next(
                (
                    name
                    for name in ("change_summary", "rationale")
                    if isinstance(candidate, dict)
                    and (
                        not isinstance(candidate.get(name), str)
                        or not candidate[name].strip()
                    )
                ),
                None,
            )
            errors.append(
                HarnessSubmissionError(
                    f"{path}/{annotation_field}" if annotation_field else path,
                    "invalid_annotation" if annotation_field else "invalid_candidate",
                    f"Candidate at index {index}: {exc}",
                    "Repair the reported entry and its research notes using exact legal task values.",
                )
            )
            continue
        key = prepared.product_id
        if key in evaluated:
            errors.append(
                HarnessSubmissionError(
                    path,
                    "historical_duplicate",
                    f"Candidate at index {index} {json.dumps(prepared.payload)} was already evaluated.",
                    "Replace this entry with a legal unseen candidate; consult validate_synthon_candidate.",
                )
            )
        elif key in first_index_by_key:
            errors.append(
                HarnessSubmissionError(
                    path,
                    "same_session_duplicate",
                    f"Candidate at index {index} duplicates index {first_index_by_key[key]} in this submission.",
                    "Replace this repeated entry and rerun uniqueness checks on the entire repaired file.",
                )
            )
        first_index_by_key.setdefault(key, index)
    return (
        HarnessSubmissionValidation("retry", tuple(errors))
        if errors
        else HarnessSubmissionValidation()
    )


def _usage_counts(usage: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        counter: int(usage[key])
        for key, counter in (
            ("providerCalls", "llm_requests"),
            ("validationSubmissions", "harness_validation_submissions"),
            ("artifactBytes", "harness_artifact_bytes"),
        )
        if key in usage
    }
    if "toolCalls" in usage:
        counts["harness_tool_calls"] = sum(int(count) for count in usage["toolCalls"].values())
    return counts


def _result_summary(
    result: HarnessTurnResult, candidate_count: int
) -> dict[str, object]:
    return {
        "profile_id": result.profile_id,
        "session_id": result.session_id,
        "session_turn_id": result.turn_id,
        "round_index": result.round_index,
        "history_from_seq": result.history_from_seq,
        "history_to_seq": result.history_to_seq,
        "history_digest": result.history_digest,
        "submission_id": result.submission_id,
        "candidate_count": candidate_count,
        "usage": result.usage,
        "tool_budget": result.tool_budget,
        "artifacts": result.artifacts,
    }
