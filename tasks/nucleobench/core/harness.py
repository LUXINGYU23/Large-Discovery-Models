"""Persistent research-harness proposal expansion for NucleoBench."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ldm_tts.harness.pi import PiGuestRuntime, load_pi_guest_runtime
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
    HarnessSubmittedArtifact,
    HarnessToolExtension,
    HarnessTurn,
    HarnessTurnResult,
    canonical_sha256,
    directory_sha256,
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
from tasks.nucleobench.core.constants import TASK_ID
from tasks.nucleobench.core.benchmark_clock import BenchmarkClock
from tasks.nucleobench.core.research import serialize_measured_observations, summarize_measured_observations, write_measured_history
from tasks.nucleobench.core.hamming_gp import HammingGPUCBConfig
from tasks.nucleobench.core.surrogate_query import write_surrogate_snapshot

HARNESS_PROFILE_IDS = (
    "target_biology",
    "regulatory_grammar",
    "sequence_landscape",
    "evidence_critic",
)
AVAILABLE_HARNESS_PROFILE_IDS = HARNESS_PROFILE_IDS + (
    "off_target_selectivity",
    "sequence_composition",
    "module_recombination",
    "alternative_programs",
    "comprehensive_research",
)
DIRECT_HARNESS_PROFILE_ID = "direct_research"
HARNESS_SKILL_IDS = (
    "biopython",
    "experimental-design",
    "scientific-critical-thinking",
    "statsmodels",
)
HARNESS_SOURCE = "nucleobench_persistent_research_harness"
HARNESS_FORBIDDEN_PATTERNS = (
    r"nucleo\s*bench",
    r"move37[-_/\s]*labs.*nucleo",
)
HARNESS_FORBIDDEN_TERMS = (
    "nucleobench",
    "move37-labs/nucleobench",
)
HARNESS_TOOL_NAMES = (
    "get_task_context",
    "get_measured_history",
    "get_sequence_window",
    "list_mutable_regions",
    "validate_mutations",
)
_PRESERVED_REJECTION_CODES = frozenset(
    {"duplicate_position", "non_mutable_position", "unchanged_base"}
)
_LOCAL_RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"
_LOCAL_PROFILE_ROOT = _LOCAL_RESOURCE_ROOT / "profiles"
_LOCAL_TOOL_PATH = _LOCAL_RESOURCE_ROOT / "tools" / "sequence_context.mjs"
_LOCAL_IMAGE_ROOT = _LOCAL_RESOURCE_ROOT / "image"
_CONTAINER_PROFILE_ROOT = Path("/resources/profiles")


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


def harness_submission_contract(candidate_count: int) -> HarnessSubmissionContract:
    if candidate_count < 1:
        raise ValueError("NucleoBench harness candidate count must be positive")
    return HarnessSubmissionContract(
        contract_id="nucleobench_candidate_batch",
        tool_name="submit_candidates",
        payload_schema={
            "type": "object",
            "properties": {
                "artifact_path": {
                    "type": "string",
                    "const": "candidates.json",
                    "description": (
                        "Workspace-relative UTF-8 JSON file containing only a candidates array "
                        f"of exactly {candidate_count} objects with mutations, change_summary, and rationale. "
                        "Write the file with code; submit its path, not its contents."
                    ),
                }
            },
            "required": ["artifact_path"],
            "additionalProperties": False,
        },
        artifact_rules=(HarnessArtifactRule("/artifact_path", (".json",), candidate_count * 65536),),
    )


def harness_profiles(
    profile_ids: Sequence[str] = HARNESS_PROFILE_IDS,
) -> tuple[HarnessProfile, ...]:
    if not profile_ids:
        raise ValueError("Harness profiles must be non-empty")
    if set(profile_ids).difference(AVAILABLE_HARNESS_PROFILE_IDS):
        raise ValueError("Unknown NucleoBench research profile")
    counts = Counter(profile_ids)
    occurrences: Counter[str] = Counter()
    profiles = []
    for profile_id in profile_ids:
        occurrences[profile_id] += 1
        session_id = (
            f"{profile_id}_{occurrences[profile_id]:02d}"
            if counts[profile_id] > 1 else profile_id
        )
        profiles.append(_profile(profile_id, session_id=session_id))
    return tuple(profiles)


def direct_harness_profile() -> tuple[HarnessProfile, ...]:
    return (_profile(DIRECT_HARNESS_PROFILE_ID),)


def _profile(profile_id: str, *, session_id: str | None = None) -> HarnessProfile:
    local_path = _LOCAL_PROFILE_ROOT / profile_id / "AGENTS.md"
    return HarnessProfile(
        session_id or profile_id,
        _CONTAINER_PROFILE_ROOT / profile_id / "AGENTS.md",
        skill_dirs=tuple(Path("/resources/skills") / name for name in HARNESS_SKILL_IDS),
        agents_sha256=file_sha256(local_path),
        skill_dir_sha256=tuple(
            directory_sha256(_LOCAL_RESOURCE_ROOT / "skills" / name)
            for name in HARNESS_SKILL_IDS
        ),
    )


def harness_tool_extensions(*, surrogate_query: bool = False) -> tuple[HarnessToolExtension, ...]:
    extensions = (
        HarnessToolExtension(
            Path("/resources/tools/sequence_context.mjs"),
            file_sha256(_LOCAL_TOOL_PATH),
            HARNESS_TOOL_NAMES,
        ),
    )
    if surrogate_query:
        extensions += (HarnessToolExtension(
            Path("/resources/tools/query_surrogate.mjs"),
            file_sha256(_LOCAL_RESOURCE_ROOT / "tools/query_surrogate.mjs"),
            ("query_surrogate",),
        ),)
    return extensions


def harness_guest_runtime() -> PiGuestRuntime:
    return load_pi_guest_runtime(TASK_ID, _LOCAL_IMAGE_ROOT)


class NucleoBenchHarnessExpander:
    """Collect one validated mutation minibatch from each persistent session."""

    def __init__(
        self,
        client: HarnessClient,
        domain: NucleoBenchCandidateDomain,
        *,
        profiles: Sequence[HarnessProfile],
        candidates_per_profile: int,
        campaign_id: str,
        first_active_round: int,
        attach_empirical_q0: bool,
        allow_repeated_occurrences: bool,
        artifact_root: Path,
        account: Callable[..., Any] | None = None,
        benchmark_clock: BenchmarkClock | None = None,
        surrogate_query_config: HammingGPUCBConfig | None = None,
    ) -> None:
        if not profiles:
            raise ValueError("NucleoBench harness requires at least one profile")
        if not campaign_id.strip():
            raise ValueError("NucleoBench harness campaign_id must not be empty")
        if first_active_round < 0:
            raise ValueError("first_active_round must be non-negative")
        if candidates_per_profile < 1:
            raise ValueError("candidates_per_profile must be positive")
        self.client = client
        self.domain = domain
        self.profiles = tuple(profiles)
        self.candidates_per_profile = candidates_per_profile
        self.campaign_id = campaign_id
        self.profile_set_sha256 = profile_set_sha256(self.profiles)
        self.first_active_round = first_active_round
        self.attach_empirical_q0 = attach_empirical_q0
        self.allow_repeated_occurrences = allow_repeated_occurrences
        self.artifact_root = artifact_root.resolve()
        self.account = account
        self.benchmark_clock = benchmark_clock
        self.surrogate_query_config = surrogate_query_config

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        expected = len(self.profiles) * self.candidates_per_profile
        if request.reservoir_size != expected:
            raise ValueError(
                "harness reservoir size must equal the configured minibatch total "
                f"({expected})"
            )
        evaluated = {
            prepare_candidate_payload(observation.candidate.payload, self.domain.context, allow_empty=True).canonical_key
            for observation in request.observations
        }
        turns = self._turns(request)
        if self.account is not None:
            self.account(
                {
                    "proposal_attempts": len(turns),
                    "harness_turns": len(turns),
                },
                usage_key=f"harness:round:{request.round_idx}",
            )
        started = time.perf_counter()
        usage_by_profile: dict[str, Mapping[str, Any]] = {}
        try:
            results = self.client.run_turn(
                turns,
                recovery_timeout_seconds=(
                    float(self.benchmark_clock.snapshot()["remaining_seconds"])
                    if self.benchmark_clock is not None
                    else 2.0 * self.client.config.limits.wall_time_seconds
                ),
                submission_validator=lambda submission: _validate_submission(
                    submission,
                    self.domain.context,
                    evaluated,
                    measured_candidate_ids={item.candidate_id for item in request.observations},
                    artifact_root=self.artifact_root,
                    candidate_count=self.candidates_per_profile,
                    allow_repeated_occurrences=self.allow_repeated_occurrences,
                ),
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
                "candidates_per_session": [self.candidates_per_profile] * len(self.profiles),
                "proposal_count": len(proposals),
                "candidate_lineage": [
                    _candidate_lineage(proposal, self.domain.context)
                    for proposal in proposals
                ],
                "sessions": [_result_summary(result, self.candidates_per_profile) for result in results],
            },
        )

    def _turns(
        self,
        request: ExpansionRequest,
    ) -> tuple[HarnessTurn, ...]:
        history = _history_delta(request, self.first_active_round)
        serialized_history = serialize_measured_observations(history, self.domain.context)
        write_measured_history(self.artifact_root, request.observations, self.domain.context)
        surrogate_query = (
            write_surrogate_snapshot(request, self.domain.context, self.surrogate_query_config, self.artifact_root)
            if self.surrogate_query_config is not None else None
        )
        history_to_seq = len(request.observations)
        history_from_seq = history_to_seq - len(history)
        history_digest = canonical_sha256(serialized_history)
        forbidden_query_terms = _forbidden_query_terms(request)
        benchmark_time = (
            None if self.benchmark_clock is None else self.benchmark_clock.snapshot()
        )
        turns = tuple(
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
                    candidate_count=self.candidates_per_profile,
                    observations=summarize_measured_observations(serialized_history),
                    session_count=len(self.profiles),
                    allow_repeated_occurrences=self.allow_repeated_occurrences,
                    attach_empirical_q0=self.attach_empirical_q0,
                    initial=request.round_idx == self.first_active_round,
                    history_from_seq=history_from_seq,
                    history_to_seq=history_to_seq,
                    history_digest=history_digest,
                    context=self.domain.context,
                    benchmark_time=benchmark_time,
                    surrogate_query=surrogate_query,
                ),
                forbidden_query_terms=forbidden_query_terms,
            )
            for profile in self.profiles
        )
        restored = []
        for turn in turns:
            input_path = self.artifact_root / "sessions" / turn.profile_id / "turns" / turn.turn_id / "input.json"
            if input_path.exists():
                saved = json.loads(input_path.read_text(encoding="utf-8"))
                old_prefix, old_body = saved["message"].split("\n\n", 1)
                prefix, body = turn.message.split("\n\n", 1)
                old_payload, payload = json.loads(old_body), json.loads(body)
                # Replay the original round clock, but reject changes to any task semantics.
                if "benchmark_time" in payload and "benchmark_time" in old_payload:
                    for name in ("elapsed_seconds", "remaining_seconds"):
                        payload["benchmark_time"][name] = old_payload["benchmark_time"][name]
                if prefix != old_prefix or payload != old_payload:
                    raise ValueError(f"Persisted Harness turn message changed: {turn.turn_id}")
                turn = replace(turn, message=saved["message"])
                if turn.to_dict() != saved:
                    raise ValueError(f"Persisted Harness turn identity changed: {turn.turn_id}")
            restored.append(turn)
        return tuple(restored)

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
        annotations_by_key: dict[str, list[dict[str, Any]]] = {}
        for profile in self.profiles:
            result = by_profile[profile.profile_id]
            if result.submission_status != "accepted":
                raise RuntimeError(
                    f"harness profile submission was rejected: {profile.profile_id}"
                )
            candidates = _read_candidate_file(
                result.submission, result.submitted_artifacts,
                self.artifact_root, self.candidates_per_profile,
            )
            profile_keys: set[str] = set()
            for index, candidate in enumerate(candidates):
                try:
                    prepared, annotation = _submission_candidate(candidate, self.domain.context)
                    if prepared.canonical_key in evaluated:
                        raise ValueError("harness candidate is already present in measured history")
                except ValueError as exc:
                    raise RuntimeError(
                        "committed harness candidate failed authoritative validation: "
                        f"{result.profile_id}[{index}]: {exc}"
                    ) from exc
                if (
                    not self.allow_repeated_occurrences
                    and prepared.canonical_key in profile_keys
                ):
                    raise RuntimeError(
                        "committed harness profile contains a duplicate occurrence: "
                        f"{result.profile_id}[{index}]"
                    )
                profile_keys.add(prepared.canonical_key)
                # Shared by equal occurrences so canonical deduplication retains every hypothesis.
                annotations = annotations_by_key.setdefault(prepared.canonical_key, [])
                annotations.append({
                    "profile_id": result.profile_id,
                    "round_index": request.round_idx,
                    "submission_id": result.submission_id,
                    "item_index": index,
                    **annotation,
                })
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
    *,
    candidate_count: int,
    observations: Sequence[dict[str, object]],
    session_count: int,
    allow_repeated_occurrences: bool,
    attach_empirical_q0: bool,
    initial: bool,
    history_from_seq: int,
    history_to_seq: int,
    history_digest: str,
    context: MutationContext,
    benchmark_time: dict[str, float | int] | None = None,
    surrogate_query: dict[str, object] | None = None,
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
        "evaluated_candidates": {
            "count": len(request.observations),
            "membership_tool": "validate_mutations",
            "membership_field": "already_evaluated",
            "identity": "SHA-256 of the complete rebuilt sequence",
        },
        "measurement_feedback": (
            "Your occurrences enter a shared pool; only selected unique candidates are measured. "
            "Fetch detailed history for informative improvements and failed controls and compare each measurement with its research_annotations: these are the original pre-evaluation "
            "design summaries and hypotheses, not verified explanations of the result. "
            "Match new measurements to your earlier submissions and revise the corresponding hypotheses. "
            "A submitted but unmeasured candidate is neither failed nor successful and remains eligible. "
            "Selection is a sampling event, not evidence of candidate quality: neither being selected "
            "nor being left unmeasured justifies extra confidence or more slots by itself. "
            "Separate expected improvement from the information value of a control. "
            "If progress stalls, investigate a contrasting hypothesis rather than only repeating "
            "near-equivalent variants."
            if attach_empirical_q0 else
            "Your distinct submitted candidates are directly measured; fetch detailed history to compare the results with their "
            "research_annotations and update your hypotheses. An annotation is a pre-evaluation hypothesis, not a measurement."
        ),
        "history_access": {
            "tool": "get_measured_history",
            "description": "Concise results list IDs, utility and mutation count. Query by candidate IDs or round, sort by utility or recency, and page with next_offset. Use response_format=detailed to read exact patches and original design notes for selected IDs. Unmeasured proposals are not shared.",
        },
        "novelty_contract": {
            "session_count": session_count,
            "evaluated_candidates_are_forbidden": True,
            "prior_unmeasured_submissions_may_be_reproposed": True,
            "required_not_evaluated_candidate_count": candidate_count,
            "same_round_cross_session_agreement_is_allowed": True,
            "same_session_repeated_occurrences_are_allowed": allow_repeated_occurrences,
            "repeated_occurrences_contribute_to_empirical_q0": attach_empirical_q0,
            "validate_before_submission": True,
        },
        "sequence_tools": list(HARNESS_TOOL_NAMES),
        "submission_contract": {
            "optional_candidate_fields": {
                "comparison_candidate_ids": "Exact measured IDs from get_measured_history; omit if no measured comparison applies.",
            },
            "tool": "submit_candidates",
            "arguments": {"artifact_path": "candidates.json"},
            "file_format": {"candidates": [{
                "mutations": [{"position": 0, "base": "C"}],
                "change_summary": "Describe the actual sequence change in one short sentence.",
                "rationale": "State the testable hypothesis, expected effect, or control purpose in one short sentence.",
            }]},
            "candidate_count": candidate_count,
            "format_example_is_not_a_candidate_recommendation": True,
            "validation": "The complete file is validated before acceptance; errors refer to JSON paths inside it.",
        },
        "time_budget": {
            "hard_wall_time_minutes": 30,
            "end_open_ended_research_by_minute": 20,
            "first_submission_by_minute": 25,
            "remaining_time_use": "repair_rejected_entries_only",
        },
        "constraints": [
            "Use zero-based positions and only the exact editable positions exposed by the structured tools.",
            "Every replacement base must differ from the paired start base at that position.",
            "Do not seek task implementations, evaluator models or weights, evaluation tables, hidden scores, or other benchmark-only assets.",
            "Campaign measurements are the only measured objective values; do not present a prediction as a measurement.",
            "Research public target biology and sequence-regulatory evidence, and use scratch code in the isolated sandbox when useful.",
            "Only already measured candidates are forbidden. evaluated_candidates describes the authoritative lookup, not an inline exclusion list. validate_mutations reports exact historical membership; submission validates the complete batch again. Earlier unmeasured proposals remain eligible.",
            "New measurements are a compact index, not a lossy scientific summary. Retrieve detailed records for improvements, disappointing variants and controls before revising a hypothesis. Use get_sequence_window for exact parent bases. Do not print whole history files or large arrays; compute aggregate analyses in the sandbox.",
            (
                "Treat the minibatch as an ordered multiset. You may allocate multiple slots to the same legal, historically unseen patch when your evidence justifies extra empirical q0 mass; use multiplicity deliberately rather than as filler."
                if allow_repeated_occurrences
                else "Your minibatch must contain distinct rebuilt sequences. Reordering a patch does not create a new candidate. Cross-session agreement remains allowed."
            ),
            "Build candidates.json with code and inspect counts and uniqueness without printing the entire array. Use validate_mutations for uncertain patches; submit_candidates validates the complete file.",
            "Every candidate must include concise English change_summary and rationale strings. Use optional comparison_candidate_ids for exact measured references copied from get_measured_history's guest_file. Unknown references are rejected. Notes do not affect identity or q0 and must match the repaired patch.",
            "Use /workspace or relative paths in sandbox commands and scripts; sidecar paths under /artifacts are not mounted inside the guest.",
            "Before writing sequence-construction or motif code, read the loaded biopython Skill once and use its construction example. Load exact bases from guest_file rather than copying DNA strings or coordinates from prose.",
            "Build and validate one hypothesis-driven candidate end to end, then save each valid entry incrementally. Maintain a complete draft panel early and improve entries in place; do not make the whole panel depend on one unfinished analysis or long generator.",
            "Task legality is mandatory: preserve length, editable coordinates, base changes, history exclusion, the requested count, and the turn's uniqueness rule. Motif absence, consensus matches, composition targets, and predicted effects are research hypotheses, not additional benchmark constraints. Diagnose a failed check on one construct: repair a coding error, or revise the contradicted design and its rationale. Keep unrelated valid entries. Never bypass task validation or claim a failed biological check passed.",
            "Before searching for a spacer or background, check that fixed inserted modules do not already violate your proposed motif exclusions. Prefer retaining the exact parent background where a redesign is unnecessary. Use short executable construction steps and concise diagnostics; reserve time to complete and submit the full panel instead of repeating an unchanged failing analysis.",
            "On rejection, edit only the reported file entries, recheck the complete batch, and submit the same file path again. Do not retranscribe candidates in tool arguments.",
        ],
    }
    if benchmark_time is not None:
        payload["benchmark_time"] = benchmark_time
    if surrogate_query is not None:
        payload["surrogate_query"] = surrogate_query
        payload["sequence_tools"].append("query_surrogate")
        payload["constraints"].append(
            "query_surrogate supplies a frozen baseline GP from this round's measured history. "
            "Compare hypotheses in batches and inspect the exported result file before retaining or revising designs. "
            "A neutral_prior has insufficient history for data-driven ranking. Keep scientifically motivated alternatives and controls; "
            "high predicted UCB is not a measurement. Explain material disagreements in your existing rationale notes. "
            "The final compiled-policy GP may differ. Queries neither submit proposals nor add frequency mass to q0."
        )
    return (
        "Continue your persistent sequence-design research role. Use your "
        "session history, new measurements, public evidence, scratch analysis, and the "
        "structured paired-start tools. Submit a complete validated minibatch before the "
        "turn deadline.\n\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _submission_candidate(
    candidate: dict[str, Any],
    context: MutationContext,
) -> tuple[PreparedMutationCandidate, dict[str, Any]]:
    fields = {"mutations", "change_summary", "rationale", "comparison_candidate_ids"}
    extra = set(candidate) - fields
    if extra or "mutations" not in candidate:
        raise CandidatePayloadError(
            "invalid_candidate",
            "Each submitted candidate requires mutations, change_summary, and rationale, with optional comparison_candidate_ids; "
            f"unexpected fields: {sorted(extra)}.",
        )
    annotation = {}
    for name in ("change_summary", "rationale"):
        value = candidate.get(name)
        if not isinstance(value, str) or not value.strip():
            raise CandidatePayloadError(
                "invalid_annotation", f"{name} must be a non-empty English research note.",
                metadata={"field": name},
            )
        annotation[name] = value.strip()
    if "comparison_candidate_ids" in candidate:
        annotation["comparison_candidate_ids"] = candidate["comparison_candidate_ids"]
    return prepare_candidate_payload({"mutations": candidate["mutations"]}, context), annotation


def _read_candidate_file(
    submission: Mapping[str, Any],
    artifacts: Sequence[HarnessSubmittedArtifact],
    artifact_root: Path,
    candidate_count: int,
) -> list[Any]:
    if dict(submission) != {"artifact_path": "candidates.json"} or len(artifacts) != 1:
        raise ValueError('Submit {"artifact_path":"candidates.json"} after writing the file.')
    artifact = artifacts[0]
    if artifact.path_pointer != "/artifact_path" or artifact.relative_path != "candidates.json":
        raise ValueError("The snapshot must reference candidates.json at /artifact_path.")
    root = artifact_root.resolve()
    path = (root / artifact.snapshot_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("The submitted file snapshot is unavailable inside the artifact root.")
    body = path.read_bytes()
    if len(body) != artifact.size_bytes or hashlib.sha256(body).hexdigest() != artifact.sha256:
        raise ValueError("The submitted file snapshot does not match its recorded digest or size.")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"candidates"} or not isinstance(payload["candidates"], list):
        raise ValueError('candidates.json must contain only the top-level field "candidates" with an array of annotated mutation patches.')
    candidates = payload["candidates"]
    if len(candidates) != candidate_count:
        raise ValueError(f"Expected exactly {candidate_count} candidates; found {len(candidates)}. Repair the array length.")
    return candidates


def _validate_submission(
    submission: HarnessSubmissionRequest,
    context: MutationContext,
    evaluated: set[str],
    *,
    measured_candidate_ids: set[str],
    artifact_root: Path,
    candidate_count: int,
    allow_repeated_occurrences: bool,
) -> HarnessSubmissionValidation:
    try:
        candidates = _read_candidate_file(
            submission.submission, submission.artifacts, artifact_root, candidate_count,
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
    errors: list[HarnessSubmissionError] = []
    first_index_by_key: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        path = f"/candidates/{index}"
        if not isinstance(candidate, dict):
            errors.append(
                HarnessSubmissionError(
                    path,
                    "invalid_candidate",
                    f"Candidate at index {index} must be an object.",
                    "Replace it with one complete legal mutation patch.",
                )
            )
            continue
        try:
            prepared, _ = _submission_candidate(candidate, context)
        except CandidatePayloadError as exc:
            code = (
                exc.reason
                if exc.reason in _PRESERVED_REJECTION_CODES or exc.reason == "invalid_annotation"
                else "invalid_candidate"
            )
            errors.append(
                HarnessSubmissionError(
                    f"{path}/{exc.metadata['field']}" if "field" in exc.metadata else path,
                    code,
                    f"Candidate at index {index} is not a legal mutation patch "
                    f"({exc.reason}): {exc}",
                    (
                        "Add or correct the short design note for this candidate without changing other entries."
                        if code == "invalid_annotation"
                        else "Use the structured sequence tools to replace this entry and update its design notes."
                    ),
                )
            )
            continue
        if prepared.canonical_key in evaluated:
            errors.append(
                HarnessSubmissionError(
                    path,
                    "historical_duplicate",
                    f"Candidate at index {index} was already evaluated in a "
                    "previous round. Replace it with a different unseen patch.",
                    "Choose a legal patch absent from evaluated_candidates.",
                )
            )
            continue
        first_index = first_index_by_key.get(prepared.canonical_key)
        if not allow_repeated_occurrences and first_index is not None:
            errors.append(
                HarnessSubmissionError(
                    path,
                    "same_session_duplicate",
                    f"Candidate at index {index} duplicates index {first_index} "
                    "in this submission. Keep the first occurrence and replace this one.",
                    "Replace only this repeated entry with another unseen legal patch.",
                )
            )
            continue
        first_index_by_key.setdefault(prepared.canonical_key, index)
        references = candidate.get("comparison_candidate_ids", [])
        if not isinstance(references, list):
            errors.append(
                HarnessSubmissionError(
                    f"{path}/comparison_candidate_ids",
                    "invalid_comparison_reference",
                    "comparison_candidate_ids must be an array of exact measured candidate IDs.",
                    "Read IDs from get_measured_history; omit the field when no measured comparison applies.",
                )
            )
            continue
        for reference_index, reference in enumerate(references):
            if not isinstance(reference, str) or reference not in measured_candidate_ids:
                errors.append(
                    HarnessSubmissionError(
                        f"{path}/comparison_candidate_ids/{reference_index}",
                        "unknown_comparison_candidate",
                        f"Comparison reference {reference!r} is not an authoritative measured candidate ID.",
                        "Copy the exact ID from get_measured_history's guest_file, or remove an unsupported comparison.",
                    )
                )
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


def _candidate_lineage(
    proposal: RawProposal,
    context: MutationContext,
) -> dict[str, object]:
    prepared = prepare_candidate_payload(proposal.payload, context)
    lineage = proposal.metadata["harness_lineage"]
    assert isinstance(lineage, dict)
    return {"canonical_key": prepared.canonical_key, **lineage}


def _result_summary(result: HarnessTurnResult, candidate_count: int) -> dict[str, object]:
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


__all__ = [
    "DIRECT_HARNESS_PROFILE_ID",
    "HARNESS_FORBIDDEN_PATTERNS",
    "HARNESS_PROFILE_IDS",
    "HARNESS_TOOL_NAMES",
    "NucleoBenchHarnessExpander",
    "direct_harness_profile",
    "harness_profiles",
    "harness_submission_contract",
    "harness_guest_runtime",
    "harness_tool_extensions",
    "write_harness_sequence_context",
]
