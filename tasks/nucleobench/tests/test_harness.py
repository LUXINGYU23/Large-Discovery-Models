"""High-risk checks for NucleoBench persistent research sessions."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ldm_tts.contracts import Candidate, EvaluationResult, Observation, RawProposal
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import (
    HarnessSubmissionRequest,
    HarnessSubmittedArtifact,
    HarnessTurnResult,
    directory_sha256,
)
from tasks.nucleobench.core.candidate import MutationContext, NucleoBenchCandidateDomain, make_start_candidate
from tasks.nucleobench.core.cases import get_case
from tasks.nucleobench.core.benchmark_clock import BenchmarkClock
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
from tasks.nucleobench.core.factory import build_proposal_expander, build_surrogate_components
from tasks.nucleobench.core.harness import (
    AVAILABLE_HARNESS_PROFILE_IDS,
    DIRECT_HARNESS_PROFILE_ID,
    HARNESS_PROFILE_IDS,
    HARNESS_SKILL_IDS,
    HARNESS_TOOL_NAMES,
    NucleoBenchHarnessExpander,
    _validate_submission,
    direct_harness_profile,
    harness_guest_runtime,
    harness_profiles,
    harness_submission_contract,
    harness_tool_extensions,
    write_harness_sequence_context,
)
from tasks.nucleobench.core.mock import MOCK_CONTEXT
from tasks.nucleobench.core.optimization_policy import policy_harness_profile
from tasks.nucleobench.core.research import MEASURED_HISTORY_FILE, serialize_measured_observations
from tasks.nucleobench.core.task_spec import build_task_spec
from tasks.nucleobench.core.workflow import describe_ldm_task, main, parse_args


class FakeHarnessClient:
    def __init__(self, artifact_root, attempts_by_profile: dict[str, list[list[dict]]]) -> None:
        self.artifact_root = artifact_root
        self.attempts_by_profile = attempts_by_profile
        self.batches = []
        self.rejections = {}
        self.config = SimpleNamespace(limits=SimpleNamespace(wall_time_seconds=120))

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds=0):
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.batches.append(turns)
        results = []
        for turn in turns:
            accepted = None
            attempt_count = 0
            for attempt_count, candidates in enumerate(
                self.attempts_by_profile[turn.profile_id], start=1
            ):
                submitted = _submission(
                    self.artifact_root, {"candidates": [_annotated(item) for item in candidates]},
                    profile_id=turn.profile_id, turn_id=turn.turn_id,
                    attempt=attempt_count,
                )
                validation = submission_validator(submitted)
                if validation.decision == "accept":
                    accepted = submitted.submission
                    break
                self.rejections.setdefault(turn.profile_id, []).append(
                    validation.errors
                )
            if accepted is None:
                raise AssertionError(
                    f"profile never submitted a valid batch: {turn.profile_id}"
                )
            results.append(
                HarnessTurnResult(
                    profile_id=turn.profile_id,
                    session_id=f"session-{turn.profile_id}",
                    turn_id=turn.turn_id,
                    round_index=turn.round_index,
                    history_from_seq=turn.history_from_seq,
                    history_to_seq=turn.history_to_seq,
                    history_digest=turn.history_digest,
                    input_digest=turn.input_digest,
                    replayed=False,
                    submission_status="accepted",
                    submission_id=f"submission-{turn.profile_id}",
                    submission_digest=submitted.digest,
                    submission=accepted,
                    submitted_artifacts=submitted.artifacts,
                    validation_errors=(),
                    usage={
                        "providerCalls": 2,
                        "toolCalls": {
                            "get_task_context": 1,
                            "submit_candidates": 1,
                        },
                        "validationSubmissions": attempt_count,
                        "artifactBytes": 50,
                    },
                    tool_budget={},
                    artifacts={"turn": f"turns/{turn.turn_id}"},
                )
            )
        return tuple(results)


@pytest.mark.parametrize("unique_count,distinct_sessions", [(256, False), (20, False), (256, True), (32, True)])
def test_eight_session_sampling_decouples_occurrences_pool_and_evaluations(tmp_path, unique_count, distinct_sessions):
    argv = [
        "--search-method", "ldm_harness_compiled",
        "--evaluations-per-round", "128",
        "--harness-candidates-per-session", "32",
        "--proposal-samples", "256", "--bo-pool-size", "176",
    ]
    roles = (
        ["comprehensive_research"] * 8 if distinct_sessions else
        [role for role in AVAILABLE_HARNESS_PROFILE_IDS if role != "comprehensive_research"]
    )
    if distinct_sessions:
        argv.append("--harness-unique-candidates")
    for profile_id in roles:
        argv.extend(("--harness-profile", profile_id))
    args = parse_args(argv)
    spec = describe_ldm_task(args)
    assert spec.reservoir.max_size == 256
    assert spec.proposal_search.parameters["profile_count"] == 8
    assert spec.proposal_search.parameters["candidates_per_session"] == 32
    assert spec.proposal_search.parameters["unique_candidates_per_session"] == distinct_sessions
    assert spec.acquisition.parameters["pool_size"] == 176
    assert spec.response_spaces[0].schema == harness_submission_contract(32).payload_schema

    context = MutationContext(
        case=get_case("malinois_k562"), start_set_digest="a" * 64,
        start_index=0, start_sequence="A" * 200, editable_positions=tuple(range(200)),
    )
    payloads = [
        {"mutations": [{"position": index // 3, "base": "CGT"[index % 3]}]}
        for index in range(unique_count)
    ]
    profiles = harness_profiles(args.harness_profile)
    assert len({profile.profile_id for profile in profiles}) == 8
    if distinct_sessions:
        assert len({profile.agents_path for profile in profiles}) == 1
        assert len({profile.agents_sha256 for profile in profiles}) == 1
    client = FakeHarnessClient(tmp_path, {
        profile.profile_id: [[payloads[(32 * index + slot) % unique_count] for slot in range(32)]]
        for index, profile in enumerate(profiles)
    })
    clock = BenchmarkClock(28_800)
    clock.start()
    expander = build_proposal_expander(
        args.search_method, context, seed=42, evaluations_per_round=128,
        harness_client=client, harness_session_profiles=profiles,
        harness_artifact_root=tmp_path,
        harness_candidates_per_session=32, campaign_id="eight-session-test",
        harness_unique_candidates=args.harness_unique_candidates,
        benchmark_clock=clock,
    )
    expanded = expander.expand(ExpansionRequest(round_idx=1, reservoir_size=256))
    assert len(expanded.proposals) == 256
    assert len(client.batches[0]) == 8
    assert 0 < client.recovery_timeout_seconds <= 28_800
    for turn in client.batches[0]:
        message = json.loads(turn.message.split("\n\n", 1)[1])
        assert 0 < message["benchmark_time"]["remaining_seconds"] <= 28_800
        assert message["novelty_contract"]["same_session_repeated_occurrences_are_allowed"] == (not distinct_sessions)
        assert message["novelty_contract"]["repeated_occurrences_contribute_to_empirical_q0"]
        assert "only selected unique candidates are measured" in message["measurement_feedback"]
    domain = NucleoBenchCandidateDomain(context)
    unique = {}
    for proposal in expanded.proposals:
        admitted = domain.admit(proposal)
        assert isinstance(admitted, Candidate)
        unique[admitted.candidate_id] = admitted
    assert len(unique) == unique_count
    assert sum(item.metadata[NUCLEOBENCH_Q0_METADATA_KEY]["occurrence_count"] for item in unique.values()) == 256
    encoder, selector = build_surrogate_components(
        args.search_method, context, evaluations_per_round=128,
        proposal_samples=256, bo_pool_size=176, seed=42,
    )
    selector.fit(())
    selected = selector.select(
        tuple(unique.values()),
        {key: encoder.encode(item) for key, item in unique.items()}, count=128,
    )
    assert selected.metadata["bo_pool_size"] == min(176, unique_count)
    assert len(set(selected.selected_candidate_ids)) == min(128, unique_count)


@pytest.mark.parametrize("extra", [
    ["--proposal-samples", "255"],
    ["--bo-pool-size", "256"],
    ["--bo-pool-size", "127"],
])
def test_invalid_parallel_harness_sizes_are_rejected(extra):
    with pytest.raises(SystemExit):
        parse_args([
            "--search-method", "ldm_harness_compiled", "--evaluations-per-round", "128",
            "--harness-candidates-per-session", "64", "--bo-pool-size", "176", *extra,
        ])


def test_parallel_sessions_preserve_within_and_cross_profile_consensus_for_q0(tmp_path) -> None:
    payloads = _payloads()
    attempts = {
        profile_id: [[payloads[0], payloads[index + 1]]]
        for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
    }
    attempts[HARNESS_PROFILE_IDS[0]] = [[payloads[0], payloads[0]]]
    client = FakeHarnessClient(tmp_path, attempts)
    usage = []

    result = _expander(client, candidates_per_profile=2, account=usage.append).expand(
        ExpansionRequest(round_idx=1, reservoir_size=8)
    )

    assert len(result.proposals) == 8
    assert len(client.batches[0]) == 4
    shared = [item for item in result.proposals if item.payload == payloads[0]]
    assert len(shared) == 5
    assert all(
        item.metadata[NUCLEOBENCH_Q0_METADATA_KEY]
        == {
            "occurrence_count": 5,
            "valid_occurrence_count": 8,
            "probability": 0.625,
        }
        for item in shared
    )
    assert result.selection_mode == "acquisition"
    for proposal in shared:
        notes = proposal.metadata["research_annotations"]
        assert len(notes) == 5
        assert {note["profile_id"] for note in notes} == set(HARNESS_PROFILE_IDS)
        admitted = NucleoBenchCandidateDomain(MOCK_CONTEXT).admit(proposal)
        assert admitted.metadata["research_annotations"] == notes
        assert admitted.canonical_key == NucleoBenchCandidateDomain(MOCK_CONTEXT).admit(
            RawProposal(proposal.payload, "without_notes")
        ).canonical_key
    assert usage == [
        {"proposal_attempts": 4, "harness_turns": 4},
        {
            "llm_requests": 8,
            "harness_tool_calls": 8,
            "harness_validation_submissions": 4,
            "harness_artifact_bytes": 200,
        },
    ]


def test_unique_parallel_batch_repairs_canonical_duplicates_but_keeps_cross_session_q0(tmp_path):
    shared = {"mutations": [{"position": 0, "base": "C"}, {"position": 2, "base": "G"}]}
    reordered = {"mutations": list(reversed(shared["mutations"]))}
    profiles = harness_profiles(["comprehensive_research"] * 2)
    client = FakeHarnessClient(tmp_path, {
        profiles[0].profile_id: [[shared, reordered], [shared, _payloads()[1]]],
        profiles[1].profile_id: [[shared, _payloads()[2]]],
    })
    result = build_proposal_expander(
        "ldm_harness_compiled", MOCK_CONTEXT, seed=42, evaluations_per_round=2,
        harness_client=client, harness_session_profiles=profiles,
        harness_artifact_root=tmp_path,
        harness_candidates_per_session=2, harness_unique_candidates=True,
        campaign_id="unique-batches",
    ).expand(ExpansionRequest(round_idx=1, reservoir_size=4))
    rejection = client.rejections[profiles[0].profile_id][0][0]
    assert rejection.code == "same_session_duplicate"
    assert rejection.path == "/candidates/1"
    assert "duplicates index 0" in rejection.message
    assert result.selection_mode == "acquisition"
    shared_proposals = [proposal for proposal in result.proposals if proposal.payload == shared]
    assert len(shared_proposals) == 2
    assert all(proposal.metadata[NUCLEOBENCH_Q0_METADATA_KEY]["probability"] == 0.5 for proposal in shared_proposals)
    assert len({turn.turn_id for turn in client.batches[0]}) == 2


def test_turn_uses_previous_round_delta_and_complete_evaluated_exclusion(tmp_path) -> None:
    payloads = _payloads()
    observations = (
        _observation({"mutations": []}, round_idx=0),
        _observation(payloads[0], round_idx=1),
    )
    client = FakeHarnessClient(
        tmp_path,
        {
            profile_id: [[payloads[index + 2]]]
            for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
        }
    )

    _expander(client).expand(
        ExpansionRequest(round_idx=2, reservoir_size=4, observations=observations)
    )

    messages = [
        json.loads(turn.message[turn.message.index("{"):])
        for turn in client.batches[0]
    ]
    assert all(message["message_type"] == "history_delta" for message in messages)
    assert all(
        message["new_measured_observations"]
        == [{"candidate_id": observations[1].candidate_id, "round_index": 1,
             "utility": 1.0, "hamming_distance": 1}]
        for message in messages
    )
    assert all(
        message["evaluated_candidates"]["count"] == 2
        and message["evaluated_candidates"]["membership_tool"] == "validate_mutations"
        for message in messages
    )
    assert all(
        message["novelty_contract"]["prior_unmeasured_submissions_may_be_reproposed"]
        for message in messages
    )
    assert all(
        message["novelty_contract"]["same_session_repeated_occurrences_are_allowed"]
        for message in messages
    )
    assert all(message["sequence_tools"] == list(HARNESS_TOOL_NAMES) for message in messages)
    assert all(
        turn.history_from_seq == 1 and turn.history_to_seq == 2
        for turn in client.batches[0]
    )


def test_submission_validation_returns_actionable_mutation_reasons(tmp_path) -> None:
    payloads = _payloads()
    historical = NucleoBenchCandidateDomain(MOCK_CONTEXT).admit(
        RawProposal(payloads[0], "test")
    )
    assert isinstance(historical, Candidate)
    validation = _validate_submission(
        _submission(
            tmp_path,
            {
                "candidates": [_annotated(item) for item in [
                    payloads[0],
                    {
                        "mutations": [
                            {"position": 0, "base": "C"},
                            {"position": 0, "base": "G"},
                        ]
                    },
                    {"mutations": [{"position": 1, "base": "C"}]},
                    {"mutations": [{"position": 2, "base": "A"}]},
                    payloads[4],
                    payloads[4],
                    {"mutations": [{"position": 2, "base": "C", "note": "extra"}]},
                ]]
            },
        ),
        MOCK_CONTEXT,
        {historical.canonical_key},
        artifact_root=tmp_path,
        candidate_count=7,
        allow_repeated_occurrences=True,
    )

    assert [item.code for item in validation.errors] == [
        "historical_duplicate",
        "duplicate_position",
        "non_mutable_position",
        "unchanged_base",
        "invalid_candidate",
    ]
    assert all(item.path.startswith("/candidates/") for item in validation.errors)

    direct_validation = _validate_submission(
        _submission(
            tmp_path,
            {"candidates": [_annotated(payloads[4]), _annotated(payloads[4])]},
            turn_id="turn-2",
        ),
        MOCK_CONTEXT,
        set(),
        artifact_root=tmp_path,
        candidate_count=2,
        allow_repeated_occurrences=False,
    )
    assert [item.code for item in direct_validation.errors] == [
        "same_session_duplicate"
    ]


def test_history_rejection_is_repaired_before_commit_and_q0(tmp_path) -> None:
    payloads = _payloads()
    attempts = {
        profile_id: [[payloads[index + 1]]]
        for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
    }
    attempts[HARNESS_PROFILE_IDS[0]] = [[payloads[0]], [payloads[5]]]
    client = FakeHarnessClient(tmp_path, attempts)

    result = _expander(client).expand(
        ExpansionRequest(
            round_idx=1,
            reservoir_size=4,
            observations=(_observation(payloads[0], round_idx=0),),
        )
    )

    rejection = client.rejections[HARNESS_PROFILE_IDS[0]][0][0]
    assert rejection.code == "historical_duplicate"
    assert "already evaluated" in rejection.message
    assert len(result.proposals) == 4
    assert all(
        item.metadata[NUCLEOBENCH_Q0_METADATA_KEY]["valid_occurrence_count"] == 4
        for item in result.proposals
    )


def test_direct_harness_uses_one_session_without_q0(tmp_path) -> None:
    payloads = _payloads()[:3]
    client = FakeHarnessClient(tmp_path, {DIRECT_HARNESS_PROFILE_ID: [payloads]})
    result = NucleoBenchHarnessExpander(
        client,
        NucleoBenchCandidateDomain(MOCK_CONTEXT),
        profiles=direct_harness_profile(),
        candidates_per_profile=3,
        campaign_id="test-campaign",
        first_active_round=1,
        attach_empirical_q0=False,
        allow_repeated_occurrences=False,
        artifact_root=tmp_path,
    ).expand(ExpansionRequest(round_idx=1, reservoir_size=3))

    assert len(client.batches[0]) == 1
    assert [item.payload for item in result.proposals] == payloads
    assert all(
        NUCLEOBENCH_Q0_METADATA_KEY not in item.metadata
        for item in result.proposals
    )
    assert result.selection_mode == "reservoir_order"
    assert client.recovery_timeout_seconds == 120


def test_task_local_harness_resources_cover_proposals_and_compiled_policy(
    tmp_path,
) -> None:
    context_path = tmp_path / "sequence_context.json"
    write_harness_sequence_context(MOCK_CONTEXT, context_path)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    profiles = harness_profiles()
    extensions = harness_tool_extensions()
    guest = harness_guest_runtime()
    proposal_contract = harness_submission_contract(2)
    policy_profiles = policy_harness_profile()

    assert set(context) == {"schema_version", "case", "paired_start"}
    assert tuple(profile.profile_id for profile in profiles) == HARNESS_PROFILE_IDS
    resource_root = Path(__file__).resolve().parents[1] / "resources" / "harness"
    expected_skills = tuple(Path("/resources/skills") / name for name in HARNESS_SKILL_IDS)
    expected_digests = tuple(
        directory_sha256(resource_root / "skills" / name) for name in HARNESS_SKILL_IDS
    )
    for profile in (*harness_profiles(AVAILABLE_HARNESS_PROFILE_IDS), *direct_harness_profile()):
        assert profile.skill_dirs == expected_skills
        assert profile.skill_dir_sha256 == expected_digests
    assert all((resource_root / "skills" / name / "LICENSE").is_file() for name in HARNESS_SKILL_IDS)
    assert all(len(profile.agents_sha256) == 64 for profile in profiles)
    assert extensions[0].tool_names == HARNESS_TOOL_NAMES
    assert proposal_contract.tool_name == "submit_candidates"
    assert proposal_contract.payload_schema["properties"]["artifact_path"]["const"] == "candidates.json"
    assert proposal_contract.artifact_rules[0].path_pointer == "/artifact_path"
    assert guest.image_ref.startswith("ldm/nucleobench-research:")
    assert policy_profiles[0].profile_id == "policy_architect"
    assert len(policy_profiles[0].skill_dirs) == 1

    compiled_spec = build_task_spec(
        MOCK_CONTEXT.case,
        search_method="ldm_harness_compiled",
        evaluations_per_round=2,
    )
    assert compiled_spec.proposal_search.parameters["profile_count"] == 4
    assert compiled_spec.proposal_search.parameters["skills_loaded"] is True
    assert compiled_spec.proposal_search.parameters["skill_ids"] == list(HARNESS_SKILL_IDS)
    assert compiled_spec.proposal_search.parameters["policy_profile_count"] == 1
    assert compiled_spec.proposal_search.parameters["policy_skills_loaded"] is True

    expander = build_proposal_expander(
        "ldm_harness",
        MOCK_CONTEXT,
        seed=0,
        evaluations_per_round=1,
        harness_client=FakeHarnessClient(
            tmp_path,
            {
                profile.profile_id: [[_payloads()[index]]]
                for index, profile in enumerate(profiles)
            }
        ),
        harness_session_profiles=profiles,
        harness_artifact_root=tmp_path,
        campaign_id="test-campaign",
        first_active_round=1,
    )
    assert isinstance(expander, NucleoBenchHarnessExpander)


def test_harness_dry_run_exposes_current_settings_without_secret(
    tmp_path,
    capsys,
) -> None:
    key_path = tmp_path / "provider.key"
    key_path.write_text("test-secret", encoding="utf-8")

    assert main(
        [
            "--dry-run",
            "--case-id",
            "malinois_k562",
            "--search-method",
            "ldm_harness_compiled",
            "--evaluations-per-round",
            "3",
            "--llm-url",
            "https://provider.example/v1",
            "--llm-model-name",
            "research-model",
            "--api-key-file",
            str(key_path),
            "--harness-tool-budget",
            "web_search=3",
            "--policy-tool-budget",
            "web_search=2",
            "--no-harness-context7",
        ]
    ) == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["proposal_provider"] == {
        "required": True,
        "configured": True,
        "wire_api": "responses",
        "reasoning": "max",
    }
    assert payload["harness"]["profile_ids"] == list(HARNESS_PROFILE_IDS)
    assert payload["harness"]["skills_loaded"] is True
    assert payload["harness"]["skill_ids"] == list(HARNESS_SKILL_IDS)
    assert payload["harness"]["tool_call_budgets"] == {"web_search": 3}
    assert payload["harness"]["policy_session"]["skills_loaded"] is True
    assert payload["harness"]["context7_enabled"] is False
    assert "test-secret" not in output


def test_local_harness_uses_host_identity(monkeypatch) -> None:
    monkeypatch.setattr("tasks.nucleobench.core.workflow.os.getuid", lambda: 1001, raising=False)
    monkeypatch.setattr("tasks.nucleobench.core.workflow.os.getgid", lambda: 1002, raising=False)

    args = parse_args(["--search-method", "harness"])

    assert args.harness_container_user == "1001:1002"


def _expander(
    client: FakeHarnessClient,
    *,
    candidates_per_profile: int = 1,
    account=None,
) -> NucleoBenchHarnessExpander:
    return NucleoBenchHarnessExpander(
        client,
        NucleoBenchCandidateDomain(MOCK_CONTEXT),
        profiles=harness_profiles(),
        candidates_per_profile=candidates_per_profile,
        campaign_id="test-campaign",
        first_active_round=1,
        attach_empirical_q0=True,
        allow_repeated_occurrences=True,
        artifact_root=client.artifact_root,
        account=account,
    )


def _submission(root, payload, *, profile_id="target_biology", turn_id="turn-1", attempt=1):
    body = json.dumps(payload).encode("utf-8")
    snapshot = root / profile_id / turn_id / str(attempt) / "snapshot.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(body)
    return HarnessSubmissionRequest(
        profile_id, turn_id, attempt, {"artifact_path": "candidates.json"},
        artifacts=(HarnessSubmittedArtifact(
            "/artifact_path", "candidates.json", snapshot.relative_to(root).as_posix(),
            hashlib.sha256(body).hexdigest(), len(body),
        ),),
    )


@pytest.mark.parametrize("payload,reason", [
    ([], "only"),
    ({"candidates": [], "notes": "extra"}, "only"),
    ({"candidates": []}, "Expected exactly 1 candidates; found 0"),
])
def test_candidate_file_shape_and_count_are_validated(tmp_path, payload, reason):
    validation = _validate_submission(
        _submission(tmp_path, payload), MOCK_CONTEXT, set(),
        artifact_root=tmp_path, candidate_count=1, allow_repeated_occurrences=True,
    )
    assert validation.decision == "retry"
    assert validation.errors[0].code == "invalid_candidate_file"
    assert reason in validation.errors[0].message


def test_candidate_file_uses_snapshot_and_rejects_corruption(tmp_path):
    submitted = _submission(tmp_path, {"candidates": [_annotated(_payloads()[0])]})
    (tmp_path / "candidates.json").write_text("invalid workspace edit")
    def validate(value):
        return _validate_submission(
            value, MOCK_CONTEXT, set(), artifact_root=tmp_path,
            candidate_count=1, allow_repeated_occurrences=False,
        )
    assert validate(submitted).decision == "accept"
    snapshot = tmp_path / submitted.artifacts[0].snapshot_path
    snapshot.write_text("{}")
    assert "digest or size" in validate(submitted).errors[0].message
    snapshot.write_bytes(b"{broken")
    malformed = replace(submitted, artifacts=(replace(
        submitted.artifacts[0], sha256=hashlib.sha256(b"{broken").hexdigest(), size_bytes=7,
    ),))
    assert validate(malformed).decision == "retry"


def _annotated(payload: dict) -> dict:
    return {
        "change_summary": "Change the specified editable bases relative to the paired start.",
        "rationale": "Test the local sequence hypothesis against the measured baseline.",
        **payload,
    }


def test_research_notes_survive_selection_history_and_policy_projection(tmp_path):
    import numpy as np
    from ldm_tts.optimization import BOObservation, BOPrediction
    from tasks.nucleobench.core.hamming_gp import HammingGPUCBConfig, NucleotideHammingEncoder
    from tasks.nucleobench.core.optimization_policy import NucleoOptimizationPolicyAdapter
    from tasks.nucleobench.core.policy_features import NucleoPolicyFeatureEncoder

    payloads = _payloads()
    client = FakeHarnessClient(tmp_path, {
        role: [[_annotated({**payloads[0], "rationale": f"Hypothesis from {role}."}), payloads[1]]]
        for role in HARNESS_PROFILE_IDS
    })
    expander = _expander(client, candidates_per_profile=2)
    first = expander.expand(ExpansionRequest(round_idx=1, reservoir_size=8))
    measured_candidate = NucleoBenchCandidateDomain(MOCK_CONTEXT).admit(first.proposals[0])
    observations = (
        _observation({"mutations": []}, round_idx=0),
        Observation(measured_candidate, EvaluationResult(
            measured_candidate.candidate_id, "succeeded", metrics={"utility": 2.0},
        ), round_idx=1),
    )
    for role in HARNESS_PROFILE_IDS:
        client.attempts_by_profile[role] = [[payloads[2], payloads[3]]]
    expander.expand(ExpansionRequest(round_idx=2, reservoir_size=8, observations=observations))
    published = json.loads((tmp_path / MEASURED_HISTORY_FILE).read_text())["observations"]
    assert len(published) == 2
    assert {row["candidate_id"] for row in published} == {item.candidate_id for item in observations}
    notes = published[1]["research_annotations"]
    assert len(notes) == 4
    assert {note["rationale"] for note in notes} == {f"Hypothesis from {role}." for role in HARNESS_PROFILE_IDS}
    for turn in client.batches[1]:
        message = json.loads(turn.message.split("\n\n", 1)[1])
        assert message["new_measured_observations"] == [{
            "candidate_id": measured_candidate.candidate_id, "round_index": 1,
            "utility": 2.0, "hamming_distance": 1,
        }]
        assert message["evaluated_candidates"]["count"] == 2
        assert message["history_access"]["tool"] == "get_measured_history"

    encoder = NucleotideHammingEncoder(MOCK_CONTEXT)
    history = tuple(BOObservation.from_observation(
        item, objective_names=("utility",), feature=encoder.encode(item.candidate),
        metadata={"round_idx": item.round_idx},
    ) for item in observations)
    query = NucleoBenchCandidateDomain(MOCK_CONTEXT).admit(RawProposal(payloads[2], "test"))
    kwargs = dict(
        history=history, candidates=(query,), representations={query.candidate_id: encoder.encode(query)},
        q0=np.asarray([1.0]), baseline_predictions=(BOPrediction.scalar(
            query.candidate_id, mean=0.0, std=1.0, acquisition_score=1.0,
        ),), valid_proposal_occurrences=8,
    )
    adapter = NucleoOptimizationPolicyAdapter(
        NucleoPolicyFeatureEncoder(MOCK_CONTEXT), seed=42, gp_config=HammingGPUCBConfig(),
        default_alpha=2.0, default_eta=0.25, measured_history_path=tmp_path / MEASURED_HISTORY_FILE,
    )
    with_notes = adapter.build_selection_round(**kwargs)
    assert with_notes.measured_observations[1]["candidate_id"] == measured_candidate.candidate_id
    assert "research_annotations" not in with_notes.measured_observations[1]
    adapter.measured_history_path = None
    without_notes = adapter.build_selection_round(**kwargs)
    np.testing.assert_array_equal(with_notes.history_features, without_notes.history_features)
    np.testing.assert_array_equal(with_notes.query_features, without_notes.query_features)
    assert with_notes.execution_context == without_notes.execution_context


@pytest.mark.parametrize("field,value", [("change_summary", None), ("rationale", " "), ("rationale", 5)])
def test_candidate_research_note_errors_are_indexed(tmp_path, field, value):
    candidate = _annotated(_payloads()[0])
    if value is None:
        del candidate[field]
    else:
        candidate[field] = value
    validation = _validate_submission(
        _submission(tmp_path, {"candidates": [candidate]}), MOCK_CONTEXT, set(),
        artifact_root=tmp_path, candidate_count=1, allow_repeated_occurrences=False,
    )
    assert validation.decision == "retry"
    assert validation.errors[0].code == "invalid_annotation"
    assert validation.errors[0].path == f"/candidates/0/{field}"


def test_history_tool_reads_filtered_fresh_snapshot(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the Pi task extension")
    version = subprocess.run([node, "--version"], check=True, capture_output=True, text=True).stdout
    if int(version.strip().lstrip("v").split(".")[0]) < 24:
        pytest.skip("The Pi task extension requires Node 24 or newer")
    context_path = tmp_path / "sequence_context.json"
    write_harness_sequence_context(MOCK_CONTEXT, context_path)
    history_path = tmp_path / "observations.json"
    history_path.write_text(json.dumps({"observations": [
        {"candidate_id": "start", "round_index": 0, "mutations": [], "research_annotations": []},
    ]}))
    extension = Path(__file__).parents[1] / "resources/harness/tools/sequence_context.mjs"
    script = """
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const {default: load} = await import(process.argv[1]);
const tools = new Map();
load({registerTool: tool => tools.set(tool.name, tool)});
const read = async args => (await tools.get('get_measured_history').execute('test', args)).details;
const window = async args => (await tools.get('get_sequence_window').execute('test', args)).details;
const start = JSON.parse(readFileSync(process.env.LDM_NUCLEOBENCH_CONTEXT)).paired_start.start_sequence;
const sha = bases => createHash('sha256').update(bases, 'ascii').digest('hex');
assert.equal((await read({})).total, 1);
const initial = await window({start: 0, end_exclusive: start.length});
assert.equal(initial.bases, start);
assert.equal(initial.bases_sha256, sha(start));
assert.deepEqual(await window({candidate_id: 'start', start: 0, end_exclusive: start.length}), initial);
const rows = [
  {candidate_id: 'start', round_index: 0, mutations: [], research_annotations: []},
  {candidate_id: 'measured', round_index: 1, mutations: [{position: 0, base: 'C'}, {position: 2, base: 'G'}], research_annotations: [{rationale: 'Original hypothesis.'}]},
];
writeFileSync(process.env.LDM_NUCLEOBENCH_HISTORY, JSON.stringify({observations: rows}));
assert.equal((await read({limit: 1})).next_offset, 1);
assert.deepEqual((await read({offset: 1, response_format: 'detailed'})).observations, [rows[0]]);
assert.deepEqual((await read({round_index: 1, response_format: 'detailed'})).observations, [rows[1]]);
assert.deepEqual((await read({candidate_ids: ['measured'], response_format: 'detailed'})).observations, [rows[1]]);
assert.equal((await read({candidate_ids: ['measured']})).observations[0].hamming_distance, 2);
assert.equal((await read({candidate_ids: ['measured']})).observations[0].mutations, undefined);
assert.equal((await tools.get('validate_mutations').execute('test', {mutations: rows[1].mutations})).details.already_evaluated, true);
assert.equal((await tools.get('validate_mutations').execute('test', {mutations: [{position: 0, base: 'G'}]})).details.already_evaluated, false);
const unseen = await read({candidate_ids: ['private-unmeasured']});
assert.deepEqual(unseen.observations, []);
assert.deepEqual(unseen.unmeasured_or_unknown_ids, ['private-unmeasured']);
const parent = await window({candidate_id: 'measured', start: 1, end_exclusive: 4});
assert.equal(parent.bases, start[1] + 'G' + start[3]);
assert.equal(parent.bases_sha256, sha(parent.bases));
await assert.rejects(window({candidate_id: 'private-unmeasured', start: 0, end_exclusive: 4}), /unmeasured or unknown/);
for (const args of [{start: -1, end_exclusive: 4}, {start: 0, end_exclusive: start.length + 1}, {start: 2, end_exclusive: 2}]) {
  await assert.rejects(window(args), /non-empty range/);
}
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script, extension.resolve().as_uri()], check=True,
        env={**os.environ, "LDM_NUCLEOBENCH_CONTEXT": str(context_path),
             "LDM_NUCLEOBENCH_HISTORY": str(history_path)},
        capture_output=True, text=True,
    )


def _payloads() -> list[dict[str, list[dict[str, object]]]]:
    return [
        {"mutations": [{"position": position, "base": base}]}
        for position, base in (
            (0, "C"),
            (0, "G"),
            (0, "T"),
            (2, "C"),
            (2, "G"),
            (2, "T"),
        )
    ]


def _observation(payload: dict, *, round_idx: int) -> Observation:
    candidate = (
        NucleoBenchCandidateDomain(MOCK_CONTEXT).admit(RawProposal(payload, "test"))
        if payload["mutations"]
        else make_start_candidate(MOCK_CONTEXT)
    )
    assert isinstance(candidate, Candidate)
    return Observation(
        candidate,
        EvaluationResult(
            candidate.candidate_id,
            "succeeded",
            metrics={"utility": 1.0},
        ),
        round_idx=round_idx,
    )
