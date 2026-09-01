"""High-risk checks for NucleoBench persistent research sessions."""

from __future__ import annotations

import json
from pathlib import Path

from ldm_tts.contracts import Candidate, EvaluationResult, Observation, RawProposal
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import HarnessSubmissionRequest, HarnessTurnResult
from tasks.nucleobench.core import workflow
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
from tasks.nucleobench.core.factory import build_proposal_expander
from tasks.nucleobench.core.harness import (
    DIRECT_HARNESS_PROFILE_ID,
    HARNESS_PROFILE_IDS,
    HARNESS_TOOL_NAMES,
    NucleoBenchHarnessExpander,
    _validate_submission,
    direct_harness_profile,
    harness_profiles,
    harness_tool_extensions,
    write_harness_sequence_context,
)
from tasks.nucleobench.core.mock import MOCK_CONTEXT
from tasks.nucleobench.core.task_spec import build_task_spec
from tasks.nucleobench.core.workflow import main, parse_args


class FakeHarnessClient:
    def __init__(self, attempts_by_profile: dict[str, list[list[dict]]]) -> None:
        self.attempts_by_profile = attempts_by_profile
        self.batches = []
        self.rejections = {}

    def run_turn(self, turns, *, submission_validator):
        self.batches.append(turns)
        results = []
        for turn in turns:
            attempts = self.attempts_by_profile[turn.profile_id]
            accepted = None
            for attempt_index, candidates in enumerate(attempts, start=1):
                validation = submission_validator(
                    HarnessSubmissionRequest(
                        turn.profile_id,
                        turn.turn_id,
                        attempt_index,
                        tuple(candidates),
                    )
                )
                if validation.accepted:
                    accepted = candidates
                    break
                self.rejections.setdefault(turn.profile_id, []).append(
                    validation.rejections
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
                    submission_id=f"submission-{turn.profile_id}",
                    candidates=tuple(accepted),
                    usage={
                        "providerCalls": 2,
                        "toolCalls": {"get_task_context": 1, "submit_candidates": 1},
                        "artifactBytes": 50,
                    },
                    tool_budget={},
                    artifacts={"turn": f"turns/{turn.turn_id}"},
                )
            )
        return tuple(results)


def test_parallel_sessions_preserve_cross_profile_consensus_for_q0() -> None:
    payloads = _payloads()
    attempts = {
        profile_id: [[payloads[0], payloads[index + 1]]]
        for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
    }
    client = FakeHarnessClient(attempts)
    usage = []
    result = _expander(client, candidates_per_turn=2, account=usage.append).expand(
        ExpansionRequest(round_idx=1, reservoir_size=8)
    )

    assert len(result.proposals) == 8
    assert len(client.batches[0]) == 4
    shared = [item for item in result.proposals if item.payload == payloads[0]]
    assert len(shared) == 4
    assert all(
        item.metadata[NUCLEOBENCH_Q0_METADATA_KEY]
        == {
            "occurrence_count": 4,
            "valid_occurrence_count": 8,
            "probability": 0.5,
        }
        for item in shared
    )
    assert result.selection_mode == "acquisition"
    assert usage == [
        {"proposal_attempts": 4, "harness_turns": 4},
        {"llm_requests": 8, "harness_tool_calls": 8, "harness_artifact_bytes": 200},
    ]


def test_turn_uses_previous_round_delta_and_complete_evaluated_exclusion() -> None:
    payloads = _payloads()
    observations = (
        _observation({"mutations": []}, round_idx=0),
        _observation(payloads[0], round_idx=1),
    )
    client = FakeHarnessClient(
        {
            profile_id: [[payloads[index + 2]]]
            for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
        }
    )
    _expander(client, candidates_per_turn=1).expand(
        ExpansionRequest(round_idx=2, reservoir_size=4, observations=observations)
    )

    messages = [
        json.loads(turn.message.split("\n\n", 1)[1]) for turn in client.batches[0]
    ]
    assert all(message["message_type"] == "history_delta" for message in messages)
    assert all(
        message["new_measured_observations"]
        == [
            {
                "mutations": payloads[0]["mutations"],
                "utility": 1.0,
            }
        ]
        for message in messages
    )
    assert all(
        message["evaluated_candidates"]
        == [
            {"mutations": []},
            payloads[0],
        ]
        for message in messages
    )
    assert all(
        message["novelty_contract"]
        == {
            "evaluated_candidates_are_forbidden": True,
            "prior_unmeasured_submissions_may_be_reproposed": True,
            "required_not_evaluated_candidate_count": 1,
            "same_round_cross_session_agreement_is_allowed": True,
            "same_session_duplicates_are_forbidden": True,
            "validate_before_submission": True,
        }
        for message in messages
    )
    assert all(
        message["sequence_tools"] == list(HARNESS_TOOL_NAMES) for message in messages
    )
    assert all(
        turn.history_from_seq == 1 and turn.history_to_seq == 2
        for turn in client.batches[0]
    )


def test_submission_validation_returns_actionable_mutation_reasons() -> None:
    historical = NucleoBenchCandidateDomain(MOCK_CONTEXT).admit(
        RawProposal(_payloads()[0], "test")
    )
    assert isinstance(historical, Candidate)
    submission = HarnessSubmissionRequest(
        "target_biology",
        "turn-1",
        1,
        (
            _payloads()[0],
            {"mutations": [{"position": 0, "base": "C"}, {"position": 0, "base": "G"}]},
            {"mutations": [{"position": 1, "base": "C"}]},
            {"mutations": [{"position": 2, "base": "A"}]},
            _payloads()[4],
            _payloads()[4],
            {"mutations": [{"position": 2, "base": "C", "note": "extra"}]},
        ),
    )

    validation = _validate_submission(
        submission, MOCK_CONTEXT, {historical.canonical_key}
    )

    assert [item.code for item in validation.rejections] == [
        "historical_duplicate",
        "duplicate_position",
        "non_mutable_position",
        "unchanged_base",
        "same_session_duplicate",
        "invalid_candidate",
    ]
    assert all(f"index {item.index}" in item.message for item in validation.rejections)
    assert "duplicates index 4" in validation.rejections[4].message


def test_history_rejection_is_repaired_before_commit_and_q0() -> None:
    payloads = _payloads()
    measured = _observation(payloads[0], round_idx=0)
    attempts = {
        profile_id: [[payloads[index + 1]]]
        for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
    }
    attempts[HARNESS_PROFILE_IDS[0]] = [[payloads[0]], [payloads[5]]]
    client = FakeHarnessClient(attempts)

    result = _expander(client, candidates_per_turn=1).expand(
        ExpansionRequest(round_idx=1, reservoir_size=4, observations=(measured,))
    )

    rejection = client.rejections[HARNESS_PROFILE_IDS[0]][0][0]
    assert rejection.code == "historical_duplicate"
    assert "already evaluated" in rejection.message
    assert len(result.proposals) == 4
    assert all(
        item.metadata[NUCLEOBENCH_Q0_METADATA_KEY]["valid_occurrence_count"] == 4
        for item in result.proposals
    )


def test_prior_unmeasured_submission_remains_eligible_next_round() -> None:
    payloads = _payloads()
    attempts = {
        profile_id: [[payloads[index]]]
        for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
    }
    client = FakeHarnessClient(attempts)
    expander = _expander(client, candidates_per_turn=1)

    first = expander.expand(ExpansionRequest(round_idx=1, reservoir_size=4))
    second = expander.expand(ExpansionRequest(round_idx=2, reservoir_size=4))

    assert [item.payload for item in second.proposals] == [
        item.payload for item in first.proposals
    ]
    assert len(client.batches) == 2
    assert not client.rejections


def test_direct_harness_uses_one_session_without_q0() -> None:
    payloads = _payloads()[:3]
    client = FakeHarnessClient({DIRECT_HARNESS_PROFILE_ID: [payloads]})
    profiles = direct_harness_profile(3, resource_root=Path("profiles"))
    result = NucleoBenchHarnessExpander(
        client,
        NucleoBenchCandidateDomain(MOCK_CONTEXT),
        profiles=profiles,
        campaign_id="test-campaign",
        first_active_round=1,
        attach_empirical_q0=False,
    ).expand(ExpansionRequest(round_idx=1, reservoir_size=3))

    assert len(client.batches[0]) == 1
    assert [item.payload for item in result.proposals] == payloads
    assert all(
        NUCLEOBENCH_Q0_METADATA_KEY not in item.metadata for item in result.proposals
    )
    assert result.selection_mode == "reservoir_order"


def test_harness_resources_task_contract_and_factory_are_task_local(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / "sequence_context.json"
    write_harness_sequence_context(MOCK_CONTEXT, context_path)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    profiles = harness_profiles(2, resource_root=Path("profiles"))
    extensions = harness_tool_extensions(resource_root=Path("tools"))

    assert set(context) == {"schema_version", "case", "paired_start"}
    assert set(context["case"]) == {
        "case_id",
        "model_family",
        "model_name",
        "target",
        "sequence_length",
        "editable_position_count",
    }
    assert set(context["paired_start"]) == {
        "start_set_digest",
        "start_index",
        "start_sequence",
        "editable_positions",
    }
    assert tuple(profile.profile_id for profile in profiles) == HARNESS_PROFILE_IDS
    assert all(not profile.skill_dirs for profile in profiles)
    assert all(len(profile.agents_sha256) == 64 for profile in profiles)
    assert extensions[0].tool_names == HARNESS_TOOL_NAMES
    assert len(extensions[0].sha256) == 64

    ldm_spec = build_task_spec(
        MOCK_CONTEXT.case,
        search_method="ldm_harness",
        evaluations_per_round=2,
    )
    direct_spec = build_task_spec(
        MOCK_CONTEXT.case,
        search_method="harness",
        evaluations_per_round=2,
    )
    assert ldm_spec.proposal_search.name == "persistent_parallel_research_sessions"
    assert ldm_spec.proposal_search.parameters == {
        "profile_count": 4,
        "candidates_per_session": 2,
        "skills_loaded": False,
    }
    assert ldm_spec.metadata["model_requests_per_round"] is None
    assert ldm_spec.metadata["model_session_turns_per_round"] == 4
    assert direct_spec.proposal_search.name == "persistent_direct_research_session"
    assert direct_spec.acquisition.name == "direct_harness_reservoir_order"
    assert direct_spec.surrogate.kind == "none"

    client = FakeHarnessClient(
        {
            profile.profile_id: [[_payloads()[index]]]
            for index, profile in enumerate(profiles)
        }
    )
    expander = build_proposal_expander(
        "ldm_harness",
        MOCK_CONTEXT,
        seed=0,
        evaluations_per_round=2,
        harness_client=client,
        harness_session_profiles=profiles,
        campaign_id="test-campaign",
        first_active_round=1,
    )
    assert isinstance(expander, NucleoBenchHarnessExpander)


def test_harness_dry_run_exposes_reproducible_settings_without_secret(
    tmp_path: Path,
    capsys,
) -> None:
    key_path = tmp_path / "provider.key"
    key_path.write_text("test-secret", encoding="utf-8")
    args = parse_args(
        [
            "--search-method",
            "ldm_harness",
            "--evaluations-per-round",
            "3",
            "--harness-thinking",
            "max",
            "--harness-tool-budget",
            "web_search=3",
            "--no-harness-context7",
        ]
    )
    assert args.harness_tool_budget == ["web_search=3"]

    assert (
        main(
            [
                "--dry-run",
                "--case-id",
                "malinois_k562",
                "--search-method",
                "ldm_harness",
                "--evaluations-per-round",
                "3",
                "--llm-url",
                "https://provider.example/v1",
                "--llm-model-name",
                "research-model",
                "--api-key-file",
                str(key_path),
                "--harness-thinking",
                "max",
                "--harness-tool-budget",
                "web_search=3",
                "--no-harness-context7",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["proposal_provider"] == {
        "required": True,
        "configured": True,
        "wire_api": "responses",
        "reasoning": "max",
    }
    assert payload["harness"]["profile_ids"] == list(HARNESS_PROFILE_IDS)
    assert payload["harness"]["candidates_per_session"] == 3
    assert payload["harness"]["tool_call_budgets"] == {"web_search": 3}
    assert payload["harness"]["context7_enabled"] is False
    assert "test-secret" not in output


def test_local_harness_uses_host_identity_and_kvm_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(workflow.os, "getuid", lambda: 1001, raising=False)
    monkeypatch.setattr(workflow.os, "getgid", lambda: 1002, raising=False)

    args = parse_args(["--search-method", "harness"])
    assert args.harness_container_user == "1001:1002"

    kvm_path = tmp_path / "kvm"
    kvm_path.touch()
    assert workflow._local_kvm_group_args(None, kvm_path) == (
        "--group-add",
        str(kvm_path.stat().st_gid),
    )

    remote = parse_args(
        [
            "--search-method",
            "harness",
            "--harness-docker-host",
            "tcp://docker.example:2375",
        ]
    )
    assert remote.harness_container_user is None
    assert workflow._local_kvm_group_args(remote.harness_docker_host, kvm_path) == ()


def _expander(
    client: FakeHarnessClient,
    *,
    candidates_per_turn: int,
    account=None,
) -> NucleoBenchHarnessExpander:
    return NucleoBenchHarnessExpander(
        client,
        NucleoBenchCandidateDomain(MOCK_CONTEXT),
        profiles=harness_profiles(candidates_per_turn, resource_root=Path("profiles")),
        campaign_id="test-campaign",
        first_active_round=1,
        attach_empirical_q0=True,
        account=account,
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
        else Candidate(
            candidate_id="start",
            payload=payload,
            canonical_key="start-key",
        )
    )
    assert isinstance(candidate, Candidate)
    return Observation(
        candidate,
        EvaluationResult(candidate.candidate_id, "succeeded", metrics={"utility": 1.0}),
        round_idx=round_idx,
    )
