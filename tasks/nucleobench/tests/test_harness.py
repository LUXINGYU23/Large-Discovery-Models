"""High-risk checks for NucleoBench persistent research sessions."""

from __future__ import annotations

import json

from ldm_tts.contracts import Candidate, EvaluationResult, Observation, RawProposal
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import (
    HarnessSubmissionRequest,
    HarnessTurnResult,
    canonical_sha256,
)
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
    harness_guest_runtime,
    harness_profiles,
    harness_submission_contract,
    harness_tool_extensions,
    write_harness_sequence_context,
)
from tasks.nucleobench.core.mock import MOCK_CONTEXT
from tasks.nucleobench.core.optimization_policy import policy_harness_profile
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
            accepted = None
            attempt_count = 0
            for attempt_count, candidates in enumerate(
                self.attempts_by_profile[turn.profile_id], start=1
            ):
                submission = {"candidates": candidates}
                validation = submission_validator(
                    HarnessSubmissionRequest(
                        turn.profile_id,
                        turn.turn_id,
                        attempt_count,
                        submission,
                    )
                )
                if validation.decision == "accept":
                    accepted = submission
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
                    submission_digest=canonical_sha256(
                        {"artifacts": [], "submission": accepted}
                    ),
                    submission=accepted,
                    submitted_artifacts=(),
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


def test_parallel_sessions_preserve_within_and_cross_profile_consensus_for_q0() -> None:
    payloads = _payloads()
    attempts = {
        profile_id: [[payloads[0], payloads[index + 1]]]
        for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
    }
    attempts[HARNESS_PROFILE_IDS[0]] = [[payloads[0], payloads[0]]]
    client = FakeHarnessClient(attempts)
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
    assert usage == [
        {"proposal_attempts": 4, "harness_turns": 4},
        {
            "llm_requests": 8,
            "harness_tool_calls": 8,
            "harness_validation_submissions": 4,
            "harness_artifact_bytes": 200,
        },
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
        == [{"mutations": payloads[0]["mutations"], "utility": 1.0}]
        for message in messages
    )
    assert all(
        message["evaluated_candidates"] == [{"mutations": []}, payloads[0]]
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


def test_submission_validation_returns_actionable_mutation_reasons() -> None:
    payloads = _payloads()
    historical = NucleoBenchCandidateDomain(MOCK_CONTEXT).admit(
        RawProposal(payloads[0], "test")
    )
    assert isinstance(historical, Candidate)
    validation = _validate_submission(
        HarnessSubmissionRequest(
            "target_biology",
            "turn-1",
            1,
            {
                "candidates": [
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
                ]
            },
        ),
        MOCK_CONTEXT,
        {historical.canonical_key},
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
        HarnessSubmissionRequest(
            DIRECT_HARNESS_PROFILE_ID,
            "turn-2",
            1,
            {"candidates": [payloads[4], payloads[4]]},
        ),
        MOCK_CONTEXT,
        set(),
        allow_repeated_occurrences=False,
    )
    assert [item.code for item in direct_validation.errors] == [
        "same_session_duplicate"
    ]


def test_history_rejection_is_repaired_before_commit_and_q0() -> None:
    payloads = _payloads()
    attempts = {
        profile_id: [[payloads[index + 1]]]
        for index, profile_id in enumerate(HARNESS_PROFILE_IDS)
    }
    attempts[HARNESS_PROFILE_IDS[0]] = [[payloads[0]], [payloads[5]]]
    client = FakeHarnessClient(attempts)

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


def test_direct_harness_uses_one_session_without_q0() -> None:
    payloads = _payloads()[:3]
    client = FakeHarnessClient({DIRECT_HARNESS_PROFILE_ID: [payloads]})
    result = NucleoBenchHarnessExpander(
        client,
        NucleoBenchCandidateDomain(MOCK_CONTEXT),
        profiles=direct_harness_profile(),
        candidates_per_profile=3,
        campaign_id="test-campaign",
        first_active_round=1,
        attach_empirical_q0=False,
    ).expand(ExpansionRequest(round_idx=1, reservoir_size=3))

    assert len(client.batches[0]) == 1
    assert [item.payload for item in result.proposals] == payloads
    assert all(
        NUCLEOBENCH_Q0_METADATA_KEY not in item.metadata
        for item in result.proposals
    )
    assert result.selection_mode == "reservoir_order"


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
    assert all(not profile.skill_dirs for profile in profiles)
    assert all(len(profile.agents_sha256) == 64 for profile in profiles)
    assert extensions[0].tool_names == HARNESS_TOOL_NAMES
    assert proposal_contract.tool_name == "submit_candidates"
    assert proposal_contract.payload_schema["properties"]["candidates"]["minItems"] == 2
    assert guest.image_ref.startswith("ldm/nucleobench-research:")
    assert policy_profiles[0].profile_id == "policy_architect"
    assert len(policy_profiles[0].skill_dirs) == 1

    compiled_spec = build_task_spec(
        MOCK_CONTEXT.case,
        search_method="ldm_harness_compiled",
        evaluations_per_round=2,
    )
    assert compiled_spec.proposal_search.parameters["profile_count"] == 4
    assert compiled_spec.proposal_search.parameters["policy_profile_count"] == 1
    assert compiled_spec.proposal_search.parameters["policy_skills_loaded"] is True

    expander = build_proposal_expander(
        "ldm_harness",
        MOCK_CONTEXT,
        seed=0,
        evaluations_per_round=1,
        harness_client=FakeHarnessClient(
            {
                profile.profile_id: [[_payloads()[index]]]
                for index, profile in enumerate(profiles)
            }
        ),
        harness_session_profiles=profiles,
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
        else Candidate("start", payload, "start-key")
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
