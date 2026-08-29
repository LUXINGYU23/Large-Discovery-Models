"""High-risk adapter checks for persistent Iron Mind harness sessions."""

from __future__ import annotations

import json

from ldm_tts.contracts import Candidate, EvaluationResult, Observation
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import HarnessSubmissionRequest, HarnessTurnResult
from tasks.iron_mind.core.candidate import (
    IRON_MIND_Q0_METADATA_KEY,
    IronMindCandidateDomain,
    prepare_candidate_payload,
)
from tasks.iron_mind.core.harness import (
    HARNESS_PROFILE_IDS,
    IronMindHarnessExpander,
    _validate_submission,
    harness_profiles,
    write_harness_space_catalog,
)
from tasks.iron_mind.core.workflow import _load_mock_table, _schema_for, describe_ldm_task, parse_args


class FakeHarnessClient:
    def __init__(self, candidates_by_profile):
        self.candidates_by_profile = candidates_by_profile
        self.batches = []

    def run_turn(self, turns, *, submission_validator):
        self.batches.append(turns)
        results = []
        for turn in turns:
            candidates = (self.candidates_by_profile[turn.profile_id],)
            validation = submission_validator(
                HarnessSubmissionRequest(turn.profile_id, turn.turn_id, 1, candidates)
            )
            assert validation.accepted
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
                    candidates=candidates,
                    usage={
                        "providerCalls": 2,
                        "webCalls": 1,
                        "context7Calls": 0,
                        "artifactBytes": 50,
                    },
                    artifacts={"turn": f"turns/{turn.turn_id}"},
                )
            )
        return tuple(results)


def test_harness_preserves_cross_session_occurrences_for_global_q0() -> None:
    domain, payloads = _domain_and_payloads()
    candidates = {
        HARNESS_PROFILE_IDS[0]: payloads[0],
        HARNESS_PROFILE_IDS[1]: payloads[0],
        HARNESS_PROFILE_IDS[2]: payloads[1],
        HARNESS_PROFILE_IDS[3]: payloads[2],
    }
    client = FakeHarnessClient(candidates)
    counts = []
    expander = IronMindHarnessExpander(
        client,
        domain,
        profiles=harness_profiles(1),
        campaign_id="campaign-test",
        first_active_round=0,
        account=counts.append,
    )

    result = expander.expand(ExpansionRequest(round_idx=0, reservoir_size=4))

    assert len(result.proposals) == 4
    assert [
        item.metadata[IRON_MIND_Q0_METADATA_KEY]["probability"]
        for item in result.proposals
    ] == [0.5, 0.5, 0.25, 0.25]
    assert result.metadata["sampling_mode"] == "persistent_parallel_research_sessions"
    assert counts[0] == {"proposal_attempts": 4, "harness_turns": 4}
    assert counts[1]["llm_requests"] == 8


def test_harness_turn_sends_history_delta_and_complete_exclusion_snapshot() -> None:
    domain, payloads = _domain_and_payloads()
    client = FakeHarnessClient(
        {profile_id: payloads[index + 2] for index, profile_id in enumerate(HARNESS_PROFILE_IDS)}
    )
    expander = IronMindHarnessExpander(
        client,
        domain,
        profiles=harness_profiles(1),
        campaign_id="campaign-test",
        first_active_round=1,
    )
    old = _observation(domain, payloads[0], round_idx=0, score=1.0)
    latest = _observation(domain, payloads[1], round_idx=1, score=2.0)

    expander.expand(
        ExpansionRequest(round_idx=2, reservoir_size=4, observations=(old, latest))
    )

    messages = [json.loads(turn.message.split("\n\n", 1)[1]) for turn in client.batches[0]]
    assert all(message["message_type"] == "history_delta" for message in messages)
    assert all(len(message["new_measured_observations"]) == 1 for message in messages)
    assert all(len(message["evaluated_candidates"]) == 2 for message in messages)
    assert all(
        message["novelty_contract"]["required_not_evaluated_candidate_count"] == 1
        for message in messages
    )
    assert all(
        message["novelty_contract"]["prior_unmeasured_submissions_may_be_reproposed"]
        for message in messages
    )
    assert all(message["reaction_space_tools"] == [
        "describe_reaction_space",
        "search_reaction_conditions",
        "validate_reaction_candidate",
    ] for message in messages)
    assert all(turn.history_from_seq == 1 and turn.history_to_seq == 2 for turn in client.batches[0])


def test_submission_validator_returns_actionable_rejection_reasons() -> None:
    domain, payloads = _domain_and_payloads()
    invalid = json.loads(json.dumps(payloads[1]))
    first_factor = domain.schema.factors[0]
    invalid["conditions"][first_factor.name] = "not-a-legal-option"
    evaluated = prepare_candidate_payload(
        payloads[0], domain.schema, domain.table
    ).canonical_key

    validation = _validate_submission(
        HarnessSubmissionRequest(
            HARNESS_PROFILE_IDS[0],
            "turn-1",
            1,
            (payloads[0], invalid, payloads[1], payloads[1]),
        ),
        domain,
        {evaluated},
    )

    assert [item.code for item in validation.rejections] == [
        "historical_duplicate",
        "invalid_candidate",
        "same_session_duplicate",
    ]
    assert "already evaluated" in validation.rejections[0].message
    assert "unknown_option" in validation.rejections[1].message
    assert "duplicates index 2" in validation.rejections[2].message


def test_catalog_and_task_spec_expose_space_without_oracle_scores(tmp_path) -> None:
    domain, _ = _domain_and_payloads()
    path = tmp_path / "reaction_space.json"
    write_harness_space_catalog(domain, path)
    catalog = json.loads(path.read_text(encoding="utf-8"))
    args = parse_args(
        [
            "--mock",
            "--proposal-backend",
            "harness",
            "--proposal-mode",
            "none",
            "--initialization-mode",
            "shared_random",
        ]
    )
    spec = describe_ldm_task(args)

    assert catalog["condition_count"] == len(catalog["candidates"])
    assert "reaction_score" not in json.dumps(catalog)
    assert spec.proposal_search.name == "persistent_parallel_research_sessions"
    assert spec.proposal_search.parameters["profile_count"] == 4
    assert spec.metadata["proposal_transport"] == "openai_responses_persistent_sessions"


def _domain_and_payloads():
    table = _load_mock_table(_schema_for("buchwald_hartwig"), candidate_count=16)
    domain = IronMindCandidateDomain(table.schema, table)
    payloads = [
        {"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)}
        for row in table.rows
    ]
    return domain, payloads


def _observation(domain, payload, *, round_idx: int, score: float) -> Observation:
    prepared = prepare_candidate_payload(payload, domain.schema, domain.table)
    candidate = Candidate(
        f"candidate-{round_idx}", prepared.payload, prepared.canonical_key, "test"
    )
    return Observation(
        candidate,
        EvaluationResult(candidate.candidate_id, "succeeded", {"reaction_score": score}),
        round_idx=round_idx,
    )
