"""High-risk adapter checks for persistent SynthonBench harness sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from synthonbench.space import Synthon, SynthonSpace

from ldm_tts.contracts import Candidate, EvaluationResult, Observation
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import HarnessSubmissionRequest, HarnessTurnResult
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.constants import Q0_METADATA_KEY
from tasks.synthonbench.core.harness import (
    HARNESS_CANDIDATE_SCHEMA,
    SynthonHarnessExpander,
    _validate_submission,
    harness_profiles,
)
from tasks.synthonbench.core.space_order import (
    ordered_positions,
    ordered_reactions,
    ordered_synthon_ids,
)
from tasks.synthonbench.core.workflow import describe_ldm_task, main, parse_args


class FakeHarnessClient:
    def __init__(
        self,
        *,
        candidate: dict[str, object] | None = None,
        initial_by_profile: dict[str, dict[str, object]] | None = None,
        replacement_by_profile: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.candidate = candidate or {"reaction_id": "r1", "synthon_ids": [3, 13]}
        self.initial_by_profile = initial_by_profile or {}
        self.replacement_by_profile = replacement_by_profile or {}
        self.batches = []
        self.rejections = {}

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def run_turn(self, turns, *, submission_validator):
        self.batches.append(turns)
        results = []
        for turn in turns:
            payload = json.loads(turn.message.split("\n\n", 1)[1])
            candidate = self.initial_by_profile.get(turn.profile_id, self.candidate)
            candidates = [
                {**candidate}
                for _ in range(payload["submission_contract"]["candidate_count"])
            ]
            validation = submission_validator(HarnessSubmissionRequest(
                turn.profile_id,
                turn.turn_id,
                1,
                tuple(candidates),
            ))
            if not validation.accepted:
                replacement = self.replacement_by_profile.get(turn.profile_id)
                if replacement is None:
                    raise AssertionError(validation.rejections)
                self.rejections[turn.profile_id] = validation.rejections
                candidates = [{**replacement} for _ in candidates]
                assert submission_validator(HarnessSubmissionRequest(
                    turn.profile_id,
                    turn.turn_id,
                    2,
                    tuple(candidates),
                )).accepted
            results.append(HarnessTurnResult(
                profile_id=turn.profile_id,
                session_id=f"session-{turn.profile_id}",
                turn_id=turn.turn_id,
                round_index=turn.round_index,
                history_from_seq=turn.history_from_seq,
                history_to_seq=turn.history_to_seq,
                history_digest=turn.history_digest,
                input_digest=turn.input_digest,
                submission_id=f"submission-{turn.profile_id}",
                candidates=tuple(candidates),
                usage={"providerCalls": 2, "webCalls": 1, "context7Calls": 0, "artifactBytes": 50},
                artifacts={"turn": f"turns/{turn.turn_id}"},
            ))
        return tuple(results)


def test_harness_expander_validates_four_minibatches_and_reuses_global_q0() -> None:
    client = FakeHarnessClient()
    expander = _expander(client)

    result = expander.expand(ExpansionRequest(round_idx=0, reservoir_size=4))

    assert len(result.proposals) == 4
    assert len(client.batches[0]) == 4
    assert {item.metadata["harness_lineage"]["profile_id"] for item in result.proposals} == {
        "target_sar",
        "reaction_feasibility",
        "scaffold_exploration",
        "property_risk",
    }
    assert all(Q0_METADATA_KEY in item.metadata for item in result.proposals)
    assert [item.metadata["harness_lineage"]["item_index"] for item in result.proposals] == [0, 0, 0, 0]
    assert len(result.metadata["candidate_lineage"]) == 4
    assert result.metadata["sampling_mode"] == "persistent_parallel_research_sessions"


def test_harness_second_turn_sends_only_the_previous_round_measurements() -> None:
    client = FakeHarnessClient()
    expander = _expander(client)
    old = _observation("old", [1, 11], round_idx=0)
    latest = _observation("latest", [2, 12], round_idx=1)

    expander.expand(ExpansionRequest(
        round_idx=2,
        reservoir_size=4,
        observations=(old, latest),
    ))

    messages = [json.loads(turn.message.split("\n\n", 1)[1]) for turn in client.batches[0]]
    assert all(message["message_type"] == "history_delta" for message in messages)
    assert all(len(message["new_measured_observations"]) == 1 for message in messages)
    assert all(message["evaluated_candidates"] == [
        {"reaction_id": "r1", "synthon_ids": [1, 11]},
        {"reaction_id": "r1", "synthon_ids": [2, 12]},
    ] for message in messages)
    assert all(message["novelty_contract"] == {
        "historical_candidates_are_forbidden": True,
        "required_unseen_candidate_count": 1,
        "same_round_cross_session_agreement_is_allowed": True,
        "same_session_duplicates_are_forbidden": True,
        "validate_before_submission": True,
    } for message in messages)
    assert all(turn.history_from_seq == 1 and turn.history_to_seq == 2 for turn in client.batches[0])
    assert len({turn.history_digest for turn in client.batches[0]}) == 1
    assert all("latest" in turn.forbidden_query_terms for turn in client.batches[0])
    assert all(
        message["new_measured_observations"][0]["synthon_ids"] == [2, 12]
        for message in messages
    )
    assert all(
        message["submission_contract"]["candidate_schema"] == HARNESS_CANDIDATE_SCHEMA
        for message in messages
    )
    assert all(
        message["synthon_space_tools"] == [
            "list_synthon_reactions",
            "search_synthon_space",
            "validate_synthon_candidate",
        ]
        for message in messages
    )


def test_submission_validator_returns_actionable_reasons() -> None:
    domain = SynthonCandidateDomain(_space(), ("r1",), "kif11")
    validation = _validate_submission(
        HarnessSubmissionRequest(
            "target_sar",
            "turn-1",
            1,
            (
                {"reaction_id": "r1", "synthon_ids": [1, 11]},
                {"reaction_id": "r1", "synthon_ids": [999, 11]},
                {"reaction_id": "r1", "synthon_ids": [2, 12]},
                {"reaction_id": "r1", "synthon_ids": [2, 12]},
            ),
        ),
        domain,
        {"r1|1_11"},
    )

    assert [item.code for item in validation.rejections] == [
        "historical_duplicate", "invalid_candidate", "same_session_duplicate",
    ]
    assert "already evaluated" in validation.rejections[0].message
    assert "synthon ID 999" in validation.rejections[1].message
    assert "duplicates index 2" in validation.rejections[2].message


def test_harness_rejects_history_and_refills_before_q0() -> None:
    initial = {
        "target_sar": {"reaction_id": "r1", "synthon_ids": [1, 11]},
        "reaction_feasibility": {"reaction_id": "r1", "synthon_ids": [2, 11]},
        "scaffold_exploration": {"reaction_id": "r1", "synthon_ids": [3, 11]},
        "property_risk": {"reaction_id": "r1", "synthon_ids": [4, 11]},
    }
    client = FakeHarnessClient(initial_by_profile=initial, replacement_by_profile={
        **initial,
        "target_sar": {"reaction_id": "r1", "synthon_ids": [5, 11]},
    })
    measured = _observation("measured", [1, 11], round_idx=0)

    result = _expander(client).expand(ExpansionRequest(
        round_idx=1,
        reservoir_size=4,
        observations=(measured,),
    ))

    assert len(result.proposals) == 4
    assert [item.metadata["harness_lineage"]["profile_id"] for item in result.proposals] == [
        "target_sar", "reaction_feasibility", "scaffold_exploration", "property_risk",
    ]
    rejection = client.rejections["target_sar"][0]
    assert rejection.code == "historical_duplicate"
    assert "already evaluated" in rejection.message
    assert all(
        proposal.metadata[Q0_METADATA_KEY]["valid_occurrence_count"] == 4
        for proposal in result.proposals
    )


def test_cross_profile_consensus_increases_shared_occurrence_probability() -> None:
    space = SynthonSpace([
        Synthon(1, 1, "r1", "CC"),
        Synthon(11, 2, "r1", "N"),
    ])
    expander = SynthonHarnessExpander(
        FakeHarnessClient(candidate={"reaction_id": "r1", "synthon_ids": [1, 11]}),
        SynthonCandidateDomain(space, ("r1",), "kif11"),
        target="kif11",
        profiles=harness_profiles(1, resource_root=Path("profiles")),
        campaign_id="test-campaign",
        first_active_round=0,
    )

    result = expander.expand(ExpansionRequest(round_idx=0, reservoir_size=4))

    assert len(result.proposals) == 4
    for proposal in result.proposals:
        q0 = proposal.metadata[Q0_METADATA_KEY]
        assert q0["occurrence_count"] == 4
        assert q0["valid_occurrence_count"] == 4
        assert q0["probability"] == pytest.approx(1.0)


def test_harness_task_spec_declares_persistent_four_profile_sampling() -> None:
    args = parse_args([
        "--mock",
        "--proposal-backend", "harness",
        "--proposal-mode", "none",
    ])

    spec = describe_ldm_task(args)

    assert spec.proposal_search.name == "persistent_parallel_research_sessions"
    assert spec.proposal_search.parameters["profile_count"] == 4
    assert spec.proposal_search.parameters["candidates_per_session"] == 16
    assert spec.proposal_search.parameters["skills_loaded"] is False
    assert spec.metadata["model_requests_per_round"] is None
    assert spec.metadata["model_session_turns_per_round"] == 4


def test_mock_campaign_routes_harness_candidates_through_the_existing_engine(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def fake_client(_args, _runtime, _provider, benchmark):
        reaction_id = ordered_reactions(benchmark.task.allowed_reactions)[0]
        candidate = {
            "reaction_id": reaction_id,
            "synthon_ids": [
                ordered_synthon_ids(benchmark.task.space, reaction_id, position)[0]
                for position in ordered_positions(benchmark.task.space, reaction_id)
            ],
        }
        return FakeHarnessClient(candidate=candidate)

    monkeypatch.setattr("tasks.synthonbench.core.workflow._harness_client", fake_client)

    assert main([
        "--mock",
        "--proposal-backend", "harness",
        "--proposal-mode", "none",
        "--proposal-samples", "4",
        "--harness-candidates-per-session", "1",
        "--bo-pool-size", "2",
        "--iterations", "1",
        "--llm-url", "http://provider.invalid/v1",
        "--llm-model-name", "fake-model",
        "--api-key", "fake-key",
        "--out-dir", str(tmp_path),
        "--run-name", "harness_mock",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    budget = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))
    assert budget["counters"]["proposal_attempts"] == 4
    assert budget["counters"]["harness_turns"] == 4
    assert budget["counters"]["llm_requests"] == 8
    assert budget["counters"]["benchmark_jobs"] == 1
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    lineage = checkpoint["state"]["observations"][0]["candidate"]["metadata"]["harness_lineage"]
    assert set(lineage) == {
        "campaign_id", "round_index", "profile_id", "session_id",
        "turn_id", "submission_id", "item_index",
    }


def _expander(client) -> SynthonHarnessExpander:
    space = _space()
    domain = SynthonCandidateDomain(space, ("r1",), "kif11")
    return SynthonHarnessExpander(
        client,
        domain,
        target="kif11",
        profiles=harness_profiles(1, resource_root=Path("profiles")),
        campaign_id="test-campaign",
        first_active_round=0,
    )


def _space() -> SynthonSpace:
    return SynthonSpace([
        Synthon(1, 1, "r1", "CC"),
        Synthon(2, 1, "r1", "CO"),
        Synthon(3, 1, "r1", "CN"),
        Synthon(4, 1, "r1", "CF"),
        Synthon(5, 1, "r1", "CCl"),
        Synthon(6, 1, "r1", "CBr"),
        Synthon(11, 2, "r1", "N"),
        Synthon(12, 2, "r1", "O"),
        Synthon(13, 2, "r1", "S"),
    ])


def _observation(name: str, ids: list[int], *, round_idx: int) -> Observation:
    candidate = Candidate(
        candidate_id=name,
        payload={"reaction_id": "r1", "synthon_ids": ids},
        canonical_key=f"r1|{'_'.join(str(item) for item in ids)}",
    )
    return Observation(
        candidate,
        EvaluationResult(name, "succeeded", metrics={"synthon_utility": -1.0}),
        round_idx=round_idx,
    )
