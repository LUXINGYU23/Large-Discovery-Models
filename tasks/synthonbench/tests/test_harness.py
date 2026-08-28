"""High-risk adapter checks for persistent SynthonBench harness sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from synthonbench.space import Synthon, SynthonSpace

from ldm_tts.contracts import Candidate, EvaluationResult, Observation
from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import HarnessTurnResult
from tasks.synthonbench.core.candidate import SynthonCandidateDomain
from tasks.synthonbench.core.catalog import SynthonProposalCatalog
from tasks.synthonbench.core.constants import Q0_METADATA_KEY
from tasks.synthonbench.core.harness import SynthonHarnessExpander, harness_profiles
from tasks.synthonbench.core.workflow import describe_ldm_task, main, parse_args


class FakeHarnessClient:
    def __init__(self, *, corrupt: bool = False) -> None:
        self.corrupt = corrupt
        self.batches = []

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def run_turn(self, turns):
        self.batches.append(turns)
        results = []
        for turn in turns:
            payload = json.loads(turn.message.split("\n\n", 1)[1])
            candidates = []
            for item_index, item in enumerate(payload["assigned_items"]):
                candidate = {
                    "item_index": item_index,
                    "option_indices": [0 for _slot in item["slot_options"]],
                }
                candidates.append(candidate)
            if self.corrupt and not results:
                candidates[0]["option_indices"][0] = 999
            results.append(HarnessTurnResult(
                profile_id=turn.profile_id,
                session_id=f"session-{turn.profile_id}",
                turn_id=turn.turn_id,
                input_digest=turn.input_digest,
                submission_id=f"submission-{turn.profile_id}",
                candidates=tuple(candidates),
                usage={"providerCalls": 2, "webCalls": 1, "context7Calls": 0, "artifactBytes": 50},
                artifacts={"turn": f"turns/{turn.turn_id}"},
            ))
        return tuple(results)


class ConsensusClient:
    def run_turn(self, turns):
        return tuple(
            HarnessTurnResult(
                profile_id=turn.profile_id,
                session_id=f"session-{turn.profile_id}",
                turn_id=turn.turn_id,
                input_digest=turn.input_digest,
                submission_id=f"submission-{turn.profile_id}",
                candidates=({"item_index": 0, "option_indices": [0, 0]},),
                usage={},
                artifacts={},
            )
            for turn in turns
        )


def test_harness_expander_validates_four_minibatches_and_reuses_global_q0() -> None:
    client = FakeHarnessClient()
    expander = _expander(client)

    result = expander.expand(ExpansionRequest(round_idx=0, reservoir_size=4))

    assert len(result.proposals) == 4
    assert len(client.batches[0]) == 4
    assert {item.metadata["profile_id"] for item in result.proposals} == {
        "target_sar",
        "reaction_feasibility",
        "scaffold_exploration",
        "property_risk",
    }
    assert all(Q0_METADATA_KEY in item.metadata for item in result.proposals)
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
    assert all(
        message["new_measured_observations"][0]["synthon_ids"] == [2, 12]
        for message in messages
    )
    assert all(
        "Treat this as a bounded selection turn, not open-ended research."
        in message["constraints"]
        for message in messages
    )
    assert all(
        any("one batch of optional tool calls" in item for item in message["constraints"])
        for message in messages
    )
    assert all(
        message["submission_contract"]["candidate_schema"] == {
            "item_index": "copy the assigned zero-based item_index",
            "option_indices": "one zero-based option_index from every slot, in slot order",
        }
        for message in messages
    )
    assert all(
        set(option) == {"option_index", "smiles"}
        for message in messages for item in message["assigned_items"]
        for slot in item["slot_options"] for option in slot["options"]
    )


def test_harness_rejects_a_candidate_outside_its_assigned_slate() -> None:
    with pytest.raises(ValueError, match=r"option_indices\[0\]"):
        _expander(FakeHarnessClient(corrupt=True)).expand(
            ExpansionRequest(round_idx=0, reservoir_size=4)
        )


def test_cross_profile_consensus_increases_shared_occurrence_probability() -> None:
    space = SynthonSpace([
        Synthon(1, 1, "r1", "CC"),
        Synthon(11, 2, "r1", "N"),
    ])
    expander = SynthonHarnessExpander(
        ConsensusClient(),
        SynthonCandidateDomain(space, ("r1",), "kif11"),
        SynthonProposalCatalog(
            space,
            allowed_reactions=("r1",),
            slate_size=1,
            seed=3,
            unique_anchors=False,
        ),
        target="kif11",
        profiles=harness_profiles(1, resource_root=Path("profiles")),
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
    client = FakeHarnessClient()
    monkeypatch.setattr(
        "tasks.synthonbench.core.workflow._harness_client",
        lambda *_args: client,
    )

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
    budget = json.loads((Path(payload["run_dir"]) / "budget.json").read_text(encoding="utf-8"))
    assert budget["counters"]["proposal_attempts"] == 4
    assert budget["counters"]["harness_turns"] == 4
    assert budget["counters"]["llm_requests"] == 8
    assert budget["counters"]["benchmark_jobs"] == 1


def _expander(client) -> SynthonHarnessExpander:
    space = _space()
    domain = SynthonCandidateDomain(space, ("r1",), "kif11")
    catalog = SynthonProposalCatalog(
        space,
        allowed_reactions=("r1",),
        slate_size=3,
        seed=7,
        unique_anchors=True,
        proposals_per_round=4,
    )
    return SynthonHarnessExpander(
        client,
        domain,
        catalog,
        target="kif11",
        profiles=harness_profiles(1, resource_root=Path("profiles")),
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
