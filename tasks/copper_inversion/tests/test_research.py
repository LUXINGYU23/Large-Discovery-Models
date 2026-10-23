"""Task adapter on the real shared research tree, with deterministic native-turn fixtures."""

import json
from dataclasses import replace

import pytest

from ldm_tts.harness import HarnessError, HarnessSubmissionRequest
from tasks.copper_inversion.core.mock import MockClient
from tasks.copper_inversion.core.workflow import main, parse_args


def load(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("strategy", ["coverage", "latest", "independent"])
def test_four_branch_workflow_and_completed_resume(tmp_path, strategy):
    root = tmp_path / "run"
    args = ["--mock", "--search-method", "ldm_harness", "--iterations", "3", "--out-dir", str(root)]
    if strategy != "coverage":
        path = tmp_path / "strategy.json"
        path.write_text(json.dumps({"name": "independent"} if strategy == "independent" else
                                  {"name": "main_forks", "fork_policy": {"name": "latest"}}))
        args += ["--harness-research-strategy-config", str(path)]
    assert main(args) == 0
    counters = load(root / "budget.json")["counters"]
    assert counters["external_evaluations"] == counters["successful_evaluations"] == 4
    assert counters["harness_turns"] == (12 if strategy == "independent" else 15)
    assert counters["main_research_turns"] == (0 if strategy == "independent" else 3)
    assert len(list((root / "candidates").glob("*/solver.py"))) == 13
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    selections = [e["payload"] for e in events if e["event_type"] == "candidates_selected"]
    assert [len(s["predictions"]) for s in selections] == [0, 4, 4, 4]
    if strategy != "independent":
        panels = [load(p) for p in sorted((root / "harness/branches").glob("*/panel.json"))]
        assert len(panels) == 3
        assert [p["history_count"] for p in panels] == [1, 2, 3]
        for panel in panels:
            assert len(panel["sessions"]) == len(panel["proposals"]) == 4
            assert len({p["payload"]["sha256"] for p in panel["proposals"]}) == 4
        if strategy == "coverage":
            assert any(s["history_from_seq"] < 2 for s in panels[-1]["sessions"])
        tree = load(root / "harness/branches/round_0003/tree.json")
        assert all(o["incumbent"] < 0 for o in tree["outcomes"])
        assert (root / "harness_provenance.json").exists()
    assert main([*args, "--resume"]) == 0
    assert load(root / "budget.json")["counters"] == counters


def test_partial_barrier_replays_accepted_peers(tmp_path, monkeypatch):
    root = tmp_path / "run"
    args = ["--mock", "--search-method", "ldm_harness", "--iterations", "3", "--out-dir", str(root)]
    original = MockClient.run_turn
    interrupted = False
    generated = []
    source = MockClient._source

    def record_source(self, turn):
        generated.append((str(self.root), turn.profile_id, turn.turn_id))
        return source(self, turn)

    def run(self, turns, **kwargs):
        nonlocal interrupted
        if len(turns) == 4 and not interrupted:
            interrupted = True
            accepted = original(self, turns[:2], **kwargs)
            raise HarnessError("injected partial barrier", turn_usage={r.profile_id: r.usage for r in accepted})
        return original(self, turns, **kwargs)

    monkeypatch.setattr(MockClient, "run_turn", run)
    monkeypatch.setattr(MockClient, "_source", record_source)
    with pytest.raises(HarnessError, match="partial barrier"):
        main(args)
    assert load(root / "budget.json")["counters"]["external_evaluations"] == 1
    assert not list((root / "harness/branches").glob("*/panel.json"))
    plan = (root / "harness/branches/round_0001/fork-plan.json").read_bytes()
    assert main([*args, "--resume"]) == 0
    assert len(generated) == len(set(generated)) == 12
    assert (root / "harness/branches/round_0001/fork-plan.json").read_bytes() == plan
    counters = load(root / "budget.json")["counters"]
    assert counters["harness_turns"] == 15 and counters["llm_requests"] == 15


def test_cross_branch_duplicate_is_rejected_before_batch_publication(tmp_path, monkeypatch):
    original = MockClient.run_turn
    rejected = []

    def run(self, turns, submission_validator, **kwargs):
        if len(turns) != 4:
            return original(self, turns, submission_validator=submission_validator, **kwargs)
        first = original(self, turns[:1], submission_validator=submission_validator, **kwargs)
        proposal = first[0]
        result = submission_validator(HarnessSubmissionRequest(
            turns[1].profile_id, turns[1].turn_id, 1, proposal.submission, proposal.submitted_artifacts))
        assert result.decision == "retry" and "duplicates" in result.errors[0].message
        rejected.append(result)
        return first + original(self, turns[1:], submission_validator=submission_validator, **kwargs)

    monkeypatch.setattr(MockClient, "run_turn", run)
    assert main(["--mock", "--search-method", "ldm_harness", "--iterations", "1",
                 "--out-dir", str(tmp_path / "run")]) == 0
    assert len(rejected) == 1


def test_single_agent_rejects_forks_and_measurement_is_disabled(tmp_path, monkeypatch):
    path = tmp_path / "strategy.json"
    path.write_text('{"name":"main_forks"}')
    with pytest.raises(ValueError, match="parallel"):
        parse_args(["--harness-research-strategy-config", str(path)])
    from ldm_tts.harness import research_strategy
    from types import SimpleNamespace
    monkeypatch.setattr(research_strategy, "load_research_strategy", lambda _: SimpleNamespace(
        requires_main=False, measurement_policy=object()))
    with pytest.raises(ValueError, match="measurement coverage"):
        parse_args(["--search-method", "ldm_harness"])
