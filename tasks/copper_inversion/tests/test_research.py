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
    from tasks.copper_inversion.core.harness import Expander
    original_init = Expander.__init__
    interrupted = False
    generated = []
    source = MockClient._source

    def record_source(self, turn):
        generated.append((str(self.root), turn.profile_id, turn.turn_id))
        return source(self, turn)

    def init(self, root, client, checks, account, **kwargs):
        calls = 0
        def wrapped(payload):
            nonlocal calls, interrupted
            calls += 1
            if calls == 3 and not interrupted:
                interrupted = True
                raise HarnessError("injected partial barrier")
            return checks(payload)
        original_init(self, root, client, wrapped, account, **kwargs)

    monkeypatch.setattr(Expander, "__init__", init)
    monkeypatch.setattr(MockClient, "_source", record_source)
    with pytest.raises(HarnessError, match="partial barrier"):
        main(args)
    assert load(root / "budget.json")["counters"]["external_evaluations"] == 1
    assert not list((root / "harness/branches").glob("*/panel.json"))
    assert len(list((root / "harness/branches/round_0001/admissions").glob("*.json"))) == 2
    plan = (root / "harness/branches/round_0001/fork-plan.json").read_bytes()
    assert main([*args, "--resume"]) == 0
    assert len(generated) == len(set(generated)) == 12
    assert (root / "harness/branches/round_0001/fork-plan.json").read_bytes() == plan
    counters = load(root / "budget.json")["counters"]
    assert counters["harness_turns"] == 15 and counters["llm_requests"] == 15


@pytest.mark.parametrize("fault", ["duplicate", "syntax"])
@pytest.mark.parametrize("count", [1, 4])
def test_cross_branch_invalid_source_prevents_all_interface_jobs(tmp_path, monkeypatch, fault, count):
    original = MockClient.run_turn
    rejected, checked = [], []
    from tasks.copper_inversion.core.harness import Expander
    original_init = Expander.__init__
    def init(self, root, client, checks, account, **kwargs):
        def counted(payload):
            checked.append(payload["sha256"])
            return checks(payload)
        original_init(self, root, client, counted, account, **kwargs)
    monkeypatch.setattr(Expander, "__init__", init)

    def run(self, turns, submission_validator=None, **kwargs):
        if len(turns) != 4:
            return original(self, turns, submission_validator=submission_validator, **kwargs)
        batch = kwargs.pop("submission_batch_validator")
        def inspect(requests):
            if len(requests) < 4:
                assert batch(requests) == {}
                assert not checked
                assert not list((self.root / "admissions").glob("*.json"))
                return {}
            first, second, *rest = requests
            duplicate = replace(second, submission=first.submission, artifacts=first.artifacts,
                                submission_digest=None)
            if fault == "syntax":
                import hashlib
                body = b"def solve(:\n"
                (self.root / "invalid-syntax.py").write_bytes(body)
                artifact = replace(second.artifacts[0], snapshot_path="invalid-syntax.py",
                                   sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body))
                duplicate = replace(second, artifacts=(artifact, *second.artifacts[1:]), submission_digest=None)
            decisions = batch((first, duplicate, *rest))
            assert decisions[second.profile_id].decision == "retry"
            assert not checked
            if fault == "duplicate":
                assert "duplicates" in decisions[second.profile_id].errors[0].message
            assert not list((self.root / "admissions").glob("*.json"))
            rejected.append(decisions)
            return batch(requests)
        return original(self, turns, submission_batch_validator=inspect, **kwargs)

    monkeypatch.setattr(MockClient, "run_turn", run)
    assert main(["--mock", "--search-method", "ldm_harness", "--iterations", "1",
                 "--out-dir", str(tmp_path / "run"), "--candidates-per-researcher", str(count)]) == 0
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
