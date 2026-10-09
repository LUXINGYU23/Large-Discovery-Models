"""Fork lineage, feedback, recovery and accounting at the experiment boundary."""

import json
from types import SimpleNamespace

import pytest

from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import HarnessError
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.forked_harness import ForkedResearchExpander, snapshot_fork, write_fork_provenance
from tasks.nucleobench.core.harness import harness_profiles
from tasks.nucleobench.core.mock import MOCK_CONTEXT
from tasks.nucleobench.core.workflow import _campaign_budget, parse_args
from tasks.nucleobench.tests.test_harness import FakeHarnessClient, _observation


def test_snapshot_is_frozen_and_copies_files(tmp_path):
    workspace = tmp_path / "main/workspace"
    workspace.mkdir(parents=True)
    (workspace / "research.md").write_text("evidence")
    (workspace / ".ldm-resources/task-data").mkdir(parents=True)
    (workspace / ".ldm-resources/task-data/history.json").write_text("exported tool data")
    session = tmp_path / "session.jsonl"
    session.write_text("native session")
    target = tmp_path / "fork-source"
    source = snapshot_fork(session, workspace, target, 1)
    (workspace / "research.md").write_text("next round")
    assert snapshot_fork(session, workspace, target, 1) == source
    assert (target / "workspace/research.md").read_text() == "evidence"
    assert (target / "workspace/.ldm-resources/task-data/history.json").read_text() == "exported tool data"
    (target / "workspace/research.md").write_text("tampered")
    with pytest.raises(ValueError, match="identity changed"):
        snapshot_fork(session, workspace, target, 1)


def test_experimental_budget_and_unsupported_modes():
    args = parse_args(["--search-method", "ldm_harness_compiled", "--harness-main-research", "--iterations", "3"])
    budget = _campaign_budget(args, None)
    assert budget["harness_turns"] == 10
    assert budget["proposal_attempts"] == 8
    assert budget["main_research_turns"] == 2
    with pytest.raises(SystemExit):
        parse_args(["--search-method", "harness", "--harness-main-research"])


def test_round_forks_carry_feedback_but_never_continue_a_previous_child(tmp_path):
    profiles = harness_profiles(("target_biology",))
    events, accounting, main_turns, branches = [], [], [], []
    def main_client():
        class Main:
            config = SimpleNamespace(limits=SimpleNamespace(wall_time_seconds=120))
            def __enter__(self): return self
            def __exit__(self, *args): events.append("main_closed")
            def run_turn(self, turns, **kwargs):
                turn, = turns
                main_turns.append(turn)
                root = tmp_path / "main/sessions/main_research"
                (root / "workspace").mkdir(parents=True, exist_ok=True)
                (root / "workspace/research.md").write_text(f"main round {turn.round_index}")
                (root / "session.jsonl").write_text(f"native main {turn.round_index}")
                return (SimpleNamespace(usage={"providerCalls": 3}, submission_status="accepted",
                    artifacts={"session": "sessions/main_research/session.jsonl"}, submission={"brief": "evidence"}),)
        return Main()
    def branch_client(root, source):
        assert events[-1] == "main_closed"
        branches.append((root, source))
        class Branch(FakeHarnessClient):
            def __enter__(self): return self
            def __exit__(self, *args): events.append("branches_closed")
        return Branch(root, {"target_biology": [[{"mutations": [{"position": 0, "base": "C"}]}]]})
    expander = ForkedResearchExpander(main_client=main_client, root=tmp_path,
        branch_client=branch_client, expander_options={
            "domain": NucleoBenchCandidateDomain(MOCK_CONTEXT), "profiles": profiles,
            "candidates_per_profile": 1, "campaign_id": "test", "first_active_round": 1,
            "attach_empirical_q0": True, "allow_repeated_occurrences": False,
            "account": lambda counts, **kw: accounting.append((counts, kw)),
        })
    measured = (_observation({"mutations": []}, round_idx=0),)
    first = ExpansionRequest(round_idx=1, reservoir_size=1, observations=measured)
    result = expander.expand(first)
    assert result.selection_mode == "acquisition"
    assert len(result.proposals) == 1
    assert events[-1] == "branches_closed"
    measured += (_observation({"mutations": [{"position": 2, "base": "T"}]}, round_idx=1),)
    second = ExpansionRequest(round_idx=2, reservoir_size=1, observations=measured)
    expander.expand(second)
    assert branches[0][0] != branches[1][0]
    assert branches[0][1].session_sha256 != branches[1][1].session_sha256
    assert (tmp_path / "main/sessions/main_research/workspace/branch_feedback/round_0001.json").exists()
    assert json.loads(main_turns[1].message)["previous_branch_panel"].endswith("round_0001.json")
    assert branches[0][1].history_cursor == 1
    assert branches[1][1].history_cursor == 2
    assert main_turns[1].history_from_seq == 1
    assert len(json.loads(main_turns[1].message)["new_measurements"]) == 1
    assert sum(c.get("llm_requests", 0) for c, _ in accounting) == 10
    assert sum(c.get("proposal_attempts", 0) for c, _ in accounting) == 2


def test_failed_main_turn_is_billed_before_any_fork(tmp_path):
    billed = []
    class Main:
        config = SimpleNamespace(limits=SimpleNamespace(wall_time_seconds=120))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def run_turn(self, *args, **kwargs):
            error = HarnessError("interrupted", turn_usage={"main_research": {"providerCalls": 7}})
            raise error
    expander = ForkedResearchExpander(main_client=Main, root=tmp_path,
        branch_client=lambda *_: pytest.fail("must not fork after failed main turn"),
        expander_options={"domain": NucleoBenchCandidateDomain(MOCK_CONTEXT),
                          "profiles": harness_profiles(), "account": lambda c, **kw: billed.append(c)})
    with pytest.raises(HarnessError):
        expander.expand(ExpansionRequest(round_idx=1, reservoir_size=4))
    assert any(c.get("llm_requests") == 7 for c in billed)


def test_tree_declaration_is_accepted_by_pilot_reporting(tmp_path):
    from ldm_tts.pilot_evaluation.execution import _mark_completed
    pool_paths = ("harness/main", "harness/branches/round_0001", "harness/branches/round_0002", "policy_harness")
    for name in pool_paths:
        path = tmp_path / name / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"campaignId": "test", "taskId": "nucleobench", "seed": 42,
                                    "profiles": [{"profileId": "research"}], "limits": {"toolCallBudgets": {}}}))
    write_fork_provenance(tmp_path / "harness", compiled_policy=True)
    entry = {}
    _mark_completed(entry, SimpleNamespace(output_root=tmp_path.parent),
                    SimpleNamespace(run_dir=tmp_path, method="ldm_harness_compiled"))
    assert entry["status"] == "completed"
    assert set(entry["harness"]) == {"main_research", "proposal_round_0001", "proposal_round_0002", "policy"}
    broken = tmp_path / "harness/branches/round_0002/manifest.json"
    data = json.loads(broken.read_text()); data["seed"] = 43; broken.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identities differ"):
        _mark_completed({}, SimpleNamespace(output_root=tmp_path.parent), SimpleNamespace(run_dir=tmp_path))
