"""Fork lineage, feedback, recovery and accounting at the experiment boundary."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import HarnessError
from ldm_tts.harness.research_tree import CoverageForks, DiverseArchive, LatestMainForks
from ldm_tts.harness.research_strategy import ResearchStrategy
from tasks.nucleobench.core.candidate import NucleoBenchCandidateDomain
from tasks.nucleobench.core.forked_harness import ForkedResearchExpander
from ldm_tts.harness.research_runtime import write_fork_provenance
from ldm_tts.harness.research_store import snapshot_fork
from tasks.nucleobench.core.harness import harness_profiles
from tasks.nucleobench.core.mock import MOCK_CONTEXT
from tasks.nucleobench.core.workflow import _campaign_budget, parse_args
from tasks.nucleobench.tests.test_harness import FakeHarnessClient, _observation


def prepare_tree(tree, request, profiles):
    from ldm_tts.harness.research_store import ResearchMeasurement
    from tasks.nucleobench.core.research import serialize_measured_observations
    measured = [o for o in request.observations if o.evaluation.succeeded]
    return tree.prepare(request.round_idx, profiles, serialize_measured_observations(request.observations, MOCK_CONTEXT),
                        [ResearchMeasurement(o.canonical_key, o.metrics["utility"], o.round_idx) for o in measured])


def commit_tree(tree, plan, expanded):
    from tasks.nucleobench.core.candidate import prepare_candidate_payload
    sessions = {s["profile_id"]: s for s in expanded.metadata["sessions"]}
    keys = {profile: [prepare_candidate_payload(p.payload, MOCK_CONTEXT).canonical_key for p in expanded.proposals
                      if p.metadata["harness_lineage"]["profile_id"] == profile] for profile in sessions}
    tree.commit(plan, sessions, keys)


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
        branch_client=branch_client, strategy=ResearchStrategy(LatestMainForks()), expander_options={
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
    assert branches[0][1]["target_biology"].session_sha256 != branches[1][1]["target_biology"].session_sha256
    assert (tmp_path / "main/sessions/main_research/workspace/branch_feedback/round_0001.json").exists()
    assert json.loads(main_turns[1].message)["previous_branch_panel"].endswith("round_0001.json")
    assert branches[0][1]["target_biology"].history_cursor == 1
    assert branches[1][1]["target_biology"].history_cursor == 2
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
        strategy=ResearchStrategy(LatestMainForks()),
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


def test_forks_resume_fixed_plan_and_continue_child_history(tmp_path):
    profiles = harness_profiles()
    seen, fail = [], [False]

    class Main:
        config = SimpleNamespace(limits=SimpleNamespace(wall_time_seconds=120))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def run_turn(self, turns, **kwargs):
            turn, = turns
            root = tmp_path / "main/sessions/main_research"
            (root / "workspace").mkdir(parents=True, exist_ok=True)
            (root / "workspace/research.md").write_text(f"main direction {turn.round_index}")
            (root / "session.jsonl").write_text(f"main session {turn.round_index}")
            return (SimpleNamespace(usage={}, submission_status="accepted",
                artifacts={"session": "sessions/main_research/session.jsonl"}, submission={"brief": "evidence"}),)

    def branches(root, sources):
        class Branch(FakeHarnessClient):
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def run_turn(self, turns, **kwargs):
                seen.append((sources, turns))
                if fail[0]:
                    fail[0] = False
                    raise HarnessError("injected interruption")
                results = super().run_turn(turns, **kwargs)
                saved = []
                for result in results:
                    child = root / "sessions" / result.profile_id
                    (child / "workspace").mkdir(parents=True, exist_ok=True)
                    (child / "workspace/research.md").write_text(f"child direction {result.round_index}")
                    (child / "session.jsonl").write_text(f"child session {result.round_index}")
                    saved.append(replace(result, artifacts={**result.artifacts,
                        "session": f"sessions/{result.profile_id}/session.jsonl"}))
                return tuple(saved)
        round_index = int(root.name.removeprefix("round_"))
        payload = {"mutations": [{"position": 2 * (round_index - 1), "base": "C"}]}
        return Branch(root, {p.profile_id: [[payload]] for p in profiles})

    expander = ForkedResearchExpander(main_client=Main, root=tmp_path, branch_client=branches,
        strategy=ResearchStrategy(CoverageForks()), expander_options={
            "domain": NucleoBenchCandidateDomain(MOCK_CONTEXT), "profiles": profiles,
            "candidates_per_profile": 1, "campaign_id": "test", "first_active_round": 1,
            "attach_empirical_q0": True, "allow_repeated_occurrences": True,
        })
    observations = (_observation({"mutations": []}, round_idx=0),)
    for r in range(1, 5):
        request = ExpansionRequest(round_idx=r, reservoir_size=4, observations=observations)
        if r == 2:
            fail[0] = True
            with pytest.raises(HarnessError, match="injected"):
                expander.expand(request)
            saved_plan = (tmp_path / "branches/round_0002/fork-plan.json").read_bytes()
        result = expander.expand(request)
        assert len(result.proposals) == 4
        assert result.selection_mode == "acquisition"
        from tasks.nucleobench.core.constants import NUCLEOBENCH_Q0_METADATA_KEY
        assert all(p.metadata[NUCLEOBENCH_Q0_METADATA_KEY]["occurrence_count"] == 4 for p in result.proposals)
        if r == 2:
            assert (tmp_path / "branches/round_0002/fork-plan.json").read_bytes() == saved_plan
            assert seen[-1][0] == seen[-2][0]
            assert [t.to_dict() for t in seen[-1][1]] == [t.to_dict() for t in seen[-2][1]]
        root = tmp_path / "branches" / f"round_{r:04d}"
        plan = json.loads((root / "fork-plan.json").read_text())
        assert len(plan["outcomes"]) == 4 * (r - 1)
        assert all(o["best_score"] == 1 and o["measured_count"] == 1 for o in plan["outcomes"])
        assert sum(a["purpose"] == "main" for a in plan["assignments"]) == 2
        if r > 1:
            continuations = [a for a in plan["assignments"] if a["purpose"] == "continue"]
            assert len(continuations) == 1
            assert len({a["lineage_id"] for a in continuations}) == len(continuations)
            for continuation in continuations:
                assert continuation["parent_id"].startswith("branch_")
                workspace = root / "fork-sources" / continuation["profile_id"] / "workspace/research.md"
                assert workspace.read_text().startswith("child direction")
            assert len(plan["archive"]) <= 3
        for turn in seen[-1][1]:
            assert turn.history_from_seq == seen[-1][0][turn.profile_id].history_cursor
            assert turn.history_to_seq == len(observations)
        # Replaying a committed round must not advance visits or replace parents.
        committed = (root / "tree.json").read_bytes()
        expander.expand(request)
        assert (root / "tree.json").read_bytes() == committed
        observations += (_observation(result.proposals[0].payload, round_idx=r),)

    with pytest.raises(ValueError, match="identity changed"):
        expander.coordinator.tree.archive_policy = DiverseArchive(capacity=2)
        expander.expand(request)


def test_coverage_configuration_and_existing_modes(tmp_path):
    budgets = {}
    configs = {"independent": {"name": "independent"}, "latest": {"name": "main_forks"},
               "coverage": {"name": "main_forks", "fork_policy": {"name": "coverage"}}}
    for name, config in configs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config))
        args = parse_args(["--search-method", "ldm_harness_compiled", "--iterations", "3",
                           "--harness-research-strategy-config", str(path)])
        assert args.harness_main_research == (name != "independent")
        budgets[name] = _campaign_budget(args, None)
    assert budgets["coverage"] == budgets["latest"]
    assert budgets["independent"]["harness_turns"] == 8
    assert budgets["coverage"]["harness_turns"] == 10
    assert budgets["coverage"]["main_research_turns"] == 2
    assert all(b["proposal_attempts"] == 8 for b in budgets.values())
    with pytest.raises(SystemExit):
        parse_args(["--search-method", "harness", "--harness-research-strategy-config", str(path)])
    with pytest.raises(SystemExit):
        parse_args(["--search-method", "ldm_harness_compiled", "--iterations", "3",
                    "--harness-research-strategy-config", str(tmp_path / "coverage.json"),
                    "--harness-profile", "target_biology"])


def test_archive_reconciles_delayed_measurements_before_parent_selection(tmp_path):
    from dataclasses import asdict
    from ldm_tts.harness.research_tree import ResearchNode
    from ldm_tts.harness.research_store import ResearchTree

    policy, archive = CoverageForks(), DiverseArchive()
    measured = _observation({"mutations": [{"position": 0, "base": "C"}]}, round_idx=1)
    unknown = ResearchNode("unknown", "snapshots/unknown", 1, 1, "unknown", proposal_keys=("unmeasured",))
    pending = ResearchNode("pending", "snapshots/pending", 1, 1, "pending", proposal_keys=(measured.canonical_key,))
    main = ResearchNode("main_0001", "snapshots/main", 1, 1, "main")
    root = tmp_path / "branches/round_0001"
    root.mkdir(parents=True)
    (root / "tree.json").write_text(json.dumps({
        "policies": {"fork_policy": policy.identity(), "archive_policy": archive.identity()},
        "main_nodes": [asdict(main)], "archive": [asdict(pending), asdict(unknown)], "outcomes": [],
    }))
    plan = prepare_tree(ResearchTree(tmp_path, policy, archive),
        ExpansionRequest(round_idx=2, reservoir_size=4, observations=(measured,)), harness_profiles())
    leaves = {n["node_id"]: n for n in plan["archive"]}
    assert leaves["pending"]["measured_count"] == 1
    assert leaves["pending"]["best_score"] == 1.0
    assert leaves["unknown"]["measured_count"] == 0
    assert leaves["unknown"]["best_score"] is None


def test_action_feedback_preserves_shared_credit_unknowns_and_decision_time_state(tmp_path):
    from dataclasses import asdict
    from ldm_tts.harness.research_tree import ResearchNode, ResearchOutcome
    from ldm_tts.harness.research_store import ResearchTree

    profiles = harness_profiles()
    policy, archive = CoverageForks(), DiverseArchive()
    parent = ResearchNode("main_0001", "snapshots/main", 1, 1, "main", best_score=-5)
    winner = _observation({"mutations": [{"position": 0, "base": "C"}]}, round_idx=1)
    old = _observation({"mutations": [{"position": 2, "base": "G"}]}, round_idx=0)
    later = _observation({"mutations": [{"position": 4, "base": "T"}]}, round_idx=2)
    failed = _observation({"mutations": [{"position": 6, "base": "C"}]}, round_idx=1)
    failed = replace(failed, evaluation=replace(failed.evaluation, status="failed", metrics={"utility": 1000}))
    batches = [ResearchOutcome(1, p.profile_id, "restart", parent, -5, keys) for p, keys in zip(profiles, (
        (winner.canonical_key,), (winner.canonical_key,),
        (old.canonical_key, later.canonical_key, failed.canonical_key), ("unknown",),
    ))]
    state = {"policies": {"fork_policy": policy.identity(), "archive_policy": archive.identity()},
             "main_nodes": [asdict(parent)], "archive": [], "outcomes": [asdict(o) for o in batches]}
    root = tmp_path / "branches/round_0002"
    root.mkdir(parents=True)
    previous = root / "tree.json"
    previous.write_text(json.dumps(state))
    tree = ResearchTree(tmp_path, policy, archive)
    request = ExpansionRequest(3, 4, observations=(old, winner, failed, later))
    plan = prepare_tree(tree, request, profiles)
    first, duplicate, unevaluated, unknown = plan["outcomes"]
    assert first["best_key"] == duplicate["best_key"] == winner.canonical_key
    assert first["measured_count"] == duplicate["measured_count"] == 1
    assert first["best_score"] == duplicate["best_score"] == 1
    assert unevaluated["best_score"] is unknown["best_score"] is None
    assert unevaluated["measured_count"] == unknown["measured_count"] == 0
    assert first["parent"]["best_score"] == -5
    assert prepare_tree(tree, request, profiles) == json.loads(json.dumps(plan))
    with pytest.raises(ValueError, match="earlier rounds"):
        prepare_tree(tree, replace(request, observations=(*request.observations, replace(winner, round_idx=3))), profiles)
    state["outcomes"][0]["incumbent"] = 100
    previous.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="identity changed"):
        prepare_tree(tree, request, profiles)


def test_three_continuations_preserve_each_lineage_snapshot_and_feedback(tmp_path):
    from dataclasses import asdict

    from ldm_tts.harness.research_tree import ForkAssignment, ResearchNode
    from ldm_tts.harness.research_store import ResearchTree

    class ContinueAll:
        def identity(self):
            return {"name": "continue_all", "version": 1}

        def plan(self, context):
            return tuple(ForkAssignment(profile, node.node_id, "continue", node.lineage_id)
                         for profile, node in zip(context.profile_ids[:3], context.archive, strict=True)) + (
                ForkAssignment(context.profile_ids[3], context.main_nodes[-1].node_id, "main", "main"),)

    profiles = harness_profiles()
    observations = tuple(_observation({"mutations": [{"position": 2 * i, "base": "G"}]}, round_idx=1)
                         for i in range(3))
    archive = tuple(ResearchNode(f"leaf_{p}", f"snapshots/{p}", 1, 3, p,
                                proposal_keys=(o.canonical_key,))
                    for p, o in zip("abc", observations))
    main = ResearchNode("main_0001", "snapshots/main", 1, 3, "main")
    tree = ResearchTree(tmp_path, ContinueAll(), DiverseArchive())
    previous = tmp_path / "branches/round_0001/tree.json"
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps({
        "policies": {"fork_policy": tree.fork_policy.identity(), "archive_policy": tree.archive_policy.identity()},
        "main_nodes": [asdict(main)], "archive": [asdict(n) for n in archive], "outcomes": [],
    }))
    plan = prepare_tree(tree, ExpansionRequest(2, 4, observations=observations), profiles)
    continued = [a for a in plan["assignments"] if a["purpose"] == "continue"]
    assert {a["parent_id"] for a in continued} == {n.node_id for n in archive}
    selected = {a["parent_id"] for a in plan["assignments"]}
    for n in (*plan["main_nodes"], *plan["archive"]):
        if n["node_id"] not in selected:
            continue
        source = tmp_path / "inputs" / n["node_id"]
        (source / "workspace").mkdir(parents=True)
        (source / "session.jsonl").write_text(n["node_id"])
        (source / "workspace/research.md").write_text(n["node_id"])
        snapshot_fork(source / "session.jsonl", source / "workspace", tmp_path / n["snapshot"], n["history_cursor"])
    sources = tree.materialize(plan)
    assert len(sources) == 4
    round_root = tmp_path / "branches/round_0002"
    sessions, proposals, new_observations = [], [], []
    for i, a in enumerate(plan["assignments"]):
        profile = a["profile_id"]
        assert (round_root / "fork-sources" / profile / "workspace/research.md").read_text() == a["parent_id"]
        child = round_root / "sessions" / profile
        (child / "workspace").mkdir(parents=True)
        (child / "session.jsonl").write_text(profile)
        (child / "workspace/research.md").write_text(profile)
        sessions.append({"profile_id": profile, "artifacts": {"session": f"sessions/{profile}/session.jsonl"}})
        payload = {"mutations": [{"position": 2 * i, "base": "C"}]}
        proposals.append(SimpleNamespace(payload=payload, metadata={"harness_lineage": {"profile_id": profile}}))
        new_observations.append(_observation(payload, round_idx=2))
    expanded = SimpleNamespace(metadata={"sessions": sessions}, proposals=proposals)
    commit_tree(tree, plan, expanded)
    committed = (round_root / "tree.json").read_bytes()
    leaves = json.loads(committed)["archive"]
    assert len(leaves) == 3
    assert {n["lineage_id"]: n["parent_id"] for n in leaves} == {n.lineage_id: n.node_id for n in archive}
    for a in continued:
        leaf = next(n for n in leaves if n["lineage_id"] == a["lineage_id"])
        assert (tmp_path / leaf["snapshot"] / "session.jsonl").read_text() == a["profile_id"]
    commit_tree(tree, plan, expanded)
    assert (round_root / "tree.json").read_bytes() == committed
    following = prepare_tree(tree, ExpansionRequest(3, 4, observations=(*observations, *new_observations)), profiles)
    assert len(following["outcomes"]) == 4
    assert all(o["best_score"] == 1 and o["measured_count"] == 1 for o in following["outcomes"])
