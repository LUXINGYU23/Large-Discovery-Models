"""Exercise task factories and native-snapshot orchestration without provider calls."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ldm_tts.engine.expansion import ExpansionRequest
from ldm_tts.harness import (HarnessSubmissionRequest, HarnessSubmittedArtifact,
    HarnessTurnResult, canonical_sha256, file_sha256)


class NativeFixture:
    def __init__(self, command, *, config, payload, events, commits, **kwargs):
        self.config, self.payload, self.events = config, payload, events
        self.commits = commits
        mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
        self.root = Path(next(m.split("src=", 1)[1].split(",", 1)[0] for m in mounts if "dst=/artifacts" in m))

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "manifest.json").write_text(json.dumps({"fixture": True}))
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    def close(self):
        pass

    def run_turn(self, turns, *, submission_validator, **kwargs):
        results = []
        for turn in turns:
            key = (self.root, turn.profile_id, turn.turn_id)
            if key in self.commits:
                committed = self.commits[key]
                assert committed.input_digest == turn.input_digest
                results.append(replace(committed, replayed=True,
                    usage={**committed.usage, "validationSubmissions": 0}))
                continue
            self.events.append((self.root, self.config, turn))
            session = self.root / "sessions" / turn.profile_id
            (session / "workspace").mkdir(parents=True, exist_ok=True)
            (session / "workspace/research.md").write_text("Task research fixture")
            (session / "session.jsonl").write_text(json.dumps({"turn": turn.turn_id}))
            if turn.profile_id == "main_research":
                submission, artifacts = {"brief": "Keep measured and unmeasured alternatives distinct."}, ()
            else:
                filename, body, extra = self.payload(self.root, turn)
                path = session / "turns" / turn.turn_id / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
                submission = {"artifact_path": filename, **extra}
                artifacts = (HarnessSubmittedArtifact("/artifact_path", filename,
                    path.relative_to(self.root).as_posix(), file_sha256(path), path.stat().st_size),)
            submitted = HarnessSubmissionRequest(turn.profile_id, turn.turn_id, 1, submission, artifacts)
            assert submission_validator(submitted).decision == "accept"
            results.append(HarnessTurnResult(
                profile_id=turn.profile_id, session_id=turn.profile_id, turn_id=turn.turn_id,
                round_index=turn.round_index, history_from_seq=turn.history_from_seq,
                history_to_seq=turn.history_to_seq, history_digest=turn.history_digest,
                input_digest=turn.input_digest, replayed=False, submission_status="accepted",
                submission_id=turn.turn_id, submission_digest=submitted.digest,
                submission=submission, submitted_artifacts=artifacts, validation_errors=(),
                usage={"providerCalls": 0, "toolCalls": {"read": 1}, "validationSubmissions": 1}, tool_budget={},
                artifacts={"session": (session / "session.jsonl").relative_to(self.root).as_posix()},
            ))
            self.commits[key] = results[-1]
        return tuple(results)


def install_fixture(monkeypatch, module, payload):
    events, commits = [], {}
    monkeypatch.setattr(module, "HarnessClient", lambda command, **kwargs:
        NativeFixture(command, payload=payload, events=events, commits=commits, **kwargs))
    return events


def check_tree(root, steps):
    for step in range(1, steps + 1):
        path = root / "branches" / f"round_{step:04d}"
        plan = json.loads((path / "fork-plan.json").read_text())
        assignments = plan["assignments"]
        assert len(assignments) == 4
        assert [a["purpose"] for a in assignments].count("main") == 2
        assert (path / "tree.json").is_file() and (path / "panel.json").is_file()
        if step > 1:
            assert [a["purpose"] for a in assignments].count("continue") == 1
        for source in (path / "fork-sources").iterdir():
            manifest = json.loads((source / "snapshot.json").read_text())
            assert manifest["sessionSha256"] == file_sha256(source / "session.jsonl")


@pytest.mark.parametrize("task", ["iron_mind", "synthonbench"])
def test_published_profiles_budget_for_main_research(task):
    from importlib import import_module
    root = Path(__file__).resolve().parents[1]
    parser = import_module(f"tasks.{task}.core.workflow_args")
    support = import_module(f"tasks.{task}.core.workflow_support")
    profiles = json.loads((root / "tasks" / task / "experiment.json").read_text())["profiles"]
    for name, profile in profiles.items():
        locked = profile.get("locked_args", {})
        if locked.get("search-method") not in {"ldm_harness", "ldm_harness_compiled"}:
            continue
        argv = [arg for key, value in locked.items()
                for item in (value if isinstance(value, list) else [value])
                for arg in ("--" + key, str(item))]
        args = parser.parse_args(argv)
        dynamic = (support.derived_budget(args, domain_size=10000) if task == "iron_mind"
                   else support.campaign_budget(args, None))
        for counter in ("harness_turns", "main_research_turns"):
            assert profile["budget"][counter] == dynamic[counter], (task, name, counter)


@pytest.mark.parametrize("task", ["iron_mind", "synthonbench"])
def test_measured_task_workflow_uses_default_tree(tmp_path, monkeypatch, capsys, task):
    from importlib import import_module
    workflow = import_module(f"tasks.{task}.core.workflow")
    argv = ["--mock", "--search-method", "ldm_harness", "--proposal-mode", "none",
            "--iterations", "2", "--proposal-samples", "4", "--harness-candidates-per-session", "1",
            "--bo-pool-size", "2", "--llm-url", "http://fixture.invalid/v1", "--llm-model-name", "fixture",
            "--api-key", "fixture", "--out-dir", str(tmp_path)]
    args = workflow.parse_args(argv)
    if task == "iron_mind":
        table = workflow._load_table(args)
        candidates = [{"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)} for row in table.rows]
    else:
        from itertools import product
        from tasks.synthonbench.core.space_order import ordered_reactions, ordered_positions, ordered_synthon_ids
        benchmark = workflow._load_benchmark(args)
        space = benchmark.task.space
        reaction = ordered_reactions(benchmark.task.allowed_reactions)[0]
        candidates = [{"reaction_id": reaction, "synthon_ids": list(ids)} for ids in product(
            *(ordered_synthon_ids(space, reaction, pos) for pos in ordered_positions(space, reaction)))]
    def payload(root, turn):
        candidate = {**candidates[turn.round_index], "change_summary": "Test a measured alternative.",
                     "rationale": "Use the task feedback to test this candidate."}
        return "candidates.json", json.dumps({"candidates": [candidate]}), {}
    events = install_fixture(monkeypatch, import_module(f"tasks.{task}.core.harness_runtime"), payload)
    assert workflow.main(argv) == 0
    root = Path(json.loads(capsys.readouterr().out)["run_dir"])
    check_tree(root / "harness", 2)
    counters = json.loads((root / "budget.json").read_text())["counters"]
    assert counters["harness_turns"] == 10 and counters["main_research_turns"] == 2
    assert counters["proposal_attempts"] == 8 and counters["benchmark_jobs"] == 2
    assert len(events) == 10
    provenance = json.loads((root / "harness_provenance.json").read_text())
    assert len(provenance["pools"]) == 3


@pytest.mark.parametrize("method", ["harness", "ldm_harness", "harness_public_audit"])
def test_atomworld_replays_committed_panel_after_task_write_or_audit_failure(tmp_path, monkeypatch, method):
    from tasks.atomworld.core import harness, harness_runtime, optimization_policy, workflow_args
    from tasks.atomworld.core.data import TASK_ROOT
    from tasks.atomworld.core.proposals import extract_cif

    fixture = json.loads((TASK_ROOT / "resources/mock_fixture.json").read_text())
    sample = fixture["public"][0]
    cif = extract_cif(fixture["mock_outputs"][sample["sample_id"]][0])
    events = install_fixture(monkeypatch, harness_runtime,
        lambda root, turn: ("answer.cif", cif, {"rationale": "Public geometry only."}))
    args = workflow_args.parse_args(["--mock", "--search-method", method, "--iterations", "2",
        "--llm-url", "http://fixture.invalid/v1", "--llm-model-name", "fixture",
        "--harness-cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path / "run")])
    def create():
        return harness.AtomWorldHarnessExpander([sample], args, tmp_path / "run",
            client_factory=harness_runtime.create_client)
    request = ExpansionRequest(round_idx=0, reservoir_size=1)
    original_write = harness.write_json
    def interrupted_write(path, value):
        if path.parent.name in {"attempts", "proposal_pools"}:
            raise OSError("interrupted after panel commit")
        return original_write(path, value)
    def interrupted_audit(*args):
        raise OSError("interrupted after panel commit")
    if method == "harness_public_audit":
        monkeypatch.setattr(optimization_policy, "select_public_answer", interrupted_audit)
    else:
        monkeypatch.setattr(harness, "write_json", interrupted_write)
    expander = create()
    with pytest.raises(OSError, match="interrupted after panel commit"):
        expander.expand(request)
    expander.close()
    panel_path, = (tmp_path / "run").glob("harness/sample_*/branches/round_0001/panel.json")
    committed_panel = panel_path.read_bytes()
    monkeypatch.setattr(harness, "write_json", original_write)
    monkeypatch.setattr(optimization_policy, "select_public_answer", lambda *args: (0, {"method": "fixture"}))
    resumed = create()
    try:
        result = resumed.expand(request)
        assert result.proposals
        assert [p.payload for p in resumed.expand(request).proposals] == [p.payload for p in result.proposals]
    finally:
        resumed.close()
    assert len(events) == 5
    assert panel_path.read_bytes() == committed_panel
    assert all(not {"replayed", "usage", "tool_budget"} & row.keys()
               for row in json.loads(committed_panel)["sessions"])


@pytest.mark.parametrize("changed_submission", [False, True])
def test_reasyn_replays_committed_panel_but_rejects_changed_submission(tmp_path, monkeypatch, changed_submission):
    from ldm_tts.transport import ProposalRequest
    from ldm_tts.harness.research_runtime import ResearchCoordinator, main_research_profile, research_contract
    from tasks.reasyn.core import harness, harness_runtime
    from tasks.reasyn.core.workflow_args import parse_args

    args = parse_args(["--mock", "--benchmark", "tdc", "--search-method", "ldm_harness",
        "--llm-url", "http://fixture.invalid/v1", "--llm-model", "fixture",
        "--harness-cache-dir", str(tmp_path / "cache")])
    args.provider_api_key = "fixture"
    def payload(root, turn):
        data = json.loads(turn.message.split("\n", 1)[1])
        return "candidates.json", json.dumps({"candidates": [
            {"target_smiles": "CCO", "rationale": "Projected hypothesis."}
            for _ in range(data["candidate_count"])]}), {}
    events = install_fixture(monkeypatch, harness_runtime, payload)
    root = tmp_path / "harness"
    def create(location, **kwargs):
        return harness_runtime.create_client(args, location, "fixture", "", **kwargs)
    def source():
        client = create(root)
        coordinator = ResearchCoordinator(root, args._research_strategy, client.config.profiles,
            lambda: create(root / "main", session_profiles=main_research_profile(root / "main", client.config.profiles[0]),
                           contract=research_contract()))
        return harness.HarnessTargetSource(client, root, args, coordinator=coordinator,
            branch_client=lambda location, sources: create(location, fork_sources=sources))
    request = ProposalRequest(messages=({"role": "user", "content": "fixture"},),
        metadata={"count": 4, "round_idx": 0, "history": []})
    original_write = harness.atomic_json_write
    def interrupted_write(path, value):
        if path.name == "response.json":
            raise OSError("interrupted after panel commit")
        return original_write(path, value)
    monkeypatch.setattr(harness, "atomic_json_write", interrupted_write)
    with pytest.raises(OSError, match="interrupted after panel commit"):
        source().propose(request)
    panel_path = root / "branches/round_0001/panel.json"
    if changed_submission:
        panel = json.loads(panel_path.read_text())
        panel["sessions"][0]["submission_digest"] = "0" * 64
        panel_path.write_text(json.dumps(panel))
    committed_panel = panel_path.read_bytes()
    monkeypatch.setattr(harness, "atomic_json_write", original_write)
    resumed = source()
    if changed_submission:
        with pytest.raises(ValueError, match="Committed research panel changed"):
            resumed.propose(request)
        assert not list(root.glob("proposal_turns/**/response.json"))
    else:
        response = resumed.propose(request)
        assert len(json.loads(response.text)["candidates"]) == 4
        assert resumed.propose(request) == response
        assert all(not {"replayed", "usage", "tool_budget"} & row.keys()
                   for row in json.loads(committed_panel)["sessions"])
    assert panel_path.read_bytes() == committed_panel
    assert len(events) == 5


@pytest.mark.parametrize("method", ["harness", "ldm_harness"])
def test_atomworld_tree_is_sample_isolated_and_keeps_feedback_boundary(tmp_path, monkeypatch, method):
    from tasks.atomworld.core import harness_runtime, workflow, workflow_args
    from tasks.atomworld.core.data import TASK_ROOT
    from tasks.atomworld.core.proposals import extract_cif
    fixture = json.loads((TASK_ROOT / "resources/mock_fixture.json").read_text())
    text = extract_cif(fixture["mock_outputs"][fixture["public"][0]["sample_id"]][0])
    events = install_fixture(monkeypatch, harness_runtime, lambda root, turn: ("answer.cif", text, {"rationale": "Public geometry only."}))
    args = workflow_args.parse_args(["--mock", "--search-method", method, "--iterations", "2",
        "--llm-url", "http://fixture.invalid/v1", "--llm-model-name", "fixture",
        "--harness-cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path / "run")])
    result = workflow.run(args, harness_client_factory=harness_runtime.create_client)
    roots = sorted((result.runtime.run_dir / "harness").glob("sample_*"))
    assert roots
    for root in roots:
        check_tree(root, 2)
    assert len([e for e in events if e[2].profile_id == "main_research"]) == len(roots) * 2
    for root in roots:
        history = json.loads((root / "main/public_history.json").read_text())["drafts"]
        assert len(history) == 1
        assert ("correct" in history[0]) == (method == "ldm_harness")
        assert not any(key in history[0] for key in ("target_cif", "judge_error", "rmsd"))
    assert len({config.campaign_id for _, config, _ in events}) == len(roots)


def test_reasyn_tree_credits_projected_occurrences_and_preserves_short_batch(tmp_path, monkeypatch):
    from ldm_tts.transport import ProposalRequest
    from ldm_tts.harness.research_runtime import ResearchCoordinator, main_research_profile, research_contract
    from tasks.reasyn.core import harness, harness_runtime
    from tasks.reasyn.core.workflow_args import parse_args
    args = parse_args(["--mock", "--benchmark", "tdc", "--search-method", "ldm_harness",
        "--llm-url", "http://fixture.invalid/v1", "--llm-model", "fixture",
        "--harness-cache-dir", str(tmp_path / "cache")])
    args.provider_api_key = "fixture"
    def payload(root, turn):
        data = json.loads(turn.message.split("\n", 1)[1])
        return "candidates.json", json.dumps({"candidates": [{"target_smiles": "CCO", "rationale": "Projected hypothesis."}
            for _ in range(data["candidate_count"])]}), {}
    events = install_fixture(monkeypatch, harness_runtime, payload)
    root = tmp_path / "harness"
    def create(location, **kwargs):
        return harness_runtime.create_client(args, location, "fixture", "", **kwargs)
    client = create(root)
    coordinator = ResearchCoordinator(root, args._research_strategy, client.config.profiles,
        lambda: create(root / "main", session_profiles=main_research_profile(root / "main", client.config.profiles[0]), contract=research_contract()))
    source = harness.HarnessTargetSource(client, root, args, coordinator=coordinator,
        branch_client=lambda location, sources: create(location, fork_sources=sources))
    def request(count, round_index, history=()):
        return ProposalRequest(messages=({"role": "user", "content": "fixture"},), metadata={
            "count": count, "round_idx": round_index, "history": list(history)})
    first = source.propose(request(4, 0))
    lineage = first.metadata["harness_lineage"][2]
    history = [{"candidate_id": "projected-product", "smiles": "CCC", "metrics": {"oracle_score": 0.8},
                "evaluation_status": "succeeded", "harness_lineage": lineage}]
    second = source.propose(request(1, 1, history))
    assert len(json.loads(second.text)["candidates"]) == 1
    assert source.propose(request(1, 1, history)) == second
    check_tree(root, 2)
    tree = json.loads((root / "branches/round_0002/tree.json").read_text())
    credited = [o for o in tree["outcomes"] if o["best_key"] == lineage["research_key"]]
    assert len(credited) == 1 and credited[0]["best_score"] == .8
    assert len(events) == 10


@pytest.mark.parametrize(("benchmark", "tree_steps"), [("reconstruction", 2), ("tdc", 3)])
def test_reasyn_tree_workflow_and_completed_resume_preserve_budget(tmp_path, monkeypatch, benchmark, tree_steps):
    from tasks.reasyn.core import harness_runtime, workflow

    preflights = []

    def preflight(args):
        preflights.append(args.llm_model)
        return {"status": "fixture"}

    def payload(root, turn):
        data = json.loads(turn.message.split("\n", 1)[1])
        smiles = ("CCO", "CCN", "CCC", "COC")[int(turn.profile_id.rsplit("_", 1)[1]) - 1]
        return "candidates.json", json.dumps({"candidates": [
            {"target_smiles": smiles, "rationale": "Test a synthesis hypothesis."}
            for _ in range(data["candidate_count"])
        ]}), {}

    monkeypatch.setattr(harness_runtime, "preflight_provider", preflight)
    events = install_fixture(monkeypatch, harness_runtime, payload)
    root = tmp_path / "run"
    argv = [
        "--mock", "--benchmark", benchmark, "--search-method", "ldm_harness",
        "--llm-url", "http://fixture.invalid/v1", "--llm-model", "fixture",
        "--harness-cache-dir", str(tmp_path / "cache"), "--iterations", "2",
        "--proposal-batch-size", "4", "--reservoir-size", "4", "--out-dir", str(root),
    ]
    assert workflow.main(argv) == 0
    # TDC excludes measured products and replenishes its second-round pool once.
    check_tree(root / "harness", tree_steps)
    assert len(events) == tree_steps * 5
    budget = json.loads((root / "budget.json").read_text())
    assert budget["counters"]["harness_turns"] == tree_steps * 5
    assert budget["counters"]["main_research_turns"] == tree_steps
    assert budget["counters"]["endpoint_preflight_requests"] == 1
    assert budget["counters"]["successful_evaluations"] == 2
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    observations = checkpoint["state"]["observations"]
    assert len(observations) == 2
    assert workflow.main([*argv, "--resume-from", str(root)]) == 0
    assert len(events) == tree_steps * 5 and preflights == ["fixture"]
    assert json.loads((root / "budget.json").read_text()) == budget
    assert json.loads((root / "checkpoint.json").read_text())["state"]["observations"] == observations


def test_alphabench_tree_workflow_includes_main_usage_and_initialization(tmp_path, monkeypatch):
    from tasks.alphabench.core import harness_runtime
    from tasks.alphabench.core.protocol import T3Protocol
    from tasks.alphabench.ldm_task.procedure import main

    protocol = T3Protocol(method="ldm_harness", cold_seed_count=1, candidates_per_session=2)
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(protocol.to_dict()))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    def payload(root, turn):
        return "candidates.json", json.dumps({"candidates": [
            {"name": f"a{i}", "expression": f"Mean($close,{51 + 2 * turn.round_index + i})"}
            for i in range(2)]}), {}
    events = install_fixture(monkeypatch, harness_runtime, payload)
    run = tmp_path / "run"
    assert main(["--mock", "--protocol-file", str(config), "--out-dir", str(run)]) == 0
    check_tree(run / "harness", 2)
    report = json.loads((run / "result.json").read_text())
    assert report["initialization"]["seed_count"] == 1
    assert report["costs"]["tool_calls"] == 10
    assert report["costs"]["tool_usage_reason"] is None
    assert report["search"]["generation_cost"]["step_count"] == 8
    assert report["budget"]["counters"]["harness_turns"] == 10
    assert report["budget"]["counters"]["proposal_attempts"] == 8
    assert report["budget"]["counters"]["dynamic_checks"] == 4
    assert len(events) == 10
    assert main(["--mock", "--protocol-file", str(config), "--resume-run", str(run)]) == 0
    assert len(events) == 10
