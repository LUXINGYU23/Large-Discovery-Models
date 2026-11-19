"""PiAgent sidecar lifetime, mounts and usage with a fake sidecar; no Docker, no model."""

import hashlib
import json
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ldm_tts.harness import (HarnessError, HarnessSubmissionRequest, HarnessSubmittedArtifact,
                             HarnessTurnResult)
from ldm_tts.harness.pi import PiForkSource
from ldm_tts.harness.protocol import canonical_sha256
from tasks.copper_inversion.rsi import agent as agent_module
from tasks.copper_inversion.rsi import protocol as protocols, workflow
from tasks.copper_inversion.rsi.agent import PiAgent, TOOL_RESOURCES, container_name, fork_sources
from tasks.copper_inversion.rsi.controller import Config, Controller, turn_root
from tasks.copper_inversion.rsi.scripted import ScriptedAgent, ScriptedEvaluator
from tasks.copper_inversion.rsi.selection import reserve_and_rank
from tasks.copper_inversion.rsi.usage import collect, recompute, response_usage, verify

FIXTURES = Path(__file__).parent / "data" / "rsi_usage"
EXAMPLE = Path(__file__).resolve().parents[1] / "resources" / "rsi" / "protocol.example.json"
CONTINUATION = "tasks.copper_inversion.rsi.continuation"


class Log:
    def __init__(self):
        self.lock, self.clients, self.events = threading.Lock(), [], []


class FakeSidecar:
    """HarnessClient plus the sidecar session store: pins the initialization identity per root,
    keeps one session file and history cursor per root, and writes provider files and commits."""

    def __init__(self, root, config, name, log, fail=None):
        self.root, self.config, self.name, self.log, self.fail = Path(root), config, name, log, fail
        self.profile = config.profiles[0].profile_id
        self.started = self.closed = False
        self.froms = []

    def start(self):
        with self.log.lock:
            if any(c.started and not c.closed and (c.name == self.name or c.root == self.root)
                   for c in self.log.clients):
                raise HarnessError("container name or artifact root already in use")
            self.log.clients.append(self)
            self.log.events.append(("start", self.profile, self.root))
        self.fresh = not (self.root / "sessions").exists()
        identity = canonical_sha256(self.config.initialize_payload())
        manifest = self.root / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text())["initializationSha256"] != identity:
                raise HarnessError("Harness initialization identity changed")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({"initializationSha256": identity}))
        profile_root = self.root / "sessions" / self.profile
        (profile_root / "workspace").mkdir(parents=True, exist_ok=True)
        self.session = profile_root / "pi-session" / "session.jsonl"
        self.session.parent.mkdir(parents=True, exist_ok=True)
        if not self.session.exists():
            self.session.write_text(json.dumps({"session_id": hashlib.sha256(str(self.root).encode()).hexdigest()[:8]})
                                    + "\n")
        self.session_id = json.loads(self.session.read_text().splitlines()[0])["session_id"]
        fork = self.config.fork_sources.get(self.profile)
        cursor = fork.history_cursor if fork else 0
        commits = [json.loads(p.read_text()) for p in sorted((self.root / "turns").glob("*/turn_committed.json"))]
        for commit in sorted(commits, key=lambda c: c["roundIndex"]):
            assert commit["historyFromSeq"] == cursor
            cursor = commit["historyToSeq"]
        self.cursor, self.started = cursor, True

    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds):
        (turn,) = turns
        assert self.started and not self.closed and turn.profile_id == self.profile
        if turn.history_from_seq != self.cursor:
            raise HarnessError("history cursor mismatch")
        if self.fail == (turn.round_index, self.profile):
            raise HarnessError("injected sidecar failure")
        self.froms.append(turn.history_from_seq)
        count = len(self.config.submission_contract.artifact_rules)
        branch = int(self.profile.rsplit("_", 1)[1])
        rows, _ = ScriptedAgent().propose(branch=branch, round_index=turn.round_index, count=count,
                                          message=turn.message)
        turn_dir = self.root / "sessions" / self.profile / "turns" / turn.turn_id
        artifacts, candidates = [], []
        for index, row in enumerate(rows):
            body = row["source"].encode()
            snapshot = turn_dir / "snapshot" / "candidates" / str(index) / "solver.py"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(body)
            artifacts.append(HarnessSubmittedArtifact(f"/candidates/{index}/artifact_path",
                                                      f"candidates/{index}/solver.py",
                                                      str(snapshot.relative_to(self.root)),
                                                      hashlib.sha256(body).hexdigest(), len(body)))
            candidates.append({"artifact_path": f"candidates/{index}/solver.py", "parent_sha256": None,
                               **{k: row[k] for k in ("change_summary", "rationale", "predicted_risk",
                                                      "interval80_low", "interval80_high")}})
        submission = {"candidates": candidates}
        if self.misnumber_first and len(candidates) > 1:
            # Entry 0 points at directory 1: the researcher must be told what to repair.
            swapped = [dict(candidates[1]), dict(candidates[0])]
            wrong = submission_validator(HarnessSubmissionRequest(
                self.profile, turn.turn_id, 1, {"candidates": swapped + candidates[2:]},
                (artifacts[1], artifacts[0], *artifacts[2:])))
            self.log.events.append(("retry", self.profile, wrong.errors[0].message))
        decision = submission_validator(HarnessSubmissionRequest(self.profile, turn.turn_id, 1, submission,
                                                                 tuple(artifacts)))
        assert decision.decision == "accept"
        provider = turn_dir / "provider"
        provider.mkdir(parents=True, exist_ok=True)
        for n, name in enumerate(("research", "compaction"), 1):
            stem = provider / f"{turn.turn_id}-provider-{n}"
            Path(f"{stem}.authorized.json").write_text("{}")
            shutil.copy(FIXTURES / f"{name}.response.bin", f"{stem}.response.bin")
        (turn_dir / "history-events.jsonl").write_text(json.dumps({"type": "compaction_committed"}) + "\n")
        with self.session.open("a") as stream:
            stream.write(json.dumps({"turn": turn.turn_id}) + "\n")
        commit = self.root / "turns" / turn.turn_id / "turn_committed.json"
        commit.parent.mkdir(parents=True, exist_ok=True)
        commit.write_text(json.dumps({"profileId": self.profile, "roundIndex": turn.round_index,
                                      "historyFromSeq": turn.history_from_seq, "historyToSeq": turn.history_to_seq,
                                      "usage": {"providerCalls": 2},
                                      "artifacts": {"session": str(self.session.relative_to(self.root))}}))
        self.cursor = turn.history_to_seq
        return (HarnessTurnResult(self.profile, self.session_id, turn.turn_id, turn.round_index,
                                  turn.history_from_seq, turn.history_to_seq, turn.history_digest, "0" * 64,
                                  False, "accepted", "submission", "0" * 64, submission, tuple(artifacts), (),
                                  {"providerCalls": 2}, {}, {"turn": str(turn_dir.relative_to(self.root))}),)

    def close(self):
        with self.log.lock:
            self.closed = True
            self.log.events.append(("close", self.profile, self.root))


class FakeExport:
    def __init__(self, catalog):
        pass

    def __call__(self, root, _problem):
        public = Path(root) / "public"
        public.mkdir(parents=True, exist_ok=True)
        (public / "cases.json").write_text("[]")


FakeSidecar.misnumber_first = False


class FakePiAgent(PiAgent):
    def __init__(self, log, fail=None, *, tmp, arm="ldm4geo", reset=2, **settings):
        values = {"base_url": "https://provider.invalid", "model": "fixture-model", "thinking": "low",
                  "bash_calls": 3, **settings}
        super().__init__(public_catalog="unused", api_key="fixture-key", sidecar_image="sha256:" + "0" * 64,
                         cache_dir=tmp / "cache", arm=arm, reset_period=reset, **values)
        self.log, self.fail = log, fail

    def _new_client(self, turn_root, config, name):
        return FakeSidecar(turn_root, config, name, self.log, self.fail)


@pytest.fixture(autouse=True)
def fake_export(monkeypatch):
    monkeypatch.setattr(agent_module, "PublicTrainingExport", FakeExport)


def scripted_select(pool, history, count):
    predictions = [{"program_id": p["program_id"], "branch": p["branch"], "mean": .5, "std": .1,
                    "acquisition": int(p["program_id"][:8], 16) / 0xffffffff} for p in pool]
    return {"predictions": predictions, "selected": reserve_and_rank(predictions, count)}


def controller(root, agent, arm, reset, rounds):
    return Controller(root, Config(arm, 7, rounds=rounds, proposals_per_branch=2, batch=4, reset_period=reset,
                                   token_cap_input=0), agent, ScriptedEvaluator(), scripted_select,
                      "Fixed task brief.", {"synthetic_test": True})


def test_model_settings_come_from_the_constructor_and_the_protocol(tmp_path):
    agent = FakePiAgent(Log(), tmp=tmp_path, base_url="https://other.invalid/v1", model="other-model",
                        thinking="high", bash_calls=5)
    config = agent.config(profile="researcher_2", run="r", epoch=1, seed=9, count=4, harness_seconds=60)
    assert (config.base_url, config.model, config.thinking) == ("https://other.invalid/v1", "other-model", "high")
    assert config.limits.tool_call_budgets == {"bash": 5, "web_search": 0, "web_fetch": 0}
    assert config.limits.wall_time_seconds == 60 and config.profiles[0].profile_id == "researcher_2"
    assert str(config.profiles[0].agents_path) == "/resources/rsi/AGENTS.md"
    with pytest.raises(ValueError):
        FakePiAgent(Log(), tmp=tmp_path, thinking="extreme")
    with pytest.raises(ValueError):
        FakePiAgent(Log(), tmp=tmp_path, bash_calls=-1)
    protocol = protocols.load(EXAMPLE)
    assert workflow.agent_settings(protocol) == {
        "base_url": protocol["model"]["base_url"], "model": protocol["model"]["name"],
        "thinking": protocol["model"]["thinking"], "bash_calls": protocol["loop"]["bash_calls"]}


def test_sidecar_mounts_only_rsi_resources_and_names_the_container_per_root(tmp_path):
    agent = FakePiAgent(Log(), tmp=tmp_path)
    root = tmp_path / "run" / "harness" / "epoch_01" / "branch_2"
    root.mkdir(parents=True)
    name = container_name(root, "researcher_2", "run", 1)
    command = agent.sidecar_command(root, name)
    at = command.index("run")
    assert command[at + 1:at + 3] == ["--name", name]
    mounts = {m.split(",dst=")[1].split(",")[0]: m for m, flag in zip(command[1:], command) if flag == "--mount"}
    assert set(mounts) == {"/artifacts", "/runtime-home/.cache/gondolin", "/resources/rsi"}
    assert f"src={root.resolve()}," in mounts["/artifacts"]
    assert mounts["/resources/rsi"] == f"type=bind,src={TOOL_RESOURCES.resolve()},dst=/resources/rsi,readonly"
    assert not any("resources/v2" in part or "resources/harness" in part for part in command)
    other = tmp_path / "run" / "harness" / "epoch_02" / "branch_2"
    assert container_name(other, "researcher_2", "run", 2) != name
    assert container_name(root, "researcher_2", "run", 1) == name


@pytest.mark.parametrize("arm,reset,rounds", [("ldm4geo", 1, 3), ("ldm4geo", 2, 5), ("restart_best", 2, 5),
                                              ("accumulated", 2, 5), ("independent_sampling", 2, 3)])
def test_one_client_per_artifact_root_fresh_after_each_boundary(tmp_path, arm, reset, rounds):
    log = Log()
    run = tmp_path / "run"
    agent = FakePiAgent(log, tmp=tmp_path, arm=arm, reset=reset)
    ctl = controller(run, agent, arm, reset, rounds)
    ctl.run()
    agent.close()
    assert all(c.closed for c in log.clients)
    for branch in range(4):
        profile = f"researcher_{branch}"
        roots = sorted({ctl.turn_root(r, branch) for r in range(1, rounds + 1)})
        mine = [c for c in log.clients if c.profile == profile]
        # One client per root, opened once, on a fresh root, and never on a previous epoch's root.
        assert sorted(c.root for c in mine) == roots and len({c.root for c in mine}) == len(mine)
        assert all(c.fresh for c in mine)
        expected = {"accumulated": 1, "independent_sampling": rounds}.get(arm, -(-rounds // reset))
        assert len(mine) == expected
        # The old client is fully closed before the branch's next client starts.
        events = [(kind, root) for kind, p, root in log.events if p == profile]
        for old, new in zip(mine, mine[1:]):
            assert events.index(("close", old.root)) < events.index(("start", new.root))
        for client in mine:  # the session continues within a root, from a zero cursor
            assert client.froms == list(range(len(client.froms)))
            assert len(client.session.read_text().splitlines()) == 1 + len(client.froms)
        sessions = [json.loads((run / "proposals" / f"round_{r:02d}" / f"branch_{branch}.json").read_text())
                    ["usage"]["session_id"] for r in range(1, rounds + 1)]
        assert len(set(sessions)) == expected
    usage = json.loads((run / "usage" / "round_01.json").read_text())
    assert sorted(usage["by_profile"]) == [f"researcher_{b}" for b in range(4)]
    assert {k: usage["by_profile"]["researcher_0"][k] for k in ("requests", "input_tokens", "cached_input_tokens",
            "output_tokens", "responses_missing_usage", "compactions")} == {
        "requests": 2, "input_tokens": 82000, "cached_input_tokens": 8000, "output_tokens": 3900,
        "responses_missing_usage": 0, "compactions": 1}
    report = verify(run)  # AT-17 on the agent path
    assert report["status"] == "passed" and report["rounds"] == list(range(1, rounds + 1))
    assert report["input_tokens"] == 82000 * 4 * rounds


def test_accumulated_session_continues_across_a_process_restart(tmp_path):
    log = Log()
    agent = FakePiAgent(log, (3, "researcher_1"), tmp=tmp_path, arm="accumulated")
    with pytest.raises(HarnessError, match="injected"):
        controller(tmp_path / "run", agent, "accumulated", 2, 4).run()
    failed = [c for c in log.clients if c.profile == "researcher_1"]
    assert len(failed) == 1 and failed[0].closed  # a failed turn releases its sidecar
    agent.close()
    restarted = Log()
    agent = FakePiAgent(restarted, tmp=tmp_path, arm="accumulated")
    controller(tmp_path / "run", agent, "accumulated", 2, 4).run(resume=True)
    agent.close()
    again = {c.profile: c for c in restarted.clients}
    assert set(again) == {f"researcher_{b}" for b in range(4)}
    assert all(not c.fresh and c.root == failed[0].root.parent / c.root.name for c in again.values())
    assert again["researcher_1"].froms == [2, 3] and again["researcher_0"].froms == [3]
    assert again["researcher_1"].session_id == failed[0].session_id
    clean = FakePiAgent(Log(), tmp=tmp_path, arm="accumulated")
    controller(tmp_path / "clean" / "run", clean, "accumulated", 2, 4).run()
    clean.close()
    assert (tmp_path / "run/ledger.jsonl").read_bytes() == (tmp_path / "clean/run/ledger.jsonl").read_bytes()
    assert verify(tmp_path / "run")["status"] == "passed"


def propose(agent, root, round_index, session_round):
    return agent.propose(branch=0, profile="researcher_0", round_index=round_index, turn_index=0, epoch=0, count=2,
                         session_round=session_round, message=f"message {round_index}", view=[], turn_root=root,
                         candidate_root=root.parents[2], known=frozenset(), seed=7, harness_seconds=60)


def test_fork_sources_are_pinned_on_every_turn_of_their_root(tmp_path, monkeypatch):
    run = tmp_path / "run"
    forked, plain = run / "harness/epoch_00/branch_0", run / "harness/epoch_01/branch_0"
    (forked / "fork-sources" / "researcher_0").mkdir(parents=True)
    source = PiForkSource("a" * 64, "b" * 64, 2)
    seen = []

    def found(root):
        seen.append(Path(root))
        return {"researcher_0": source} if (Path(root) / "fork-sources").is_dir() else {}

    monkeypatch.setitem(sys.modules, CONTINUATION, SimpleNamespace(fork_sources=found))
    log = Log()
    agent = FakePiAgent(log, tmp=tmp_path, arm="accumulated")
    propose(agent, forked, 11, 2)
    propose(agent, forked, 12, 3)
    assert len(log.clients) == 1 and log.clients[0].config.fork_sources == {"researcher_0": source}
    assert log.clients[0].froms == [2, 3] and seen == [forked.resolve()] * 2
    agent.close()
    restarted = Log()
    agent = FakePiAgent(restarted, tmp=tmp_path, arm="accumulated")
    propose(agent, forked, 13, 4)  # a new process pins the same identity, or the fake sidecar refuses it
    assert restarted.clients[0].config.fork_sources == {"researcher_0": source}
    propose(agent, plain, 14, 0)
    assert restarted.clients[1].config.fork_sources == {}
    agent.close()


def test_fork_sources_default_to_none_without_the_continuation_module(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, CONTINUATION, None)
    assert fork_sources(tmp_path) == {}


def test_concurrent_branch_turns_never_share_a_client_or_root(tmp_path):
    log = Log()
    barrier = threading.Barrier(4, timeout=30)

    class Waiting(FakeSidecar):
        def run_turn(self, *args, **kwargs):
            barrier.wait()
            return super().run_turn(*args, **kwargs)

    agent = FakePiAgent(log, tmp=tmp_path)
    agent._new_client = lambda root, config, name: Waiting(root, config, name, log)
    controller(tmp_path / "run", agent, "ldm4geo", 2, 1).run()
    agent.close()
    assert len({c.name for c in log.clients}) == len({c.root for c in log.clients}) == 4
    assert len({c.config.campaign_id for c in log.clients}) == 4


def test_a_structural_defect_gets_an_actionable_retry_not_the_novelty_reply(tmp_path, monkeypatch):
    """The rehearsal lost a run: entry i pointed at another directory and every retry said only
    "not novel or invalid", so the researcher never found the defect."""
    monkeypatch.setattr(FakeSidecar, "misnumber_first", True)
    log = Log()
    with FakePiAgent(log, tmp=tmp_path) as agent:
        controller(tmp_path / "run", agent, "ldm4geo", 2, 1).run()
    reasons = [e[2] for e in log.events if e[0] == "retry"]
    assert reasons and all("candidates[0].artifact_path must be candidates/0/solver.py" in r for r in reasons)
    assert not any("novel" in r for r in reasons)


def test_turn_index_other_than_zero_is_refused(tmp_path):
    agent = FakePiAgent(Log(), tmp=tmp_path)
    with pytest.raises(ValueError, match="extension arm"):
        agent.propose(branch=0, profile="researcher_0", round_index=1, turn_index=1, epoch=0, count=2,
                      session_round=1, message="m", view=[], turn_root=tmp_path / "r", candidate_root=tmp_path,
                      known=frozenset(), seed=0, harness_seconds=60)


def test_collect_counts_missing_usage_and_never_fails_on_a_call_count_mismatch(tmp_path):
    shutil.copytree(FIXTURES, tmp_path / "provider")
    (tmp_path / "history-events.jsonl").write_text(
        "\n".join(json.dumps({"type": t}) for t in ("compaction_started", "compaction_committed",
                                                     "compaction_committed")) + "\n")
    expected = {"input_tokens": 82800, "cached_input_tokens": 8100, "output_tokens": 3960, "compactions": 2}
    for calls, requests in ((None, 6), (6, 6), (8, 8), (4, 6), ("x", 6)):
        usage = collect(tmp_path, calls)
        assert {k: usage[k] for k in expected} == expected
        assert usage["requests"] == requests and usage["responses_missing_usage"] == requests - 4
        assert usage["response_files"] == 6


def test_independent_parser_agrees_on_recorded_fixtures():
    values = {p.name: response_usage(p.read_bytes()) for p in FIXTURES.iterdir()}
    assert values == {"research.response.bin": (12000, 8000, 900), "compaction.response.bin": (70000, 0, 3000),
                      "crlf.response.bin": (500, 0, 40), "plain.response.bin": (300, 100, 20),
                      "truncated.response.bin": None, "no-usage.response.bin": None}
    assert response_usage(b'data:{"type":"response.completed","response":{"usage":'
                          b'{"input_tokens":3,"output_tokens":1}}}\n\n') == (3, 0, 1)
    assert response_usage(b'{"usage":{"input_tokens":1,"output_tokens":1,'
                          b'"input_tokens_details":{"cached_tokens":2}}}') is None


def write_turn(run, round_index, profile, files, authorized, committed):
    root = run / "harness" / "epoch_00" / f"branch_{profile[-1]}"
    turn = f"round_{round_index:03d}"
    provider = root / "sessions" / profile / "turns" / turn / "provider"
    provider.mkdir(parents=True)
    for n, name in enumerate(files, 1):
        shutil.copy(FIXTURES / name, provider / f"{turn}-provider-{n}.response.bin")
    for n in range(1, authorized + 1):
        (provider / f"{turn}-provider-{n}.authorized.json").write_text("{}")
    commit = root / "turns" / turn / "turn_committed.json"
    commit.parent.mkdir(parents=True)
    commit.write_text(json.dumps({"usage": {"providerCalls": committed}}))
    return collect(provider.parent, committed)


def test_at17_recount_matches_usage_files_and_reports_every_difference(tmp_path):
    run = tmp_path / "run"
    keys = ("requests", "input_tokens", "cached_input_tokens", "output_tokens", "responses_missing_usage",
            "compactions")
    first = {"researcher_0": write_turn(run, 1, "researcher_0", ["research.response.bin", "truncated.response.bin"],
                                        3, 3),
             "researcher_1": write_turn(run, 1, "researcher_1", ["plain.response.bin", "crlf.response.bin"], 2, 2)}
    second = {"researcher_0": write_turn(run, 2, "researcher_0", ["no-usage.response.bin"], 1, 1)}
    assert first["researcher_0"]["responses_missing_usage"] == 2  # one cut stream, one call without a file
    total = 0
    for round_index, by_profile in ((1, first), (2, second)):
        total += sum(u["input_tokens"] for u in by_profile.values())
        (run / "usage").mkdir(exist_ok=True)
        (run / "usage" / f"round_{round_index:02d}.json").write_text(json.dumps(
            {"round": round_index, "by_profile": {p: {k: u[k] for k in keys} for p, u in by_profile.items()},
             "run_total_input_tokens": total}))
    (run / "result.json").write_text(json.dumps({"input_tokens": total, "responses_missing_usage": 3}))
    counted, log = recompute(run)
    assert counted[1]["researcher_1"]["input_tokens"] == 800 and log == []
    report = verify(run)
    assert report["status"] == "passed" and report["responses_missing_usage"] == 3 and report["input_tokens"] == 12800
    path = run / "usage" / "round_01.json"
    record = json.loads(path.read_text())
    record["by_profile"]["researcher_1"]["output_tokens"] += 1
    path.write_text(json.dumps(record))
    commit = run / "harness/epoch_00/branch_0/turns/round_002/turn_committed.json"
    commit.write_text(json.dumps({"usage": {"providerCalls": 4}}))
    report = verify(run)
    assert report["status"] == "failed"
    assert report["differences"] == [{"round": 1, "profile": "researcher_1", "field": "output_tokens",
                                      "usage_file": 61, "recount": 60}]
    assert report["sidecar_log_mismatches"][0]["committed_provider_calls"] == 4


def test_live_audit_expects_the_manifest_model():
    from tasks.copper_inversion.tests.rsi_live_audit import expected_provider
    manifest = {"model": {"base_url": "https://x.invalid", "name": "m", "thinking": "medium"}}
    assert expected_provider(manifest) == ("m", "medium")
    assert expected_provider({"model": {**manifest["model"], "thinking": "off"}}) == ("m", None)
    assert expected_provider(manifest, "high") == ("m", "high")


class NonSubmittingSidecar(FakeSidecar):
    """Simulate the sidecar's per-turn recovery block with saved provider traces."""
    def run_turn(self, turns, *, submission_validator, recovery_timeout_seconds):
        (turn,) = turns
        if (turn.round_index, self.profile) not in self.log.fail_turns:
            return super().run_turn(turns, submission_validator=submission_validator,
                                    recovery_timeout_seconds=recovery_timeout_seconds)
        if turn.history_from_seq != self.cursor:
            raise HarnessError("history cursor mismatch")
        self.froms.append(turn.history_from_seq)
        turn_dir = self.root / "sessions" / self.profile / "turns" / turn.turn_id
        if (turn_dir / "history-recovery.json").exists():
            self.log.events.append(("blocked_replay", self.profile, turn.round_index))
            raise HarnessError("Error: " + agent_module.EXHAUSTED, code="sidecar_error",
                               turn_usage={self.profile:{"providerCalls":2}})
        self.log.events.append(("non_submission", self.profile, turn.round_index))
        provider = turn_dir / "provider"
        provider.mkdir(parents=True, exist_ok=True)
        for n, name in enumerate(("research", "compaction"), 1):
            stem = provider / f"{turn.turn_id}-provider-{n}"
            Path(f"{stem}.authorized.json").write_text("{}")
            shutil.copy(FIXTURES / f"{name}.response.bin", f"{stem}.response.bin")
        (turn_dir / "history-events.jsonl").write_text(json.dumps({"type":"history_blocked"}) + "\n")
        (turn_dir / "history-recovery.json").write_text(json.dumps({"continuations":2,
            "failure":{"message":agent_module.EXHAUSTED,"compactionId":None}}))
        with self.session.open("a") as stream:
            stream.write(json.dumps({"turn":turn.turn_id,"failed":True}) + "\n")
        raise HarnessError("Error: " + agent_module.EXHAUSTED, code="sidecar_error",
                           turn_usage={self.profile:{"providerCalls":2}})


class NonSubmittingAgent(FakePiAgent):
    def _new_client(self, turn_root, config, name):
        return NonSubmittingSidecar(turn_root, config, name, self.log, self.fail)


def test_no_submission_keeps_session_usage_and_replays_after_crash(tmp_path):
    log = Log(); log.fail_turns = {(2, "researcher_1")}
    agent = NonSubmittingAgent(log, tmp=tmp_path, arm="accumulated", reset=2)
    class FailOnce(ScriptedEvaluator):
        failed = False
        def evaluate_source(self, path):
            if not self.failed and (tmp_path / "run/proposals/round_02/branch_1.json").is_file():
                self.failed = True
                raise RuntimeError("injected evaluation interruption")
            return super().evaluate_source(path)
    run = tmp_path / "run"
    ctl = controller(run, agent, "accumulated", 2, 3)
    ctl.evaluator = FailOnce()
    with pytest.raises(RuntimeError, match="evaluation interruption"):
        ctl.run()
    receipt = json.loads((run / "proposals/round_02/branch_1.json").read_text())
    assert receipt["status"] == "no_submission" and receipt["proposals"] == []
    assert receipt["usage"]["requests"] == 2 and not receipt["cursor_advanced"]
    before = sum(e[0] == "non_submission" for e in log.events)
    result = ctl.run(resume=True)
    assert result["rounds_completed"] == 3 and result["charged_records"] == 13
    assert sum(e[0] == "non_submission" for e in log.events) == before == 1
    client = next(c for c in log.clients if c.profile == "researcher_1")
    assert client.froms == [0, 1, 1]  # failed turn did not advance its existing session
    assert len([e for e in log.events if e[0] == "start" and e[1] == "researcher_1"]) == 1
    assert json.loads((run / "usage/round_02.json").read_text())["by_profile"]["researcher_1"]["requests"] == 2
    assert verify(run)["status"] == "passed"
    from tasks.copper_inversion.experiments.rsi.analysis.rundir import Run
    from tasks.copper_inversion.experiments.rsi.analysis.measures import RunMeasures
    from tasks.copper_inversion.experiments.rsi.analysis.bundle import Analysis
    item = Run(run); measures = RunMeasures(item)
    assert measures.no_submission_turns == 1 and measures.rounds[2]["no_submission_turns"] == 1
    analysis = Analysis.__new__(Analysis); analysis.measures = {item.run_id: measures}
    analysis.arm_runs = lambda experiment, arm: [item] if (experiment, arm) == ("M1", "accumulated") else []
    assert analysis.process()["accumulated"]["no_submission_turns"] == 1
    agent.close()


@pytest.mark.parametrize("failed_branches, expected", [([0,1,2], 2), ([0,1,2,3], 0)])
def test_small_or_empty_pool_leaves_slots_unspent(tmp_path, failed_branches, expected):
    log = Log(); log.fail_turns = {(1, f"researcher_{b}") for b in failed_branches}
    agent = NonSubmittingAgent(log, tmp=tmp_path)
    run = tmp_path / "run"; ctl = controller(run, agent, "ldm4geo", 2, 2)
    result = ctl.run()
    selected = json.loads((run/"selection/round_01.json").read_text())["selected"]
    assert len(selected) == expected
    assert result["charged_records"] == 1 + expected + 4
    assert result["rounds_completed"] == 2
    for b in failed_branches:
        assert json.loads((run/f"proposals/round_01/branch_{b}.json").read_text())["status"] == "no_submission"
    assert verify(run)["status"] == "passed"
    agent.close()


def test_exhaustion_requires_exact_sidecar_code_and_durable_state(tmp_path):
    turn = tmp_path / "turn"; turn.mkdir()
    (turn/"history-recovery.json").write_text(json.dumps({"continuations":2,
        "failure":{"message":agent_module.EXHAUSTED,"compactionId":None}}))
    usage = {"researcher_0":{"providerCalls":2}}
    correct = HarnessError("Error: " + agent_module.EXHAUSTED, code="sidecar_error", turn_usage=usage)
    assert agent_module._exhausted_submission(correct, turn, "researcher_0")
    assert not agent_module._exhausted_submission(HarnessError(str(correct), code="network_error", turn_usage=usage), turn, "researcher_0")
    assert not agent_module._exhausted_submission(HarnessError("provider failed", code="sidecar_error", turn_usage=usage), turn, "researcher_0")
    assert not agent_module._exhausted_submission(HarnessError(str(correct), code="sidecar_error"), turn, "researcher_0")


@pytest.mark.parametrize("failed_branches", [[1], [0,1,2,3]])
def test_checkpoint_and_conversation_fork_after_no_submission(tmp_path, failed_branches):
    from tasks.copper_inversion.rsi import continuation
    log = Log(); log.fail_turns = {(1, f"researcher_{b}") for b in failed_branches}
    agent = NonSubmittingAgent(log, tmp=tmp_path, arm="accumulated", reset=2)
    run = tmp_path / "source"
    wrapped = continuation.CheckpointingAgent(agent, run, 1)
    original = controller(run, wrapped, "accumulated", 2, 2)
    original.run()
    checkpoint = continuation.open_checkpoint(run / "checkpoints/round_01")
    assert len(checkpoint.records) == 1 + (0 if len(failed_branches) == 4 else 4)
    for branch in range(4):
        entry = checkpoint.snapshot["sessions"][f"researcher_{branch}"]
        assert entry["historyCursor"] == (0 if branch in failed_branches else 1)
    log.fail_turns.clear()
    fork_agent = NonSubmittingAgent(log, tmp=tmp_path, arm="accumulated", reset=2)
    start = continuation.continuation_start(checkpoint, "conversation")
    dest = tmp_path / "fork"
    config = Config("accumulated", 7, rounds=2, proposals_per_branch=2, batch=4, reset_period=2, token_cap_input=0)
    fork = Controller(dest, config, fork_agent, ScriptedEvaluator(), scripted_select,
                      "Fixed task brief.", {"synthetic_test": True}, start=start)
    result = continuation.run_continuation(fork, checkpoint, "conversation")
    assert result["rounds_completed"] == 2
    for branch in range(4):
        peer = next(c for c in log.clients if c.profile == f"researcher_{branch}" and c.root.is_relative_to(dest))
        assert peer.froms == [0 if branch in failed_branches else 1]
    agent.close(); fork_agent.close()


def test_resume_after_crash_before_failed_turn_receipt(tmp_path):
    log = Log(); log.fail_turns = {(2, "researcher_1")}
    agent = NonSubmittingAgent(log, tmp=tmp_path, arm="accumulated", reset=2)
    class Interrupt(ScriptedEvaluator):
        def evaluate_source(self, path):
            if (tmp_path/"run/proposals/round_02/branch_1.json").is_file():
                raise RuntimeError("interrupt")
            return super().evaluate_source(path)
    run = tmp_path/"run"; first = controller(run, agent, "accumulated", 2, 3); first.evaluator = Interrupt()
    with pytest.raises(RuntimeError, match="interrupt"): first.run()
    (run/"proposals/round_02/branch_1.json").unlink()  # crash before the controller could publish this receipt
    agent.close()
    resumed_agent = NonSubmittingAgent(log, tmp=tmp_path, arm="accumulated", reset=2)
    resumed = controller(run, resumed_agent, "accumulated", 2, 3)
    assert resumed.run(resume=True)["rounds_completed"] == 3
    assert sum(e[0] == "non_submission" for e in log.events) == 1  # no extra provider request
    assert sum(e[0] == "blocked_replay" for e in log.events) == 1
    peers = [c for c in log.clients if c.profile == "researcher_1"]
    assert len(peers) == 2 and peers[0].session_id == peers[1].session_id
    assert peers[1].froms == [1, 1]
    assert verify(run)["status"] == "passed"
    resumed_agent.close()
