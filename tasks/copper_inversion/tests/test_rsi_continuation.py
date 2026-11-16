"""Checkpoints, continuations and the planted note (WP6) with synthetic collaborators."""

import hashlib
import json
import os
import re
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from tasks.copper_inversion.rsi import continuation as continuations, intervention as interventions
from tasks.copper_inversion.rsi.audit import reconcile_budget, verify_briefs
from tasks.copper_inversion.rsi.controller import Config, Controller, turn_root
from tasks.copper_inversion.rsi.ledger import read
from tasks.copper_inversion.rsi.scripted import ScriptedAgent, ScriptedEvaluator
from tasks.copper_inversion.rsi.selection import reserve_and_rank

BRIEF = "Fixed task brief."


def scripted_select(pool, history, count):
    predictions = [{"program_id": p["program_id"], "branch": p["branch"], "mean": .5, "std": .1,
                    "acquisition": int(p["program_id"][:8], 16) / 0xffffffff} for p in pool]
    return {"predictions": predictions, "selected": reserve_and_rank(predictions, count)}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SessionAgent(ScriptedAgent):
    """Emulates the Pi session store: one profile per turn root, fork initialization, history cursor.

    Every turn appends its message to ``sessions/<profile>/pi-session/session.jsonl`` and
    proposes from the whole session text, so later records depend on the full conversation.
    """

    def propose(self, *, branch, round_index, count, message, view, turn_root, session_round, profile, **_):
        root = Path(turn_root)
        home, fork = root / "sessions" / profile, root / "fork-sources" / profile
        if fork.is_dir() and not home.exists():  # initializeFork
            saved = json.loads((fork / "snapshot.json").read_text())
            assert _sha(fork / "session.jsonl") == saved["sessionSha256"]
            shutil.copytree(fork / "workspace", home / "workspace")
            (home / "pi-session").mkdir()
            shutil.copyfile(fork / "session.jsonl", home / "pi-session/session.jsonl")
            (home / "fork.json").write_text(json.dumps(saved))
        commits = [json.loads(p.read_text()) for p in root.glob("turns/*/turn_committed.json")]
        start = json.loads((fork / "snapshot.json").read_text())["historyCursor"] if fork.is_dir() else 0
        cursor = max([c["historyToSeq"] for c in commits], default=start)
        if session_round != cursor:  # acceptCursor
            raise ValueError(f"history cursor mismatch: expected {cursor}, received {session_round}")
        session = home / "pi-session/session.jsonl"
        session.parent.mkdir(parents=True, exist_ok=True)
        with session.open("a") as stream:
            stream.write(json.dumps({"role": "user", "content": message}) + "\n")
        (home / "workspace").mkdir(exist_ok=True)
        with (home / "workspace/notes.md").open("a") as stream:
            stream.write(f"round {round_index}\n")
        rows, usage = super().propose(branch=branch, round_index=round_index, count=count,
                                      message=session.read_text(), view=view)
        commit = root / f"turns/round_{round_index:03d}/turn_committed.json"
        commit.parent.mkdir(parents=True, exist_ok=True)
        commit.write_text(json.dumps({"profileId": profile, "roundIndex": round_index,
                                      "historyFromSeq": session_round, "historyToSeq": session_round + 1,
                                      "artifacts": {"session": str(session.relative_to(root)),
                                                    "turn": f"sessions/{profile}/turns/round_{round_index:03d}"}}))
        return rows, usage


class FailingAfter(ScriptedAgent):
    """Every proposal after round ``after`` fails all its solves (for finalist provenance)."""

    def __init__(self, after):
        super().__init__()
        self.after = after

    def propose(self, *, round_index, **settings):
        rows, usage = super().propose(round_index=round_index, **settings)
        if round_index > self.after:
            for index, row in enumerate(rows):
                row["source"] = ("def solve(problem, budget, seed):\n    raise RuntimeError('synthetic failure')\n"
                                 f"marker_{settings['branch']}_{round_index}_{index} = 1\n")
        return rows, usage


def config(arm="accumulated", *, rounds=5, reset=5, seed=7, **settings):
    return Config(arm, seed, rounds=rounds, proposals_per_branch=2, batch=4, reset_period=reset,
                  token_cap_input=0, **settings)


def controller(root, cfg, agent, *, evaluator=None, start=None, intervention=None, identity=None):
    return Controller(root, cfg, agent, evaluator or ScriptedEvaluator(), scripted_select, BRIEF,
                      identity or {"synthetic_test": True, "run_id": Path(root).name}, intervention=intervention,
                      start=start)


def source_run(root, *, reset=5, at=2, agent=None, rounds=5):
    agent = continuations.CheckpointingAgent(agent or SessionAgent(), root, at)
    result = controller(root, config(reset=reset, rounds=rounds), agent).run()
    return result, continuations.open_checkpoint(continuations.checkpoint_dir(root, at))


def continue_run(root, checkpoint, mode, *, agent=None, intervention=None, evaluator=None):
    cfg = replace(checkpoint.config, arm=continuations.MODES[mode], policy=None)
    ctrl = controller(root, cfg, agent or SessionAgent(), evaluator=evaluator, intervention=intervention,
                      start=continuations.continuation_start(checkpoint, mode))
    return ctrl, continuations.run_continuation(ctrl, checkpoint, mode)


def briefs(root):
    found = {}
    for path in sorted(Path(root, "harness").glob("epoch_*/branch_*/briefs/round_*.json")):
        found[(int(path.parent.parent.name[7:]), int(path.stem[6:]))] = path.read_bytes()
    return found


def tree(root, skip=()):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(Path(root).rglob("*"))
            if p.is_file() and not p.relative_to(root).as_posix().startswith(skip)}


# Checkpoint --------------------------------------------------------------

def test_checkpoint_is_a_side_copy_that_leaves_the_source_run_unchanged(tmp_path):
    result, checkpoint = source_run(tmp_path / "source")
    plain = tmp_path / "plain" / "source"
    plain.parent.mkdir()
    controller(plain, config(), SessionAgent(), identity={"synthetic_test": True, "run_id": "source"}).run()
    assert tree(tmp_path / "source", ("checkpoints/",)) == tree(plain)
    ledger = (tmp_path / "source/ledger.jsonl").read_bytes().splitlines(keepends=True)
    folder = checkpoint.path
    assert folder == (tmp_path / "source/checkpoints/round_02").resolve()
    assert (folder / "ledger-prefix.jsonl").read_bytes() == b"".join(ledger[:9])
    assert (folder / "candidates.txt").read_text().split() == [r["program_id"] for r in read(plain / "ledger.jsonl")][:9]
    assert len(checkpoint.records) == 9 and checkpoint.round == 2 and len(checkpoint.previous_pool) == 8
    assert checkpoint.sha256 == _sha(folder / "snapshot.json")
    for branch in range(4):
        profile = f"researcher_{branch}"
        entry = checkpoint.snapshot["sessions"][profile]
        assert entry["sidecar_profile"] == profile and entry["historyCursor"] == 2
        session = (folder / f"sessions/{profile}/session.jsonl").read_text()
        assert session.count("\n") == 2  # rounds 1 and 2 only, although the source went on to round 5
        live = plain / f"harness/epoch_00/branch_{branch}/sessions/{profile}/pi-session/session.jsonl"
        assert live.read_text().startswith(session) and live.read_text().count("\n") == 5
    # Idempotent; any changed byte is refused; sessions advanced past t cannot be checkpointed afterwards.
    assert continuations.take_checkpoint(tmp_path / "source", 2) == checkpoint
    with pytest.raises(ValueError, match="not at the checkpoint round"):
        continuations.take_checkpoint(plain, 2)
    (folder / "sessions/researcher_1/workspace/notes.md").write_text("changed\n")
    with pytest.raises(ValueError, match="differ from its snapshot"):
        continuations.open_checkpoint(folder)


def test_checkpoint_without_sessions_can_be_taken_after_the_run(tmp_path):
    root = tmp_path / "run"
    controller(root, config(), ScriptedAgent()).run()
    checkpoint = continuations.take_checkpoint(root, 3)
    assert checkpoint.snapshot["sessions"] == {} and len(checkpoint.records) == 13
    with pytest.raises(ValueError, match="no sessions"):
        continuations.continuation_start(checkpoint, "conversation")
    with pytest.raises(ValueError, match="inside the campaign"):
        continuations.take_checkpoint(root, 5)


# AT-25 -------------------------------------------------------------------

@pytest.mark.parametrize("reset", [5, 2])  # round 3 is a round message (unevaluated list) or a boundary
def test_at25_conversation_continuation_reproduces_the_source_run(tmp_path, reset):
    source = tmp_path / "source"
    result, checkpoint = source_run(source, reset=reset)
    target = tmp_path / "ck7-conversation-r1"
    ctrl, continued = continue_run(target, checkpoint, "conversation")
    original = briefs(source)
    assert briefs(target) == {k: v for k, v in original.items() if k[1] >= 3}
    if reset == 5:
        assert json.loads(original[(0, 3)])["kind"] == "round" and json.loads(original[(0, 3)])["unevaluated"]
    assert (target / "ledger.jsonl").read_bytes() == (source / "ledger.jsonl").read_bytes()
    assert continued["training_finalists"] == result["training_finalists"]
    assert continued["charged_records"] == result["charged_records"] == 21
    for branch in range(4):
        profile = f"researcher_{branch}"
        child = target / f"harness/epoch_00/branch_{branch}/sessions/{profile}/pi-session/session.jsonl"
        parent = source / f"harness/epoch_00/branch_{branch}/sessions/{profile}/pi-session/session.jsonl"
        assert child.read_bytes() == parent.read_bytes()  # the continuation read the pre-checkpoint turns
    assert verify_briefs(target)["briefs_checked"] == 4 * 3
    assert json.loads((target / "manifest.json").read_text())["start"]["fresh_sessions"] is False
    # Resuming the finished continuation changes nothing.
    before = tree(target)
    continuations.run_continuation(ctrl, checkpoint, "conversation", resume=True)
    assert tree(target) == before


def test_history_and_best_program_modes_start_fresh_from_the_prefix(tmp_path):
    _, checkpoint = source_run(tmp_path / "source", reset=2)
    source = tmp_path / "source"
    for mode in ("history", "best_program"):
        target = tmp_path / f"ck7-{mode}-r1"
        _, result = continue_run(target, checkpoint, mode)
        rows = read(target / "ledger.jsonl")
        assert rows[:9] == list(checkpoint.records) and len(rows) == 21
        assert rows != read(source / "ledger.jsonl")
        assert not list(target.rglob("fork-sources"))
        message = json.loads(briefs(target)[(0, 3)])
        assert message["kind"] == "boundary" and message["brief"].startswith(BRIEF)
        session = next(target.glob("harness/epoch_*/branch_0/sessions/researcher_0/pi-session/session.jsonl"))
        lines = [json.loads(line)["content"] for line in session.read_text().splitlines()]
        assert json.loads(lines[0])["round"] == 3  # no turn from before the checkpoint
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["start"]["round"] == 3 and manifest["start"]["fresh_sessions"] is True
        if mode == "history":
            assert "# history v1  records=9  rounds=0..2" in message["brief"]
            assert verify_briefs(target)["briefs_checked"] == 12
        else:
            assert "records=1 " in message["brief"]


def test_continuations_refuse_changed_settings_and_mismatched_starts(tmp_path):
    _, checkpoint = source_run(tmp_path / "source")
    with pytest.raises(ValueError, match="settings differ"):
        continuations.check_config(checkpoint, replace(checkpoint.config, batch=8, arm="ldm4geo", policy=None),
                                   "history")
    with pytest.raises(ValueError, match="source arm"):
        continuations.check_config(checkpoint, replace(checkpoint.config, arm="ldm4geo", policy=None),
                                   "conversation")
    cfg = replace(checkpoint.config, arm="ldm4geo", policy=None)
    wrong = controller(tmp_path / "wrong", cfg, SessionAgent(),
                       start=continuations.continuation_start(checkpoint, "conversation"))
    with pytest.raises(ValueError, match="start differs"):
        continuations.run_continuation(wrong, checkpoint, "history")
    assert not (tmp_path / "wrong").exists()


def test_service_check_requires_exactly_the_imported_prefix(tmp_path):
    _, checkpoint = source_run(tmp_path / "source")
    continuations.check_service({"used_slots": 9}, checkpoint)
    continuations.check_service({"used_slots": 13}, checkpoint, resume=True)
    for used in (0, 13):
        with pytest.raises(ValueError, match="imported prefix"):
            continuations.check_service({"used_slots": used}, checkpoint)
    digest = checkpoint.snapshot["ledger_prefix_sha256"]
    continuations.check_service({"used_slots": 9, "import_ledger_prefix_sha256": digest}, checkpoint)
    for other in (None, "0" * 64):
        with pytest.raises(ValueError, match="different ledger prefix"):
            continuations.check_service({"used_slots": 9, "import_ledger_prefix_sha256": other}, checkpoint)


def test_fork_sources_helper_returns_the_placed_snapshots(tmp_path):
    _, checkpoint = source_run(tmp_path / "source")
    cfg = checkpoint.config
    ctrl = controller(tmp_path / "ck", cfg, SessionAgent(), start=continuations.continuation_start(checkpoint,
                                                                                                    "conversation"))
    ctrl.prepare()
    continuations.place_forks(ctrl, checkpoint)
    for branch in range(4):
        sources = continuations.fork_sources(ctrl.turn_root(3, branch))
        entry = checkpoint.snapshot["sessions"][f"researcher_{branch}"]
        assert {k: v.to_dict() for k, v in sources.items()} == {
            f"researcher_{branch}": {k: entry[k] for k in ("sessionSha256", "workspaceSha256", "historyCursor")}}
    assert continuations.fork_sources(tmp_path) == {}
    continuations.place_forks(ctrl, checkpoint)  # idempotent


def test_checkpoint_waits_for_one_copy_when_branches_run_concurrently(tmp_path):
    calls, lock = [], threading.Lock()

    class Counting(SessionAgent):
        def propose(self, **settings):
            if settings["round_index"] == 3:
                with lock:
                    calls.append(continuations.checkpoint_dir(tmp_path / "source", 2).is_dir())
            return super().propose(**settings)
    source_run(tmp_path / "source", agent=Counting())
    assert calls == [True] * 4


def test_prefix_encodings_are_reused_by_the_continuation(tmp_path):
    root = tmp_path / "source"
    agent = continuations.CheckpointingAgent(SessionAgent(), root, 2)

    def select(pool, history, count):
        for row in history:  # the stand-in surrogate caches one encoding per charged program
            path = root / "encodings" / f"{row['program_id']}.json"
            if not path.exists():
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({"program": row["program_id"], "source": "source-run"}))
        return scripted_select(pool, history, count)
    Controller(root, config(), agent, ScriptedEvaluator(), select, BRIEF, {"synthetic_test": True}).run()
    checkpoint = continuations.open_checkpoint(root / "checkpoints/round_02")
    assert checkpoint.snapshot["encodings"] == 9
    target = tmp_path / "ck"
    continue_run(target, checkpoint, "history")
    for row in checkpoint.records:
        name = f"encodings/{row['program_id']}.json"
        assert (target / name).read_bytes() == (root / name).read_bytes()


# AT-27 -------------------------------------------------------------------

def test_at27_imported_service_charges_the_prefix_and_seals_prefix_finalists(tmp_path):
    pytest.importorskip("numpy")
    from tasks.copper_inversion.tests.test_rsi_service import LoopbackClient, make_service, write_suite
    from tasks.copper_inversion.v2.training_receipts import eligible
    train = write_suite(tmp_path / "release", "train")
    source_service = make_service(tmp_path, train, name="source", slots=17)
    source = tmp_path / "accumulated-s7"
    cfg = config(rounds=4, reset=2)
    Controller(source, cfg, continuations.CheckpointingAgent(ScriptedAgent(), source, 2),
               LoopbackClient(source, source_service), scripted_select, BRIEF, {"synthetic_test": True}).run()
    checkpoint = continuations.open_checkpoint(source / "checkpoints/round_02")
    prefix = checkpoint.path / "ledger-prefix.jsonl"
    spec = {"source_root": str(source_service.root), "ledger_prefix": str(prefix),
            "ledger_prefix_sha256": checkpoint.snapshot["ledger_prefix_sha256"]}
    service = make_service(tmp_path, train, name="continued", slots=17, **{"import": spec})
    info = service.dispatch({"operation": "info"})
    assert info["used_slots"] == len(checkpoint.records) == 9 and info["slots"] == 17
    continuations.check_service(info, checkpoint)
    fresh = make_service(tmp_path, train, name="fresh", slots=17)
    with pytest.raises(ValueError, match="imported prefix"):
        continuations.check_service(fresh.dispatch({"operation": "info"}), checkpoint)
    prefix_ids = {r["program_id"] for r in checkpoint.records}
    assert prefix_ids - {r["program_id"] for r in checkpoint.records if r["status"] == "rejected"} \
        <= set(eligible(service.root, train))
    target = tmp_path / "ck7-history-r1"
    client = LoopbackClient(target, service)
    _, result = continue_run(target, checkpoint, "history", agent=FailingAfter(2), evaluator=client)
    rows = read(target / "ledger.jsonl")
    assert len(rows) == result["charged_records"] == 17
    assert {r["status"] for r in rows if r["round"] > 2} == {"failed"}
    sealed = client.seal_training()
    assert sealed["used_slots"] == 17
    assert sealed["source_sha256s"] == result["training_finalists"]
    assert set(sealed["source_sha256s"]) <= prefix_ids and len(sealed["source_sha256s"]) == 3
    assert reconcile_budget(target, service.root)["charged_sources"] == 17
    assert set(eligible(service.root, train)) >= prefix_ids


# AT-28 -------------------------------------------------------------------

def parity_tagger(source):
    found = re.search(r"ident_b\dr\d\dp(\d\d)x", source)
    if found is None:
        return {"baseline", "even", "odd"}
    return {"baseline", "even" if int(found.group(1)) % 2 == 0 else "odd"}


def test_target_rule_v1_takes_the_most_frequent_eligible_tag_of_the_best_program():
    sets = [{"a", "b"}, {"a"}, {"a", "c"}, {"b"}, {"a", "b", "c"}, set(), {"d"}, {"a"}, {"a"}, {"b"}]
    # shares: a .6, b .4, c .2, d .1
    assert interventions.choose_target(sets, {"a", "b", "c"}) == "a"
    assert interventions.choose_target(sets, {"b", "c"}) == "b"
    assert interventions.choose_target(sets, {"c", "d"}) is None
    assert interventions.choose_target([{"x"}] * 3 + [set()] * 7, {"x"}) == "x"  # 30% is inside
    assert interventions.choose_target([{"x"}] * 9 + [set()], {"x"}) == "x"  # 90% is inside
    assert interventions.choose_target([{"x"}] * 10, {"x"}) is None
    assert interventions.choose_target([{"x", "y"}] * 5 + [set()] * 5, {"y", "x"}) == "x"  # ties by name


def test_note_is_the_fixed_template_and_records_its_rule():
    note = interventions.planted_note("fast_marching", 11)
    assert note["note"] == interventions.TEMPLATE.format(design="fast marching", tag="fast_marching")
    assert interventions.validate(note, 20) == note
    identity = interventions.identity(note)
    assert identity == {"kind": "planted_note", "round": 11, "target_tag": "fast_marching",
                        "note_sha256": hashlib.sha256(note["note"].encode()).hexdigest(), "selection_rule": "v1"}
    with pytest.raises(ValueError, match="fixed template"):
        interventions.validate({**note, "note": "free text"}, 20)
    assert "iou" not in note["note"].lower()


def test_at28_planted_note_only_in_the_planted_round_messages(tmp_path):
    source = tmp_path / "source"
    _, checkpoint = source_run(source, reset=2)
    record = interventions.select_target(checkpoint, parity_tagger, {"spec": "test:parity"})
    best = min((r for r in checkpoint.records if r["status"] == "valid"), key=lambda r: (r["risk"], r["program_id"]))
    assert record["best_program"] == best["program_id"] and record["target_tag"] in ("even", "odd")
    assert record["shares"]["baseline"] == 1 and record["proposals"] == 16 and record["planted_round"] == 3
    assert interventions.select_target(checkpoint, parity_tagger, {"spec": "test:parity"}) == record
    assert continuations.open_checkpoint(checkpoint.path).sha256 == checkpoint.sha256  # selection is not hashed
    note = interventions.from_checkpoint(checkpoint)
    text = note["note"]
    assert note["target_tag"] == record["target_tag"] and note["round"] == 3
    for mode in ("conversation", "history"):
        target = tmp_path / f"ck7-{mode}-planted-r1"
        continue_run(target, checkpoint, mode, intervention=note)
        hits = []
        for path in sorted(target.rglob("*")):
            parts = path.relative_to(target).parts
            # Agent transcripts carry the messages they received; everything else is scanned.
            if path.is_file() and not {"sessions", "fork-sources", "turns"} & set(parts):
                if text.encode() in path.read_bytes():
                    hits.append(path.relative_to(target).as_posix())
        messages = sorted(p for p in hits if "/briefs/" in p)
        assert len(messages) == 4 and all(p.endswith("/briefs/round_03.json") for p in messages)
        assert sorted(set(hits) - set(messages)) == ["intervention/note.txt"]
        for key, body in briefs(target).items():
            assert ("carried_note" in json.loads(body)) == (key[1] == 3)
            if key[1] == 3:
                assert json.loads(body)["carried_note"] == text
        manifest = json.loads((target / "manifest.json").read_text())
        assert manifest["intervention"] == interventions.identity(note)
        assert verify_briefs(target)["briefs_checked"] == 12
        # Unplanted control: identical round-3 messages apart from the carried note.
        control = tmp_path / f"ck7-{mode}-r1"
        continue_run(control, checkpoint, mode)
        for branch in range(4):
            planted = json.loads(briefs(target)[(branch, 3)])
            assert {k: v for k, v in planted.items() if k != "carried_note"} == json.loads(briefs(control)[(branch, 3)])
    with pytest.raises(ValueError, match="evidence-only or accumulated"):
        continue_run(tmp_path / "ck7-best_program-planted-r1", checkpoint, "best_program", intervention=note)


def test_selection_without_an_eligible_tag_is_refused(tmp_path):
    _, checkpoint = source_run(tmp_path / "source")
    with pytest.raises(ValueError, match="no design tag"):
        interventions.select_target(checkpoint, lambda source: {"everywhere"})
    with pytest.raises(ValueError, match="no target selection"):
        interventions.from_checkpoint(checkpoint)


def test_tagger_loader_accepts_a_class_with_tags(tmp_path, monkeypatch):
    module = tmp_path / "fixture_tagger.py"
    module.write_text("class Tagger:\n    sha256 = 'r' * 64\n    def tags(self, source):\n        return ['x']\n"
                      "def plain(source):\n    return ['y']\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    function, identity = interventions.load_tagger("fixture_tagger:Tagger")
    assert function("s") == ["x"] and identity["rules_sha256"] == "r" * 64
    assert identity["module_sha256"] == _sha(module)
    assert interventions.load_tagger("fixture_tagger:plain")[0]("s") == ["y"]


# Entry point -------------------------------------------------------------

protocol_tests = pytest.importorskip("tasks.copper_inversion.tests.test_rsi_protocol")
from tasks.copper_inversion.tests.test_rsi_protocol import NOW, example, sources, write  # noqa: E402,F401
from tasks.copper_inversion.rsi import protocol as protocols, workflow  # noqa: E402


def protocol_source(tmp_path, path, experiment="smoke", seed=101):
    """A scripted accumulated source run under the protocol, checkpointed at round 1."""
    value = protocols.load(path)
    cfg = Config.from_protocol(value, "accumulated", "S", seed=seed)
    root = tmp_path / "runs" / experiment / f"accumulated-s{seed}"
    identity = {"synthetic_test": True, "run_id": root.name, "experiment": experiment, "arm": "accumulated",
                "seed": seed, "tier": "S", "judge": "standard", "protocol_id": protocols.protocol_id(value)}
    Controller(root, cfg, continuations.CheckpointingAgent(SessionAgent(), root, 1), ScriptedEvaluator(),
               scripted_select, BRIEF, identity).run()
    return continuations.checkpoint_dir(root, 1)


def prepare(tmp_path, path, checkpoint, *, mode="history", arm=None, replicate=1, planted=False,
            experiment="smoke", seed=101, name=None):
    extra = ["--continuation", str(checkpoint), "--continuation-mode", mode, "--replicate", str(replicate)]
    extra += ["--planted"] if planted else []
    name = name or workflow.continuation_run_id(seed, mode, replicate, planted)
    argv = protocol_tests._argv(tmp_path, path, experiment=experiment, arm=arm or continuations.MODES[mode],
                                seed=seed, out=tmp_path / "runs" / name, extra=extra)
    return workflow.prepare(workflow.parse_args(argv))


def test_entry_point_builds_the_continuation_manifest(tmp_path):
    checkpoint = protocol_source(tmp_path, protocol_tests.EXAMPLE)
    opened = continuations.open_checkpoint(checkpoint)
    plan = prepare(tmp_path, protocol_tests.EXAMPLE, checkpoint)
    manifest = plan["manifest"]
    assert manifest["run_id"] == "ck101-history-r1" and manifest["arm"] == "ldm4geo"
    assert manifest["continuation"] == {"source_run": "accumulated-s101", "round": 1, "mode": "history",
                                        "replicate": 1, "snapshot_sha256": opened.sha256}
    assert "intervention" not in manifest and plan["continuation"]["intervention"] is None
    assert prepare(tmp_path, protocol_tests.EXAMPLE, checkpoint, mode="conversation",
                   replicate=3)["manifest"]["run_id"] == "ck101-conversation-r3"
    for kwargs, message in [({"arm": "accumulated"}, "runs arm ldm4geo"),
                            ({"name": "ck101-history-r2"}, "must be named ck101-history-r1"),
                            ({"seed": 102}, "source run's seed"),
                            ({"planted": True}, "no target selection"),
                            ({"planted": True, "mode": "best_program"}, "conversation and history"),
                            ({"replicate": 0}, "positive --replicate")]:
        with pytest.raises(ValueError, match=message):
            prepare(tmp_path, protocol_tests.EXAMPLE, checkpoint, **kwargs)
    interventions.select_target(opened, parity_tagger)
    plan = prepare(tmp_path, protocol_tests.EXAMPLE, checkpoint, planted=True)
    assert plan["manifest"]["run_id"] == "ck101-history-planted-r1"
    note = plan["continuation"]["intervention"]
    assert note == interventions.from_checkpoint(opened) and note["round"] == 2
    with pytest.raises(ValueError, match="only E1, E2"):
        prepare(tmp_path, protocol_tests.EXAMPLE, checkpoint, experiment="M1")


def test_entry_point_checkpoint_round_is_recorded_and_bounded(tmp_path):
    argv = protocol_tests._argv(tmp_path, protocol_tests.EXAMPLE, arm="accumulated",
                                extra=["--checkpoint-round", "1"])
    assert workflow.prepare(workflow.parse_args(argv))["manifest"]["checkpoint_round"] == 1
    assert protocol_tests._prepare(tmp_path, protocol_tests.EXAMPLE)["manifest"]["checkpoint_round"] is None
    argv = protocol_tests._argv(tmp_path, protocol_tests.EXAMPLE, extra=["--checkpoint-round", "3"])
    with pytest.raises(ValueError, match="inside the campaign"):
        workflow.prepare(workflow.parse_args(argv))


def test_entry_point_e1_and_e2_continue_frozen_m1_checkpoints(tmp_path, sources):
    path = write(tmp_path / "protocol.json", example())
    protocols.freeze(path, now=NOW, **sources)
    checkpoint = protocol_source(tmp_path, path, experiment="M1")
    plan = prepare(tmp_path, path, checkpoint, experiment="E1")
    assert plan["manifest"]["experiment"] == "E1" and plan["manifest"]["continuation"]["mode"] == "history"
    with pytest.raises(ValueError, match="E2 continuations are planted"):
        prepare(tmp_path, path, checkpoint, experiment="E2")
    interventions.select_target(continuations.open_checkpoint(checkpoint), parity_tagger)
    with pytest.raises(ValueError, match="E2 continuations are planted"):
        prepare(tmp_path, path, checkpoint, experiment="E1", planted=True)
    assert prepare(tmp_path, path, checkpoint, experiment="E2", planted=True,
                   mode="conversation")["manifest"]["run_id"] == "ck101-conversation-planted-r1"
    with pytest.raises(ValueError, match="start from a checkpoint"):
        protocol_tests._prepare(tmp_path, path, experiment="E1")
    smoke = protocol_source(tmp_path, path, experiment="smoke")
    with pytest.raises(ValueError, match="checkpoints of M1 runs"):
        prepare(tmp_path, path, smoke, experiment="E1")


# AT-26 (H20 phase) -------------------------------------------------------

def _strings(path, root):
    """Every string in a Pi session file and, through parentSession, in its fork parent."""
    found, pending = [], [Path(path)]
    while pending:
        current = pending.pop()
        for line in current.read_text().splitlines():
            entry = json.loads(line)
            stack = [entry]
            while stack:
                value = stack.pop()
                if isinstance(value, str):
                    found.append(value)
                elif isinstance(value, dict):
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
            parent = entry.get("parentSession") if entry.get("type") == "session" else None
            if parent and parent.startswith("/artifacts/"):
                pending.append(Path(root) / parent.removeprefix("/artifacts/"))
    return found


@pytest.mark.skipif(not os.environ.get("RSI_AT26_SOURCE"), reason="needs real-sidecar runs (H20 phase)")
def test_at26_real_sidecar_fork_reads_the_conversation_before_the_checkpoint():
    """Set RSI_AT26_SOURCE, RSI_AT26_CONVERSATION, RSI_AT26_HISTORY to the three run directories."""
    source = Path(os.environ["RSI_AT26_SOURCE"])
    runs = {mode: Path(os.environ["RSI_AT26_" + mode.upper()]) for mode in ("conversation", "history")}
    manifest = json.loads((runs["conversation"] / "manifest.json").read_text())
    first = manifest["continuation"]["round"] + 1
    for branch in range(4):
        earliest = json.loads((source / f"harness/epoch_00/branch_{branch}/briefs/round_01.json").read_text())
        message = json.dumps(earliest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for mode, run in runs.items():
            cfg = Config(**json.loads((run / "manifest.json").read_text())["config"])
            root = turn_root(run, cfg, first, branch)
            sessions = sorted(root.glob("sessions/*/pi-session/*.jsonl"))
            assert len(sessions) == 1
            seen = any(message in value for value in _strings(sessions[0], root))
            assert seen == (mode == "conversation"), (mode, branch)
