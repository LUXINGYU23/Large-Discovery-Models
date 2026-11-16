"""Round checkpoints and continuations from them (contracts §7).

A checkpoint is a side copy taken after round t's evaluation and before any turn
of round t+1. It holds the ledger prefix, the sources that the prefix and the last
two proposal rounds need, and the prefix encodings, so every continuation fits the
same surrogate. It also holds each branch's Pi session and workspace, frozen with
``snapshot_fork``. The source run is only read.

A continuation is a new run that starts at round t+1 from the prefix (``Start``),
against a service that imported the prefix (161 used slots for t=10, B=16):

- ``conversation``: the four sessions fork from the snapshots and continue as the
  source arm.
- ``history``: ``ldm4geo`` with fresh sessions, opening with c0 + rho(prefix).
- ``best_program``: ``restart_best`` with fresh sessions.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness.pi import PiForkSource
from ldm_tts.harness.research_store import snapshot_fork
from ..core.candidate import validate_source
from .controller import Config, Start, profile_name, session_round, turn_root
from .ledger import read, validate as validate_record

SCHEMA = "rsi-checkpoint-v1"
# Mode -> the arm whose controller definition the continuation runs (spec §5 E1).
MODES = {"conversation": "accumulated", "history": "ldm4geo", "best_program": "restart_best"}
# Written into a checkpoint after its snapshot by intervention.select_target; not hashed.
SELECTION_FILE = "intervention-target.json"
_PROFILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")


def checkpoint_dir(run, round_index):
    return Path(run) / "checkpoints" / f"round_{round_index:02d}"


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _inside(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("checkpoint input escaped its run directory")
    return path


def _files(root):
    """Hash of every checkpoint file except the snapshot itself and the later target selection."""
    found = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint must contain ordinary files: " + str(path))
        name = path.relative_to(root).as_posix()
        if path.is_file() and name not in ("snapshot.json", SELECTION_FILE):
            found[name] = _sha(path.read_bytes())
    return found


def _put(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _sessions(run, config, round_index, target):
    """Freeze each branch's session as it stood after its round-t turn."""
    found = {}
    for branch in range(config.policy.branches):
        root = turn_root(run, config, round_index, branch)
        commits = [json.loads(p.read_text()) for p in sorted((root / "turns").glob("*/turn_committed.json"))]
        if not commits:
            continue
        last = max(commits, key=lambda c: c["roundIndex"])
        if len({c["profileId"] for c in commits}) != 1 or _PROFILE.fullmatch(last["profileId"]) is None:
            raise ValueError("branch turn root holds an unexpected session profile")
        if last["roundIndex"] != round_index:
            raise ValueError("branch session is not at the checkpoint round; take checkpoints during the run")
        cursor = last["historyToSeq"]
        if cursor != session_round(config, round_index) + 1:
            raise ValueError("branch session history differs from the controller schedule")
        session = _inside(root, root / last["artifacts"]["session"])
        workspace = _inside(root, root / "sessions" / last["profileId"] / "workspace")
        fork = snapshot_fork(session, workspace, target / profile_name(branch), cursor)
        found[profile_name(branch)] = {"sidecar_profile": last["profileId"], **fork.to_dict()}
    if found and len(found) != config.policy.branches:
        raise ValueError("some branches have no committed session to checkpoint")
    return found


def take_checkpoint(run, round_index):
    """Copy round 0..t of a run into checkpoints/round_tt (idempotent; atomic rename)."""
    run = Path(run).resolve()
    final = checkpoint_dir(run, round_index)
    if final.exists():
        return open_checkpoint(final)
    manifest_body = (run / "manifest.json").read_bytes()
    config = Config(**json.loads(manifest_body)["config"])
    if type(round_index) is not int or not 1 <= round_index < config.rounds:
        raise ValueError("checkpoint round must lie inside the campaign")
    if not (run / "usage" / f"round_{round_index:02d}.json").is_file():
        raise ValueError("checkpoint round has not finished its evaluation")
    rows = read(run / "ledger.jsonl")
    prefix = [r for r in rows if r["round"] <= round_index]
    lines = (run / "ledger.jsonl").read_bytes().splitlines(keepends=True)[:len(prefix)]
    if (rows[:len(prefix)] != prefix or [json.loads(line) for line in lines] != prefix
            or sum(r["round"] == round_index for r in prefix) != config.batch):
        raise ValueError("ledger prefix of the checkpoint round is incomplete")
    staging = final.with_name(final.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _put(staging / "ledger-prefix.jsonl", b"".join(lines))
    _put(staging / "candidates.txt", "".join(r["program_id"] + "\n" for r in prefix).encode())
    _put(staging / "source-manifest.json", manifest_body)
    ids = [r["program_id"] for r in prefix]
    for number in range(max(1, round_index - 1), round_index + 1):
        name = f"proposals/round_{number:02d}.jsonl"
        body = (run / name).read_bytes()
        _put(staging / name, body)
        ids += [json.loads(line)["program_id"] for line in body.splitlines()]
    for sha in dict.fromkeys(ids):
        path = run / "candidates" / sha / "solver.py"
        body = path.read_bytes()
        if path.is_symlink() or validate_source(body) != sha:
            raise ValueError("stored candidate changed")
        _put(staging / "candidates" / sha / "solver.py", body)
    encodings = 0
    for row in prefix:
        path = run / "encodings" / f"{row['program_id']}.json"
        if path.is_file() and not path.is_symlink():
            _put(staging / "encodings" / path.name, path.read_bytes())
            encodings += 1
    sessions = _sessions(run, config, round_index, staging / "sessions")
    atomic_json_write(staging / "snapshot.json", {
        "schema": SCHEMA, "source_run": run.name, "round": round_index, "records": len(prefix),
        "ledger_prefix_sha256": _sha(b"".join(lines)), "source_manifest_sha256": _sha(manifest_body),
        "encodings": encodings, "sessions": sessions, "files": _files(staging)})
    staging.rename(final)
    return open_checkpoint(final)


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    snapshot: dict
    sha256: str
    records: tuple
    previous_pool: tuple
    source_manifest: dict

    @property
    def round(self):
        return self.snapshot["round"]

    @property
    def config(self):
        return Config(**self.source_manifest["config"])

    @property
    def source_run(self):
        return self.source_manifest.get("run_id") or self.snapshot["source_run"]

    @property
    def source_seed(self):
        return self.config.seed

    def identity(self, mode, replicate):
        """The manifest ``continuation`` field (contracts §7.2)."""
        return {"source_run": self.source_run, "round": self.round, "mode": mode,
                "replicate": replicate, "snapshot_sha256": self.sha256}


def open_checkpoint(path):
    """Load a checkpoint after checking every file against its snapshot."""
    path = Path(path).resolve()
    body = (path / "snapshot.json").read_bytes()
    snapshot = json.loads(body)
    if snapshot.get("schema") != SCHEMA or _files(path) != snapshot["files"]:
        raise ValueError("checkpoint files differ from its snapshot")
    prefix = (path / "ledger-prefix.jsonl").read_bytes()
    records = [validate_record(json.loads(line)) for line in prefix.splitlines()]
    t = snapshot["round"]
    if (_sha(prefix) != snapshot["ledger_prefix_sha256"] or len(records) != snapshot["records"]
            or [r["seq"] for r in records] != list(range(len(records))) or records[-1]["round"] != t):
        raise ValueError("checkpoint ledger prefix differs from its snapshot")
    manifest = (path / "source-manifest.json").read_bytes()
    if _sha(manifest) != snapshot["source_manifest_sha256"]:
        raise ValueError("checkpoint source manifest differs from its snapshot")
    for profile, entry in snapshot["sessions"].items():
        saved = json.loads((path / "sessions" / profile / "snapshot.json").read_text())
        if saved != {k: entry[k] for k in ("sessionSha256", "workspaceSha256", "historyCursor")}:
            raise ValueError("checkpoint session snapshot differs")
    pool = [json.loads(line) for line in (path / f"proposals/round_{t:02d}.jsonl").read_text().splitlines()]
    return Checkpoint(path, snapshot, _sha(body), tuple(records), tuple(pool), json.loads(manifest))


class CheckpointingAgent:
    """Delegate every turn; before the first turn after round t, take the round-t checkpoint once.

    The controller asks for round t+1 only after round t's evaluation and usage
    record, and no round-t turn is still running, so the copy sees a settled run.
    Sibling branches wait for the copy, then proceed unchanged.
    """

    def __init__(self, agent, run, round_index):
        self.agent, self.run, self.round = agent, Path(run), round_index
        self._lock, self._done = threading.Lock(), False

    def propose(self, *, round_index, **settings):
        if round_index > self.round and not self._done:
            with self._lock:
                if not self._done:
                    take_checkpoint(self.run, self.round)
                    self._done = True
        return self.agent.propose(round_index=round_index, **settings)

    def __getattr__(self, name):
        return getattr(self.__dict__["agent"], name)


def check_config(checkpoint, config, mode):
    """A continuation keeps every source setting except the arm definition of its mode."""
    if mode not in MODES:
        raise ValueError("unknown continuation mode: " + str(mode))
    def shared(value):
        return {k: v for k, v in asdict(value).items() if k not in ("arm", "policy")}
    source = checkpoint.config
    if shared(source) != shared(config):
        raise ValueError("continuation settings differ from the source run")
    if mode == "conversation" and asdict(source.policy) != asdict(config.policy):
        raise ValueError("a conversation continuation runs the source arm's definition")
    if config.rounds <= checkpoint.round:
        raise ValueError("continuation has no rounds left after its checkpoint")


def continuation_start(checkpoint, mode):
    if mode not in MODES:
        raise ValueError("unknown continuation mode: " + str(mode))
    if mode == "conversation" and not checkpoint.snapshot["sessions"]:
        raise ValueError("checkpoint holds no sessions to fork")
    return Start(checkpoint.round + 1, checkpoint.records, str(checkpoint.path), checkpoint.previous_pool,
                 fresh_sessions=mode != "conversation")


def check_service(info, checkpoint, *, resume=False):
    """A new continuation needs a service that imported exactly the prefix."""
    used, prefix = info["used_slots"], len(checkpoint.records)
    if "import_ledger_prefix_sha256" in info and info["import_ledger_prefix_sha256"] != checkpoint.snapshot[
            "ledger_prefix_sha256"]:
        raise ValueError("continuation service imported a different ledger prefix")
    if used < prefix or (not resume and used != prefix):
        raise ValueError(f"continuation service must start with the {prefix} imported prefix slots, has {used}")


def place_forks(controller, checkpoint):
    """Put each branch's frozen session where the sidecar initializes its fork."""
    first, config = controller.start.round, controller.config
    sessions = checkpoint.snapshot["sessions"]
    if set(sessions) != {profile_name(b) for b in range(config.policy.branches)}:
        raise ValueError("checkpoint sessions do not cover the branches")
    for branch in range(config.policy.branches):
        entry = sessions[profile_name(branch)]
        expected = PiForkSource(entry["sessionSha256"], entry["workspaceSha256"], entry["historyCursor"])
        if _PROFILE.fullmatch(entry["sidecar_profile"]) is None:
            raise ValueError("invalid sidecar profile in checkpoint")
        if expected.history_cursor != session_round(config, first, controller.start):
            raise ValueError("forked session history differs from the continuation schedule")
        source = checkpoint.path / "sessions" / profile_name(branch)
        target = controller.turn_root(first, branch) / "fork-sources" / entry["sidecar_profile"]
        if snapshot_fork(source / "session.jsonl", source / "workspace", target, expected.history_cursor) != expected:
            raise ValueError("placed fork differs from the checkpoint")


def fork_sources(turn_root):
    """Fork sources placed in a turn root, for the agent's PiHarnessConfig (pass on every turn there)."""
    found = {}
    for path in sorted((Path(turn_root) / "fork-sources").glob("*/snapshot.json")):
        saved = json.loads(path.read_text())
        found[path.parent.name] = PiForkSource(saved["sessionSha256"], saved["workspaceSha256"],
                                               saved["historyCursor"])
    return found


def seed_encodings(run, checkpoint):
    """Reuse the source's prefix encodings so every mode fits the surrogate on identical vectors."""
    for path in sorted((checkpoint.path / "encodings").glob("*.json")):
        target = Path(run) / "encodings" / path.name
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise ValueError("continuation encoding differs from the checkpoint")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(path.read_bytes())
        os.replace(temporary, target)


def run_continuation(controller, checkpoint, mode, *, resume=False):
    """Start (or resume) a continuation run whose controller was built with ``continuation_start``."""
    check_config(checkpoint, controller.config, mode)
    if controller.start is None or controller.start.identity() != continuation_start(checkpoint, mode).identity():
        raise ValueError("controller start differs from the checkpoint")
    if not resume:
        controller.prepare()
    seed_encodings(controller.root, checkpoint)
    if mode == "conversation":
        place_forks(controller, checkpoint)
    return controller.run(resume=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Checkpoints of rsi runs (contracts §7.1)")
    commands = parser.add_subparsers(dest="command", required=True)
    take = commands.add_parser("checkpoint", help="copy a finished round of a run without sessions advanced")
    take.add_argument("--run", type=Path, required=True)
    take.add_argument("--round", type=int, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    value = take_checkpoint(args.run, args.round) if args.command == "checkpoint" else open_checkpoint(args.checkpoint)
    print(json.dumps({"checkpoint": str(value.path), "snapshot_sha256": value.sha256, "round": value.round,
                      "records": len(value.records), "sessions": sorted(value.snapshot["sessions"])},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
