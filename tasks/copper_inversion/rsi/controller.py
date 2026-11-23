"""Evidence-only epoch controller driven by the six controller fields (contracts §2.2).

Thread-safety contract for ``agent.propose`` (one call per branch, run concurrently):

- Calls for different branches of one round run at the same time in separate
  threads; calls for one branch never overlap. The controller waits for every
  branch, then admits results in branch order, so completion order is irrelevant.
- Each call receives its own ``turn_root`` (``harness/epoch_kk/branch_b``) and
  ``profile`` (``researcher_<b>``). The agent may write only below ``turn_root``;
  sidecar container names, session ids and temporary paths must be derived from
  the run, ``profile`` and ``turn_root`` so that branches never collide.
- ``candidate_root`` is read-only for the agent (the controller seals sources),
  ``view`` is a private list; neither may be retained. ``known`` is the round's
  ``RoundClaims``: an agent that validates submissions inside the turn claims its
  sources there, so a source another branch already submitted is refused in the
  turn (contracts §5: "not novel") instead of failing admission afterwards.
  Shared agent attributes (key, image, catalog export) must be read-only after
  construction; any shared cache must tolerate concurrent readers and writers.
- The call returns ``(proposals, usage)`` for its own profile only. A call may be
  repeated after a crash when no receipt exists; it must accept existing files in
  ``turn_root``. Exceptions propagate; sibling results are still kept.
"""

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ..core.candidate import seal, validate_source
from ..core.execution import source_identity
from .arms import Arm, policy
from .case_isolation import check_source_case_ids
from .exposure import validate_feedback
from .ledger import append, read, validate as validate_record
from .intervention import identity as intervention_identity, validate as validate_intervention
from .render import render, stripped_source, write_view
from .selection import reserve_and_rank


INITIAL_SOURCE = (b"from tasks.copper_inversion.v2.baselines import initialize\n"
                  b"def solve(problem,budget,seed):\n    return initialize(problem)\n")
LOOP_KEYS = ("rounds", "branches", "proposals_per_branch", "batch", "reserved_per_branch",
             "reset_period", "finalists", "harness_seconds", "bash_calls")
USAGE_COUNTS = ("requests", "input_tokens", "cached_input_tokens", "output_tokens",
                "responses_missing_usage", "compactions")


SCHEDULING = frozenset({"evaluation_concurrency"})  # Config fields that never change results


def profile_name(branch):
    return f"researcher_{branch}"


@dataclass(frozen=True)
class Config:
    arm: str
    seed: int
    rounds: int = 20
    proposals_per_branch: int = 8
    batch: int = 16
    reset_period: int = 5
    harness_seconds: int = 1800
    token_cap_input: int = 200_000_000  # 0: no cap
    policy: Arm = None  # defaults to the named arm of the built-in table
    reserved_per_branch: int = 1
    finalists: int = 3
    bash_calls: int = 8
    token_cap_counts_cached: bool = True  # open author decision; prototype counts cached input
    tier: str = None
    solver_seconds: int = None
    model: dict = None
    evaluation_concurrency: int = 1  # scheduling only: not in the run identity, may change on resume

    def __post_init__(self):
        if not isinstance(self.arm, str) or not self.arm:
            raise ValueError("invalid controller setting: arm")
        if self.policy is None:
            object.__setattr__(self, "policy", policy(self.arm))
        elif isinstance(self.policy, dict):
            object.__setattr__(self, "policy", Arm.from_dict(self.policy))
        elif not isinstance(self.policy, Arm):
            raise ValueError("invalid controller setting: policy")
        lower = {"seed": 0, "token_cap_input": 0, "reserved_per_branch": 0, "bash_calls": 0}
        for name in ("seed", "rounds", "proposals_per_branch", "batch", "reset_period", "harness_seconds",
                     "token_cap_input", "reserved_per_branch", "finalists", "bash_calls", "evaluation_concurrency"):
            value = getattr(self, name)
            if type(value) is not int or value < lower.get(name, 1):
                raise ValueError("invalid controller setting: " + name)
        if type(self.token_cap_counts_cached) is not bool:
            raise ValueError("invalid controller setting: token_cap_counts_cached")
        if self.tier is not None and (not isinstance(self.tier, str) or not self.tier):
            raise ValueError("invalid controller setting: tier")
        if self.solver_seconds is not None and (type(self.solver_seconds) is not int or self.solver_seconds < 1):
            raise ValueError("invalid controller setting: solver_seconds")
        if self.model is not None and (not isinstance(self.model, dict)
                                       or not {"base_url", "name", "thinking"} <= set(self.model)
                                       or any(not isinstance(k, str) or not isinstance(v, str)
                                              for k, v in self.model.items())):
            raise ValueError("invalid controller setting: model")
        pool = self.policy.turns_per_round * self.proposals_per_branch
        if not self.policy.branches * min(self.reserved_per_branch, pool) <= self.batch <= self.policy.branches * pool:
            raise ValueError("batch must hold the per-branch reservations and fit the proposal pool")

    @property
    def slots(self):
        return 1 + self.rounds * self.batch

    @classmethod
    def from_protocol(cls, protocol, arm, tier, *, seed=0):
        """Controller settings from protocol.json (contracts §2.1): loop, tiers[tier], cap, model, arms."""
        try:
            loop, settings, definition = protocol["loop"], protocol["tiers"][tier], protocol["arms"][arm]
            cap, model, seconds = protocol["token_cap_input"], protocol["model"], settings["solver_seconds"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"protocol lacks controller setting {exc}") from exc
        if not isinstance(loop, dict) or set(loop) != set(LOOP_KEYS):
            raise ValueError("protocol loop requires exactly: " + ", ".join(LOOP_KEYS))
        config = cls(arm, seed, rounds=loop["rounds"], proposals_per_branch=loop["proposals_per_branch"],
                     batch=loop["batch"], reset_period=loop["reset_period"],
                     harness_seconds=loop["harness_seconds"], token_cap_input=cap,
                     policy=Arm.from_dict(definition), reserved_per_branch=loop["reserved_per_branch"],
                     finalists=loop["finalists"], bash_calls=loop["bash_calls"],
                     token_cap_counts_cached=protocol.get("token_cap_counts_cached", True),
                     tier=tier, solver_seconds=seconds, model=dict(model) if isinstance(model, dict) else model)
        if "slots" in settings and settings["slots"] != config.slots:
            raise ValueError("tier slot count differs from 1 + rounds * batch")
        return config


@dataclass(frozen=True)
class Start:
    """Begin a run at ``round`` from a preloaded ledger prefix (continuations, contracts §7.2).

    ``records`` are the charged rows of rounds < ``round`` (seq 0..n-1); their sources are
    copied from ``candidate_root/candidates``. ``previous_pool`` holds the proposal rows of
    round ``round - 1`` (only ``program_id`` and ``branch`` are used, for ``unevaluated``).
    ``fresh_sessions``: sessions begin at ``round`` with a boundary message (history and
    best_program modes); otherwise sessions forked into ``turn_root(round, b)`` continue.
    """
    round: int
    records: tuple
    candidate_root: str
    previous_pool: tuple = ()
    fresh_sessions: bool = False

    def __post_init__(self):
        records = tuple(validate_record(dict(r)) for r in self.records)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "previous_pool", tuple(dict(p) for p in self.previous_pool))
        if (type(self.round) is not int or self.round < 1 or not records or records[0]["round"] != 0
                or [r["seq"] for r in records] != list(range(len(records)))
                or any(r["round"] >= self.round for r in records)
                or any(p.get("branch") is None or not p.get("program_id") for p in self.previous_pool)
                or type(self.fresh_sessions) is not bool):
            raise ValueError("invalid continuation start")

    def identity(self):
        return {"round": self.round, "records": len(self.records), "fresh_sessions": self.fresh_sessions,
                "records_sha256": _digest(list(self.records)),
                "previous_pool_sha256": _digest([[p["branch"], p["program_id"]] for p in self.previous_pool])}


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


class NoSubmission(Exception):
    """A completed branch turn yielded no accepted proposals; preserve its usage."""
    def __init__(self, usage, *, cursor_advanced=False):
        if type(cursor_advanced) is not bool:
            raise ValueError("cursor_advanced must be a boolean")
        super().__init__("no accepted submission")
        self.usage = usage
        self.cursor_advanced = cursor_advanced


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def epoch_index(config, round_index):
    """Ledger epoch: K-based for every arm, so rendered histories agree across arms."""
    return 0 if round_index == 0 else (round_index - 1) // config.reset_period


def harness_epoch(config, round_index):
    """Harness directory index; a new directory is a new session and workspace."""
    clear = config.policy.clear
    return round_index - 1 if clear == "every_round" else epoch_index(config, round_index) if clear == "every_k" else 0


def epoch_start(config, round_index):
    if config.policy.clear == "every_round":
        return round_index
    return epoch_index(config, round_index) * config.reset_period + 1


def is_boundary(config, round_index, start=None):
    return (config.policy.clear == "every_round" or (round_index - 1) % config.reset_period == 0
            or (start is not None and start.fresh_sessions and round_index == start.round))


def session_round(config, round_index, start=None, turn_index=0):
    """Turns already in the session that receives this turn."""
    clear = config.policy.clear
    first = round_index if clear == "every_round" else 1 if clear == "never" else epoch_start(config, round_index)
    if start is not None and start.fresh_sessions and round_index >= start.round:
        first = max(first, start.round)
    return (round_index - first) * config.policy.turns_per_round + turn_index


def actual_session_round(root, config, round_index, branch, start=None):
    """Uncommitted no-submission turns do not advance the sidecar's cursor."""
    nominal = session_round(config, round_index, start)
    current = turn_root(root, config, round_index, branch)
    first = round_index - nominal
    if start is not None and not start.fresh_sessions and round_index >= start.round:
        fork = current / "fork-sources" / profile_name(branch) / "snapshot.json"
        if fork.is_file():
            nominal = json.loads(fork.read_text())["historyCursor"] + round_index - start.round
            first = start.round
    for previous in range(max(1, first), round_index):
        if turn_root(root, config, previous, branch) != current:
            continue
        receipt = Path(root) / "proposals" / f"round_{previous:02d}" / f"branch_{branch}.json"
        if receipt.is_file():
            value = json.loads(receipt.read_text())
            if value.get("status") == "no_submission" and not value.get("cursor_advanced", False):
                nominal -= 1
    if nominal < 0:
        raise ValueError("no-submission receipts produce a negative session cursor")
    return nominal


def turn_root(root, config, round_index, branch):
    return Path(root) / "harness" / f"epoch_{harness_epoch(config, round_index):02d}" / f"branch_{branch}"


def _write_once(path, content):
    """Publish a complete file once (temporary inode plus link); a replay must match it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = content.encode("utf-8")
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    if path.read_bytes() != body:
        raise ValueError("saved branch message changed on replay")


def _replace_text(path, content):
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def _best(rows):
    choices = [r for r in rows if r["status"] == "valid"]
    if not choices:
        raise ValueError("no valid measured program for restart")
    return min(choices, key=lambda r: (r["risk"], r["program_id"]))


def _source(root, row):
    path = root / "candidates" / row["program_id"] / "solver.py"
    if path.is_symlink() or validate_source(path.read_bytes()) != row["program_id"]:
        raise ValueError("stored candidate changed")
    return stripped_source(path.read_bytes())


def _fed(config, rows, branch):
    feed = config.policy.round_feed
    return list(rows) if feed == "all" else [r for r in rows if r["branch"] == branch] if feed == "own" else []


def _view(config, root, prefix, branch, round_index):
    start = epoch_start(config, round_index)
    payload = config.policy.boundary_payload
    within = _fed(config, [r for r in prefix if r["round"] >= start], branch)
    if payload == "history":
        return [r for r in prefix if r["round"] < start] + within
    if payload == "initial_program":
        return [r for r in prefix if r["round"] == 0] + within
    if payload == "best_program":
        return [_best([r for r in prefix if r["round"] < start])] + within
    raise ValueError("unsupported boundary payload: " + payload)


def make_message(config, root, prefix, branch, round_index, task_brief, previous_pool, intervention=None,
                 start=None):
    payload = config.policy.boundary_payload
    view = _view(config, root, prefix, branch, round_index)
    boundary = is_boundary(config, round_index, start)
    message = {"kind": "boundary" if boundary else "round", "round": round_index,
               "candidates_required": config.proposals_per_branch,
               "research_wall_time_seconds": config.harness_seconds}
    if boundary:
        if payload == "history":
            brief = render(view)
        else:
            reference = view[0]
            brief = _source(root, reference)
            if payload == "best_program":
                brief += "\n" + render([reference])
        message["brief"] = task_brief.rstrip() + "\n\n" + brief
    else:
        latest = [r for r in prefix if r["round"] == round_index - 1]
        message["new_records"] = render(_fed(config, latest, branch)) if config.policy.round_feed != "none" else ""
        evaluated = {r["program_id"] for r in latest}
        message["unevaluated"] = [p["program_id"][:12] for p in previous_pool
                                  if p["branch"] == branch and p["program_id"] not in evaluated]
    if intervention is not None and intervention["round"] == round_index:
        message["carried_note"] = intervention["note"]
    return message, view


def _validate_submission(row):
    expected = {"source", "parent_sha256", "change_summary", "rationale", "predicted_risk",
                "interval80_low", "interval80_high"}
    if not isinstance(row, dict) or set(row) != expected or not isinstance(row["source"], str):
        raise ValueError("invalid proposal submission")
    for name, limit in (("change_summary", 1000), ("rationale", 2000)):
        if not isinstance(row[name], str) or not 1 <= len(row[name]) <= limit:
            raise ValueError("invalid proposal annotation")
    if row["parent_sha256"] is not None and (not isinstance(row["parent_sha256"], str)
                                             or len(row["parent_sha256"]) != 64):
        raise ValueError("invalid proposed parent")
    values = [row[k] for k in ("interval80_low", "predicted_risk", "interval80_high")]
    if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or values != sorted(values)):
        raise ValueError("invalid predicted risk interval")


def _validate_usage(usage):
    """Counts as reported by the agent; missing provider usage is counted there, never estimated."""
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in USAGE_COUNTS):
        raise ValueError("invalid provider usage record")
    seconds = usage.get("wall_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid provider usage record")
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        raise ValueError("cached input exceeds input tokens")
    return usage


def unsupported(arm):
    """Reason the controller cannot run this definition, or None."""
    if arm.turns_per_round != 1:
        return "turns_per_round > 1"
    if arm.budget_split != "shared":
        return "budget_split = separate"
    if arm.boundary_payload == "parents":
        return "boundary_payload = parents"
    return None


class RoundClaims:
    """Sources taken in one round: the known ones plus each branch's accepted submission.

    Thread-safe. ``claim`` succeeds only if no source is known or held by another
    branch; a branch may re-claim its own. Iteration and ``in`` see every source.
    """

    def __init__(self, known):
        self._known, self._owner, self._lock = frozenset(known), {}, threading.Lock()

    def excluding(self, branch):
        with self._lock:
            return self._known | {sha for sha, owner in self._owner.items() if owner != branch}

    def claim(self, branch, shas):
        with self._lock:
            if any(sha in self._known or self._owner.get(sha, branch) != branch for sha in shas):
                return False
            self._owner.update((sha, branch) for sha in shas)
            return True

    def __contains__(self, sha):
        with self._lock:
            return sha in self._known or sha in self._owner

    def __iter__(self):
        return iter(self.excluding(None))

    def __len__(self):
        return len(self.excluding(None))


class Controller:
    """Run or replay a campaign; each completed step has a durable receipt."""

    def __init__(self, root, config, agent, evaluator, select, task_brief, identity, intervention=None,
                 start=None, forbidden_case_tokens=()):
        self.root, self.config = Path(root).resolve(), config
        self.agent, self.evaluator, self.select = agent, evaluator, select
        self.task_brief = task_brief
        self.forbidden_case_tokens = frozenset(forbidden_case_tokens)
        reason = unsupported(config.policy)
        if reason is not None:
            raise ValueError(f"controller does not implement {reason} (extension arm)")
        self.intervention = None if intervention is None else validate_intervention(intervention, config.rounds)
        if self.intervention is not None and (config.policy.boundary_payload != "history"
                                              or config.policy.clear == "every_round"):
            raise ValueError("planted notes require an evidence-only or accumulated arm")
        if start is not None and (not isinstance(start, Start) or start.round > config.rounds):
            raise ValueError("invalid continuation start")
        self.start = start
        if {"schema", "config", "implementation_sha256", "task_brief_sha256", "intervention", "start"} & set(identity):
            raise ValueError("run identity cannot override controller fields")
        scientific = {k: v for k, v in asdict(config).items() if k not in SCHEDULING}
        self.identity = {"schema": "rsi-controller-v2", "config": scientific,
                         "implementation_sha256": source_identity(),
                         "task_brief_sha256": hashlib.sha256(task_brief.encode()).hexdigest(),
                         "intervention": intervention_identity(self.intervention) if self.intervention else None,
                         **identity}
        if start is not None:
            self.identity["start"] = start.identity()
        self.ledger = self.root / "ledger.jsonl"

    def turn_root(self, round_index, branch):
        return turn_root(self.root, self.config, round_index, branch)

    def _manifest(self, resume):
        path = self.root / "manifest.json"
        if resume:
            if not path.is_file() or json.loads(path.read_text()) != self.identity:
                raise ValueError("resume identity differs from the original run")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            atomic_json_write(path, self.identity)
        _write_once(self.root / "task_brief.txt", self.task_brief)
        if self.intervention:
            _write_once(self.root / "intervention/note.txt", self.intervention["note"])

    def _initialize(self):
        """Charge the initial solver, or import a continuation prefix (idempotent)."""
        rows = read(self.ledger)
        if self.start is None:
            if not rows:
                payload = seal(self.root, INITIAL_SOURCE)
                source = self.root / payload["source_path"]
                self._record(self.evaluator.evaluate_source(source), round_index=0, branch=-1, reason="initial")
            return
        prefix = list(self.start.records)
        if rows[:len(prefix)] != prefix[:len(rows)]:
            raise ValueError("ledger differs from the continuation prefix")
        source_root = Path(self.start.candidate_root)
        for row in prefix[len(rows):]:
            path = source_root / "candidates" / row["program_id"] / "solver.py"
            if path.is_symlink():
                raise ValueError("candidate source cannot be a symlink")
            if seal(self.root, path.read_bytes())["sha256"] != row["program_id"]:
                raise ValueError("continuation source differs from its record")
            append(self.ledger, row)
        if self.start.previous_pool:
            folder = self.root / "proposals" / f"round_{self.start.round - 1:02d}"
            for branch in sorted({p["branch"] for p in self.start.previous_pool}):
                body = {"imported": True, "message_sha256": None, "usage": None,
                        "proposals": [p for p in self.start.previous_pool if p["branch"] == branch]}
                _write_once(folder / f"branch_{branch}.json", _canonical(body) + "\n")

    def prepare(self):
        """Create the run directory and its first records; sessions may then be placed and run(resume=True)."""
        self._manifest(False)
        self._initialize()

    def _record(self, feedback, *, round_index, branch, reason):
        feedback = validate_feedback(feedback)
        rows = read(self.ledger)
        previous = next((r for r in rows if r["program_id"] == feedback["source_sha256"]), None)
        row = {"seq": len(rows), "program_id": feedback["source_sha256"],
               "epoch": epoch_index(self.config, round_index),
               "round": round_index, "branch": branch, "selection": reason,
               **{k: v for k, v in feedback.items() if k != "source_sha256"}}
        if previous is not None:
            row["seq"] = previous["seq"]
        return append(self.ledger, row)

    def _turn(self, round_index, branch, job, known):
        started = time.monotonic()
        values, usage = self.agent.propose(
            branch=branch, profile=profile_name(branch), round_index=round_index, turn_index=0,
            epoch=harness_epoch(self.config, round_index), count=self.config.proposals_per_branch,
            session_round=actual_session_round(self.root, self.config, round_index, branch, self.start),
            message=job["message"], view=list(job["view"]), turn_root=job["turn_root"],
            candidate_root=self.root, known=known, seed=self.config.seed,
            harness_seconds=self.config.harness_seconds)
        usage = dict(usage)
        usage.setdefault("wall_seconds", time.monotonic() - started)
        return values, usage

    def _admit(self, values, branch, view, claimed):
        if not isinstance(values, list) or len(values) != self.config.proposals_per_branch:
            raise ValueError("researcher did not submit the required proposal count")
        visible, local, rows = {r["program_id"] for r in view}, set(claimed), []
        for value in values:
            _validate_submission(value)
            body = value["source"].encode("utf-8")
            sha = validate_source(body)
            check_source_case_ids(body, self.forbidden_case_tokens)
            if sha in local:
                raise ValueError("source rejected: not novel")
            if value["parent_sha256"] is not None and value["parent_sha256"] not in visible:
                raise ValueError("source rejected: parent is not visible")
            seal(self.root, body)
            local.add(sha)
            rows.append({"program_id": sha, "branch": branch, **{k: v for k, v in value.items() if k != "source"}})
        return rows

    def _proposals(self, round_index, prefix, previous_pool):
        # Sources already known (prefix and saved receipts) plus those claimed by in-turn validation.
        known = frozenset(r["program_id"] for r in prefix)
        folder = self.root / "proposals" / f"round_{round_index:02d}"
        jobs = {}
        for branch in range(self.config.policy.branches):
            message, view = make_message(self.config, self.root, prefix, branch, round_index, self.task_brief,
                                         previous_pool, self.intervention, self.start)
            text = _canonical(message)
            root = self.turn_root(round_index, branch)
            _write_once(root / "briefs" / f"round_{round_index:02d}.json", text + "\n")
            write_view(root / "public" / "evidence", view, self.root)
            jobs[branch] = {"message": text, "view": view, "turn_root": root,
                            "sha": hashlib.sha256(text.encode()).hexdigest(),
                            "receipt": folder / f"branch_{branch}.json"}
        saved, claimed = {}, set(known)
        for branch, job in jobs.items():
            if job["receipt"].exists():
                value = json.loads(job["receipt"].read_text())
                if value["message_sha256"] != job["sha"]:
                    raise ValueError("proposal receipt message differs")
                saved[branch] = value
                claimed.update(r["program_id"] for r in value["proposals"])
        claims = RoundClaims(claimed)
        pending = [b for b in jobs if b not in saved]
        results, errors = {}, {}
        if pending:
            with ThreadPoolExecutor(max_workers=len(pending), thread_name_prefix="rsi-branch") as pool:
                futures = {b: pool.submit(self._turn, round_index, b, jobs[b], claims) for b in pending}
            for branch in pending:
                try:
                    results[branch] = futures[branch].result()
                except NoSubmission as exc:
                    results[branch] = (None, exc.usage, exc.cursor_advanced)
                except Exception as exc:
                    errors[branch] = exc
        for branch in sorted(results):  # admission order is the branch order, never completion order
            result = results[branch]
            values, usage = result[:2]
            try:
                rows = [] if values is None else self._admit(values, branch, jobs[branch]["view"], claimed)
                _validate_usage(usage)
            except (ValueError, SyntaxError, UnicodeError) as exc:
                errors[branch] = exc
                continue
            claimed.update(r["program_id"] for r in rows)
            saved[branch] = {"message_sha256": jobs[branch]["sha"], "proposals": rows, "usage": usage}
            if values is None:
                saved[branch].update(status="no_submission", cursor_advanced=result[2])
            atomic_json_write(jobs[branch]["receipt"], saved[branch])
        if errors:
            raise errors[min(errors)]
        all_rows, usages, seen = [], {}, set(known)
        for branch in sorted(saved):
            rows, usage = saved[branch]["proposals"], _validate_usage(saved[branch]["usage"])
            status = saved[branch].get("status", "accepted")
            if status not in ("accepted", "no_submission") or (status == "accepted" and len(rows) != self.config.proposals_per_branch) or (status == "no_submission" and (rows or type(saved[branch].get("cursor_advanced")) is not bool)):
                raise ValueError("saved proposal batch is incomplete")
            for row in rows:
                if row["branch"] != branch or row["program_id"] in seen:
                    raise ValueError("saved proposal identity differs")
                path = self.root / "candidates" / row["program_id"] / "solver.py"
                if path.is_symlink() or validate_source(path.read_bytes()) != row["program_id"]:
                    raise ValueError("saved proposal source changed")
                seen.add(row["program_id"])
            all_rows.extend(rows)
            usages[profile_name(branch)] = usage
        _replace_text(self.root / "proposals" / f"round_{round_index:02d}.jsonl",
                      "".join(_canonical(r) + "\n" for r in all_rows))
        return all_rows, usages

    def _selection(self, round_index, pool, prefix, existing):
        path = self.root / "selection" / f"round_{round_index:02d}.json"
        batch, reserve = min(self.config.batch, len(pool)), self.config.reserved_per_branch
        if path.exists():
            chosen = json.loads(path.read_text())
        else:
            if any(r["round"] == round_index for r in existing):
                raise ValueError("selected batch missing during recovery")
            scored = self.select(pool, prefix, batch) if batch else {"predictions": [], "selected": []}
            chosen = {**scored, "selected": reserve_and_rank(scored["predictions"], batch, reserve) if batch else []}
            atomic_json_write(path, chosen)
        selected = chosen["selected"]
        pool_map = {p["program_id"]: p["branch"] for p in pool}
        prediction_map = {p["program_id"]: p["branch"] for p in chosen["predictions"]}
        per_branch = Counter(p["branch"] for p in pool)
        if (len(chosen["predictions"]) != len(pool) or prediction_map != pool_map
                or len(selected) != batch
                or len({p["program_id"] for p in selected}) != len(selected)
                or any(pool_map.get(p["program_id"]) != p["branch"] for p in selected)
                or Counter(p["branch"] for p in selected if p["selection"] == "reserved")
                != Counter({b: min(reserve, n) for b, n in per_branch.items() if min(reserve, n)})
                or (reserve_and_rank(chosen["predictions"], batch, reserve) if batch else []) != selected):
            raise ValueError("selection receipt is inconsistent with proposal pool")
        return selected

    def _evaluate(self, round_index, selected, existing):
        """Evaluate with bounded concurrency; records are appended strictly in selection order.

        A record's seq is its position in the selection, whatever the completion order. After a
        failure, queued evaluations are cancelled, in-flight ones finish, and only the completed
        prefix before the first failure is recorded; the rest are resumed (the client replays
        saved feedback, so nothing charged is evaluated twice).
        """
        recorded, pending = {r["program_id"]: r for r in existing}, []
        for item in selected:
            row = recorded.get(item["program_id"])
            if row is not None:  # charged before an interruption; never evaluated twice
                if pending or (row["round"], row["branch"], row["selection"]) != (
                        round_index, item["branch"], item["selection"]):
                    raise ValueError("replayed record changed")
                continue
            pending.append(item)

        def evaluate(item):
            return self.evaluator.evaluate_source(self.root / "candidates" / item["program_id"] / "solver.py")

        def record(item, feedback):
            self._record(feedback, round_index=round_index, branch=item["branch"], reason=item["selection"])

        workers = min(self.config.evaluation_concurrency, len(pending))
        if workers <= 1:
            for item in pending:
                record(item, evaluate(item))
            return
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rsi-evaluation")
        try:
            futures = [pool.submit(evaluate, item) for item in pending]
            for item, future in zip(pending, futures):
                record(item, future.result())
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def run(self, *, resume=False):
        self._manifest(resume)
        self._initialize()
        rows = read(self.ledger)
        if (rows[0]["round"] != 0 or any(a["round"] > b["round"] for a, b in zip(rows, rows[1:]))
                or rows[-1]["round"] > self.config.rounds):
            raise ValueError("ledger round sequence differs from the campaign")
        basis = "input_tokens" if self.config.token_cap_counts_cached else "uncached_input_tokens"
        previous_pool = list(self.start.previous_pool) if self.start else []
        totals = {"input_tokens": 0, "uncached_input_tokens": 0, "responses_missing_usage": 0}
        stop_reason, completed = "completed", 0
        for round_index in range(self.start.round if self.start else 1, self.config.rounds + 1):
            existing = read(self.ledger)
            prefix = [r for r in existing if r["round"] < round_index]
            pool, usages = self._proposals(round_index, prefix, previous_pool)
            previous_pool = pool
            selected = self._selection(round_index, pool, prefix, existing)
            self._evaluate(round_index, selected, existing)
            completed = round_index
            for usage in usages.values():
                totals["input_tokens"] += usage["input_tokens"]
                totals["uncached_input_tokens"] += usage["input_tokens"] - usage["cached_input_tokens"]
                totals["responses_missing_usage"] += usage["responses_missing_usage"]
            usage_record = {"round": round_index, "by_profile": usages,
                            "run_total_input_tokens": totals["input_tokens"],
                            "run_total_uncached_input_tokens": totals["uncached_input_tokens"],
                            "run_total_responses_missing_usage": totals["responses_missing_usage"],
                            "token_cap_basis": basis}
            usage_path = self.root / "usage" / f"round_{round_index:02d}.json"
            if usage_path.exists() and json.loads(usage_path.read_text()) != usage_record:
                raise ValueError("provider usage changed on replay")
            atomic_json_write(usage_path, usage_record)
            # Missing usage is only counted. The cap is checked after the round's evaluation.
            if (self.config.token_cap_input and totals[basis] >= self.config.token_cap_input
                    and round_index < self.config.rounds):
                stop_reason = "token_cap"
                break
        records = read(self.ledger)
        finalists = sorted((r for r in records if r["status"] == "valid"),
                           key=lambda r: (r["risk"], -r["legal_rate"], r["seconds_per_solve"],
                                          r["program_id"]))[:self.config.finalists]
        report = {"status": "completed" if stop_reason == "completed" else "stopped",
                  "stop_reason": stop_reason, "charged_records": len(records), "rounds_completed": completed,
                  "input_tokens": totals["input_tokens"], "uncached_input_tokens": totals["uncached_input_tokens"],
                  "token_cap_basis": basis, "responses_missing_usage": totals["responses_missing_usage"],
                  "training_finalists": [r["program_id"] for r in finalists],
                  "scope": "training feedback only; validation selection and sealed evaluation not run"}
        if self.identity.get("synthetic_test"):
            report["scope"] = "synthetic integration fixture only; not a scientific result"
        atomic_json_write(self.root / "result.json", report)
        return report
