"""Four independent proposal sessions with evidence-only epoch boundaries."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ..core.candidate import seal, validate_source
from ..core.execution import source_identity
from .arms import policy
from .exposure import validate_feedback
from .ledger import append, read
from .intervention import identity as intervention_identity, validate as validate_intervention
from .render import render, stripped_source, visible_history
from .selection import reserve_and_rank


INITIAL_SOURCE = (b"from tasks.copper_inversion.v2.baselines import initialize\n"
                  b"def solve(problem,budget,seed):\n    return initialize(problem)\n")


@dataclass(frozen=True)
class Config:
    arm: str
    seed: int
    rounds: int = 20
    proposals_per_branch: int = 8
    batch: int = 16
    reset_period: int = 5
    harness_seconds: int = 1800
    token_cap_input: int = 200_000_000

    def __post_init__(self):
        policy(self.arm)
        for name in ("seed", "rounds", "proposals_per_branch", "batch", "reset_period",
                     "harness_seconds", "token_cap_input"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError("invalid controller setting: " + name)
        if not 4 <= self.batch <= 4 * self.proposals_per_branch:
            raise ValueError("batch must reserve one slot per branch and fit the proposal pool")

    @property
    def slots(self):
        return 1 + self.rounds * self.batch


def _write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError("saved branch message changed on replay")
    else:
        with path.open("xb") as stream:
            stream.write(body)


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


def _view(config, root, prefix, branch, round_index):
    arm = policy(config.arm)
    start = ((round_index - 1) // config.reset_period) * config.reset_period + 1
    if arm.clear == "every_round":
        start = round_index
    if arm.boundary_payload == "history":
        return visible_history(prefix, branch=branch, epoch_start_round=start)
    if arm.boundary_payload == "initial_program":
        return [r for r in prefix if r["round"] == 0]
    before = [r for r in prefix if r["round"] < start]
    initial = _best(before)
    return [initial] + [r for r in prefix if r["round"] >= start and r["branch"] == branch]


def make_message(config, root, prefix, branch, round_index, task_brief, previous_pool, intervention=None):
    arm = policy(config.arm)
    boundary = arm.clear == "every_round" or (round_index - 1) % config.reset_period == 0
    view = _view(config, root, prefix, branch, round_index)
    message = {"kind": "boundary" if boundary else "round", "round": round_index,
               "candidates_required": config.proposals_per_branch,
               "research_wall_time_seconds": config.harness_seconds}
    if boundary:
        if arm.boundary_payload == "history":
            brief = render(view)
        else:
            reference = view[0]
            brief = _source(root, reference)
            if arm.boundary_payload == "best_program":
                brief += "\n" + render([reference])
        message["brief"] = task_brief.rstrip() + "\n\n" + brief
    else:
        new = [r for r in prefix if r["round"] == round_index - 1 and r["branch"] == branch]
        message["new_records"] = render(new) if arm.round_feed == "own" else ""
        evaluated = {r["program_id"] for r in new}
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


class Controller:
    """Run or replay a campaign; each completed step has a durable receipt."""

    def __init__(self, root, config, agent, evaluator, select, task_brief, identity, intervention=None):
        self.root, self.config = Path(root).resolve(), config
        self.agent, self.evaluator, self.select = agent, evaluator, select
        self.task_brief = task_brief
        self.intervention = None if intervention is None else validate_intervention(intervention, config.rounds)
        if self.intervention is not None and config.arm not in {"ldm4geo", "accumulated"}:
            raise ValueError("planted notes require an evidence-only or accumulated arm")
        if {"schema", "config", "implementation_sha256", "task_brief_sha256", "intervention"} & set(identity):
            raise ValueError("run identity cannot override controller fields")
        self.identity = {"schema": "rsi-controller-v1", "config": asdict(config),
                         "implementation_sha256": source_identity(),
                         "task_brief_sha256": hashlib.sha256(task_brief.encode()).hexdigest(),
                         "intervention": intervention_identity(self.intervention) if self.intervention else None,
                         **identity}
        self.ledger = self.root / "ledger.jsonl"

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

    def _record(self, feedback, *, round_index, branch, reason):
        feedback = validate_feedback(feedback)
        rows = read(self.ledger)
        previous = next((r for r in rows if r["program_id"] == feedback["source_sha256"]), None)
        epoch = 0 if round_index == 0 else (round_index - 1) // self.config.reset_period
        row = {"seq": len(rows), "program_id": feedback["source_sha256"], "epoch": epoch,
               "round": round_index, "branch": branch, "selection": reason,
               **{k: v for k, v in feedback.items() if k != "source_sha256"}}
        if previous is not None:
            row["seq"] = previous["seq"]
        return append(self.ledger, row)

    def _proposals(self, round_index, prefix, previous_pool):
        all_rows, usages = [], {}
        known = {r["program_id"] for r in prefix}
        for branch in range(4):
            message, view = make_message(self.config, self.root, prefix, branch, round_index,
                                         self.task_brief, previous_pool, self.intervention)
            message_text = json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            epoch = ((round_index - 1) if policy(self.config.arm).clear == "every_round"
                     else (round_index - 1) // self.config.reset_period
                     if policy(self.config.arm).clear == "every_k" else 0)
            turn_root = self.root / "harness" / f"epoch_{epoch:02d}" / f"branch_{branch}"
            _write_once(turn_root / "briefs" / f"round_{round_index:02d}.json", message_text + "\n")
            receipt = self.root / "proposals" / f"round_{round_index:02d}" / f"branch_{branch}.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["message_sha256"] != hashlib.sha256(message_text.encode()).hexdigest():
                    raise ValueError("proposal receipt message differs")
                rows, usage = saved["proposals"], saved["usage"]
            else:
                values, usage = self.agent.propose(
                    branch=branch, round_index=round_index, epoch=epoch, count=self.config.proposals_per_branch,
                    session_round=(round_index - 1 if policy(self.config.arm).clear == "never"
                                   else 0 if policy(self.config.arm).clear == "every_round"
                                   else (round_index - 1) % self.config.reset_period),
                    message=message_text, view=view, turn_root=turn_root, candidate_root=self.root,
                    known=frozenset(known), seed=self.config.seed, harness_seconds=self.config.harness_seconds)
                if len(values) != self.config.proposals_per_branch:
                    raise ValueError("researcher did not submit the required proposal count")
                rows, admitted = [], set(known)
                for value in values:
                    _validate_submission(value)
                    body = value["source"].encode("utf-8")
                    sha = validate_source(body)
                    if sha in admitted:
                        raise ValueError("source rejected: not novel")
                    if value["parent_sha256"] is not None and value["parent_sha256"] not in {r["program_id"] for r in view}:
                        raise ValueError("source rejected: parent is not visible")
                    seal(self.root, body)
                    admitted.add(sha)
                    rows.append({"program_id": sha, "branch": branch,
                                 **{k: v for k, v in value.items() if k != "source"}})
                atomic_json_write(receipt, {"message_sha256": hashlib.sha256(message_text.encode()).hexdigest(),
                                            "proposals": rows, "usage": usage})
            if len(rows) != self.config.proposals_per_branch:
                raise ValueError("saved proposal batch is incomplete")
            for row in rows:
                if row["branch"] != branch or row["program_id"] in known:
                    raise ValueError("saved proposal identity differs")
                path = self.root / "candidates" / row["program_id"] / "solver.py"
                if path.is_symlink() or validate_source(path.read_bytes()) != row["program_id"]:
                    raise ValueError("saved proposal source changed")
                known.add(row["program_id"])
            all_rows.extend(rows)
            usages[str(branch)] = usage
        return all_rows, usages

    def run(self, *, resume=False):
        self._manifest(resume)
        rows = read(self.ledger)
        if rows and (rows[0]["round"] != 0
                     or any(a["round"] > b["round"] for a, b in zip(rows, rows[1:]))
                     or rows[-1]["round"] > self.config.rounds):
            raise ValueError("ledger round sequence differs from the campaign")
        if not rows:
            payload = seal(self.root, INITIAL_SOURCE)
            source = self.root / payload["source_path"]
            self._record(self.evaluator.evaluate_source(source), round_index=0, branch=-1, reason="initial")
        previous_pool = []
        total_tokens = 0
        stop_reason = "completed"
        for round_index in range(1, self.config.rounds + 1):
            existing = read(self.ledger)
            prefix = [r for r in existing if r["round"] < round_index]
            pool, usages = self._proposals(round_index, prefix, previous_pool)
            previous_pool = pool
            selection_path = self.root / "selection" / f"round_{round_index:02d}.json"
            if selection_path.exists():
                chosen = json.loads(selection_path.read_text())
            else:
                if any(r["round"] == round_index for r in existing):
                    raise ValueError("selected batch missing during recovery")
                chosen = self.select(pool, prefix, self.config.batch)
                atomic_json_write(selection_path, chosen)
            selected = chosen["selected"]
            pool_map = {p["program_id"]: p["branch"] for p in pool}
            prediction_map = {p["program_id"]: p["branch"] for p in chosen["predictions"]}
            if (len(chosen["predictions"]) != len(pool) or prediction_map != pool_map
                    or len(selected) != self.config.batch
                    or len({p["program_id"] for p in selected}) != len(selected)
                    or any(pool_map.get(p["program_id"]) != p["branch"] for p in selected)
                    or {p["branch"] for p in selected if p["selection"] == "reserved"} != set(range(4))
                    or len([p for p in selected if p["selection"] == "reserved"]) != 4
                    or reserve_and_rank(chosen["predictions"], self.config.batch) != selected):
                raise ValueError("selection receipt is inconsistent with proposal pool")
            for item in selected:
                source = self.root / "candidates" / item["program_id"] / "solver.py"
                self._record(self.evaluator.evaluate_source(source), round_index=round_index,
                             branch=item["branch"], reason=item["selection"])
            total_tokens += sum(v["input_tokens"] for v in usages.values())
            usage_path = self.root / "usage" / f"round_{round_index:02d}.json"
            usage_record = {"round": round_index, "by_profile": usages, "run_total_input_tokens": total_tokens}
            if usage_path.exists() and json.loads(usage_path.read_text()) != usage_record:
                raise ValueError("provider usage changed on replay")
            atomic_json_write(usage_path, usage_record)
            if any(v["responses_missing_usage"] for v in usages.values()):
                stop_reason = "token_usage_unavailable"
                break
            if total_tokens >= self.config.token_cap_input:
                stop_reason = "token_cap"
                break
        records = read(self.ledger)
        finalists = sorted((r for r in records if r["status"] == "valid"),
                           key=lambda r: (r["risk"], -r["legal_rate"], r["seconds_per_solve"], r["program_id"]))[:3]
        report = {"status": "completed" if stop_reason == "completed" else "stopped",
                  "stop_reason": stop_reason, "charged_records": len(records),
                  "input_tokens": total_tokens, "training_finalists": [r["program_id"] for r in finalists],
                  "scope": "training feedback only; validation selection and sealed evaluation not run"}
        if self.identity.get("synthetic_test"):
            report["scope"] = "synthetic integration fixture only; not a scientific result"
        atomic_json_write(self.root / "result.json", report)
        return report
