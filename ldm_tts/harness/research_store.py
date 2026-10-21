"""Durable research plans and immutable native Pi snapshots, shared by tasks."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import canonical_sha256, directory_sha256, file_sha256
from ldm_tts.harness.pi import PiForkSource
from ldm_tts.harness.research_tree import (
    ArchivePolicy, ForkAssignment, ForkContext, ForkPolicy, ResearchNode, ResearchOutcome,
)

@dataclass(frozen=True)
class ResearchMeasurement:
    canonical_key: str
    score: float
    round_index: int

    def __post_init__(self):
        if not self.canonical_key or not math.isfinite(self.score):
            raise ValueError("Research measurements require a key and finite maximization score")


def snapshot_fork(source_session: Path, workspace: Path, target: Path, cursor: int) -> PiForkSource:
    """Freeze conversation and ordinary files atomically; never share writable inodes."""
    manifest = target / "snapshot.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        source = PiForkSource(saved["sessionSha256"], saved["workspaceSha256"], saved["historyCursor"])
        if (source.history_cursor != cursor or file_sha256(target / "session.jsonl") != source.session_sha256
                or directory_sha256(target / "workspace") != source.workspace_sha256):
            raise ValueError("Frozen fork snapshot identity changed")
        return source
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copyfile(source_session, staging / "session.jsonl")
    # Include task-tool exports: inherited messages/scripts can reference these files.
    # Pi replaces only the child's role and skill snapshots during initialization.
    for path in workspace.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Fork workspace must contain ordinary files: {path}")
    shutil.copytree(workspace, staging / "workspace")
    source = PiForkSource(file_sha256(staging / "session.jsonl"), directory_sha256(staging / "workspace"), cursor)
    atomic_json_write(staging / "snapshot.json", source.to_dict())
    staging.rename(target)
    return source


def _node(value: dict) -> ResearchNode:
    return ResearchNode(**{**value, "proposal_keys": tuple(value["proposal_keys"])})


def _outcome(value: dict) -> ResearchOutcome:
    return ResearchOutcome(**{**value, "parent": _node(value["parent"]),
                              "proposal_keys": tuple(value["proposal_keys"])})


class ResearchTree:
    def __init__(self, root: Path, fork_policy: ForkPolicy, archive_policy: ArchivePolicy):
        self.root = root
        self.fork_policy = fork_policy
        self.archive_policy = archive_policy

    def prepare(self, round_index, profiles, history, measurements) -> dict:
        round_root = self.root / "branches" / f"round_{round_index:04d}"
        if round_index < 1 or any(o.round_index >= round_index for o in measurements):
            raise ValueError("Research planning requires observations from earlier rounds")
        previous = self.root / "branches" / f"round_{round_index - 1:04d}" / "tree.json"
        identity = {
            "evidence_version": 2,
            "fork_policy": self.fork_policy.identity(), "archive_policy": self.archive_policy.identity(),
            "profiles": [p.profile_id for p in profiles], "round_index": round_index,
            "history_digest": canonical_sha256(history),
            "measurements_digest": canonical_sha256([asdict(o) for o in measurements]),
            "previous_tree_sha256": file_sha256(previous) if round_index > 1 else None,
        }
        path = round_root / "fork-plan.json"
        if path.exists():
            plan = json.loads(path.read_text())
            if plan["identity"] != identity:
                raise ValueError("Persisted fork plan identity changed")
            return plan
        main_nodes, archive, outcomes = [], [], []
        if round_index > 1:
            state = json.loads(previous.read_text())
            if state["policies"] != {k: identity[k] for k in ("fork_policy", "archive_policy")}:
                raise ValueError("Research tree policies changed during campaign")
            main_nodes = [_node(n) for n in state["main_nodes"]]
            archive = [_node(n) for n in state["archive"]]
            outcomes = [_outcome(o) for o in state["outcomes"]]
        scores = {o.canonical_key: o.score for o in measurements}
        incumbent = max(scores.values(), default=None)
        main_nodes.append(ResearchNode(
            node_id=f"main_{round_index:04d}",
            snapshot=(round_root / "fork-source").relative_to(self.root).as_posix(),
            round_index=round_index, history_cursor=len(history), lineage_id="main",
            parent_id=main_nodes[-1].node_id if main_nodes else None,
            measured_count=len(scores), best_score=incumbent,
        ))
        archive = [replace(n, measured_count=sum(k in scores for k in n.proposal_keys),
                           best_score=max((scores[k] for k in n.proposal_keys if k in scores), default=None))
                   for n in archive]
        archive = list(self.archive_policy.retain(tuple(archive)))
        by_round = {}
        for o in measurements:
            by_round.setdefault(o.round_index, {})[o.canonical_key] = o.score
        for i, outcome in enumerate(outcomes):
            batch_scores = by_round.get(outcome.round_index, {})
            measured_keys = set(outcome.proposal_keys) & batch_scores.keys()
            best_key = max(measured_keys, key=lambda k: (batch_scores[k], k), default=None)
            outcomes[i] = replace(outcome, measured_count=len(measured_keys), best_key=best_key,
                                  best_score=batch_scores[best_key] if best_key is not None else None)
        available = {n.node_id: n for n in (*main_nodes, *archive)}
        assignments = self.fork_policy.plan(ForkContext(
            round_index, tuple(identity["profiles"]), tuple(main_nodes), tuple(archive),
            tuple(outcomes), incumbent))
        if len(assignments) != len(profiles) or {a.profile_id for a in assignments} != set(identity["profiles"]):
            raise ValueError("Fork policy must assign every profile exactly once")
        for a in assignments:
            if a.parent_id not in available or not a.lineage_id or a.purpose not in {"main", "restart", "continue"}:
                raise ValueError("Fork policy returned an invalid parent assignment")
            if a.purpose == "continue" and a.lineage_id != available[a.parent_id].lineage_id:
                raise ValueError("Continuation must preserve its parent lineage")
        plan = {"identity": identity, "main_nodes": [asdict(n) for n in main_nodes],
                "archive": [asdict(n) for n in archive], "assignments": [asdict(a) for a in assignments],
                "outcomes": [asdict(o) for o in outcomes], "incumbent": incumbent}
        atomic_json_write(path, plan)
        return plan

    def materialize(self, plan: dict) -> dict[str, PiForkSource]:
        """Copy selected immutable sources into the pool before any child runs."""
        nodes = {n["node_id"]: _node(n) for n in (*plan["main_nodes"], *plan["archive"])}
        round_root = self.root / "branches" / f"round_{plan['identity']['round_index']:04d}"
        sources = {}
        for value in plan["assignments"]:
            a = ForkAssignment(**value)
            node = nodes[a.parent_id]
            source = (self.root / node.snapshot).resolve()
            if not source.is_relative_to(self.root.resolve()):
                raise ValueError("Research node snapshot escaped tree root")
            # Validate the original too, not just an already-materialized child copy.
            original = snapshot_fork(source / "session.jsonl", source / "workspace", source, node.history_cursor)
            sources[a.profile_id] = snapshot_fork(
                source / "session.jsonl", source / "workspace",
                round_root / "fork-sources" / a.profile_id, node.history_cursor,
            )
            if sources[a.profile_id] != original:
                raise ValueError("Materialized fork differs from selected parent")
        return sources

    def commit(self, plan: dict, sessions: dict, proposal_keys: dict) -> None:
        """Commit all leaves together; retries reuse the same plan and snapshots."""
        round_index = plan["identity"]["round_index"]
        round_root = self.root / "branches" / f"round_{round_index:04d}"
        main_nodes = [_node(n) for n in plan["main_nodes"]]
        archive = [_node(n) for n in plan["archive"]]
        nodes = {n.node_id: n for n in (*main_nodes, *archive)}
        expected = {a["profile_id"] for a in plan["assignments"]}
        if set(sessions) != expected or set(proposal_keys) != expected:
            raise ValueError("Research commit requires every planned branch exactly once")
        replacements = {}
        children = []
        outcomes = list(plan["outcomes"])
        for value in plan["assignments"]:
            a = ForkAssignment(**value)
            keys = set(proposal_keys[a.profile_id])
            outcomes.append(asdict(ResearchOutcome(
                round_index=round_index, profile_id=a.profile_id, purpose=a.purpose,
                parent=nodes[a.parent_id], incumbent=plan["incumbent"], proposal_keys=tuple(sorted(keys)),
            )))
            if a.purpose == "main":
                continue
            parent = replacements.get(a.parent_id, nodes[a.parent_id])
            replacements[a.parent_id] = replace(parent, expansions=parent.expansions + 1, last_expanded=round_index)
            session = (round_root / sessions[a.profile_id]["artifacts"]["session"]).resolve()
            if not session.is_relative_to(round_root.resolve()):
                raise ValueError("Child session escaped round root")
            target = round_root / "nodes" / a.profile_id
            snapshot_fork(session, round_root / "sessions" / a.profile_id / "workspace", target,
                          main_nodes[-1].history_cursor)
            continuing = a.purpose == "continue"
            if continuing:
                keys.update(parent.proposal_keys)
            children.append(ResearchNode(
                node_id=f"branch_{round_index:04d}_{a.profile_id}", snapshot=target.relative_to(self.root).as_posix(),
                round_index=round_index, history_cursor=main_nodes[-1].history_cursor,
                lineage_id=a.lineage_id, parent_id=a.parent_id, proposal_keys=tuple(sorted(keys)),
                measured_count=parent.measured_count if continuing else 0,
                best_score=parent.best_score if continuing else None,
                expansions=parent.expansions + 1 if continuing else 1, last_expanded=round_index,
            ))
        continued = {n.lineage_id for n in children}
        leaves = tuple(n for n in archive if n.lineage_id not in continued) + tuple(children)
        # Keep new leaves pending until next round, when this round's measurements
        # exist. They are not selectable yet; the active archive remains bounded.
        state = {"policies": {k: plan["identity"][k] for k in ("fork_policy", "archive_policy")},
                 "main_nodes": [asdict(replacements.get(n.node_id, n)) for n in main_nodes],
                 "archive": [asdict(n) for n in leaves], "outcomes": outcomes}
        path = round_root / "tree.json"
        if path.exists() and canonical_sha256(json.loads(path.read_text())) != canonical_sha256(state):
            raise ValueError("Committed research tree changed on replay")
        atomic_json_write(path, state)
