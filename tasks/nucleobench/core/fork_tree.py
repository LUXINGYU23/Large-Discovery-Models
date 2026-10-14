"""Durable task-side research tree; selection policies are pure and injectable."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import canonical_sha256, directory_sha256, file_sha256
from ldm_tts.harness.pi import PiForkSource
from ldm_tts.harness.research_tree import (
    ArchivePolicy, ForkAssignment, ForkContext, ForkPolicy, ResearchNode, ResearchOutcome,
)
from tasks.nucleobench.core.candidate import prepare_candidate_payload
from tasks.nucleobench.core.research import serialize_measured_observations


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
    return ResearchNode(**{**value, "proposal_keys": tuple(value["proposal_keys"]),
                           "descriptor": tuple(value["descriptor"])})


def _outcome(value: dict) -> ResearchOutcome:
    return ResearchOutcome(**{**value, "parent": _node(value["parent"]),
                              "proposal_keys": tuple(value["proposal_keys"])})


def _descriptor(payloads, context) -> tuple[float, ...]:
    """Per-position base frequencies; a leaf describes its latest batch only."""
    positions = {p: i for i, p in enumerate(context.editable_positions)}
    reference = ["ACGT".index(context.start_sequence[p]) for p in positions]
    counts = [0.0] * (4 * len(positions))
    for payload in payloads:
        bases = reference.copy()
        for mutation in payload["mutations"]:
            bases[positions[mutation["position"]]] = "ACGT".index(mutation["base"])
        for i, base in enumerate(bases):
            counts[4 * i + base] += 1 / (len(payloads) * math.sqrt(len(positions)))
    return tuple(counts)


class ResearchTree:
    def __init__(self, root: Path, fork_policy: ForkPolicy, archive_policy: ArchivePolicy):
        self.root = root
        self.fork_policy = fork_policy
        self.archive_policy = archive_policy

    def prepare(self, request, profiles, context) -> dict:
        round_root = self.root / "branches" / f"round_{request.round_idx:04d}"
        if any(o.round_idx is None or o.round_idx >= request.round_idx for o in request.observations):
            raise ValueError("Research planning requires observations from earlier rounds")
        previous = self.root / "branches" / f"round_{request.round_idx - 1:04d}" / "tree.json"
        identity = {
            "evidence_version": 1,
            "fork_policy": self.fork_policy.identity(), "archive_policy": self.archive_policy.identity(),
            "profiles": [p.profile_id for p in profiles], "round_index": request.round_idx,
            "history_digest": canonical_sha256(serialize_measured_observations(request.observations, context)),
            "previous_tree_sha256": file_sha256(previous) if request.round_idx > 1 else None,
        }
        path = round_root / "fork-plan.json"
        if path.exists():
            plan = json.loads(path.read_text())
            if plan["identity"] != identity:
                raise ValueError("Persisted fork plan identity changed")
            return plan
        main_nodes, archive, outcomes = [], [], []
        if request.round_idx > 1:
            state = json.loads(previous.read_text())
            if state["policies"] != {k: identity[k] for k in ("fork_policy", "archive_policy")}:
                raise ValueError("Research tree policies changed during campaign")
            main_nodes = [_node(n) for n in state["main_nodes"]]
            archive = [_node(n) for n in state["archive"]]
            outcomes = [_outcome(o) for o in state["outcomes"]]
        measured = [o for o in request.observations if o.evaluation.status == "succeeded"
                    and math.isfinite(float(o.evaluation.metrics["utility"]))]
        scores = {o.canonical_key: float(o.evaluation.metrics["utility"]) for o in measured}
        incumbent = max(scores.values(), default=None)
        leader = max(measured, key=lambda o: (scores[o.canonical_key], o.canonical_key), default=None)
        descriptor = _descriptor([leader.candidate.payload if leader else {"mutations": []}], context)
        main_nodes.append(ResearchNode(
            node_id=f"main_{request.round_idx:04d}",
            snapshot=(round_root / "fork-source").relative_to(self.root).as_posix(),
            round_index=request.round_idx, history_cursor=len(request.observations), lineage_id="main",
            parent_id=main_nodes[-1].node_id if main_nodes else None,
            measured_count=len(scores), best_score=incumbent, descriptor=descriptor,
        ))
        archive = [replace(n, measured_count=sum(k in scores for k in n.proposal_keys),
                           best_score=max((scores[k] for k in n.proposal_keys if k in scores), default=None))
                   for n in archive]
        archive = list(self.archive_policy.retain(tuple(archive)))
        by_round = {}
        for o in measured:
            by_round.setdefault(o.round_idx, {})[o.canonical_key] = float(o.evaluation.metrics["utility"])
        for i, outcome in enumerate(outcomes):
            batch_scores = by_round.get(outcome.round_index, {})
            measured_keys = set(outcome.proposal_keys) & batch_scores.keys()
            best_key = max(measured_keys, key=lambda k: (batch_scores[k], k), default=None)
            outcomes[i] = replace(outcome, measured_count=len(measured_keys), best_key=best_key,
                                  best_score=batch_scores[best_key] if best_key is not None else None)
        available = {n.node_id: n for n in (*main_nodes, *archive)}
        assignments = self.fork_policy.plan(ForkContext(
            request.round_idx, tuple(identity["profiles"]), tuple(main_nodes), tuple(archive),
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

    def commit(self, plan: dict, expanded, context) -> None:
        """Commit all leaves together; retries reuse the same plan and snapshots."""
        round_index = plan["identity"]["round_index"]
        round_root = self.root / "branches" / f"round_{round_index:04d}"
        main_nodes = [_node(n) for n in plan["main_nodes"]]
        archive = [_node(n) for n in plan["archive"]]
        nodes = {n.node_id: n for n in (*main_nodes, *archive)}
        sessions = {s["profile_id"]: s for s in expanded.metadata["sessions"]}
        replacements = {}
        children = []
        outcomes = list(plan["outcomes"])
        for value in plan["assignments"]:
            a = ForkAssignment(**value)
            payloads = [prepare_candidate_payload(p.payload, context) for p in expanded.proposals
                        if p.metadata["harness_lineage"]["profile_id"] == a.profile_id]
            keys = {p.canonical_key for p in payloads}
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
                descriptor=_descriptor([p.payload for p in payloads], context),
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
