"""Main research, branch snapshots and provenance for task-owned proposal batches."""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import replace
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (
    HarnessError, HarnessSubmissionContract, HarnessSubmissionValidation,
    HarnessTurn, HarnessTurnResult, canonical_sha256, file_sha256,
)
from ldm_tts.harness.research_store import ResearchMeasurement, ResearchTree, snapshot_fork


def research_contract():
    return HarnessSubmissionContract(
        contract_id="research_brief", tool_name="submit_research",
        payload_schema={"type": "object", "additionalProperties": False,
                        "properties": {"brief": {"type": "string", "minLength": 1, "maxLength": 4096}},
                        "required": ["brief"]},
    )


def main_research_profile(root, template):
    """Share task tools and skills, with one task-neutral coordinator instruction."""
    path = Path(root) / "main-research.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Main researcher\n\n"
        "Maintain the common research state for this task. Read the task contract through its tools. "
        "Use only feedback supplied by the host and public task resources. Never seek hidden labels, "
        "private evaluator inputs, benchmark answers or credentials.\n\n"
        "Reconcile earlier hypotheses with available measurements. An unmeasured proposal is not "
        "a failure. Keep alternative explanations in research.md and preserve useful scripts. "
        "Side branches inherit this state and create the candidate submissions. Your submission is "
        "a concise research brief through submit_research, not a candidate batch.\n\n"
        "Every script and search loop must have finite bounds. Check feasibility before sampling; "
        "use a bounded construction when constraints may be impossible. Reserve time for submission. "
        "Do not run external evaluations or increase the host budgets.\n",
        encoding="utf-8",
    )
    return (replace(template, profile_id="main_research", agents_path=Path("/artifacts/main-research.md"),
                    agents_sha256=file_sha256(path)),)


def usage_counts(usage):
    counts = {counter: int(usage[key]) for key, counter in (
        ("providerCalls", "llm_requests"), ("validationSubmissions", "harness_validation_submissions"),
        ("artifactBytes", "harness_artifact_bytes"),
    ) if key in usage}
    if "toolCalls" in usage:
        counts["harness_tool_calls"] = sum(int(v) for v in usage["toolCalls"].values())
    return counts


def proposal_turn_summary(result: HarnessTurnResult, candidate_count: int) -> dict[str, object]:
    return {
        "profile_id": result.profile_id,
        "session_id": result.session_id,
        "session_turn_id": result.turn_id,
        "round_index": result.round_index,
        "history_from_seq": result.history_from_seq,
        "history_to_seq": result.history_to_seq,
        "history_digest": result.history_digest,
        "submission_id": result.submission_id,
        "candidate_count": candidate_count,
        "usage": result.usage,
        "tool_budget": result.tool_budget,
        "artifacts": result.artifacts,
    }


class ResearchCoordinator:
    """Tasks supply authorized evidence and canonical keys; policies only select nodes."""

    def __init__(self, root, strategy, profiles, main_client, *, account=None,
                 before_main=None, after_main=None, provider_authorizer=None):
        if strategy.fork_policy is None:
            raise ValueError("A research tree requires a fork policy")
        self.root = Path(root).resolve()
        self.tree = ResearchTree(self.root, strategy.fork_policy, strategy.archive_policy)
        self.profiles, self.main_client = tuple(profiles), main_client
        self.account = account
        self.before_main, self.after_main = before_main, after_main
        self.provider_authorizer = provider_authorizer

    def prepare(self, round_index, history, measurements, *, message, forbidden_query_terms=()):
        main_root = self.root / "main"
        workspace = main_root / "sessions/main_research/workspace"
        round_root = self.root / "branches" / f"round_{round_index:04d}"
        previous = self.root / "branches" / f"round_{round_index - 1:04d}" / "panel.json"
        cursor = 0
        if round_index > 1:
            panel = json.loads(previous.read_text(encoding="utf-8"))
            cursor = panel["history_count"]
            if not 0 <= cursor <= len(history) or panel["history_digest"] != canonical_sha256(history[:cursor]):
                raise ValueError("Research history prefix changed")
            inbox = workspace / "branch_feedback" / previous.parent.name
            inbox.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous, inbox.with_suffix(".json"))
        payload = {**message, "round_index": round_index,
                   "branch_roles": [p.profile_id for p in self.profiles],
                   "previous_branch_panel": f"/workspace/branch_feedback/round_{round_index - 1:04d}.json" if round_index > 1 else None,
                   "instructions": "Reconcile available feedback, preserve alternatives in research.md, and finish with submit_research({brief: ...}); at most 4096 characters."}
        turn = HarnessTurn(
            profile_id="main_research", turn_id=f"main_round_{round_index:04d}", round_index=round_index,
            history_from_seq=cursor, history_to_seq=len(history), history_digest=canonical_sha256(history[cursor:]),
            message=json.dumps(payload, sort_keys=True), forbidden_query_terms=tuple(forbidden_query_terms),
        )
        if self.before_main:
            self.before_main(turn)
        if self.account:
            self.account({"harness_turns": 1, "main_research_turns": 1}, usage_key=f"main:{turn.turn_id}:count")
        usage, started = {}, time.perf_counter()
        try:
            with self.main_client() as client:
                kwargs = {"submission_validator": lambda _: HarnessSubmissionValidation("accept"),
                          "recovery_timeout_seconds": 2.0 * client.config.limits.wall_time_seconds}
                if self.provider_authorizer:
                    kwargs["provider_authorizer"] = self.provider_authorizer
                result, = client.run_turn((turn,), **kwargs)
                usage = result.usage
        except HarnessError as exc:
            usage = exc.turn_usage.get("main_research", {})
            raise
        finally:
            if self.account:
                self.account(usage_counts(usage), usage_key=f"main:{turn.turn_id}")
                self.account({"harness_wall_time_seconds": time.perf_counter() - started})
            if self.after_main:
                self.after_main(turn, usage)
        if result.submission_status != "accepted":
            raise ValueError("Main research brief was rejected")
        session = (main_root / result.artifacts["session"]).resolve()
        if not session.is_relative_to(main_root):
            raise ValueError("Main session escaped its artifact root")
        source = snapshot_fork(session, workspace, round_root / "fork-source", len(history))
        plan = self.tree.prepare(round_index, self.profiles, history, measurements)
        sources = self.tree.materialize(plan)
        return {"root": round_root, "plan": plan, "sources": sources, "source": source,
                "main_turn_id": turn.turn_id, "brief": result.submission["brief"],
                "history_count": len(history), "history_digest": canonical_sha256(history)}

    def commit(self, prepared, sessions, proposal_keys, *, proposals):
        self.tree.commit(prepared["plan"], sessions, proposal_keys)
        panel = {k: prepared[k] for k in ("main_turn_id", "brief", "history_count", "history_digest")}
        panel.update(round_index=prepared["plan"]["identity"]["round_index"],
                     main_session_sha256=prepared["source"].session_sha256,
                     proposals=proposals, sessions=list(sessions.values()),
                     fork_plan=prepared["plan"]["assignments"],
                     status="branches_closed; selection_and_evaluation_follow")
        path = prepared["root"] / "panel.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != panel:
            raise ValueError("Committed research panel changed on replay")
        atomic_json_write(path, panel)


class MeasuredResearchExpander:
    """Adapt measured candidate batches while leaving task validation and q0 intact."""

    def __init__(self, coordinator, branch_client, branch_expander, *, serialize_history,
                 write_history, proposal_key, objective_name, first_active_round):
        self.coordinator, self.branch_client, self.branch_expander = coordinator, branch_client, branch_expander
        self.serialize_history, self.write_history = serialize_history, write_history
        self.proposal_key, self.objective_name = proposal_key, objective_name
        self.first_active_round = first_active_round

    def expand(self, request):
        step = request.round_idx - self.first_active_round + 1
        history = self.serialize_history(request.observations)
        self.write_history(self.coordinator.root, request.observations)
        self.write_history(self.coordinator.root / "main", request.observations)
        measurements = tuple(ResearchMeasurement(o.canonical_key, float(o.metrics[self.objective_name]),
                             o.round_idx - self.first_active_round + 1)
                             for o in request.observations if o.evaluation.succeeded
                             and math.isfinite(float(o.metrics[self.objective_name])))
        prepared = self.coordinator.prepare(step, history, measurements,
            message={"objective": {"name": self.objective_name, "direction": "maximize"},
                     "new_measurements": [{"candidate_id": row["candidate_id"],
                                           "round_index": row["round_index"],
                                           self.objective_name: row.get(self.objective_name)}
                                          for row in history[-16:]],
                     "measured_history": "/artifacts/measured_history/observations.json"},
            forbidden_query_terms=tuple(sorted({key for o in request.observations
                                               for key in (o.candidate_id, o.canonical_key)})))
        with self.branch_client(prepared["root"], prepared["sources"]) as client:
            expanded = self.branch_expander(client, prepared["root"],
                {p: source.history_cursor for p, source in prepared["sources"].items()}).expand(request)
        assignments = {a["profile_id"]: a for a in prepared["plan"]["assignments"]}
        expanded = replace(expanded, proposals=tuple(replace(p, metadata={
            **p.metadata, "research_fork": assignments[p.metadata["harness_lineage"]["profile_id"]],
        }) for p in expanded.proposals))
        keys = {profile: tuple(self.proposal_key(p) for p in expanded.proposals
                               if p.metadata["harness_lineage"]["profile_id"] == profile) for profile in assignments}
        self.coordinator.commit(prepared, {s["profile_id"]: s for s in expanded.metadata["sessions"]}, keys,
            proposals=[{"payload": p.payload, "annotations": p.metadata.get("research_annotations", []),
                        "lineage": p.metadata["harness_lineage"]} for p in expanded.proposals])
        return replace(expanded, metadata={**expanded.metadata, "sampling_mode": "main_research_round_forks",
                       "main_turn_id": prepared["main_turn_id"], "fork_source": prepared["source"].to_dict(),
                       "fork_policy": self.coordinator.tree.fork_policy.identity(),
                       "fork_plan": prepared["plan"]["assignments"]})


def write_fork_provenance(root: Path, *, compiled_policy: bool) -> None:
    root = root.resolve()
    branches = sorted((root / "branches").glob("round_*/manifest.json"))
    if not branches:
        raise ValueError("A completed research tree requires branch manifests")
    paths = {"main_research": root / "main/manifest.json",
             **{f"proposal_{path.parent.name}": path for path in branches}}
    if compiled_policy:
        paths["policy"] = root.parent / "policy_harness/manifest.json"
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("A completed research tree is missing a pool manifest")
    atomic_json_write(root.parent / "harness_provenance.json", {
        "schema_version": 1,
        "pools": {name: [path.relative_to(root.parent).as_posix()] for name, path in paths.items()},
    })
