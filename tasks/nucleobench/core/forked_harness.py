"""Experimental main-session research with one-round native Pi forks."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ldm_tts.engine.expansion import ExpansionRequest, ExpansionResult
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (
    HarnessClient, HarnessError, HarnessSubmissionContract, HarnessSubmissionValidation,
    HarnessTurn, canonical_sha256, directory_sha256, file_sha256,
)
from ldm_tts.harness.pi import PiForkSource
from tasks.nucleobench.core.harness import (
    NucleoBenchHarnessExpander, _forbidden_query_terms, _history_delta, _profile,
    _usage_counts,
)
from tasks.nucleobench.core.research import (
    serialize_measured_observations, summarize_measured_observations, write_measured_history,
)


def main_research_profile():
    return (_profile("main_research"),)


def research_contract() -> HarnessSubmissionContract:
    return HarnessSubmissionContract(
        contract_id="nucleobench_research_brief", tool_name="submit_research",
        payload_schema={
            "type": "object", "additionalProperties": False,
            "properties": {"brief": {"type": "string", "minLength": 1, "maxLength": 4096}},
            "required": ["brief"],
        },
    )


def write_fork_provenance(root: Path, *, compiled_policy: bool) -> None:
    """Declare the tree through the pilot runner's existing multi-pool contract."""
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


class ForkedResearchExpander:
    """Keep one main identity; archive all branch evidence, then prune live children."""

    def __init__(self, *, main_client: Callable[[], HarnessClient], root: Path,
                 branch_client: Callable[[Path, PiForkSource], HarnessClient],
                 expander_options: dict[str, Any]) -> None:
        self.main_client = main_client
        self.root = root.resolve()
        self.branch_client = branch_client
        self.options = expander_options
        self.domain = expander_options["domain"]
        self.account = expander_options.get("account")

    def expand(self, request: ExpansionRequest) -> ExpansionResult:
        main_root = self.root / "main"
        workspace = main_root / "sessions/main_research/workspace"
        round_root = self.root / "branches" / f"round_{request.round_idx:04d}"
        write_measured_history(self.root, request.observations, self.domain.context)
        write_measured_history(main_root, request.observations, self.domain.context)
        previous = self.root / "branches" / f"round_{request.round_idx - 1:04d}" / "panel.json"
        inbox = workspace / "branch_feedback" / f"round_{request.round_idx - 1:04d}.json"
        if request.round_idx > 1 and not previous.exists():
            raise ValueError("Previous branch panel is required to continue main research")
        if previous.exists():
            inbox.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(previous, inbox)
        history = _history_delta(request, 1)
        rows = serialize_measured_observations(history, self.domain.context)
        payload = {
            "round_index": request.round_idx,
            "new_measurements": summarize_measured_observations(rows),
            "previous_branch_panel": str(Path("/workspace") / inbox.relative_to(workspace)) if previous.exists() else None,
            "branch_roles": [p.profile_id for p in self.options["profiles"]],
            "instructions": (
                "Use the task context and measured-history tools for exact data. Reconcile prior branch hypotheses "
                "with the new measurements; the panel archive contains all proposals, including unmeasured ones. "
                "Update research.md and shared scripts only where evidence warrants. Preserve alternatives. "
                "Prepare a brief common research foundation; branches will independently construct candidates. "
                "Finish with submit_research({brief: ...}), at most 4096 characters."
            ),
        }
        turn = HarnessTurn(
            profile_id="main_research", turn_id=f"main_round_{request.round_idx:04d}",
            round_index=request.round_idx, history_from_seq=len(request.observations) - len(history),
            history_to_seq=len(request.observations), history_digest=canonical_sha256(rows),
            message=json.dumps(payload, sort_keys=True), forbidden_query_terms=_forbidden_query_terms(request),
        )
        if self.account:
            self.account({"harness_turns": 1, "main_research_turns": 1}, usage_key=f"main:{turn.turn_id}:count")
        usage = {}
        started = time.perf_counter()
        try:
            with self.main_client() as client:
                result, = client.run_turn(
                    (turn,), submission_validator=lambda _: HarnessSubmissionValidation("accept"),
                    recovery_timeout_seconds=2.0 * client.config.limits.wall_time_seconds,
                )
                usage = result.usage
        except HarnessError as exc:
            usage = exc.turn_usage.get("main_research", {})
            raise
        finally:
            if self.account:
                self.account(_usage_counts(usage), usage_key=f"main:{turn.turn_id}")
                self.account({"harness_wall_time_seconds": time.perf_counter() - started})
        if result.submission_status != "accepted":
            raise ValueError("Main research brief was rejected")
        session = (main_root / result.artifacts["session"]).resolve()
        if not session.is_relative_to(main_root.resolve()):
            raise ValueError("Main session escaped its artifact root")
        source = snapshot_fork(session, workspace, round_root / "fork-source", len(request.observations))
        with self.branch_client(round_root, source) as client:
            expander = NucleoBenchHarnessExpander(
                client, artifact_root=round_root, inherited_history_count=len(request.observations),
                **self.options,
            )
            expanded = expander.expand(request)
        panel = {
            "round_index": request.round_idx, "main_turn_id": turn.turn_id,
            "main_session_sha256": source.session_sha256, "brief": result.submission["brief"],
            "proposals": [{"payload": p.payload, "annotations": p.metadata["research_annotations"],
                           "lineage": p.metadata["harness_lineage"]} for p in expanded.proposals],
            "sessions": expanded.metadata["sessions"],
            "status": "branches_closed; selection_and_evaluation_follow",
        }
        atomic_json_write(round_root / "panel.json", panel)
        return ExpansionResult(
            proposals=expanded.proposals, selection_mode=expanded.selection_mode,
            metadata={**expanded.metadata, "sampling_mode": "main_research_round_forks",
                      "main_turn_id": turn.turn_id, "fork_source": source.to_dict()},
        )
