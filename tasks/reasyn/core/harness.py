"""Persistent, independent ReaSyn research sessions using the shared Harness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (
    HarnessArtifactRule, HarnessClient, HarnessError, HarnessLimits,
    HarnessNetworkPolicy, HarnessProfile, HarnessSubmissionContract,
    HarnessSubmissionError, HarnessSubmissionValidation, HarnessToolExtension,
    HarnessTurn, canonical_sha256, directory_sha256, file_sha256,
    load_harness_mcp_config, parse_tool_call_budgets, policy_submission_contract,
)
from ldm_tts.harness.container import docker_identity_args, resolve_container_user
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime, policy_mcp_server
from ldm_tts.transport import ProposalResponse
from .chemistry import canonicalize

RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources/harness"
TOOL_NAMES = ("describe_reasyn_task", "get_measured_history", "check_measured_product")


def profiles(count=4, *, policy=False):
    role = "policy_architect" if policy else "molecular_research"
    skills = ("compile-ldm-policy",) if policy else ("molecular-design",)
    return tuple(HarnessProfile(
        "policy_architect" if policy else f"molecular_research_{i + 1:02d}",
        Path(f"/resources/profiles/{role}/AGENTS.md"),
        agents_sha256=file_sha256(RESOURCE_ROOT / "profiles" / role / "AGENTS.md"),
        skill_dirs=tuple(Path("/resources/skills") / name for name in skills),
        skill_dir_sha256=tuple(directory_sha256(RESOURCE_ROOT / "skills" / name) for name in skills),
    ) for i in range(1 if policy else count))


def submission_contract():
    return HarnessSubmissionContract(
        contract_id="reasyn_projection_targets", tool_name="submit_candidates",
        payload_schema={"type": "object", "properties": {
            "artifact_path": {"type": "string", "const": "candidates.json"}},
            "required": ["artifact_path"], "additionalProperties": False},
        artifact_rules=(HarnessArtifactRule("/artifact_path", (".json",), 1048576),),
        max_validation_attempts=4,
    )


def read_submission(submission, artifacts, root, count, *, mock=False):
    if dict(submission) != {"artifact_path": "candidates.json"} or len(artifacts) != 1:
        raise ValueError("Submit only the path to candidates.json, containing the requested candidate count.")
    artifact = artifacts[0]
    path = (root / artifact.snapshot_path).resolve()
    if (artifact.path_pointer != "/artifact_path" or artifact.relative_path != "candidates.json"
            or not path.is_relative_to(root.resolve()) or not path.is_file()):
        raise ValueError("Candidate snapshot must be inside this Harness artifact root.")
    body = path.read_bytes()
    if len(body) != artifact.size_bytes or hashlib.sha256(body).hexdigest() != artifact.sha256:
        raise ValueError("Candidate snapshot digest/size mismatch.")
    data = json.loads(body)
    if not isinstance(data, dict) or set(data) != {"candidates"} or not isinstance(data["candidates"], list) or len(data["candidates"]) != count:
        raise ValueError(f"Expected exactly {count} candidates.")
    normalized = []
    for i, row in enumerate(data["candidates"]):
        if not isinstance(row, dict) or set(row) != {"target_smiles", "rationale"}:
            raise ValueError(f"candidates[{i}] requires only target_smiles and rationale.")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip() or len(row["rationale"]) > 2000:
            raise ValueError(f"candidates[{i}].rationale must be a short nonempty research note.")
        try:
            smiles = canonicalize(row["target_smiles"], mock=mock)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"candidates[{i}].target_smiles: {exc}") from exc
        normalized.append({"target_smiles": smiles, "rationale": row["rationale"]})
    return normalized


def create_client(args, root, campaign_id, target, *, policy=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    context = {"benchmark": args.benchmark, "oracle": args.oracle if not target else None,
               "original_target": target, "mock": args.mock,
               "feedback": "Only completed task measurements are available. Projection is not an oracle score."}
    atomic_json_write(root / "context.json", context)
    history_root = root.parent / "harness" if policy else root
    history_root.mkdir(parents=True, exist_ok=True)
    if not (history_root / "measured_history.json").exists():
        atomic_json_write(history_root / "measured_history.json", {"observations": []})
    cache = (args.harness_cache_dir or Path.home() / ".cache/ldm-gondolin").expanduser().resolve()
    (cache / "runtime-overlays").mkdir(parents=True, exist_ok=True)
    command = ["docker"]
    if args.harness_docker_host:
        command += ["--host", args.harness_docker_host]
    command += ["run", "--rm", "-i"]
    command += docker_identity_args(resolve_container_user(args.harness_container_user, args.harness_docker_host), args.harness_docker_host)
    command += ["--device", "/dev/kvm"]
    env = {"HOME": "/runtime-home", "XDG_CACHE_HOME": "/runtime-home/.cache",
           "GONDOLIN_IMAGE_STORE": "/runtime-home/.cache/gondolin/images",
           "GONDOLIN_SESSIONS_DIR": "/runtime-home/.cache/gondolin/sessions",
           "LDM_HARNESS_CACHE_ROOT": "/runtime-home/.cache/gondolin",
           "TMPDIR": "/runtime-home/.cache/gondolin/runtime-overlays",
           "LDM_REASYN_CONTEXT": "/artifacts/context.json",
           "LDM_REASYN_HISTORY": "/measured_history/measured_history.json" if policy else "/artifacts/measured_history.json"}
    for name, value in env.items():
        command += ["--env", f"{name}={value}"]
    mounts = [(root, "/artifacts", False), (cache, "/runtime-home/.cache/gondolin", False), (RESOURCE_ROOT, "/resources", True)]
    if policy:
        mounts.append((history_root, "/measured_history", True))
    for source, destination, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")]
    command.append(args.harness_sidecar_image)
    mcp = load_harness_mcp_config(args.harness_mcp_config)
    contract = policy_submission_contract() if policy else submission_contract()
    return HarnessClient(command, api_key=args.provider_api_key,
        named_secrets=mcp.named_secrets, response_timeout_seconds=args.harness_response_timeout,
        config=PiHarnessConfig(
            artifact_root=Path("/artifacts"), base_url=args.llm_url, model=args.llm_model,
            profiles=profiles(args.harness_sessions, policy=policy), campaign_id=campaign_id,
            task_id="reasyn", case_id=f"{args.benchmark}:{target or args.oracle}" + (":policy" if policy else ""),
            seed=args.seed, submission_contract=contract,
            guest_runtime=load_pi_guest_runtime("reasyn", RESOURCE_ROOT / "image"),
            tool_extensions=(HarnessToolExtension(Path("/resources/tools/molecular_research.mjs"),
                file_sha256(RESOURCE_ROOT / "tools/molecular_research.mjs"), TOOL_NAMES),),
            mcp_servers=(*mcp.servers, policy_mcp_server(
                diagnostics_path="/resources/policy_diagnostics.py",
                diagnostics_sha256=file_sha256(RESOURCE_ROOT / "policy_diagnostics.py"),
            )) if policy else mcp.servers,
            thinking=args.harness_thinking, context7_enabled=False,
            limits=HarnessLimits(wall_time_seconds=args.harness_wall_time_seconds,
                tool_call_budgets=parse_tool_call_budgets(args.harness_tool_budget, excluded_tools=(contract.tool_name,))),
            network_policy=HarnessNetworkPolicy(forbidden_query_patterns=("benchmark_result", "oracle_cache", "checkpoint")),
        ))


class HarnessTargetSource:
    """Run a strict barrier of independent persistent molecular research sessions.

    Immutable turn inputs and a durable normalized response make the cursor and
    projection handoff replayable. Usage is cumulative per native turn, including
    task validation attempts observed before an interrupted provider response.
    """
    def __init__(self, client, root, args, *, target="", account=None):
        self.client, self.root, self.args, self.target = client, Path(root), args, target
        self.root.mkdir(parents=True, exist_ok=True)
        self.account = account
        self.session_profiles = tuple(client.config.profiles)
        if not self.session_profiles:
            raise ValueError("ReaSyn Harness needs at least one independent session")

    def propose(self, request):
        meta = request.metadata
        count = meta["count"]
        round_index, batch = meta["round_idx"], meta.get("minibatch_index", 0)
        recovery_pass = meta.get("proposal_recovery_pass", 0)
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in (count, round_index, batch, recovery_pass)
            )
            or count < 1
        ):
            raise ValueError("Harness count must be positive; round and minibatch indices nonnegative")
        history = list(meta.get("history", []))
        self._write_history(history)
        selected = self.session_profiles[:min(count, len(self.session_profiles))]
        counts = {p.profile_id: count // len(selected) + int(i < count % len(selected)) for i, p in enumerate(selected)}
        turn_root = self.root / "proposal_turns"
        if recovery_pass:
            turn_root = turn_root / f"recovery-{recovery_pass:06d}"
        batch_root = turn_root / f"round-{round_index:06d}" / f"batch-{batch:06d}"
        request_digest = canonical_sha256({
            "metadata": meta, "profiles": [profile.to_dict() for profile in selected], "counts": counts,
            "benchmark": self.args.benchmark, "target": self.target, "oracle": self.args.oracle,
        })
        turns = self._turns(batch_root, request_digest, meta, history, selected, counts)
        response_path = batch_root / "response.json"
        if response_path.exists():
            record = json.loads(response_path.read_text())
            if record["request_sha256"] != request_digest:
                raise ValueError("Harness cached response request changed on resume")
            self._commit_cursors(selected, history)
            data = record["response"]
            data["tool_calls"] = tuple(data.get("tool_calls", ()))
            return ProposalResponse(**data)
        expected_turns = {turn.profile_id: turn for turn in turns}

        def validate(submission):
            if submission.profile_id not in expected_turns or submission.turn_id != expected_turns[submission.profile_id].turn_id:
                raise HarnessError("ReaSyn submission does not match the active independent session")
            if self.account:
                self.account({"harness_validation_submissions": submission.attempt_index},
                             usage_key=f"harness:{submission.turn_id}")
            try:
                read_submission(submission.submission, submission.artifacts, self.root,
                                counts[submission.profile_id], mock=self.args.mock)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return HarnessSubmissionValidation("retry", (HarnessSubmissionError(
                    "/artifact_path", "invalid_projection_targets", str(exc),
                    "Use RDKit in the sandbox to repair the specified entry, then resubmit candidates.json."),))
            return HarnessSubmissionValidation()

        if self.account:
            for turn in turns:
                self.account({"harness_turns": 1}, usage_key=f"harness:{turn.turn_id}")
        usage = {}
        try:
            results = self.client.run_turn(turns, submission_validator=validate,
                recovery_timeout_seconds=self.args.harness_wall_time_seconds * 2)
            usage = {r.profile_id: r.usage for r in results}
        except HarnessError as exc:
            usage = exc.turn_usage
            raise
        finally:
            if self.account:
                for turn in turns:
                    item = usage.get(turn.profile_id, {})
                    self.account({
                        "harness_provider_calls": int(item.get("providerCalls", 0)),
                        "harness_tool_calls": sum(int(v) for v in item.get("toolCalls", {}).values()),
                        "harness_validation_submissions": int(item.get("validationSubmissions", 0)),
                    }, usage_key=f"harness:{turn.turn_id}")
        by_profile = {r.profile_id: r for r in results}
        if len(results) != len(selected) or len(by_profile) != len(selected) or set(by_profile) != set(counts):
            raise HarnessError("ReaSyn research barrier requires exactly one result per independent session")
        candidates, lineage = [], []
        for profile in selected:
            result = by_profile[profile.profile_id]
            turn = expected_turns[profile.profile_id]
            if result.submission_status != "accepted" or result.turn_id != turn.turn_id or result.input_digest != turn.input_digest:
                raise HarnessError("ReaSyn research submission is not accepted for this frozen turn")
            rows = read_submission(result.submission, result.submitted_artifacts, self.root,
                                   counts[profile.profile_id], mock=self.args.mock)
            for index, row in enumerate(rows):
                candidates.append({"target_smiles": row["target_smiles"]})
                lineage.append({
                    "profile_id": profile.profile_id, "session_id": result.session_id,
                    "turn_id": result.turn_id, "submission_id": result.submission_id,
                    "submission_digest": result.submission_digest, "item_index": index,
                    "rationale": row["rationale"], "artifacts": result.artifacts,
                    "submitted_artifacts": [a.to_dict() for a in result.submitted_artifacts],
                })
        response = ProposalResponse(text=json.dumps({"candidates": candidates}),
            metadata={"harness_lineage": lineage, "mode": "persistent_harness"})
        atomic_json_write(response_path, {"request_sha256": request_digest, "response": response.to_dict()})
        self._commit_cursors(selected, history)
        return response

    def _write_history(self, history):
        path = self.root / "measured_history.json"
        if path.exists():
            previous = json.loads(path.read_text())["observations"]
            if history[:len(previous)] != previous:
                raise ValueError("Authoritative measured history changed or shrank on Harness resume")
        atomic_json_write(path, {"observations": history})

    def _commit_cursors(self, selected, history):
        for profile in selected:
            atomic_json_write(self.root / f"history_cursor_{profile.profile_id}.json", {
                "history_to_seq": len(history), "history_prefix_sha256": canonical_sha256(history),
            })

    def _turns(self, batch_root, request_digest, meta, history, selected, counts):
        path = batch_root / "input.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request_sha256"] != request_digest:
                raise ValueError("Harness minibatch input changed on resume")
            turns = tuple(HarnessTurn(
                profile_id=row["profileId"], turn_id=row["turnId"], round_index=row["roundIndex"],
                history_from_seq=row["historyFromSeq"], history_to_seq=row["historyToSeq"],
                history_digest=row["historyDigest"], message=row["message"],
                forbidden_query_terms=tuple(row.get("forbiddenQueryTerms", ())),
            ) for row in saved["turns"])
            if any(turn.input_digest != row["inputDigest"] for turn, row in zip(turns, saved["turns"])):
                raise ValueError("Frozen Harness turn input digest mismatch")
            return turns
        turns = []
        for profile in selected:
            cursor_path = self.root / f"history_cursor_{profile.profile_id}.json"
            cursor_state = json.loads(cursor_path.read_text()) if cursor_path.exists() else {"history_to_seq": 0}
            cursor = cursor_state["history_to_seq"]
            if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor <= len(history):
                raise ValueError("Invalid Harness measured-history cursor")
            if cursor_state.get("history_prefix_sha256", canonical_sha256(history[:cursor])) != canonical_sha256(history[:cursor]):
                raise ValueError("Measured history prefix changed on Harness resume")
            delta = history[cursor:]
            compact = [{k: row[k] for k in ("candidate_id", "smiles", "sampling_seed", "metrics") if k in row}
                       for row in delta[-8:]]
            feedback = list(meta.get("rejection_feedback", []))
            payload = {
                "task": "reasyn", "benchmark": self.args.benchmark,
                "goal": {"original_target": self.target} if self.target else {"oracle": self.args.oracle},
                "message_type": "campaign_bootstrap" if cursor == 0 else "history_delta",
                "new_measured_observations": compact,
                "history_from_seq": cursor, "history_to_seq": len(history),
                "history_digest": canonical_sha256(delta),
                "omitted_delta_observations": max(0, len(delta) - len(compact)),
                "candidate_count": counts[profile.profile_id],
                "rejection_feedback": feedback[-16:], "rejection_count": len(feedback),
                "history_query_tool": "get_measured_history", "product_membership_tool": "check_measured_product",
                "history_query": {"offset": cursor, "limit": 32, "total_observations": len(history)},
                "research_annotations": "Query measured candidates for original independent hypotheses and source notes.",
                "submission": {"artifact_path": "candidates.json", "fields": ["target_smiles", "rationale"]},
                "occurrences": "Repeated unmeasured targets are legal independent draws and contribute frequency. Already measured TDC products are rejected after projection with bounded replenishment.",
            }
            turn_id = f"r{meta['round_idx']:06d}_b{meta.get('minibatch_index', 0):04d}_{profile.profile_id}_{request_digest[:16]}"
            turns.append(HarnessTurn(profile.profile_id, turn_id, meta["round_idx"], cursor, len(history),
                canonical_sha256(delta), "Research independently using the sandbox, molecular tools and measured history. "
                "Write and submit the complete candidate file; repair indexed validation errors in this session.\n"
                + json.dumps(payload, separators=(",", ":"))))
        atomic_json_write(path, {"request_sha256": request_digest, "turns": [turn.to_dict() for turn in turns]})
        return tuple(turns)
