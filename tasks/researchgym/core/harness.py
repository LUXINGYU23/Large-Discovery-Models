"""Persistent ResearchGym research sessions through the shared Pi Harness.

Each profile is an independent persistent session with its own workspace. A
turn delivers a compact measured-history delta; sessions query details through
task tools, write candidates.json with a script, and repair indexed validation
errors in the same session. Committed turns replay on resume and unfinished
sessions continue; a partial barrier is never accepted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ldm_tts.engine.expansion import attach_proposal_attempt_receipt
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (
    HarnessArtifactRule, HarnessClient, HarnessError, HarnessLimits, HarnessNetworkPolicy, HarnessProfile,
    HarnessSubmissionContract, HarnessSubmissionError, HarnessSubmissionValidation, HarnessToolExtension,
    HarnessTurn, canonical_sha256, directory_sha256, file_sha256, load_harness_mcp_config,
    parse_tool_call_budgets, policy_submission_contract,
)
from ldm_tts.harness.container import docker_identity_args, resolve_container_user
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime, policy_mcp_server
from ldm_tts.transport import ProposalResponse
from ldm_tts.transport.openai import generation_body

from .cases import CaseSpec
from .clock import WallClockExhausted
from .history import compact, write_history
from .proposals import ProposalBatch, ProposalExhausted, validate_rows
from .usage import ApiPricing, account_tokens, harness_turn_tokens

RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"
TOOL_FILE = "tools/researchgym_tools.mjs"
TOOL_NAMES = ("describe_researchgym_case", "get_measured_history", "check_candidate_programs")
PROPOSAL_SKILLS = ("researchgym-method-research", "ml-experiment-analysis")
POLICY_SKILLS = ("compile-ldm-policy",)
ARTIFACT = "candidates.json"


def profiles(count: int, *, policy: bool = False) -> tuple[HarnessProfile, ...]:
    role, skills = ("policy_architect", POLICY_SKILLS) if policy else ("program_research", PROPOSAL_SKILLS)
    return tuple(HarnessProfile(
        "policy_architect" if policy else f"program_research_{i + 1:02d}",
        Path(f"/resources/profiles/{role}/AGENTS.md"),
        agents_sha256=file_sha256(RESOURCE_ROOT / "profiles" / role / "AGENTS.md"),
        skill_dirs=tuple(Path("/resources/skills") / name for name in skills),
        skill_dir_sha256=tuple(directory_sha256(RESOURCE_ROOT / "skills" / name) for name in skills),
    ) for i in range(1 if policy else count))


def submission_contract(max_attempts: int) -> HarnessSubmissionContract:
    return HarnessSubmissionContract(
        contract_id="researchgym_candidate_programs", tool_name="submit_candidates",
        payload_schema={"type": "object", "properties": {"artifact_path": {"type": "string", "const": ARTIFACT}},
                        "required": ["artifact_path"], "additionalProperties": False},
        artifact_rules=(HarnessArtifactRule("/artifact_path", (".json",), 4 * 1024 * 1024),),
        max_validation_attempts=max_attempts,
    )


class SubmissionRejected(ValueError):
    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{e['path']} {e['code']}" for e in errors[:8]))


def read_submission(submission, artifacts, root: Path, *, case: CaseSpec, count: int,
                    measured_keys: set[str], measured_ids: set[str]) -> list[dict]:
    """Validate the immutable snapshot with the Campaign admission rules."""
    if dict(submission) != {"artifact_path": ARTIFACT} or len(artifacts) != 1:
        raise SubmissionRejected([{"path": "/artifact_path", "code": "invalid_submission",
                                   "message": f"submit only artifact_path={ARTIFACT}", "hint": ""}])
    artifact = artifacts[0]
    path = (root / artifact.snapshot_path).resolve()
    if artifact.path_pointer != "/artifact_path" or artifact.relative_path != ARTIFACT \
            or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise SubmissionRejected([{"path": "/artifact_path", "code": "invalid_snapshot",
                                   "message": "candidate snapshot is outside this Harness root", "hint": ""}])
    body = path.read_bytes()
    if len(body) != artifact.size_bytes or hashlib.sha256(body).hexdigest() != artifact.sha256:
        raise SubmissionRejected([{"path": "/artifact_path", "code": "snapshot_digest_mismatch",
                                   "message": "candidate snapshot digest or size mismatch", "hint": ""}])
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SubmissionRejected([{"path": "/artifact_path", "code": "invalid_json", "message": str(exc),
                                   "hint": "Write the file with json.dump from a script."}]) from exc
    if not isinstance(data, dict) or set(data) != {"candidates"}:
        raise SubmissionRejected([{"path": "/artifact_path", "code": "invalid_candidates_file",
                                   "message": "the file must be an object with only a candidates array", "hint": ""}])
    rows, errors = validate_rows(data["candidates"], case=case, count=count, measured_keys=measured_keys,
                                 measured_ids=measured_ids)
    if errors:
        raise SubmissionRejected([{**e, "path": "/artifact_path" + e["path"]} for e in errors])
    return rows


def create_client(args, root: Path, campaign_id: str, case: CaseSpec, *, policy: bool = False, command=None) -> HarnessClient:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    history_root = root.parent / "harness" if policy else root
    history_root.mkdir(parents=True, exist_ok=True)
    if not (history_root / "measured_history.json").exists():
        write_history(history_root / "measured_history.json", [])
    mcp = load_harness_mcp_config(args.harness_mcp_config)
    contract = policy_submission_contract(args.policy_submission_attempts) if policy \
        else submission_contract(args.harness_submission_attempts)
    budgets = args.policy_tool_budget if policy else args.harness_tool_budget
    # A supplied command is the mock-only protocol fixture, which runs on the host.
    artifact_root = root if command is not None else Path("/artifacts")
    if command is None:
        command = docker_command(args, root, history_root, policy=policy)
    return HarnessClient(command, api_key=args.provider_api_key, named_secrets=mcp.named_secrets,
        response_timeout_seconds=args.harness_response_timeout,
        config=PiHarnessConfig(
            artifact_root=artifact_root, base_url=args.llm_url or "http://fixture.invalid/v1",
            model=args.llm_model or "synthetic-fixture",
            profiles=profiles(args.harness_sessions, policy=policy), campaign_id=campaign_id,
            task_id="researchgym", case_id=case.case_id + (":policy" if policy else ""), seed=args.seed,
            submission_contract=contract, guest_runtime=load_pi_guest_runtime("researchgym", RESOURCE_ROOT / "image"),
            tool_extensions=(HarnessToolExtension(Path("/resources") / TOOL_FILE, file_sha256(RESOURCE_ROOT / TOOL_FILE),
                                                  TOOL_NAMES),),
            mcp_servers=(*mcp.servers, policy_mcp_server(
                diagnostics_path="/resources/policy_diagnostics.py",
                diagnostics_sha256=file_sha256(RESOURCE_ROOT / "policy_diagnostics.py"))) if policy else mcp.servers,
            thinking=args.harness_thinking, context7_enabled=args.harness_context7,
            provider_request_body={"max_output_tokens": args.llm_max_tokens,
                                   **generation_body(wire_api="responses", reasoning=args.llm_reasoning,
                                                     extra_body=json.loads(args.llm_extra_body_json))},
            limits=HarnessLimits(wall_time_seconds=args.harness_wall_time_seconds,
                                 tool_call_budgets=parse_tool_call_budgets(budgets, excluded_tools=(contract.tool_name,))),
            # Candidate code may read public benchmark code; campaign records are off limits.
            network_policy=HarnessNetworkPolicy(forbidden_query_patterns=("grade_summary", "checkpoint.json", "events.jsonl")),
        ))


def docker_command(args, root: Path, history_root: Path, *, policy: bool) -> list[str]:
    cache = (args.harness_cache_dir or Path.home() / ".cache/ldm-gondolin").expanduser().resolve()
    (cache / "runtime-overlays").mkdir(parents=True, exist_ok=True)
    command = ["docker"] + (["--host", args.harness_docker_host] if args.harness_docker_host else [])
    command += ["run", "--rm", "-i", *docker_identity_args(
        resolve_container_user(args.harness_container_user, args.harness_docker_host), args.harness_docker_host),
        "--device", "/dev/kvm"]
    env = {"HOME": "/runtime-home", "XDG_CACHE_HOME": "/runtime-home/.cache",
           "GONDOLIN_IMAGE_STORE": "/runtime-home/.cache/gondolin/images",
           "GONDOLIN_SESSIONS_DIR": "/runtime-home/.cache/gondolin/sessions",
           "LDM_HARNESS_CACHE_ROOT": "/runtime-home/.cache/gondolin",
           "TMPDIR": "/runtime-home/.cache/gondolin/runtime-overlays",
           "LDM_RG_CONTEXT": "/artifacts/context.json",
           "LDM_RG_HISTORY": "/measured_history/measured_history.json" if policy else "/artifacts/measured_history.json"}
    for name, value in env.items():
        command += ["--env", f"{name}={value}"]
    mounts = [(root, "/artifacts", False), (cache, "/runtime-home/.cache/gondolin", False), (RESOURCE_ROOT, "/resources", True)]
    if policy:
        mounts.append((history_root, "/measured_history", True))
    for source, destination, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")]
    return command + [args.harness_sidecar_image]


def write_context(root: Path, case: CaseSpec, references: dict[str, str], facts: dict) -> None:
    contract = case.public_contract()
    atomic_json_write(Path(root) / "context.json", {
        **contract, "proposal_facts": facts, "reference_sources": references,
        "checker_case": {"entry": case.entry, "forbidden_import_prefixes": case.raw["forbidden_import_prefixes"]},
    })


class HarnessProgramSource:
    """Strict barrier over independent persistent sessions."""

    def __init__(self, client: HarnessClient, root: Path, args, case: CaseSpec, *, pricing: ApiPricing) -> None:
        self.client, self.root, self.args, self.case, self.pricing = client, Path(root), args, case, pricing
        self.session_profiles = tuple(client.config.profiles)
        self.runtime = None
        self.clock = None

    def collect(self, *, round_idx, count, batch_size, records, measured_keys, recovery_pass,
                refill=0, avoid=()) -> ProposalBatch:
        del batch_size  # Each session submits its share of the occurrences in one file.
        if self.clock is not None and self.clock.remaining() <= 0:
            raise WallClockExhausted("campaign wall-clock allowance is exhausted before research turns")
        if self.runtime is not None:
            self.pricing.check(self.runtime)
        history_digest = write_history(self.root / "measured_history.json", records)
        selected = self.session_profiles[:min(count, len(self.session_profiles))]
        counts = {p.profile_id: count // len(selected) + int(i < count % len(selected)) for i, p in enumerate(selected)}
        measured_ids = {r["candidate_id"] for r in records}
        # Sessions accepted earlier for this round and history are never re-run,
        # including after another session exhausted its validation attempts.
        phase = f"round-{round_idx:06d}" + (f"-refill-{refill:02d}" if refill else "")
        accepted_dir = self.root / "proposal_turns" / f"{phase}-accepted"
        accepted = {}
        for profile in selected:
            path = accepted_dir / f"{profile.profile_id}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["history_sha256"] == history_digest and saved["count"] == counts[profile.profile_id]:
                    accepted[profile.profile_id] = saved
        pending = tuple(p for p in selected if p.profile_id not in accepted)
        if pending:
            self._run_pending(pending, round_idx, recovery_pass, records, counts, history_digest, measured_keys,
                              measured_ids, accepted_dir, accepted, phase, refill, list(avoid))
        rows, turn_records = [], []
        for profile in selected:
            saved = accepted[profile.profile_id]
            rows.extend(saved["rows"])
            turn_records.append(saved["turn"])
        self._commit_cursors(selected, records)
        return self._batch(rows, turn_records, counts)

    def _run_pending(self, pending, round_idx, recovery_pass, records, counts, history_digest, measured_keys,
                     measured_ids, accepted_dir, accepted, phase, refill, avoid) -> None:
        request_digest = canonical_sha256({"round": round_idx, "recovery_pass": recovery_pass, "history": history_digest,
                                           "refill": refill, "avoid": avoid,
                                           "counts": {p.profile_id: counts[p.profile_id] for p in pending},
                                           "case": self.case.case_id, "profiles": [p.to_dict() for p in pending]})
        batch_root = self.root / "proposal_turns" / f"pass-{recovery_pass:03d}" / phase
        turns = self._turns(batch_root, request_digest, round_idx, recovery_pass, records, pending, counts,
                            refill, avoid)
        expected = {turn.profile_id: turn for turn in turns}

        def validate(submission):
            turn = expected.get(submission.profile_id)
            if turn is None or submission.turn_id != turn.turn_id:
                raise HarnessError("submission does not match the active independent session")
            if self.runtime is not None:
                self.runtime.consume_many({"harness_validation_submissions": submission.attempt_index},
                                          usage_key=f"harness:{submission.turn_id}")
            try:
                read_submission(submission.submission, submission.artifacts, self.root, case=self.case,
                                count=counts[submission.profile_id], measured_keys=measured_keys,
                                measured_ids=measured_ids)
            except SubmissionRejected as exc:
                return HarnessSubmissionValidation("retry", tuple(
                    HarnessSubmissionError(e["path"], e["code"], e["message"], e["hint"]) for e in exc.errors[:32]))
            return HarnessSubmissionValidation()

        if self.runtime is not None:
            for turn in turns:
                self.runtime.consume_many({"harness_turns": 1}, usage_key=f"harness:{turn.turn_id}")
        usage, results = {}, ()
        recovery = min(self.args.harness_wall_time_seconds * 2, self.clock.remaining() if self.clock else float("inf"))
        try:
            results = self.client.run_turn(turns, submission_validator=validate, recovery_timeout_seconds=recovery)
            usage = {r.profile_id: r.usage for r in results}
        except HarnessError as exc:
            usage = exc.turn_usage
            raise
        finally:
            self._account(turns, usage)
        by_profile = {r.profile_id: r for r in results}
        if set(by_profile) != set(expected) or len(results) != len(turns):
            raise HarnessError("research barrier requires exactly one committed result per session")
        rejected = []
        for profile in pending:
            result, turn = by_profile[profile.profile_id], expected[profile.profile_id]
            if result.turn_id != turn.turn_id or result.input_digest != turn.input_digest:
                raise HarnessError("committed result does not match its frozen turn")
            if result.submission_status != "accepted":
                rejected.append({"profile_id": profile.profile_id,
                                 "errors": [e.to_dict() for e in result.validation_errors][:24]})
                continue
            rows = read_submission(result.submission, result.submitted_artifacts, self.root, case=self.case,
                                   count=counts[profile.profile_id], measured_keys=measured_keys,
                                   measured_ids=measured_ids)
            saved = {"history_sha256": history_digest, "count": counts[profile.profile_id],
                     "turn": {"turn_id": result.turn_id, "profile_id": profile.profile_id,
                              "session_id": result.session_id, "usage": result.usage},
                     "rows": [{**row, "note": {
                         "source": "persistent_harness", "profile_id": profile.profile_id,
                         "session_id": result.session_id, "turn_id": result.turn_id,
                         "submission_id": result.submission_id, "submission_digest": result.submission_digest,
                         "item_index": index, "artifacts": dict(result.artifacts)}} for index, row in enumerate(rows)]}
            atomic_json_write(accepted_dir / f"{profile.profile_id}.json", saved)
            accepted[profile.profile_id] = saved
        if rejected:
            issued = [{"turn_id": r.turn_id, "profile_id": r.profile_id, "session_id": r.session_id} for r in results]
            raise ProposalExhausted({"reason": "harness_validation_attempts_exhausted", "sessions": rejected},
                                    attempts=self._batch([], issued, counts).attempts)

    def _batch(self, rows, turn_records, counts) -> ProposalBatch:
        attempts = [attach_proposal_attempt_receipt(ProposalResponse(text=json.dumps(
            {"turn_id": t["turn_id"], "profile_id": t["profile_id"], "occurrences": counts[t["profile_id"]]}),
            metadata={"mode": "persistent_harness", "session_id": t["session_id"]}), f"harness:{t['turn_id']}")
            for t in turn_records]
        return ProposalBatch(rows, attempts, {"sessions": len(turn_records), "occurrences_per_session": counts})

    def _account(self, turns, usage) -> None:
        if self.runtime is None:
            return
        for turn in turns:
            key = f"harness:{turn.turn_id}"
            item = usage.get(turn.profile_id)
            if item is None:
                self.runtime.consume_many({"harness_usage_unknown": 1}, usage_key=key)
            else:
                self.runtime.consume_many({
                    "harness_provider_calls": int(item.get("providerCalls", 0)),
                    "harness_tool_calls": sum(int(v) for v in item.get("toolCalls", {}).values()),
                    "harness_artifact_bytes": int(item.get("artifactBytes", 0)),
                }, usage_key=key)
            account_tokens(self.runtime, self.pricing, "harness", harness_turn_tokens(self.root, turn.turn_id),
                           usage_key=key)

    def _commit_cursors(self, selected, records) -> None:
        for profile in selected:
            atomic_json_write(self.root / f"history_cursor_{profile.profile_id}.json", {
                "history_to_seq": len(records), "history_prefix_sha256": canonical_sha256(records)})

    def _turns(self, batch_root, request_digest, round_idx, recovery_pass, records, selected, counts,
               refill=0, avoid=()):
        path = batch_root / "input.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["request_sha256"] != request_digest:
                raise ValueError("Harness turn input changed on resume")
            turns = tuple(HarnessTurn(row["profileId"], row["turnId"], row["roundIndex"], row["historyFromSeq"],
                                      row["historyToSeq"], row["historyDigest"], row["message"])
                          for row in saved["turns"])
            if any(t.input_digest != row["inputDigest"] for t, row in zip(turns, saved["turns"])):
                raise ValueError("frozen Harness turn input digest mismatch")
            return turns
        turns = []
        wall = self.args.harness_wall_time_seconds
        for profile in selected:
            cursor_path = self.root / f"history_cursor_{profile.profile_id}.json"
            cursor_state = json.loads(cursor_path.read_text()) if cursor_path.exists() else {"history_to_seq": 0}
            cursor = cursor_state["history_to_seq"]
            if not 0 <= cursor <= len(records) or cursor_state.get(
                    "history_prefix_sha256", canonical_sha256(records[:cursor])) != canonical_sha256(records[:cursor]):
                raise ValueError("measured-history prefix changed on Harness resume")
            delta = records[cursor:]
            payload = {
                "task": "researchgym", "case_id": self.case.case_id,
                "message_type": "history_delta" if cursor_path.exists() else "campaign_bootstrap",
                "new_measured_candidates": [compact(r) for r in delta[-24:]],
                "omitted_delta_candidates": max(0, len(delta) - 24),
                "history_from_seq": cursor, "history_to_seq": len(records),
                "candidate_count": counts[profile.profile_id],
                "submission": {"artifact_path": ARTIFACT, "fields": ["program", "change_summary", "rationale"],
                               "optional_fields": ["comparison_candidate_ids"]},
                "occurrence_rules": ("Entries of your file must be distinct programs. Programs in the measured history "
                                     "are rejected; unmeasured ideas from earlier turns remain eligible. Other sessions "
                                     "work independently; agreement between sessions is recorded as proposal frequency."),
                "time_plan": {"turn_wall_seconds": wall, "finish_research_by_seconds": int(wall * 0.5),
                              "first_submission_by_seconds": int(wall * 0.7)},
                "tools": list(TOOL_NAMES),
            }
            if refill:
                payload["replenishment"] = {
                    "reason": "independent sessions proposed too few distinct programs for the evaluation batch",
                    "already_proposed_this_round": avoid[-32:],
                    "request": "submit programs that differ from these"}
            refill_part = f"_f{refill:02d}" if refill else ""
            turn_id = f"p{recovery_pass:03d}_r{round_idx:06d}{refill_part}_{profile.profile_id}_{request_digest[:16]}"
            message = ("Continue your independent research. Use describe_researchgym_case and get_measured_history, "
                       "analyze in the sandbox, write candidates.json with a script, run check_candidate_programs, "
                       "then submit_candidates. Repair indexed errors in this session.\n"
                       + json.dumps(payload, separators=(",", ":")))
            turns.append(HarnessTurn(profile.profile_id, turn_id, round_idx, cursor, len(records),
                                     canonical_sha256(delta), message))
        atomic_json_write(path, {"request_sha256": request_digest, "turns": [t.to_dict() for t in turns]})
        return tuple(turns)
