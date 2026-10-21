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
import fcntl
from pathlib import Path

from ldm_tts.engine.expansion import attach_proposal_attempt_receipt
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (
    HarnessArtifactRule, HarnessClient, HarnessError, HarnessLimits, HarnessNetworkPolicy, HarnessProfile,
    HarnessProviderAuthorizationRequest, HarnessSubmissionContract, HarnessSubmissionError,
    HarnessSubmissionValidation, HarnessToolExtension,
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
from .usage import ApiPricing, account_tokens, harness_turn_tokens, provider_response_tokens

RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "harness"
TOOL_FILE = "tools/researchgym_tools.mjs"
TOOL_NAMES = ("describe_researchgym_case", "get_measured_history", "check_candidate_programs")
PROPOSAL_SKILLS = ("researchgym-method-research", "ml-experiment-analysis")
POLICY_SKILLS = ("compile-ldm-policy",)
ARTIFACT = "candidates.json"


class ProviderAuthorizationLedger:
    """Host-owned, durable authorizations for individual Pi provider requests.

    A Harness turn can make several provider calls. Checking a dollar cap only
    once before ``run_turn`` therefore lets every call in that turn pass. This
    ledger is called by the shared 0.2 Pi authorization handshake for every
    request, records an idempotent receipt, and keeps its conservative reserve
    live until the sidecar capture has been accounted for.
    """

    def __init__(self, *, root: Path, runtime, pricing: ApiPricing, campaign_id: str,
                 usage_prefix: str, reserve_usd: float | None) -> None:
        if usage_prefix not in {"harness", "policy"}:
            raise ValueError("unknown provider authorization usage prefix")
        if reserve_usd is not None and reserve_usd <= 0:
            raise ValueError("provider request reservation must be positive")
        self.root = Path(root) / "provider_authorizations"
        self.runtime, self.pricing = runtime, pricing
        self.campaign_id, self.usage_prefix, self.reserve_usd = campaign_id, usage_prefix, reserve_usd
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _identity(request: HarnessProviderAuthorizationRequest) -> dict[str, str]:
        return {
            "campaign_id": request.campaign_id,
            "profile_id": request.profile_id,
            "turn_id": request.turn_id,
            "provider_request_id": request.provider_request_id,
        }

    def _path(self, identity: dict[str, str]) -> Path:
        return self.root / (canonical_sha256(identity) + ".json")

    def _records(self) -> list[tuple[Path, dict]]:
        records = []
        for path in sorted(self.root.glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise HarnessError(f"invalid provider authorization receipt: {path}") from exc
            if not isinstance(record, dict):
                raise HarnessError(f"invalid provider authorization receipt: {path}")
            records.append((path, record))
        return records

    def _outstanding_usd(self) -> float:
        # Both Pi pools belong to one campaign and must share the same reserve.
        outstanding = 0.0
        for pool in ("harness", "policy_harness"):
            for path in (self.root.parent.parent / pool / "provider_authorizations").glob("*.json"):
                try:
                    record = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise HarnessError(f"invalid provider authorization receipt: {path}") from exc
                if not isinstance(record, dict):
                    raise HarnessError(f"invalid provider authorization receipt: {path}")
                if record.get("campaign_id") == self.campaign_id and record.get("status") == "authorized":
                    outstanding += float(record.get("reserved_usd", 0.0))
        return outstanding

    def _settle_completed_requests_locked(self) -> None:
        """Account captured requests before admitting more calls in any pool.

        A turn may contain many model calls and several turns run concurrently.
        Waiting for the whole turn to finish can exhaust the budget in *reserves*
        even when measured spending is tiny.  The shared campaign lock covers
        both the receipt transition and the budget update, while the usage key
        makes a crash between those writes safe to replay.
        """
        if self.pricing.budget_usd is None:
            return
        campaign_root = self.root.parent.parent
        for pool, prefix in (("harness", "harness"), ("policy_harness", "policy")):
            pool_root = campaign_root / pool
            indices: dict[Path, list[dict]] = {}
            for receipt_path in sorted((pool_root / "provider_authorizations").glob("*.json")):
                try:
                    receipt = json.loads(receipt_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise HarnessError(f"invalid provider authorization receipt: {receipt_path}") from exc
                if receipt.get("campaign_id") != self.campaign_id or receipt.get("status") != "authorized":
                    continue
                turn_root = pool_root / "sessions" / receipt["profile_id"] / "turns" / receipt["turn_id"]
                index = turn_root / "provider_index.jsonl"
                if index not in indices:
                    if not index.is_file():
                        indices[index] = []
                    else:
                        rows = []
                        # Ignore an incomplete trailing write, but reject a malformed
                        # completed line instead of silently releasing its reserve.
                        for line in index.read_text().splitlines(keepends=True):
                            if not line.endswith("\n"):
                                continue
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError as exc:
                                raise HarnessError(f"invalid provider capture index: {index}") from exc
                        indices[index] = rows
                matches = [row for row in indices[index]
                           if row.get("requestId") == receipt["provider_request_id"]
                           and row.get("type") in {"provider_exchange", "provider_transport_error"}]
                if not matches:
                    continue
                if len(matches) != 1:
                    raise HarnessError("duplicate provider capture for one authorization")
                capture = matches[0]
                if capture.get("turnId") != receipt["turn_id"] or capture.get("profileId", receipt["profile_id"]) != receipt["profile_id"]:
                    raise HarnessError("provider capture identity does not match authorization")
                if capture["type"] == "provider_transport_error":
                    continue  # The call may have reached the provider; retain its reserve.
                response = capture.get("response", {})
                status = response.get("status")
                if not isinstance(status, int) or not 200 <= status < 300:
                    raise HarnessError(f"provider returned HTTP {status}; inspect the redacted capture before retrying")
                artifact = f"provider/{receipt['provider_request_id']}.response.bin"
                if response.get("artifact") != artifact:
                    raise HarnessError("provider capture artifact does not match authorization")
                tokens = provider_response_tokens(turn_root / artifact)
                if tokens is None:
                    continue  # Unknown usage must keep the full reservation live.
                cost = self.pricing.cost(tokens)
                if cost is None:
                    raise HarnessError("API budget requires configured provider token prices")
                amounts = {f"{prefix}_input_tokens": tokens["input_tokens"],
                           f"{prefix}_output_tokens": tokens["output_tokens"], "api_cost_usd": cost}
                self.runtime.consume_many(amounts, usage_key=f"{prefix}:provider_usage:{receipt_path.stem}")
                atomic_json_write(receipt_path, {**receipt, "status": "reconciled"})

    def settle_completed_requests(self) -> None:
        lock_path = self.root.parent.parent / "provider_authorizations.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._settle_completed_requests_locked()

    def authorize(self, request: HarnessProviderAuthorizationRequest) -> bool:
        if request.campaign_id != self.campaign_id:
            raise HarnessError("provider authorization belongs to a different campaign")
        identity = self._identity(request)
        path = self._path(identity)
        record = {
            **identity,
            "request_digest": request.request_digest,
            "reserved_usd": 0.0 if self.pricing.budget_usd is None else self.reserve_usd,
            "status": "authorized",
        }
        lock_path = self.root.parent.parent / "provider_authorizations.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._settle_completed_requests_locked()
            if path.exists():
                saved = json.loads(path.read_text())
                expected = {key: value for key, value in record.items() if key != "status"}
                previous = {key: value for key, value in saved.items() if key != "status"}
                if previous != expected or saved.get("status") not in {"authorized", "reconciled"}:
                    raise HarnessError("provider authorization identity was reused with a different request")
            else:
                if self.pricing.budget_usd is not None:
                    if self.reserve_usd is None:
                        raise ValueError("Harness API budgets require --harness-provider-reserve-usd")
                    self.pricing.authorize_reservation(
                        self.runtime, outstanding_usd=self._outstanding_usd(), request_usd=self.reserve_usd,
                    )
                atomic_json_write(path, record)
            self.runtime.consume_many({
                f"{self.usage_prefix}_provider_requests": 1,
                f"{self.usage_prefix}_provider_reservation_usd": record["reserved_usd"],
            }, usage_key=f"{self.usage_prefix}:provider:{canonical_sha256(identity)}")
        return True

    def reconcile_turn(self, turn_id: str, *, reported_provider_calls: int | None,
                        tokens: dict[str, float] | None) -> None:
        """Settle only a fully captured turn; unknown calls keep their reserve live."""
        if self.pricing.budget_usd is not None:
            self.settle_completed_requests()
        all_rows = [(path, record) for path, record in self._records() if record.get("turn_id") == turn_id]
        rows = [(path, record) for path, record in all_rows if record.get("status") == "authorized"]
        if not all_rows:
            return
        if reported_provider_calls is not None and (
                isinstance(reported_provider_calls, bool) or not isinstance(reported_provider_calls, int)
                or reported_provider_calls < 0 or reported_provider_calls > len(all_rows)):
            raise HarnessError("sidecar provider usage exceeds host authorizations")
        if self.pricing.budget_usd is not None:
            # Known usage was already accounted once per request.  Unresolved
            # captures remain reserved and are visible as unknown usage.
            if reported_provider_calls == len(all_rows):
                self.runtime.consume_many({f"{self.usage_prefix}_usage_unknown": len(rows)},
                                          usage_key=f"{self.usage_prefix}:unknown:{turn_id}")
            return
        # A lower reported count or absent/invalid capture may still contain an
        # in-flight paid request, so leave all reservations outstanding.
        if reported_provider_calls != len(all_rows) or tokens is None or tokens.get("unknown_calls", 0):
            return
        for path, record in rows:
            atomic_json_write(path, {**record, "status": "reconciled"})

    def settle_captured_turns(self) -> None:
        """Best-effort settlement for policy turns, whose controller owns usage."""
        turn_ids = {record.get("turn_id") for _, record in self._records()
                    if record.get("status") == "authorized" and isinstance(record.get("turn_id"), str)}
        for turn_id in sorted(turn_ids):
            captures = 0
            for index in self.root.parent.rglob("provider_index.jsonl"):
                for line in index.read_text().splitlines():
                    try:
                        if json.loads(line).get("turnId") == turn_id:
                            captures += 1
                    except json.JSONDecodeError as exc:
                        raise HarnessError(f"invalid provider capture index: {index}") from exc
            self.reconcile_turn(turn_id, reported_provider_calls=captures,
                                tokens=harness_turn_tokens(self.root.parent, turn_id))


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
            # The deployed Pi sidecar's request-body override is applied after
            # its first-turn hook.  This endpoint rejects required mode when
            # the research agent exposes several tools.
            provider_request_body={"tool_choice": "auto", "max_output_tokens": args.llm_max_tokens,
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
        self.provider_meter: ProviderAuthorizationLedger | None = None

    def configure_provider_authorization(self, meter: ProviderAuthorizationLedger) -> None:
        if meter.campaign_id != self.client.config.campaign_id:
            raise ValueError("provider authorization ledger campaign does not match Harness client")
        self.provider_meter = meter

    def collect(self, *, round_idx, count, batch_size, records, measured_keys, recovery_pass,
                refill=0, avoid=()) -> ProposalBatch:
        del batch_size  # Each session submits its share of the occurrences in one file.
        if self.clock is not None and self.clock.remaining() <= 0:
            raise WallClockExhausted("campaign wall-clock allowance is exhausted before research turns")
        if self.runtime is not None:
            self.pricing.check(self.runtime)
        history_digest = write_history(self.root / "measured_history.json", records)
        # Pi's runTurns contract is the complete initialized profile set, even
        # when a refill asks for fewer candidates than there are sessions.
        selected = self.session_profiles
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
            self._run_pending(selected, pending, round_idx, recovery_pass, records, counts, history_digest, measured_keys,
                              measured_ids, accepted_dir, accepted, phase, refill, list(avoid))
        rows, turn_records = [], []
        for profile in selected:
            saved = accepted[profile.profile_id]
            rows.extend(saved["rows"])
            turn_records.append(saved["turn"])
        self._commit_cursors(selected, records)
        return self._batch(rows, turn_records, counts)

    def _run_pending(self, selected, pending, round_idx, recovery_pass, records, counts, history_digest, measured_keys,
                     measured_ids, accepted_dir, accepted, phase, refill, avoid) -> None:
        request_digest = canonical_sha256({"round": round_idx, "recovery_pass": recovery_pass, "history": history_digest,
                                           "refill": refill, "avoid": avoid,
                                           "counts": counts,
                                           "case": self.case.case_id, "profiles": [p.to_dict() for p in selected]})
        batch_root = self.root / "proposal_turns" / f"pass-{recovery_pass:03d}" / phase
        new_turns = self._turns(batch_root, request_digest, round_idx, recovery_pass, records, selected, counts,
                                refill, avoid)
        turns = []
        for turn in new_turns:
            previous = accepted.get(turn.profile_id)
            if previous is not None:
                frozen = previous.get("input")
                if frozen is None:
                    # Older receipts predate the frozen input field. Recover it
                    # from the persisted request, never synthesize a new turn.
                    for path in sorted((self.root / "proposal_turns").glob("pass-*/round-*/input.json")):
                        for row in json.loads(path.read_text()).get("turns", []):
                            if row.get("turnId") == previous["turn"]["turn_id"]:
                                frozen = row
                                break
                        if frozen is not None:
                            break
                if frozen is None:
                    raise HarnessError("accepted session has no committed turn input to replay")
                turn = HarnessTurn(frozen["profileId"], frozen["turnId"], frozen["roundIndex"],
                                   frozen["historyFromSeq"], frozen["historyToSeq"],
                                   frozen["historyDigest"], frozen["message"],
                                   tuple(frozen.get("forbiddenQueryTerms", ())))
                if turn.input_digest != frozen["inputDigest"]:
                    raise HarnessError("accepted session committed turn digest mismatch")
            turns.append(turn)
        turns = tuple(turns)
        expected = {turn.profile_id: turn for turn in turns}

        def validate(submission):
            turn = expected.get(submission.profile_id)
            if turn is None or submission.turn_id != turn.turn_id:
                raise HarnessError("submission does not match the active independent session")
            if submission.profile_id in accepted:
                raise HarnessError("a committed accepted session attempted a new submission")
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
            kwargs = {"submission_validator": validate, "recovery_timeout_seconds": recovery}
            if self.provider_meter is not None:
                kwargs["provider_authorizer"] = self.provider_meter.authorize
            results = self.client.run_turn(turns, **kwargs)
            usage = {r.profile_id: r.usage for r in results}
        except HarnessError as exc:
            usage = exc.turn_usage
            raise
        finally:
            self._account(turns, usage)
        by_profile = {r.profile_id: r for r in results}
        if set(by_profile) != set(expected) or len(results) != len(self.session_profiles):
            raise HarnessError("research barrier requires exactly one committed result per session")
        for profile_id in accepted:
            if not by_profile[profile_id].replayed:
                raise HarnessError("accepted session was executed instead of replayed")
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
                     "input": turn.to_dict(),
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
            tokens = harness_turn_tokens(self.root, turn.turn_id)
            if self.provider_meter is None or self.pricing.budget_usd is None:
                account_tokens(self.runtime, self.pricing, "harness", tokens, usage_key=key)
            if self.provider_meter is not None:
                calls = None if item is None else item.get("providerCalls")
                self.provider_meter.reconcile_turn(turn.turn_id, reported_provider_calls=calls, tokens=tokens)

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
