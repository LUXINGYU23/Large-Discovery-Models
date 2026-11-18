"""History-aware Pi research and a staged, independently persisted research panel."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import tempfile
import threading

from ldm_tts.async_runtime.lane import ResearchOutcome
from ldm_tts.contracts import CandidateRejection, RawProposal
from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import (HarnessClient, HarnessLimits, HarnessProfile, HarnessSubmissionContract,
                             HarnessToolExtension, directory_sha256, file_sha256)
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime
from ldm_tts.harness.pi_research import PiResearchSource, ProposalContract
from ldm_tts.transport.openai import OpenAICompatibleProposalClient
from tasks.antibody.core.engine_adapters import AntibodyCandidateDomain

RESOURCES = Path(__file__).resolve().parents[1] / "resources/harness"
TOOLS = ("design_h3", "sequence_h3")
PANEL = ("explorer", "refiner", "evidence_reviewer")


def profile(role="cdrh3_research"):
    role_dir = "research" if role == "cdrh3_research" else role
    agents = f"profiles/{role_dir}/AGENTS.md"
    skills = ["skills/cdrh3-design"]
    if role in ("explorer", "refiner"):
        skills.append("skills/cdrh3-experiment-design")
    elif role == "evidence_reviewer":
        skills.append("skills/cdrh3-evidence-review")
    return HarnessProfile(role, Path("/resources") / agents, tuple(Path("/resources") / p for p in skills),
                          file_sha256(RESOURCES / agents), tuple(directory_sha256(RESOURCES / p) for p in skills))


def submission_contract(count, *, review=False):
    properties = {"design_ids": {"type": "array", "minItems": 1, "maxItems": count,
                                  "items": {"type": "string"}}}
    required = ["design_ids"]
    if review:
        properties.update(evidence={"type": "string", "minLength": 1},
                          hypotheses={"type": "string", "minLength": 1})
        required.extend(("evidence", "hypotheses"))
    return HarnessSubmissionContract("cdrh3_design_ids", "submit_designs", {
        "type": "object", "additionalProperties": False, "required": required, "properties": properties,
    }, max_validation_attempts=5)


@contextmanager
def worker_socket(worker):
    from socketserver import UnixStreamServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/" or not 0 < length <= 4096:
                    raise ValueError("Invalid design request")
                request = json.loads(self.rfile.read(length))
                rows = worker.execute(**request)
                eligible = worker.eligible(rows)
                data = {"backbone_id": rows[0]["backbone_id"], "candidates": eligible,
                        "sampled": len(rows), "eligible_unique_new": len(eligible),
                        "remaining": worker.remaining(),
                        "role_remaining": worker.role_remaining(request.get("profile_id"))}
                if not eligible:
                    data["next_action"] = "Choose an earlier eligible ID or sample another design."
                status = 200
            except (TypeError, ValueError, RuntimeError, OSError) as exc:
                status, data = 400, {"error": str(exc)}
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with tempfile.TemporaryDirectory(prefix="h3-") as directory:
        with UnixStreamServer(str(Path(directory) / "design.sock"), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield Path(directory)
            finally:
                server.shutdown()
                thread.join()


class AntibodyPiResearch(PiResearchSource):
    """One profile's persistent session, proposal validator, and watermark chain."""

    def __init__(self, client, service, worker, args, *, workspace=None):
        self.worker, self.args = worker, args
        if len(client.config.profiles) != 1:
            raise ValueError("Each antibody research client must own exactly one profile")
        self.role = client.config.profiles[0].profile_id
        self.workspace = workspace
        self.required = 1
        self.peer_catalog = None
        self.peer_ids = None
        super().__init__(client, service, self.make_contract(), name=f"antibody-{args.method}", instructions=(
            "Read the cdrh3-design skill, use experiment_history to inspect measured and pending sequences, "
            "then submit issued eligible design IDs. Lower energy is better. A proposal is not a measurement."))

    def make_contract(self):
        # Close over the original turn's state: settlement can replay different roles and generations.
        generation_round = getattr(self.worker, "round", 0)
        role = getattr(self, "role", "cdrh3_research")
        peer_ids = None if getattr(self, "peer_ids", None) is None else frozenset(self.peer_ids)
        peer_only = role == "evidence_reviewer" or (role == "refiner" and
            (generation_round == 0 or self.worker.role_budget(role)["sequences"] == 0))
        required = self.required

        def admit(design_id):
            if not isinstance(design_id, str):
                return None
            catalog = self.worker.candidates()
            row = catalog.get(design_id)
            if row is None or row.get("violations"):
                return None
            if peer_only and (peer_ids is None or design_id not in peer_ids):
                return None
            if role in ("explorer", "refiner") and not peer_only:
                if not any(j.get("profile_id") == role and j["backbone_id"] and
                           any(r["design_id"] == design_id for r in j.get("candidates", ()))
                           for j in self.worker.records() if j.get("status") == "succeeded"):
                    return None
            if generation_round == 0 and row != next((r for r in catalog.values() if not r["violations"]), None):
                return None
            result = AntibodyCandidateDomain(11).admit(RawProposal(row, self.args.method))
            return None if isinstance(result, CandidateRejection) else result

        minimum = required if role == "cdrh3_research" else 1
        return ProposalContract(submission_contract(required, review=role == "evidence_reviewer"), admit,
                                (minimum, required), "design_ids")

    def message(self, round_input, header):
        text = super().message(round_input, header)
        details = {"antigen": self.args.antigen, "oracle": self.args.oracle, "method": self.args.method,
                   "role": self.role, "remaining": self.worker.remaining(), "required_candidates": self.required}
        if self.role in PANEL:
            details["role_remaining"] = self.worker.role_remaining(self.role)
            details["candidate_count"] = {"minimum": 1, "maximum": self.required}
        if self.peer_catalog:
            details["peer_catalog"] = self.peer_catalog
        if self.worker.round == 0:
            if self.role in ("explorer", "cdrh3_research"):
                details["initialization"] = (f"Read the skill, then call design_h3 with count={self.args.samples_per_call} "
                    f"and temperature={self.args.temperature}. Submit the first eligible design ID.")
            else:
                details["initialization"] = "Read the peer catalog and submit its first design ID; do not generate."
        return text + "\n\nTask settings: " + json.dumps(details, sort_keys=True)

    def turns(self, round_input, sessions=None):
        turns = super().turns(round_input, sessions)
        # The same in-flight generation must replay exactly, including its pre-spend budget and catalog digest.
        frozen = []
        for turn in turns:
            sent = self.service.sent_turn(turn.profile_id, turn.turn_id)
            if sent is None:
                frozen.append(turn)
            else:
                if (sent["roundIndex"], sent["historyFromSeq"], sent["historyToSeq"], sent["historyDigest"]) != (
                    turn.round_index, turn.history_from_seq, turn.history_to_seq, turn.history_digest):
                    raise ValueError("Saved research turn does not match its frozen history window")
                frozen.append(replace(turn, message=sent["message"],
                                      forbidden_query_terms=tuple(sent.get("forbiddenQueryTerms") or ())))
        return tuple(frozen)

    def _catalog_ids(self, info):
        if not info:
            return None
        if self.workspace is None:
            raise ValueError("Peer catalog has no profile workspace")
        path = self.workspace / Path(info["path"]).relative_to("/workspace")
        if file_sha256(path) != info["sha256"]:
            raise ValueError("Peer catalog differs from the saved turn")
        return {r["design_id"] for r in json.loads(path.read_text())["candidates"]}

    def settle(self, round_input):
        # Every client owns exactly one profile. Restore this particular turn's validator, not another role's.
        role = self.profiles[0]
        previous = None
        for generation in sorted(round_input.get("unsettled") or (), reverse=True):
            turn_id = f"g{int(generation):06d}-{role}"
            sent = self.service.sent_turn(role, turn_id)
            if sent is not None and self.service.terminal_decision(role, turn_id) is not None:
                settings = json.loads(sent["message"].rsplit("\n\nTask settings: ", 1)[1])
                previous = (int(generation) - 1, int(settings["required_candidates"]),
                            self._catalog_ids(settings.get("peer_catalog")))
                break
        if previous is None:
            return super().settle(round_input)
        current = (self.worker.round, self.required, self.peer_ids, self.contract)
        try:
            self.worker.round, self.required, self.peer_ids = previous
            self.contract = self.make_contract()
            return super().settle(round_input)
        finally:
            self.worker.round, self.required, self.peer_ids, self.contract = current

    def research(self, round_input, history):
        self.worker.round = round_input["generation"] - 1
        self.worker.evaluated = {o.canonical_key for o in history} | set(round_input.get("pending_keys", ()))
        if self.role == "cdrh3_research":
            self.required = 1
        self.contract = self.make_contract()
        return super().research(round_input, history)


class RecordingClient:
    """Keep the terminal result for panel review metadata while using the normal client protocol."""

    def __init__(self, client):
        self.client = client
        self.last_results = ()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def run_turn(self, *args, **kwargs):
        self.last_results = self.client.run_turn(*args, **kwargs)
        return self.last_results


class AntibodyResearchPanel:
    """Generate with two lineages, then independently review their submitted designs."""

    version = "1"

    def __init__(self, sources, worker, args, root):
        self.sources, self.worker, self.args, self.root = sources, worker, args, Path(root)
        self.name = "antibody-ldm_harness"

    def catalog(self, role, generation, base_seq, proposals):
        rows = {}
        for proposal in proposals:
            payload = dict(proposal.candidate.payload)
            design_id = payload["design_id"]
            if design_id not in rows:
                job = next((j for j in self.worker.records() if any(
                    c["design_id"] == design_id for c in j.get("candidates", ()))), {})
                rows[design_id] = {**payload, "proposed_by": [], "generation_job": {
                    key: job.get(key) for key in ("profile_id", "round", "action", "count", "temperature", "seed")},
                    "evidence_status": "proposal, not an energy measurement"}
            rows[design_id]["proposed_by"].append(proposal.lineage)
        document = {"generation": generation, "recipient": role, "base_history_seq": base_seq,
                    "candidates": list(rows.values())}
        frozen = self.root / "review_inputs" / role / f"g{generation:06d}.json"
        if frozen.exists():
            if json.loads(frozen.read_text()) != document:
                raise ValueError("Peer catalog changed while recovering a research generation")
        else:
            atomic_json_write(frozen, document)
        relative = Path("review") / f"g{generation:06d}.json"
        workspace = self.sources[role].workspace
        atomic_json_write(workspace / relative, document)
        info = {"path": f"/workspace/{relative.as_posix()}", "sha256": file_sha256(frozen),
                "record": frozen.relative_to(self.root.parent).as_posix()}
        return info, set(rows)

    def research(self, round_input, history):
        generation = int(round_input["generation"])
        self.worker.round = generation - 1
        outputs, proposals, catalogs = [], [], {}
        for role in PANEL:
            source = self.sources[role]
            source.peer_catalog, source.peer_ids = None, None
            if role != "explorer":
                if not proposals:
                    break
                source.peer_catalog, source.peer_ids = self.catalog(
                    role, generation, round_input["watermarks"]["base_history_seq"], proposals)
                catalogs[role] = source.peer_catalog
            if generation == 1:
                source.required = 1
            elif role == "explorer":
                source.required = max(1, (self.args.pool_size + 1) // 2)
            elif role == "refiner":
                source.required = max(1, self.args.pool_size // 2)
            else:
                source.required = min(self.args.pool_size, len(source.peer_ids))
            outcome = source.research(round_input, history)
            outputs.append(outcome)
            proposals.extend(outcome.proposals)
        sessions, settled = {}, {}
        for outcome in outputs:
            sessions.update(outcome.metadata.get("sessions") or {})
            settled.update(outcome.metadata.get("settled") or {})
        delivered = set(outputs[0].metadata.get("task_messages") or ()) if outputs else set()
        for outcome in outputs[1:]:
            delivered.intersection_update(outcome.metadata.get("task_messages") or ())
        metadata = {"sessions": sessions, "task_messages": sorted(delivered),
                    "base_history_seq": round_input["watermarks"]["base_history_seq"],
                    "bootstrap": generation == 1, "peer_catalogs": catalogs}
        if settled:
            metadata["settled"] = settled
        reviewer = self.sources["evidence_reviewer"].client
        if "evidence_reviewer" in sessions and reviewer.last_results:
            result = reviewer.last_results[0]
            if result.submission_status == "accepted":
                metadata["review"] = dict(result.submission)
        failed = [role for role in PANEL if sessions.get(role, {}).get("submission_status") != "accepted"]
        accepted = bool(proposals) and not failed
        reasons = [o.reason for o in outputs if o.reason]
        if failed:
            reasons.insert(0, "panel roles missing an accepted submission: " + ", ".join(failed))
        if not proposals:
            reasons.append("no profile submitted an admissible design")
        return ResearchOutcome("committed" if accepted else "rejected", tuple(proposals) if accepted else (),
                               tuple(w for o in outputs for w in o.withdrawals),
                               "" if accepted else "; ".join(reasons),
                               min((o.final_watermark for o in outputs if o.final_watermark is not None), default=None),
                               metadata)


@contextmanager
def open_agent(args, directory, worker, bridge, campaign_id):
    url, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL_NAME")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not (url and model and key):
        raise ValueError("Pi requires LLM_BASE_URL, LLM_MODEL_NAME and LLM_API_KEY")
    OpenAICompatibleProposalClient(url=url, model=model, api_key=key,
                                   wire_api="responses", max_retries=0).preflight()
    root = Path(directory) / "harness"
    root.mkdir(exist_ok=True)
    extension = "tools/design.mjs"
    roles = PANEL if args.method == "ldm_harness" else ("cdrh3_research",)
    with worker_socket(worker) as socket_dir, ExitStack() as stack:
        sources = {}
        for role in roles:
            artifact_root = root / role if len(roles) > 1 else root
            artifact_root.mkdir(exist_ok=True)
            budgets = {"web_search": 0, "web_fetch": 0, "write": 0, "bash": 0}
            if role == "evidence_reviewer":
                budgets.update(design_h3=0, sequence_h3=0)
            config = PiHarnessConfig(artifact_root=Path("/artifacts"), profiles=(profile(role),),
                campaign_id=campaign_id, task_id="antibody_rfd3", case_id=args.antigen, seed=args.seed,
                submission_contract=submission_contract(args.pool_size, review=role == "evidence_reviewer"),
                limits=HarnessLimits(wall_time_seconds=900, tool_call_budgets=budgets),
                base_url=url.removesuffix("/responses"), model=model,
                thinking=os.environ.get("LLM_REASONING_EFFORT", "off"), context7_enabled=False,
                force_first_tool_call=False, guest_runtime=load_pi_guest_runtime("antibody_rfd3", RESOURCES / "image"),
                tool_extensions=(HarnessToolExtension(Path("/resources") / extension,
                                 file_sha256(RESOURCES / extension), TOOLS),), history=bridge.service.config)
            cmd = pi_sidecar_command(artifact_root, args.sidecar_image, cache_root=args.guest_cache,
                                     mounts=((RESOURCES, "/resources", True), (socket_dir, "/worker", True)))
            client = stack.enter_context(HarnessClient(cmd, api_key=key, config=config, history=bridge.service))
            bridge.add_client(client)
            wrapped = RecordingClient(client) if len(roles) > 1 else client
            workspace = artifact_root / "sessions" / role / "workspace"
            sources[role] = AntibodyPiResearch(wrapped, bridge.service, worker, args, workspace=workspace)
        if len(roles) == 1:
            yield sources[roles[0]]
        else:
            yield AntibodyResearchPanel(sources, worker, args, root)
