"""History-aware Pi research, with design tools hosted by the campaign worker."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import tempfile
import threading

from ldm_tts.contracts import CandidateRejection, RawProposal
from ldm_tts.harness import (HarnessClient, HarnessLimits, HarnessProfile, HarnessSubmissionContract,
                             HarnessToolExtension, directory_sha256, file_sha256)
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime
from ldm_tts.harness.pi_research import PiResearchSource, ProposalContract
from ldm_tts.transport.openai import OpenAICompatibleProposalClient
from tasks.antibody.core.engine_adapters import AntibodyCandidateDomain

RESOURCES = Path(__file__).resolve().parents[1] / "resources/harness"
TOOLS = ("design_h3", "sequence_h3")
SKILL = "skills/cdrh3-design"
ROLE = "profiles/research/AGENTS.md"


def profile():
    return HarnessProfile("cdrh3_research", Path("/resources") / ROLE,
                          (Path("/resources") / SKILL,), file_sha256(RESOURCES / ROLE),
                          (directory_sha256(RESOURCES / SKILL),))


def submission_contract(count):
    return HarnessSubmissionContract("cdrh3_design_ids", "submit_designs", {
        "type": "object", "additionalProperties": False, "required": ["design_ids"],
        "properties": {"design_ids": {"type": "array", "minItems": count, "maxItems": count,
                                      "items": {"type": "string"}}}}, max_validation_attempts=5)


@contextmanager
def worker_socket(worker):
    from socketserver import UnixStreamServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/" or not 0 < length <= 4096:
                    raise ValueError("Invalid design request")
                rows = worker.execute(**json.loads(self.rfile.read(length)))
                eligible = worker.eligible(rows)
                data = {"backbone_id": rows[0]["backbone_id"], "candidates": eligible,
                        "sampled": len(rows), "eligible_unique_new": len(eligible),
                        "remaining": worker.remaining()}
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
    """Resolve issued design IDs and let the shared history validator enforce CAS/dedup."""

    def __init__(self, client, service, worker, args):
        self.worker, self.args = worker, args
        self.required = 1
        super().__init__(client, service, self.make_contract(), name=f"antibody-{args.method}", instructions=(
            "Read the cdrh3-design skill, use experiment_history to inspect measured and pending sequences, "
            "then use design_h3 or sequence_h3 and submit issued eligible design IDs. Lower energy is better."))

    def make_contract(self):
        def admit(design_id):
            if not isinstance(design_id, str):
                return None
            catalog = self.worker.candidates()
            row = catalog.get(design_id)
            if row is None or row.get("violations"):
                return None
            if self.worker.round == 0 and row != next((r for r in catalog.values() if not r["violations"]), None):
                return None
            result = AntibodyCandidateDomain(11).admit(RawProposal(row, self.args.method))
            return None if isinstance(result, CandidateRejection) else result

        return ProposalContract(submission_contract(self.required), admit, (self.required, self.required), "design_ids")

    def message(self, round_input, header):
        text = super().message(round_input, header)
        details = {"antigen": self.args.antigen, "oracle": self.args.oracle, "method": self.args.method,
                   "remaining": self.worker.remaining(), "required_candidates": self.required}
        if self.worker.round == 0:
            details["initialization"] = (f"Read the skill, then call design_h3 with count={self.args.samples_per_call} "
                f"and temperature={self.args.temperature}. Submit the first eligible design ID.")
        return text + "\n\nTask settings: " + json.dumps(details, sort_keys=True)

    def settle(self, round_input):
        # An abandoned turn is replayed with its own generation's exact count
        # and initialization rule, not the current generation's validator.
        previous = None
        for generation in sorted(round_input.get("unsettled") or (), reverse=True):
            turn_id = f"g{int(generation):06d}-{self.profiles[0]}"
            sent = self.service.sent_turn(self.profiles[0], turn_id)
            if sent is not None and self.service.terminal_decision(self.profiles[0], turn_id) is not None:
                settings = json.loads(sent["message"].rsplit("\n\nTask settings: ", 1)[1])
                previous = (int(generation) - 1, int(settings["required_candidates"]))
                break
        if previous is None:
            return super().settle(round_input)
        current = (self.worker.round, self.required, self.contract)
        try:
            self.worker.round, self.required = previous
            self.contract = self.make_contract()
            return super().settle(round_input)
        finally:
            self.worker.round, self.required, self.contract = current

    def research(self, round_input, history):
        self.worker.round = round_input["generation"] - 1
        self.worker.evaluated = {o.canonical_key for o in history} | set(round_input.get("pending_keys", ()))
        successful = any(o.evaluation.succeeded for o in history)
        self.required = self.args.pool_size if successful and self.args.method == "ldm_harness" else 1
        self.contract = self.make_contract()
        return super().research(round_input, history)


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
    # The host validator refines this schema's count for each research turn.
    contract = HarnessSubmissionContract("cdrh3_design_ids", "submit_designs", {
        "type": "object", "additionalProperties": False, "required": ["design_ids"],
        "properties": {"design_ids": {"type": "array", "minItems": 1, "maxItems": args.pool_size,
                                      "items": {"type": "string"}}}}, max_validation_attempts=5)
    config = PiHarnessConfig(artifact_root=Path("/artifacts"), profiles=(profile(),), campaign_id=campaign_id,
        task_id="antibody_rfd3", case_id=args.antigen, seed=args.seed, submission_contract=contract,
        limits=HarnessLimits(wall_time_seconds=900, tool_call_budgets={"web_search": 0, "web_fetch": 0}),
        base_url=url.removesuffix("/responses"), model=model,
        thinking=os.environ.get("LLM_REASONING_EFFORT", "off"), context7_enabled=False,
        force_first_tool_call=False, guest_runtime=load_pi_guest_runtime("antibody_rfd3", RESOURCES / "image"),
        tool_extensions=(HarnessToolExtension(Path("/resources") / extension, file_sha256(RESOURCES / extension), TOOLS),),
        history=bridge.service.config)
    with worker_socket(worker) as socket_dir:
        cmd = pi_sidecar_command(root, args.sidecar_image, cache_root=args.guest_cache,
                                 mounts=((RESOURCES, "/resources", True), (socket_dir, "/worker", True)))
        with HarnessClient(cmd, api_key=key, config=config, history=bridge.service) as client:
            bridge.add_client(client)
            yield AntibodyPiResearch(client, bridge.service, worker, args)
