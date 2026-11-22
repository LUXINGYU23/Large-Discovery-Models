"""A persistent Pi proposal session backed by the host design worker."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import tempfile
import threading

from ldm_tts.harness import (
    HarnessClient, HarnessLimits, HarnessProfile, HarnessSubmissionContract,
    HarnessSubmissionError, HarnessSubmissionValidation, HarnessToolExtension, HarnessTurn,
    canonical_sha256, directory_sha256, file_sha256,
)
from ldm_tts.harness.container import pi_sidecar_command
from ldm_tts.harness.pi import PiHarnessConfig, load_pi_guest_runtime
from ldm_tts.transport.openai import OpenAICompatibleProposalClient

RESOURCES = Path(__file__).resolve().parents[1] / "resources/harness"
TOOLS = ("design_h3", "sequence_h3", "sequence_h3_soluble", "score_interface")
SKILL = "skills/cdrh3-design"
ROLE = "profiles/research/AGENTS.md"


def profile():
    return HarnessProfile("cdrh3_research", Path("/resources") / ROLE,
                          (Path("/resources") / SKILL,), file_sha256(RESOURCES / ROLE),
                          (directory_sha256(RESOURCES / SKILL),))


def submission_contract():
    return HarnessSubmissionContract("cdrh3_design_ids", "submit_designs", {
        "type": "object", "additionalProperties": False, "required": ["design_ids"],
        "properties": {"design_ids": {"type": "array", "minItems": 1,
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
                params = json.loads(self.rfile.read(length))
                if params.get("action") == "score_interface":
                    from .interface import score_interface
                    data = score_interface(worker, params.get("artifact_id"))
                    status = 200
                    body = json.dumps(data).encode()
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                rows = worker.execute(**params)
                eligible = worker.eligible(rows)
                data = {"backbone_id": rows[0]["backbone_id"], "candidates": eligible,
                        "sampled": len(rows), "eligible_unique_new": len(eligible),
                        "remaining": worker.remaining()}
                if not eligible:
                    data["next_action"] = "Choose eligible IDs from earlier calls or sample again, for example with sequence_h3 on another existing backbone."
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


@contextmanager
def open_agent(args, runtime, worker):
    url, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL_NAME")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not (url and model and key):
        raise ValueError("Pi requires LLM_BASE_URL, LLM_MODEL_NAME and LLM_API_KEY")
    probe = OpenAICompatibleProposalClient(url=url, model=model, api_key=key,
                                          wire_api="responses", max_retries=0)
    probe.preflight()
    root = runtime.run_dir / "harness"
    root.mkdir(exist_ok=True)
    extension = "tools/design.mjs"
    config = PiHarnessConfig(
        artifact_root=Path("/artifacts"), profiles=(profile(),), campaign_id=runtime.run_id,
        task_id="antibody_rfd3", case_id=args.antigen, seed=args.seed,
        submission_contract=submission_contract(), limits=HarnessLimits(wall_time_seconds=900,
            tool_call_budgets={"web_search": 0, "web_fetch": 0}),
        base_url=url.removesuffix("/responses"), model=model, context7_enabled=False,
        force_first_tool_call=False, guest_runtime=load_pi_guest_runtime("antibody_rfd3", RESOURCES / "image"),
        tool_extensions=(HarnessToolExtension(Path("/resources") / extension,
                                              file_sha256(RESOURCES / extension), TOOLS),))
    with worker_socket(worker) as socket_dir:
        cmd = pi_sidecar_command(root, args.sidecar_image, cache_root=args.guest_cache,
                                 mounts=((RESOURCES, "/resources", True), (socket_dir, "/worker", True)))
        with HarnessClient(cmd, api_key=key, config=config) as client:
            yield PiAgent(client, worker, runtime)


class PiAgent:
    def __init__(self, client, worker, runtime):
        self.client, self.worker, self.runtime = client, worker, runtime
        self.cursor = 0

    def propose(self, round_idx, history, count):
        seen = self.worker.evaluated | {row["sequence"] for row in history}
        message = {"antigen": self.worker.args.antigen, "oracle": self.worker.args.oracle,
                   "method": self.worker.args.method, "measured_history": history,
                   "remaining": self.worker.remaining(), "required_candidates": count,
                   "instructions": "Read cdrh3-design on your first turn. Call design_h3 or sequence_h3. "
                   "Submit exactly the requested number of distinct eligible design IDs. "
                   "Only IDs listed as eligible may be submitted. If a call produces no eligible candidates, "
                   "sample another backbone or resample a previously measured backbone. Earlier eligible, unmeasured "
                   "designs may also be submitted. Allow time to repair rejected submissions."}
        if getattr(self.worker.args, "research_instructions", ""):
            message["research_guidance"] = self.worker.args.research_instructions
        if round_idx == 0:
            message["first_turn_steps"] = (
                "1. First call the Pi read tool with path "
                "/workspace/.ldm-resources/skills/0/cdrh3-design/SKILL.md and read its full instructions. "
                f"2. Then call design_h3(count={self.worker.args.samples_per_call}, temperature={self.worker.args.temperature}). "
                "Submit the first eligible returned design ID. Repeat only if all candidates are invalid.")
        turn = HarnessTurn("cdrh3_research", f"design-{round_idx}", round_idx, self.cursor,
                           len(history), canonical_sha256(history), json.dumps(message))

        def resolve(payload):
            ids = payload.get("design_ids")
            if not isinstance(ids, list) or len(ids) != count:
                raise ValueError(f"Submit exactly {count} candidate IDs")
            catalog = self.worker.candidates()
            rows = [catalog[key] for key in ids]
            if round_idx == 0 and rows[0] != next((r for r in catalog.values() if not r["violations"]), None):
                raise ValueError("Select the first eligible design from the shared initialization")
            sequences = {row["sequence"] for row in rows}
            if len(sequences) != count or sequences & seen or any(row["violations"] for row in rows):
                available = [r["design_id"] for r in self.worker.eligible(catalog.values())
                             if r["sequence"] not in seen][:16]
                raise ValueError(f"Submitted IDs repeat a measured H3, duplicate a sequence or fail constraints. Eligible unmeasured IDs: {available}")
            return rows

        def validate(submission):
            try:
                resolve(submission.submission)
            except (ValueError, KeyError, TypeError) as exc:
                return HarnessSubmissionValidation("retry", (HarnessSubmissionError(
                    "/design_ids", "invalid_design_ids", str(exc), "Choose eligible IDs returned by a tool."),))
            return HarnessSubmissionValidation()

        result, = self.client.run_turn((turn,), submission_validator=validate)
        if (result.submission_status != "accepted" or result.turn_id != turn.turn_id
                or result.input_digest != turn.input_digest):
            raise ValueError("Pi did not commit the requested turn")
        self.cursor = len(history)
        self.runtime.consume_many({"harness_turns": 1, "harness_provider_calls": result.usage.get("providerCalls", 0),
                                   "harness_tool_calls": sum(result.usage.get("toolCalls", {}).values())},
                                  usage_key=turn.turn_id)
        self.runtime.record("pi_design_turn", {"turn": turn.turn_id, "usage": result.usage,
                            "history_digest": turn.history_digest, "design_ids": result.submission["design_ids"]})
        return resolve(result.submission)
