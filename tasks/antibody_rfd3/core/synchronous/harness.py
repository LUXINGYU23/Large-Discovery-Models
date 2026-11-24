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

from ..optional_tools import active_tools, optional_host_flags
from ..interface import score_interface

RESOURCES = Path(__file__).resolve().parents[2] / "resources/baseline_harness"
TOOLS = ("design_h3", "sequence_h3", "mutate_h3", "register_h3", "candidate_catalog")
SKILL = "skills/cdrh3-design"
ROLE = "profiles/research/AGENTS.md"
TURN_RECOVERY_SECONDS = 600


def profile(args=None):
    skills = [SKILL]
    if getattr(args, "enable_protein_tools", False) or getattr(args, "enable_boltzgen", False):
        skills.append("skills/cdrh3-protein-tools")
    return HarnessProfile("cdrh3_research", Path("/resources") / ROLE,
                          tuple(Path("/resources") / skill for skill in skills), file_sha256(RESOURCES / ROLE),
                          tuple(directory_sha256(RESOURCES / skill) for skill in skills))


def submission_contract():
    return HarnessSubmissionContract("cdrh3_design_ids", "submit_designs", {
        "type": "object", "additionalProperties": False, "required": ["design_ids"],
        "properties": {"design_ids": {"type": "array", "minItems": 1, "uniqueItems": True,
                                      "items": {"type": "string"}}}}, max_validation_attempts=5)


@contextmanager
def worker_socket(worker):
    from socketserver import UnixStreamServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/" or not 0 < length <= 65536:
                    raise ValueError("Invalid design request")
                params = json.loads(self.rfile.read(length))
                if params.get("action") == "score_interface":
                    if not getattr(worker.args, "enable_protein_tools", False):
                        raise ValueError("Optional protein tools must be enabled by the host")
                    if set(params) != {"action", "artifact_id", "call_id", "profile_id"}:
                        raise ValueError("Invalid interface scoring request")
                    if params.get("profile_id") not in ("cdrh3_research",):
                        raise ValueError("Unknown scoring role")
                    data = {"interface_score": score_interface(worker, params.get("artifact_id"))}
                else:
                    if params["action"] == "candidate_catalog":
                        rows = [row for job in worker.records() if job["round"] == worker.round
                                and job["status"] == "succeeded" for row in job.get("candidates", [])]
                    else:
                        rows = worker.execute(**params)
                    eligible = worker.eligible(rows)
                    data = {"backbone_id": rows[0]["backbone_id"] if rows else None, "candidates": eligible,
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
        optional_host_flags(worker, directory)
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
        artifact_root=Path("/artifacts"), profiles=(profile(args),), campaign_id=runtime.run_id,
        task_id="antibody_rfd3", case_id=args.antigen, seed=args.seed,
        submission_contract=submission_contract(), limits=HarnessLimits(wall_time_seconds=900,
            tool_call_budgets={"web_search": 0, "web_fetch": 0}),
        base_url=url.removesuffix("/responses"), model=model, context7_enabled=False,
        force_first_tool_call=False, guest_runtime=load_pi_guest_runtime("antibody_rfd3", RESOURCES / "image"),
        tool_extensions=(HarnessToolExtension(Path("/resources") / extension,
                                              file_sha256(RESOURCES / extension), active_tools(args, TOOLS)),))
    with worker_socket(worker) as socket_dir:
        image = os.environ.get("LDM_PI_RESUME_IMAGE") if args.resume_run else None
        environment = {
            name: os.environ[name] for name in (
                "LDM_PI_PRIOR_IMPLEMENTATION_SHA256", "LDM_PI_RECOVER_TIMEOUT_TURN",
            ) if image and name in os.environ
        }
        if image and worker.prelaunch_retry:
            environment["LDM_PI_PRELAUNCH_RECOVERY_TURN"] = f"design-{worker.prelaunch_retry['round']}"
        cmd = pi_sidecar_command(root, image or args.sidecar_image, cache_root=args.guest_cache,
                                 environment=environment,
                                 mounts=((RESOURCES, "/resources", True), (socket_dir, "/worker", True)))
        with HarnessClient(cmd, api_key=key, config=config) as client:
            yield PiAgent(client, worker, runtime)


class PiAgent:
    def __init__(self, client, worker, runtime):
        self.client, self.worker, self.runtime = client, worker, runtime
        self.cursor = None

    def _turn(self, round_idx, history, message):
        role = "cdrh3_research"
        root = self.runtime.run_dir / "harness"
        def commits(index):
            prefix = f"design-{index}"
            result = []
            for path in (root / "turns").glob(prefix + "*/turn_committed.json"):
                name = path.parent.name
                suffix = name.removeprefix(prefix)
                if suffix and (not suffix.startswith("-retry-") or not suffix[7:].isdigit()):
                    continue
                result.append((0 if not suffix else int(suffix[7:]), json.loads(path.read_text())))
            return sorted(result, key=lambda item: item[0])

        if self.cursor is None:
            previous = commits(round_idx - 1)
            self.cursor = previous[-1][1]["historyToSeq"] if previous else 0
        current = commits(round_idx)
        attempt = 0
        for number, committed in current:
            if committed["submissionStatus"] == "accepted":
                attempt = number
                self.cursor = committed["historyFromSeq"]
                break
            if committed["submissionStatus"] != "rejected":
                raise ValueError("Unexpected committed Pi submission status")
            self.cursor = committed["historyToSeq"]
            attempt = number + 1
        turn_id = f"design-{round_idx}" + (f"-retry-{attempt}" if attempt else "")
        digest = canonical_sha256(history)
        frozen = root / "sessions" / role / "turns" / turn_id / "input.json"
        saved = json.loads(frozen.read_text()) if frozen.is_file() else None
        turn = HarnessTurn(role, turn_id, round_idx, self.cursor, len(history), digest,
                           saved["message"] if saved else json.dumps(message))
        if saved and turn.input_digest != saved["inputDigest"]:
            raise ValueError(f"Saved Pi turn does not match the campaign checkpoint: {turn_id}")
        return turn

    def propose(self, round_idx, history, count):
        seen = self.worker.evaluated | {row["sequence"] for row in history}
        message = {"antigen": self.worker.args.antigen, "oracle": self.worker.args.oracle,
                   "method": self.worker.args.method, "measured_history": history,
                   "remaining": self.worker.remaining(), "required_candidates": count,
                   "instructions": "Read cdrh3-design on your first turn. Use the available sequence or structure tools. "
                   "Submit exactly the requested number of distinct eligible design IDs. "
                   "Only IDs listed as eligible may be submitted. If a call produces no eligible candidates, "
                   "sample another backbone or resample a previously measured backbone. Earlier eligible, unmeasured "
                   "designs may also be submitted. Allow time to repair rejected submissions."}
        eligible = self.worker.eligible(
            row for job in self.worker.records() if job["round"] == round_idx
            and job["status"] == "succeeded" for row in job.get("candidates", ()))
        if len(eligible) >= count:
            message["eligible_current_round_ids"] = [row["design_id"] for row in eligible]
        if round_idx == 0:
            message["first_turn_steps"] = (
                "1. First call the Pi read tool with path "
                "/workspace/.ldm-resources/skills/0/cdrh3-design/SKILL.md and read its full instructions. "
                f"2. Then call design_h3(count={self.worker.args.samples_per_call}, temperature={self.worker.args.temperature}). "
                "Submit the first eligible returned design ID. Repeat only if all candidates are invalid.")
        turn = self._turn(round_idx, history, message)

        def resolve(payload):
            ids = payload.get("design_ids")
            if not isinstance(ids, list) or len(ids) != count:
                raise ValueError(f"Submit exactly {count} candidate IDs")
            catalog = self.worker.candidates()
            rows = [catalog[key] for key in ids]
            if round_idx == 0 and not self.worker.args.seed_bank and rows[0] != next((r for r in catalog.values() if not r["violations"]), None):
                raise ValueError("Select the first eligible design from the shared initialization")
            sequences = {row["sequence"] for row in rows}
            if len(sequences) != count or sequences & seen or any(row["violations"] for row in rows):
                available = [r["design_id"] for r in self.worker.eligible(catalog.values())
                             if r["sequence"] not in seen]
                raise ValueError(f"Submitted IDs repeat a measured H3, duplicate a sequence or fail constraints. Eligible unmeasured IDs: {available}")
            return rows

        def validate(submission):
            try:
                resolve(submission.submission)
            except (ValueError, KeyError, TypeError) as exc:
                return HarnessSubmissionValidation("retry", (HarnessSubmissionError(
                    "/design_ids", "invalid_design_ids", str(exc), "Choose eligible IDs returned by a tool."),))
            return HarnessSubmissionValidation()

        for attempt in range(3):
            result, = self.client.run_turn((turn,), submission_validator=validate,
                                           recovery_timeout_seconds=TURN_RECOVERY_SECONDS)
            if result.turn_id != turn.turn_id or result.input_digest != turn.input_digest:
                raise ValueError("Pi committed a different turn")
            if result.submission_status == "accepted":
                break
            if result.submission_status != "rejected":
                raise ValueError("Pi did not commit the requested turn")
            self.runtime.consume_many({
                "harness_turns": 1,
                "harness_provider_calls": result.usage.get("providerCalls", 0),
                "harness_tool_calls": sum(result.usage.get("toolCalls", {}).values()),
            }, usage_key=turn.turn_id)
            self.runtime.record("pi_design_turn_rejected", {"turn": turn.turn_id, "usage": result.usage})
            if attempt == 2:
                raise ValueError("Pi did not commit the requested turn")
            self.cursor = turn.history_to_seq
            eligible = self.worker.eligible(
                row for job in self.worker.records() if job["round"] == round_idx
                and job["status"] == "succeeded" for row in job.get("candidates", ()))
            message["eligible_current_round_ids"] = [row["design_id"] for row in eligible]
            turn = self._turn(round_idx, history, message)
        self.cursor = len(history)
        self.runtime.consume_many({"harness_turns": 1, "harness_provider_calls": result.usage.get("providerCalls", 0),
                                   "harness_tool_calls": sum(result.usage.get("toolCalls", {}).values())},
                                  usage_key=turn.turn_id)
        self.runtime.record("pi_design_turn", {"turn": turn.turn_id, "usage": result.usage,
                            "history_digest": turn.history_digest, "design_ids": result.submission["design_ids"]})
        return resolve(result.submission)
