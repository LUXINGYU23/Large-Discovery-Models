"""Host-only durable oracle with a permit recorded before each physical worker."""

import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import atomic_json_write
from .oracle_identity import oracle_environment_digest
from .protocol import T3Protocol, digest
from .receipts import Receipts


class OracleService:
    def __init__(self, config_path, root):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if self.config["backend"] not in {"qlib", "assay"}:
            raise ValueError("unknown oracle backend")
        if not self.config.get("data_manifest"):
            raise ValueError("oracle requires a frozen data manifest")
        manifest = json.loads(Path(self.config["data_manifest"]).read_text(encoding="utf-8"))
        if (digest(manifest) != self.config["data_digest"]
                or (manifest.get("backend"), manifest.get("market")) != (self.config["backend"], self.config["market"])):
            raise ValueError("oracle data manifest identity mismatch")
        if self.config["backend"] == "qlib" and (
                Path(self.config["data_root"]).resolve() != Path(manifest["data_root"]).resolve()
                or self.config["benchmark"].lower() != manifest["benchmark"].lower()):
            raise ValueError("Qlib oracle data root or benchmark differs from the manifest")
        if oracle_environment_digest(self.config) != self.config["environment_digest"]:
            raise ValueError("oracle scoring environment identity mismatch")
        self.config_digest = digest(self.config)
        self.root = Path(root)
        self.receipts = Receipts(self.root / "requests")

    def reconciled(self, identity):
        record = self.receipts.load(identity)
        output = self.root / "jobs" / identity / "response.json"
        if record and record["state"] == "dispatch_intent" and output.exists():
            if not (output.parent / "permit.json").exists():
                raise ValueError("worker output lacks its durable permit")
            response = json.loads(output.read_text(encoding="utf-8"))
            if response.get("request_id") != identity:
                raise ValueError("worker output identity mismatch")
            response = wire_response(record["request"], response)
            record.update(state="completed", response=response, response_digest=digest(response))
            atomic_json_write(self.receipts.path(identity), record)
        return record

    def execute(self, request):
        protocol = T3Protocol(**request["protocol"])
        for key in ("backend", "market", "data_digest", "environment_digest"):
            if getattr(protocol, key) != self.config[key]:
                raise ValueError("oracle configuration identity mismatch: " + key)
        try:
            if oracle_environment_digest(self.config) != protocol.environment_digest:
                raise ValueError("oracle scoring environment changed after startup")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise EvaluationPaused("oracle scoring environment integrity failure: " + str(exc),
                                   status="paused_data_integrity") from exc
        identity = request["request_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", identity) or type(request["job_permits"]) is not int or request["job_permits"] < 1:
            raise ValueError("valid request identity and worker permit required")
        directory = self.root / "jobs" / identity
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            record = self.reconciled(identity)
            if record and record["state"] == "dispatch_intent" and not (directory / "permit.json").exists():
                if ((directory / "worker.log").exists() or (directory / "response.json").exists()
                        or (directory / "request.json").exists() and
                        json.loads((directory / "request.json").read_text(encoding="utf-8")) != request):
                    raise ValueError("unpermitted request has worker artifacts or a different payload")
                if (record["identity"] != identity or record["request"] != request or
                        record["request_digest"] != digest(request)):
                    raise ValueError("unpermitted receipt differs from the request")
                # The permit precedes Popen, so this dispatch could not have launched a worker.
                record["state"] = "reserved"
                atomic_json_write(self.receipts.path(identity), record)
            def operation():
                atomic_json_write(directory / "request.json", request)
                atomic_json_write(directory / "permit.json", {"request_digest": digest(request), "job_id": identity + ":0"})
                with (directory / "worker.log").open("ab") as log:
                    process = subprocess.Popen([sys.executable, "-m", "tasks.alphabench.core.oracle_worker",
                        str(directory / "request.json"), str(self.config_path), str(directory / "response.json"),
                        "--config-digest", self.config_digest],
                        stdout=log, stderr=log, start_new_session=True,
                        env={key: value for key, value in os.environ.items() if key not in {"DEEPSEEK_API_KEY", "OPENAI_API_KEY"}})
                    try:
                        code = process.wait(timeout=protocol.worker_timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                        return {"request_id": identity, "success": False, "error": "worker_timeout",
                                "check_kind": protocol.check_kind if request["operation"] == "check" else None,
                                "elapsed_seconds": protocol.worker_timeout, "metrics": {},
                                "jobs": [{"job_id": identity + ":0", "operation": request["operation"]}]}
                if code or not (directory / "response.json").exists():
                    raise EvaluationPaused("worker exit requires reconciliation")
                return wire_response(request, json.loads((directory / "response.json").read_text(encoding="utf-8")))

            return self.receipts.execute(identity, request, reserve=lambda: None, operation=operation)

    def health(self):
        if oracle_environment_digest(self.config) != self.config["environment_digest"]:
            raise ValueError("oracle scoring environment changed after startup")
        return {key: self.config[key] for key in ("backend", "market", "data_digest", "environment_digest")} | {
            "config_digest": self.config_digest,
            "capabilities": ["durable_requests", "worker_permits", "daily_ic", "factor_scores", "portfolio", "dynamic_check"]
                + (["lint_check"] if self.config["backend"] == "assay" else [])}


def wire_response(request, response, *, full_digest=None):
    if request.get("phase") != "search" or "scores" not in response:
        return response
    return {key: value for key, value in response.items() if key != "scores"} | {
        "full_response_digest": full_digest or digest(response)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19779)
    args = parser.parse_args(argv)
    service = OracleService(args.config.resolve(), args.root.resolve())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            raw = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/t3/health":
                try:
                    return self.reply(200, service.health())
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    return self.reply(409, {"status": "paused_data_integrity", "error": str(exc)})
            if self.path.startswith("/t3/requests/"):
                identity = self.path.removeprefix("/t3/requests/")
                if not re.fullmatch(r"[a-f0-9]{64}", identity):
                    return self.reply(400, {"error": "invalid identity"})
                record = service.reconciled(identity)
                if record and record["state"] == "completed":
                    response = wire_response(record["request"], record["response"],
                                             full_digest=record["response_digest"])
                    if response is not record["response"]:
                        record = record | {"response": response, "response_digest": digest(response)}
                return self.reply(200 if record else 404, record or {"state": "missing"})
            return self.reply(404, {"error": "unknown route"})

        def do_POST(self):
            if self.path != "/t3/execute":
                return self.reply(404, {"error": "unknown route"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("request size outside contract")
                request = json.loads(self.rfile.read(size))
                result = service.execute(request)
            except EvaluationPaused as exc:
                return self.reply(409, {"status": exc.status, "error": str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": str(exc)})
            return self.reply(200, wire_response(request, result))

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
