"""Host-owned public research tools for persistent AlphaBench sessions."""

import json
import os
import socketserver
import tempfile
from pathlib import Path
from threading import Thread
from uuid import uuid4

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError, atomic_json_write
from ldm_tts.optimization.records import BOObservation
from .candidate import FactorDomain
from .grammar import REGISTRY, parse_expression
from .protocol import digest
from .selection import FactorEncoder, FactorSelector


def public_observation(observation):
    return {"candidate_id": observation.candidate_id, "expression": observation.candidate.payload["expression"],
            "status": observation.evaluation.status, "metrics": dict(observation.metrics), "round_idx": observation.round_idx}


def check_position(round_idx, candidate):
    return ["harness_check", round_idx, candidate.candidate_id]


class HarnessToolService:
    def __init__(self, protocol, gateway, socket_root, *, surrogate_query):
        self.protocol, self.gateway = protocol, gateway
        Path(socket_root).mkdir(parents=True, exist_ok=True)
        self.socket_dir = Path(tempfile.mkdtemp(prefix="t3-", dir=socket_root))
        self.socket_path = self.socket_dir / "host.sock"
        if len(os.fsencode(self.socket_path)) >= 108:
            self.socket_dir.rmdir()
            raise ValueError("Harness Host socket path exceeds the Unix limit")
        self.domain = FactorDomain(protocol.backend, protocol.grammar_depth)
        self.encoder = FactorEncoder() if surrogate_query else None
        self.active = None
        self.failure = None
        try:
            self.server = _UnixServer(str(self.socket_path), _ToolHandler)
        except BaseException:
            self.socket_dir.rmdir()
            raise
        self.server.service = self
        self.thread = Thread(target=self.server.serve_forever, name="t3-harness-tools", daemon=True)
        self.thread.start()
        self.socket_path.chmod(0o600)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.socket_path.unlink(missing_ok=True)
        self.socket_dir.rmdir()

    def prepare(self, request):
        history = [public_observation(item) for item in request.observations]
        identity = {"round_idx": request.round_idx, "history": history,
                    "protocol": self.protocol.identity, "feature_version": self.encoder.describe().version if self.encoder else None}
        snapshot_id = digest(identity)
        document = {"snapshot_id": snapshot_id, **identity}

        def freeze():
            path = self.gateway.runtime.run_dir / "private" / "harness" / "rounds" / f"round_{request.round_idx:04d}.json"
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != document:
                    raise ValueError("Harness research snapshot changed across recovery")
            else:
                atomic_json_write(path, document)
            selector = None
            if self.encoder:
                objective = ("mock_" if self.gateway.mock else "") + self.protocol.objective
                measured = tuple(BOObservation.from_observation(item, objective_names=(objective,),
                    feature=item.surrogate or self.encoder.encode(item.candidate))
                    for item in request.observations if item.evaluation.succeeded)
                selector = FactorSelector(objective, seed=self.protocol.random_seed)
                selector.fit(measured)
            self.active = {"id": snapshot_id, "request": request, "history": history,
                           "evaluated": {item.canonical_key for item in request.observations}, "selector": selector}
            self.failure = None
            return {"snapshot_id": snapshot_id, "fit_status": selector.gp.fit_status if selector else None,
                    "history_digest": digest(history)}

        return self.gateway.host.call(freeze)

    def clear(self):
        self.gateway.host.call(lambda: setattr(self, "active", None))

    def execute(self, tool, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        active = self.active
        if active is None:
            raise ValueError("no active Harness research round")
        if tool == "get_contract":
            dialects = ("qlib", "assay") if self.protocol.backend == "assay" else ("qlib",)
            return {"backend": self.protocol.backend, "market": self.protocol.market,
                "objective": self.protocol.objective, "filter_profile": self.protocol.filter_profile,
                "operator_signatures": {dialect: REGISTRY[dialect] for dialect in dialects},
                "fields": {dialect: (["open", "high", "low", "close", "volume", "vwap"]
                    if dialect == "assay" else ["$open", "$high", "$low", "$close", "$volume"])
                    for dialect in dialects},
                "snapshot_id": active["id"], "history_digest": digest(active["history"]),
                "dynamic_checks_remaining": self.gateway.runtime.budget.remaining(self.protocol.check_kind + "_checks")}
        if tool == "get_history":
            offset, limit = arguments.get("offset", 0), arguments.get("limit", 20)
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
                raise ValueError("history offset/limit are out of range")
            history = active["history"]
            return {"total": len(history), "offset": offset,
                "next_offset": offset + limit if offset + limit < len(history) else None,
                "observations": history[offset:offset + limit], "snapshot_id": active["id"]}
        if tool == "get_observation":
            candidate_id = arguments.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("candidate_id is required")
            for item in active["history"]:
                if item["candidate_id"] == candidate_id:
                    return {"observation": item, "snapshot_id": active["id"]}
            raise ValueError("candidate_id is not in evaluated search history")
        if tool in {"validate_expression", "check_expression", "query_surrogate"}:
            expression = arguments.get("expression")
            if not isinstance(expression, str):
                raise ValueError("expression must be a string")
            parsed = parse_expression(expression, backend=self.protocol.backend, max_depth=self.protocol.grammar_depth)
            candidate = self.domain.admit(RawProposal({"name": "research", "expression": parsed.canonical}, "harness_tool"))
            if not isinstance(candidate, Candidate):
                raise ValueError("expression failed admission")
            identity = {"canonical_expression": parsed.canonical, "candidate_id": candidate.candidate_id,
                        "depth": parsed.depth, "nodes": parsed.nodes,
                        "already_evaluated": candidate.canonical_key in active["evaluated"]}
            if tool == "validate_expression":
                return identity
            if tool == "check_expression":
                raw = self.gateway.evaluate(candidate, phase="check",
                    position=check_position(active["request"].round_idx, candidate))
                return {**identity, "check_kind": raw["check_kind"], "success": self.protocol.check_passed(raw),
                    "nan_ratio": raw.get("nan_ratio"), "non_finite_ratio": raw.get("non_finite_ratio"),
                    "elapsed_seconds": raw["elapsed_seconds"], "error": raw.get("error")}
            selector = active["selector"]
            if selector is None:
                raise ValueError("surrogate query is unavailable for direct Harness")
            if arguments.get("snapshot_id") != active["id"]:
                raise ValueError("surrogate snapshot ID differs from the current round")
            self.gateway.runtime.consume_many({"surrogate_queries": 1},
                usage_key="task:harness:gp-query:" + uuid4().hex)
            vector = self.encoder.encode(candidate)
            prediction = selector.gp.predict_record(candidate.candidate_id, vector.values, beta=selector.beta)
            return {**identity, "snapshot_id": active["id"], "history_digest": digest(active["history"]),
                "fit_status": selector.gp.fit_status, "mean": prediction.scalar_mean,
                "latent_std": prediction.scalar_std, "ucb": prediction.acquisition_score}
        raise ValueError("unknown or unavailable Harness tool")


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class _ToolHandler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(65537)
        if len(raw) > 65536 or not raw.endswith(b"\n"):
            return
        try:
            request = json.loads(raw)
            if not isinstance(request, dict) or set(request) != {"tool", "arguments"}:
                raise ValueError("tool request fields are invalid")
            service = self.server.service
            result = service.gateway.host.call(lambda: service.execute(request["tool"], request["arguments"]))
            response = {"ok": True, "result": result}
        except Exception as exc:
            if isinstance(exc, (BudgetExceededError, EvaluationPaused)):
                try:
                    service.gateway.host.call(lambda: setattr(service, "failure", exc))
                except RuntimeError:
                    pass
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(response, allow_nan=False).encode() + b"\n")
