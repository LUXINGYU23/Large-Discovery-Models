"""One Host-owned request meter for checks and all oracle phases."""

import json
import math
import urllib.error
import urllib.request

from ldm_tts.contracts import EvaluationResult
from ldm_tts.contracts.evaluation import EVALUATION_ATTEMPT_RECEIPT_KEY, EvaluationPaused
from .protocol import digest
from .receipts import Receipts
from .host import HostDispatcher


class OracleGateway:
    def __init__(self, protocol, runtime, *, endpoint=None, mock=False):
        self.protocol, self.runtime = protocol, runtime
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.mock = mock
        self.receipts = Receipts(runtime.run_dir / "private" / "oracle")
        self.host = HostDispatcher()
        self.before_dispatch = None

    def preflight(self):
        if self.mock:
            return {"mock": True}
        if not self.endpoint:
            raise ValueError("a controlled T3 oracle endpoint is required")
        with urllib.request.urlopen(self.endpoint + "/t3/health", timeout=10) as response:
            health = json.load(response)
        required = {"durable_requests", "worker_permits", "daily_ic", "factor_scores", "portfolio", "dynamic_check"}
        if not required <= set(health.get("capabilities", [])):
            raise ValueError("oracle lacks full T3 capabilities")
        if any(health.get(key) != getattr(self.protocol, key) for key in ("backend", "market", "data_digest")):
            raise ValueError("oracle identity differs from the frozen backend/data")
        if health.get("environment_digest") != self.protocol.environment_digest:
            raise ValueError("oracle environment differs from the frozen contract")
        return health

    def identity(self, phase, position, candidate):
        return {"run": self.runtime.run_id, "protocol": self.protocol.identity, "phase": phase,
                "position": position, "candidate": candidate.candidate_id}

    def evaluate(self, candidate, *, phase, position, fast=True):
        logical, request, counter = self.evaluation_request(candidate, phase=phase, position=position, fast=fast)
        return self._execute(logical, request, counter, charge_jobs=phase != "search")

    def evaluation_request(self, candidate, *, phase, position, fast=True):
        protocol = self.protocol
        logical = self.identity(phase, position, candidate)
        start, end = protocol.interval("search" if phase == "initialization" else "check" if phase in {"check", "quality"} else phase)
        operation = "check" if phase in {"check", "quality"} else "evaluate"
        request = {"request_id": digest(logical), "protocol": protocol.to_dict(), "operation": operation,
                   "expression": candidate.payload["expression"], "dialect": candidate.payload["dialect"],
                   "phase": phase, "start": start, "end": end, "fast": fast,
                   "job_permits": 8 if not fast else 2}
        counter = {"check": "dynamic_checks", "quality": "quality_checks", "initialization": "initialization_evaluations",
                   "validation": "validation_evaluations", "test": "test_evaluations", "search": None}[phase]

        return logical, request, counter

    @staticmethod
    def reservation(logical, request, counter):
        amounts = {"oracle_job_slots": request["job_permits"]}
        if counter:
            amounts[counter] = 1
        return "task:oracle:" + digest(logical), amounts

    def combine(self, candidates, *, selection_digest):
        start, end = self.protocol.interval("test")
        logical = {"run": self.runtime.run_id, "protocol": self.protocol.identity, "phase": "analysis",
                   "selection": selection_digest, "candidates": [candidate.candidate_id for candidate in candidates]}
        request = {"request_id": digest(logical), "protocol": self.protocol.to_dict(), "operation": "combine",
                   "expressions": [candidate.payload["expression"] for candidate in candidates],
                   "phase": "analysis", "start": start, "end": end, "fast": False, "job_permits": 8}
        return self._execute(logical, request, "analysis_jobs")

    def _execute(self, logical, request, counter, *, charge_jobs=True):
        def reserve():
            key, amounts = self.reservation(logical, request, counter)
            self.runtime.consume_many(amounts, usage_key=key)

        try:
            result = self.receipts.execute(logical, request, reserve=reserve,
                operation=lambda: self._send(request), reconcile=lambda: self._reconcile(request), owner=self.host,
                authorize=self.before_dispatch if logical["phase"] in {"search", "check"} else None)
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise EvaluationPaused("oracle dispatch requires reconciliation") from exc
        if not isinstance(result, dict) or result.get("request_id") != request["request_id"]:
            raise EvaluationPaused("oracle returned a different request identity")
        jobs = result.get("jobs")
        if (not isinstance(jobs, list) or len(jobs) > request["job_permits"] or
                any(not isinstance(job, dict) or not isinstance(job.get("job_id"), str) or not job["job_id"] for job in jobs) or
                len({job["job_id"] for job in jobs}) != len(jobs)):
            raise EvaluationPaused("oracle job accounting is incomplete or exceeds permits")
        if charge_jobs:
            self.host.call(lambda: self.runtime.consume_many({"benchmark_jobs": len(jobs)}, usage_key="task:oracle-result:" + digest(logical)))
        return result

    def _reconcile(self, request):
        if self.mock:
            return None
        try:
            with urllib.request.urlopen(self.endpoint + "/t3/requests/" + request["request_id"], timeout=10) as response:
                receipt = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404: return None
            raise
        if receipt["request_digest"] != digest(request):
            raise ValueError("server receipt belongs to another request")
        if receipt["state"] == "completed" and digest(receipt["response"]) == receipt["response_digest"]:
            return receipt["response"]
        return None

    def _send(self, request):
        if self.mock:
            return mock_oracle(request)
        encoded = json.dumps(request, allow_nan=False).encode()
        http_request = urllib.request.Request(self.endpoint + "/t3/execute", data=encoded,
                                             headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http_request, timeout=self.protocol.request_timeout) as response:
            return json.load(response)


class FactorEvaluator:
    def __init__(self, gateway):
        self.gateway = gateway

    def evaluation_attempt_usage_key(self, candidate):
        return digest(self.gateway.identity("search", candidate.metadata["attempt_position"], candidate))

    def evaluate(self, candidate):
        position = candidate.metadata["attempt_position"]
        raw = self.gateway.evaluate(candidate, phase="search", position=position)
        if raw["success"]:
            self.gateway.evaluate(candidate, phase="validation", position=position)
        prefix = "mock_" if self.gateway.mock else ""
        metrics = {prefix + key: value for key, value in raw.get("metrics", {}).items()
                   if type(value) in (int, float) and math.isfinite(value)}
        return EvaluationResult(candidate.candidate_id, "succeeded" if raw["success"] else "invalid",
                                metrics=metrics, error=raw.get("error", ""),
                                resource_usage={"benchmark_jobs": len(raw["jobs"])},
                                metadata={EVALUATION_ATTEMPT_RECEIPT_KEY: self.evaluation_attempt_usage_key(candidate)})


def mock_oracle(request):
    """Synthetic transport fixture; never scientific evidence or backend substitution."""
    seed = int(digest(request.get("expression", request.get("expressions")))[:8], 16)
    ic = (seed % 1000 - 400) / 10000
    daily = [{"date": f"{request['start'][:4]}-01-{day:02d}", "ic": ic + (day - 8) / 1000,
              "rank_ic": ic * .9 + (day - 8) / 1100} for day in range(1, 15)]
    return {"mock": True, "request_id": request["request_id"], "success": True,
            "jobs": [{"job_id": request["request_id"] + ":0", "operation": request["operation"]}],
            "metrics": {"ic": ic, "rank_ic": ic * .9, "icir": ic / .01, "rank_icir": ic * 90},
            "daily": daily, "nan_ratio": 0.0, "elapsed_seconds": 0.001,
            "scores": [{"date": row["date"], "instrument": f"mock-{stock}", "score": (seed % 11 + stock) * (index+1)}
                       for index, row in enumerate(daily) for stock in range(3)],
            "portfolio": None if request["fast"] else {"mock": True, "daily": daily, "holdings": [], "actions": [],
                "benchmark": {}, "pure": {}, "excess_with_cost": {}, "excess_without_cost": {}}}
