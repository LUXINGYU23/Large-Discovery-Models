"""Short authenticated requests for long private evaluations.

Task identity is the exact source digest. Scientific work and durable receipts
remain owned by the existing evaluator; reconnecting never creates a new source.
"""
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from ldm_tts.contracts.evaluation import EvaluationPaused
from ..core.execution import digest


class AsyncServiceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tasks = {}
        self._tasks_lock = threading.RLock()
        self._task_pool = ThreadPoolExecutor(max_workers=16)
        self._progress = 0

    def task_progress(self):
        with self._tasks_lock:
            self._progress += 1

    def dispatch(self, request):
        if not isinstance(request, dict):
            return {"status": "rejected_request"}
        operation = request.get("operation")
        if operation not in {"submit", "status"}:
            if operation == "evaluate":
                return {"status": "rejected_request"}
            with self._tasks_lock:
                if operation == "seal_training" and any(not f.done() for f in self._tasks.values()):
                    return {"status": "training_active"}
                return super().dispatch(request)
        fields = {"operation", "contract_digest", "source" if operation == "submit" else "source_sha256"}
        if set(request) != fields:
            return {"status": "rejected_request"}
        if request["contract_digest"] != digest(self.identity):
            return {"status": "contract_mismatch"}
        source = request.get("source")
        if operation == "submit":
            if not isinstance(source, str):
                return {"status": "rejected_request"}
            key = hashlib.sha256(source.encode()).hexdigest()
        else:
            key = request["source_sha256"]
            if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                return {"status": "rejected_request"}
        with self._tasks_lock:
            if operation == "submit" and (self.root / "training-sealed.json").exists():
                return {"status": "training_sealed"}
            if operation == "submit" and key not in self._tasks:
                if len(self._tasks) >= self.slot_limit:
                    return {"status": "budget_exhausted"}
                evaluate = super().dispatch
                self._tasks[key] = self._task_pool.submit(evaluate, dict(request, operation="evaluate"))
            future = self._tasks.get(key)
            if future is None:
                return {"status": "unknown_task", "source_sha256": key}
            if not future.done():
                return {"status": "pending", "source_sha256": key, "progress": self._progress}
            try:
                result = future.result()
            except Exception:
                result = {"status": "infrastructure_error"}
            return {"status": "completed", "source_sha256": key, "result": result}


def evaluate_request(client, payload, request_once, *, poll_seconds=5, stall_seconds=1800):
    """No total suite deadline; fail closed after bounded service inactivity."""
    if payload.get("operation") != "evaluate":
        return request_once(payload)
    key = hashlib.sha256(payload["source"].encode()).hexdigest()
    request = dict(payload, operation="submit")
    progress = None
    changed = time.monotonic()
    while True:
        try:
            response = request_once(request)
        except EvaluationPaused as exc:
            if "outcome unavailable" not in str(exc):
                raise
            # Reissue only the same idempotent operation; never change source.
            if time.monotonic() - changed >= stall_seconds:
                raise
            time.sleep(poll_seconds)
            continue
        if response.get("status") not in {"pending", "completed", "unknown_task"}:
            return response
        if response.get("source_sha256") != key:
            raise EvaluationPaused("private task source identity differs")
        if response["status"] == "completed":
            return response["result"]
        if response["status"] == "unknown_task":
            request = dict(payload, operation="submit")
        else:
            value = response.get("progress")
            if type(value) is not int or value < 0:
                raise EvaluationPaused("invalid private task progress")
            if value != progress:
                changed = time.monotonic()
                progress = value
            request = dict(operation="status", source_sha256=key, contract_digest=payload["contract_digest"])
        if time.monotonic() - changed >= stall_seconds:
            raise EvaluationPaused("private evaluator has no completed-job progress")
        time.sleep(poll_seconds)


def install():
    """Operational overlay; leave frozen scientific modules and receipts intact."""
    from . import service_client
    from ..experiments.v2 import parallel_runtime
    base = parallel_runtime.ParallelService

    class AsyncService(AsyncServiceMixin, base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            run = self.evaluator.execution.run
            def tracked(*a, **kw):
                result = run(*a, **kw)
                self.task_progress()
                return result
            self.evaluator.execution.run = tracked

    parallel_runtime.ParallelService = AsyncService
    original = service_client.ServiceEvaluator.request
    def request(self, payload):
        transport = SimpleNamespace(socket=self.socket, service_uid=self.service_uid, timeout=30)
        return evaluate_request(self, payload, lambda p: original(transport, p))
    service_client.ServiceEvaluator.request = request
