"""Record native worker boundaries in the Campaign's existing event journal."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
from threading import Condition, Lock, local
import time
from types import SimpleNamespace
import uuid

from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError
from .protocol import digest


class NativeStop(BaseException):
    """Must pass through the upstream algorithms' broad Exception handlers."""

    def __init__(self, cause):
        self.cause = cause
        super().__init__(str(cause))


class NativeRuntime:
    def __init__(self, campaign, host):
        self.campaign, self.host = campaign, host
        self.recorded = [event["payload"] for event in campaign.events()
                         if event["event_type"] == "native_boundary"]
        self.by_id = {event["id"]: event for event in self.recorded}
        if len(self.by_id) != len(self.recorded):
            raise EvaluationPaused("duplicate native boundary identity")
        if any(digest(event["output"]) != event["output_digest"] for event in self.recorded):
            raise EvaluationPaused("native output integrity failure")
        self.cursor = 0
        self.condition = Condition()
        self.stop_lock = Lock()
        self.thread = local()
        self.stopped = None

    @property
    def scope(self):
        return self.thread.scope

    @property
    def role(self):
        return getattr(self.thread, "role", "search")

    @contextmanager
    def seed(self):
        previous = self.role
        self.thread.role = "seed"
        try:
            yield
        finally:
            self.thread.role = previous

    def stop(self, cause, *, notify=True):
        with self.stop_lock:
            if self.stopped is None:
                self.stopped = cause if isinstance(cause, NativeStop) else NativeStop(cause)
        # Host callbacks cannot acquire a lock held by a worker awaiting that Host.
        if notify:
            with self.condition:
                self.condition.notify_all()
        return self.stopped

    def _check(self):
        if self.stopped is not None:
            raise self.stopped

    def _scope(self, scope, operation):
        self.thread.scope, self.thread.sequence, self.thread.objects = scope, 0, 0
        try:
            self._check()
            return operation()
        except (NativeStop, EvaluationPaused, BudgetExceededError, KeyboardInterrupt, SystemExit) as exc:
            raise self.stop(exc)

    def run(self, operation):
        def stage():
            result = self._scope("root", operation)
            if self.cursor != len(self.recorded):
                raise self.stop(EvaluationPaused("native replay ended before its recorded boundaries"))
            return result
        try:
            return self.host.run(stage)
        except NativeStop as exc:
            raise exc.cause

    def _identity(self):
        identity = f"{self.scope}/{self.thread.sequence}"
        self.thread.sequence += 1
        return identity

    def _object(self, kind):
        identity = f"{self.scope}/{kind}{self.thread.objects}"
        self.thread.objects += 1
        return identity

    def _wait(self):
        # A divergent replay must pause instead of hanging on a missing worker.
        if not self.condition.wait(timeout=60):
            raise self.stop(EvaluationPaused("native replay made no progress for 60 seconds"))

    def boundary(self, kind, inputs, output=lambda: None, *, identity=None, ready=lambda: True):
        identity = identity or self._identity()
        request = {"id": identity, "kind": kind, "input_digest": digest(inputs)}
        with self.condition:
            known = self.by_id.get(identity)
            if known and any(known[key] != value for key, value in request.items()):
                raise self.stop(EvaluationPaused(f"native boundary changed: {identity} ({kind})"))
            while True:
                self._check()
                if self.cursor < len(self.recorded):
                    event = self.recorded[self.cursor]
                    if event["id"] != identity:
                        self._wait()
                        continue
                    if not ready():
                        raise self.stop(EvaluationPaused(f"native replay lock conflict: {identity}"))
                    result = event["output"]
                    self.cursor += 1
                else:
                    if not ready():
                        self._wait()
                        continue
                    result = json.loads(json.dumps(output(), allow_nan=False))
                    event = {**request, "output": result, "output_digest": digest(result)}
                    self.host.call(lambda: self.campaign.record(
                        "native_boundary", event, event_key=f"native:{identity}"))
                self.condition.notify_all()
                # Algorithm code may attach metrics to returned dictionaries.
                return json.loads(json.dumps(result))

    def callback(self, kind, operation):
        """A leaf callback; its operation must use durable receipts for incomplete calls."""
        def invoke(*args, **kwargs):
            identity = self._identity()
            inputs = {"args": args, "kwargs": kwargs, "role": self.role}
            self.boundary(kind + ".begin", inputs, identity=identity)
            completed_id = self._identity()
            if completed_id in self.by_id:
                return self.boundary(kind + ".end", {"begin": identity}, identity=completed_id)
            try:
                self._check()
                answer = operation(identity, *args, **kwargs)
            except (NativeStop, EvaluationPaused, BudgetExceededError, KeyboardInterrupt, SystemExit) as exc:
                raise self.stop(exc)
            return self.boundary(kind + ".end", {"begin": identity}, lambda: answer, identity=completed_id)
        return invoke

    def executor(self, *args, **kwargs):
        scheduler = self

        class Executor(ThreadPoolExecutor):
            def __init__(self):
                super().__init__(*args, **kwargs)
                self.identity, self.submitted = scheduler._object("executor"), 0

            def submit(self, function, /, *arguments, **keywords):
                scheduler._check()
                scope = f"{self.identity}/worker{self.submitted}"
                self.submitted += 1
                future = super().submit(scheduler._scope, scope, lambda: function(*arguments, **keywords))
                future.native_scope = scope
                original_result = future.result

                def result(timeout=None):
                    try:
                        answer = original_result(timeout)
                    except NativeStop as exc:
                        raise scheduler.stop(exc)
                    except Exception as exc:
                        scheduler.boundary("future.result", {"worker": scope, "error": type(exc).__name__,
                                                             "message": str(exc)})
                        raise
                    scheduler.boundary("future.result", {"worker": scope, "result_digest": digest(answer)})
                    return answer

                future.result = result
                return future

        return Executor()

    def completed(self, futures):
        futures = {future.native_scope: future for future in futures}
        physical = iter(as_completed(futures.values()))
        remaining = set(futures)
        while remaining:
            identity = self._identity()
            if identity in self.by_id:
                selected = self.boundary("future.deliver", sorted(remaining), identity=identity)
            else:
                future = next(physical)
                while future.native_scope not in remaining:
                    future = next(physical)
                selected = self.boundary("future.deliver", sorted(remaining),
                                         lambda: future.native_scope, identity=identity)
            if selected not in remaining:
                raise self.stop(EvaluationPaused("native completion references an unknown worker"))
            remaining.remove(selected)
            yield futures[selected]

    def lock(self):
        scheduler, identity = self, self._object("lock")

        class Lock:
            held = False

            def __enter__(self):
                with scheduler.condition:
                    scheduler.boundary("lock.acquire", identity, ready=lambda: not self.held)
                    self.held = True
                return self

            def __exit__(self, *_):
                with scheduler.condition:
                    try:
                        if scheduler.stopped is None:
                            scheduler.boundary("lock.release", identity)
                    finally:
                        self.held = False
                        scheduler.condition.notify_all()

        return Lock()

    def bindings(self):
        return {
            "ThreadPoolExecutor": self.executor,
            "as_completed": self.completed,
            "native_seed": self.seed,
            "native_state": lambda kind, state: self.boundary(
                "state", {"kind": kind, "state": state}, lambda: {"kind": kind, "state": state}),
            "threading": SimpleNamespace(Lock=self.lock),
            "time": SimpleNamespace(
                time=lambda: self.boundary("time", None, time.time),
                strftime=lambda fmt: self.boundary("strftime", fmt, lambda: time.strftime(fmt))),
            "uuid": SimpleNamespace(uuid4=lambda: uuid.UUID(self.boundary("uuid4", None, lambda: str(uuid.uuid4())))),
        }
