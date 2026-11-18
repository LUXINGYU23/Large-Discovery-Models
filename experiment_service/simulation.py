"""Simulated executor for deterministic simulation (design 8.3).

Every random draw is a pure function of ``(seed, label, key[, attempt])``: durations and
outcomes depend on the spec key, not on the experiment id, so the same candidate takes the
same time in every experiment arm. Dispatch uncertainty depends on the experiment id.
"""
from __future__ import annotations

import hashlib
import math
import random
import threading

from .executors import (FAILURE_CLASSES, AttemptStatus, DispatchUncertain, ExecutorCapabilities)
from .service import canonical


def fixed(value):
    return lambda rng: float(value)


def uniform(low, high):
    return lambda rng: rng.uniform(low, high)


def exponential(mean):
    return lambda rng: rng.expovariate(1.0 / mean)


def heavy_tail(scale, alpha):
    """Pareto with minimum ``scale`` and tail index ``alpha`` (infinite variance for alpha <= 2)."""
    return lambda rng: scale * rng.paretovariate(alpha)


def lognormal(median, sigma):
    return lambda rng: rng.lognormvariate(math.log(median), sigma)


_NAMED = {"fixed": fixed, "uniform": uniform, "exponential": exponential, "heavy_tail": heavy_tail,
          "lognormal": lognormal}


def distribution(spec):
    """A number, a callable ``rng -> float``, or ``(name, *params)`` / ``[name, *params]`` with
    name in fixed, uniform, exponential, heavy_tail, lognormal."""
    if callable(spec):
        return spec
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return fixed(spec)
    if isinstance(spec, (tuple, list)) and spec and spec[0] in _NAMED:
        return _NAMED[spec[0]](*spec[1:])
    raise ValueError(f"unsupported distribution {spec!r}")


class SimulatedExecutor:
    """Executor whose attempts run on an injected clock.

    ``failures`` maps failure classes to probabilities, or is ``callable(spec, attempt_index)``
    returning a class or ``None``. ``cancel_running`` is the declared and simulated cancel
    behaviour (``cooperative`` stops ``cancel_delay`` after the request). ``dispatch_uncertain``
    is the probability that the first ``start`` of an attempt raises ``DispatchUncertain``;
    ``launch_on_uncertain`` is the probability that such an attempt was launched anyway.
    ``launches`` records every physical launch, so tests can assert at-most-once dispatch.
    """

    def __init__(self, clock, *, kind="sim.objective", kind_version="1", seed=0, duration=10.0, queue_delay=0.0,
                 key=None, objective=None, artifacts=None, failures=None, cancel_running="cooperative",
                 cancel_delay=1.0, dispatch="at_most_once", dispatch_uncertain=0.0, launch_on_uncertain=0.5,
                 retry_safe=False, in_process=False, pool="default", units=1, validate=None, schema=None):
        self.clock, self.kind, self.kind_version, self.seed, self.schema = clock, kind, kind_version, seed, schema
        self.capabilities = ExecutorCapabilities(cancel_running=cancel_running, dispatch=dispatch,
                                                 retry_safe=retry_safe, pool=pool, units=units, in_process=in_process,
                                                 cancel_idempotent=True)
        self._duration, self._queue_delay = distribution(duration), distribution(queue_delay)
        self._key, self._objective, self._artifacts = key or canonical, objective, artifacts
        if isinstance(failures, dict) and (set(failures) - set(FAILURE_CLASSES) or sum(failures.values()) > 1):
            raise ValueError("failures maps failure classes to probabilities summing to at most 1")
        self._failures, self._cancel_delay, self._validate = failures or {}, cancel_delay, validate
        self._uncertain, self._launch_on_uncertain = dispatch_uncertain, launch_on_uncertain
        self._runs, self._starts, self._unreachable = {}, {}, set()
        self.launches: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def _rng(self, *labels):
        material = canonical([self.seed, *labels]).encode()
        return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))

    def plan(self, spec, attempt_index=1):
        """The deterministic fate of one attempt: duration, queue delay and outcome."""
        key = self._key(spec)
        duration = max(0.0, self._duration(self._rng("duration", key)))
        delay = max(0.0, self._queue_delay(self._rng("queue_delay", key)))
        if callable(self._failures):
            failure = self._failures(spec, attempt_index)
        else:
            # infeasibility is a property of the candidate; the other classes vary by attempt
            failure = "infeasible" if self._rng("infeasible", key).random() < self._failures.get("infeasible", 0) else None
            draw, cumulative = self._rng("failure", key, attempt_index).random(), 0.0
            for name in FAILURE_CLASSES[1:]:
                cumulative += self._failures.get(name, 0.0)
                if failure is None and draw < cumulative:
                    failure = name
        return {"key": key, "duration": duration, "queue_delay": delay, "failure": failure}

    def validate(self, spec):
        problems = [] if isinstance(spec, dict) else ["spec must be a JSON object"]
        return problems + (list(self._validate(spec)) if self._validate and not problems else [])

    def start(self, attempt):
        base = f"sim:{attempt.experiment_id}#{attempt.attempt_index}"
        with self._lock:
            calls = self._starts[base] = self._starts.get(base, 0) + 1
            if self.capabilities.dispatch == "idempotent" and base in self._runs:
                return base
            ref = base if base not in self._runs else f"{base}/{calls}"
            if calls == 1 and self._rng("dispatch", attempt.experiment_id, attempt.attempt_index).random() < self._uncertain:
                if self._rng("launched", attempt.experiment_id, attempt.attempt_index).random() < self._launch_on_uncertain:
                    self._launch(ref, attempt)
                raise DispatchUncertain(f"simulated dispatch timeout for {base}")
            self._launch(ref, attempt)
            return ref

    def _launch(self, ref, attempt):
        plan, now = self.plan(dict(attempt.spec), attempt.attempt_index), self.clock.now()
        self._runs[ref] = {"spec": dict(attempt.spec), "plan": plan, "started": now,
                           "active_at": now + plan["queue_delay"], "end": now + plan["queue_delay"] + plan["duration"],
                           "stop": None, "cancel_requested": None}
        self.launches.append((attempt.experiment_id, attempt.attempt_index))

    def poll(self, executor_ref):
        now = self.clock.now()
        with self._lock:
            run = self._runs.get(executor_ref)
            if run is None or executor_ref in self._unreachable or "*" in self._unreachable:
                return AttemptStatus.unreachable("simulated node unreachable" if run else "unknown reference")
            usage = {"simulated_seconds": min(now, run["end"]) - run["started"]}
            if run["stop"] is not None and run["stop"] < run["end"] and now >= run["stop"]:
                return AttemptStatus.stopped({"simulated_seconds": run["stop"] - run["started"]})
            if now < run["end"]:
                return AttemptStatus.active("waiting" if now < run["active_at"] else "active")
            failure = run["plan"]["failure"]
            if failure:
                return AttemptStatus.failed(failure, f"simulated {failure}", usage)
            result = dict(self._objective(run["spec"])) if self._objective else \
                {"value": self._rng("objective", run["plan"]["key"]).random()}
            artifacts = tuple(self._artifacts(run["spec"], result)) if self._artifacts else ()
            return AttemptStatus.succeeded(result, artifacts, usage)

    def cancel(self, executor_ref):
        now = self.clock.now()
        with self._lock:
            run = self._runs.get(executor_ref)
            if run is None or run["cancel_requested"] is not None:
                return
            run["cancel_requested"] = now
            if self.capabilities.cancel_running == "immediate":
                run["stop"] = now
            elif self.capabilities.cancel_running == "cooperative":
                run["stop"] = now + self._cancel_delay

    def find(self, experiment_id, attempt_index):
        base = f"sim:{experiment_id}#{attempt_index}"
        with self._lock:
            return base if base in self._runs else None

    def next_event_time(self):
        now, times = self.clock.now(), []
        with self._lock:
            for run in self._runs.values():
                done = run["stop"] if run["stop"] is not None and run["stop"] < run["end"] else run["end"]
                if done > now:
                    times.append(done)
        return min(times) if times else None

    def set_unreachable(self, executor_ref="*", unreachable=True):
        with self._lock:
            (self._unreachable.add if unreachable else self._unreachable.discard)(executor_ref)


def simulated_executor_factory(context):
    """Manifest/deployment hook: ``"experiment_service.simulation:simulated_executor_factory"``;
    options come from the deployment config's ``executor_options[kind]``."""
    return SimulatedExecutor(context.clock, kind=context.kind, **dict(context.options))
