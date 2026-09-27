"""Native callback batches share the Host ledger, receipts and private validation."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import math
from pathlib import Path

from ldm_tts.contracts import Candidate, EvaluationResult, Observation, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from ldm_tts.engine.run_store import BudgetExceededError
from ldm_tts.engine.runtime import LDMEngineState
from .candidate import FactorDomain
from .protocol import digest
from .receipts import Receipts


class NativeSearchBudgetExhausted(BudgetExceededError):
    """The matched search evaluation allowance is exhausted."""


class NativeEvaluator:
    def __init__(self, scheduler, gateway, initial_observations, *, workers):
        self.scheduler, self.gateway = scheduler, gateway
        self.runtime, self.protocol = gateway.runtime, gateway.protocol
        if self.protocol.profile not in {"ldm_matched_v1", "upstream_searcher_v1", "upstream_benchmark_v1"}:
            raise ValueError("native evaluator requires an implemented complete entry protocol")
        if self.runtime.budget.limits.get("expensive_evaluation_attempts") != self.protocol.evaluations:
            raise ValueError("native runtime must freeze the complete search attempt limit")
        if type(workers) is not int or workers < 1:
            raise ValueError("native oracle workers must be a positive integer")
        self.workers = workers
        self.initial = {item.candidate_id: item for item in initial_observations if item.evaluation.succeeded}
        self.initial_digest = digest([item.to_dict() for item in initial_observations])
        Receipts(self.runtime.run_dir / "private/stages").accept("native_evaluation_contract",
            {"protocol": self.protocol.identity, "initial_information_digest": self.initial_digest,
             "workers": workers, "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest()})
        self.stopped = next((event["payload"] for event in self.runtime.events()
                             if event["event_type"] == "native_search_stopped"), None)
        self.settling = False
        gateway.before_dispatch = self.authorize

    def authorize(self):
        if self.settling:
            return
        self.scheduler._check()
        if self.stopped:
            raise self.scheduler.stop(NativeSearchBudgetExhausted(self.stopped["reason"]), notify=False)

    def callbacks(self):
        return {
            "evaluate_fn": self.scheduler.callback("evaluate_fn", lambda identity, expression:
                self.evaluate(identity, [{"name": expression, "expression": expression}])[0]),
            "batch_evaluate_fn": self.scheduler.callback("batch_evaluate_fn", self.evaluate),
            "batch_evaluate_fn_dict": self.scheduler.callback("batch_evaluate_fn_dict", lambda identity, factors:
                {factor["name"]: result for factor, result in zip(factors, self.evaluate(identity, factors))}),
        }

    def evaluate(self, identity, factors):
        role = self.scheduler.role
        domain = FactorDomain(self.protocol.backend, None if role == "seed" else self.protocol.grammar_depth)
        candidates = []
        for index, factor in enumerate(factors):
            candidate = domain.admit(RawProposal(factor, "native_search",
                {"attempt_position": ["native", identity, index]}))
            if not isinstance(candidate, Candidate):
                raise EvaluationPaused("native evaluation received an unadmitted expression: " + candidate.reason)
            candidates.append(candidate)
        if role == "seed":
            results = []
            for candidate in candidates:
                observation = self.initial.get(candidate.candidate_id)
                if observation is None:
                    raise EvaluationPaused("native seed lacks a verified successful initial observation")
                metrics = observation.evaluation.metrics
                prefix = "mock_" if self.gateway.mock else ""
                results.append({"name": candidate.payload["name"], "expression": candidate.payload["expression"],
                    "success": True, "metrics": {key.removeprefix(prefix): value for key, value in metrics.items()},
                    "error": "", "cached": True})
            self.gateway.host.call(lambda: self.runtime.record("native_seed_reused",
                {"batch": identity, "initial_information_digest": self.initial_digest,
                 "candidates": [item.candidate_id for item in candidates]}, event_key="native_seed:" + identity))
            return results

        def reserve():
            self.authorize()
            groups = {}
            selected = []
            usage = self.runtime.budget.metadata.get("cumulative_usage", {})
            remaining = self.runtime.budget.remaining("expensive_evaluation_attempts")
            for candidate in candidates:
                logical, request, counter = self.gateway.evaluation_request(candidate, phase="search",
                    position=candidate.metadata["attempt_position"])
                key, amounts = self.gateway.reservation(logical, request, counter)
                charged = usage.get(key, {}).get("expensive_evaluation_attempts", 0)
                if not charged and remaining == 0 and self.protocol.profile == "ldm_matched_v1":
                    break
                if not charged:
                    remaining -= 1
                groups[key] = {**amounts, "expensive_evaluation_attempts": 1,
                               "external_evaluations": 1, "selected_candidates": 1}
                selected.append(candidate)
            if not selected and candidates:
                self.stopped = {"batch": identity, "requested": len(candidates), "remaining": 0,
                                "reason": "matched search evaluation allowance exhausted"}
                self.runtime.record("native_search_stopped", self.stopped, event_key="native_search_stopped")
                raise self.scheduler.stop(NativeSearchBudgetExhausted(self.stopped["reason"]), notify=False)
            try:
                self.runtime.consume_groups(groups)
            except BudgetExceededError as exc:
                cause = EvaluationPaused("native batch exceeds the frozen execution budget: " + str(exc),
                                         status="paused_budget")
                raise self.scheduler.stop(cause, notify=False)
            submitted = next(event["sequence"] for event in self.runtime.events()
                if event["event_type"] == "native_boundary" and event["payload"]["id"] == identity)
            self.runtime.record("native_batch_reserved", {"batch": identity, "submitted": submitted,
                "requested": len(candidates), "candidates": [item.to_dict() for item in selected]},
                event_key="native_batch:" + identity)
            return submitted, selected

        submitted, selected = self.gateway.host.call(reserve)

        def measure(index, candidate):
            try:
                return self._measure(identity, submitted, index, candidate)
            except (EvaluationPaused, BudgetExceededError) as exc:
                raise self.scheduler.stop(exc)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(measure, index, candidate)
                       for index, candidate in enumerate(selected)]
            results = [future.result() for future in futures]
        if len(selected) < len(candidates):
            def stop():
                self.stopped = {"batch": identity, "requested": len(candidates),
                                "accepted": len(selected), "remaining": 0,
                                "reason": "matched search evaluation allowance exhausted"}
                self.runtime.record("native_search_stopped", self.stopped, event_key="native_search_stopped")
            self.gateway.host.call(stop)
            raise self.scheduler.stop(NativeSearchBudgetExhausted(self.stopped["reason"]), notify=False)
        return results

    def _measure(self, identity, submitted, index, candidate):
        position = candidate.metadata["attempt_position"]
        raw = self.gateway.evaluate(candidate, phase="search", position=position)
        logical = self.gateway.identity("search", position, candidate)
        self.gateway.host.call(lambda: self.runtime.consume_many({"benchmark_jobs": len(raw["jobs"])},
            usage_key="task:oracle-result:" + digest(logical)))
        if type(raw.get("success")) is not bool or not isinstance(raw.get("metrics"), dict):
            raise EvaluationPaused("native search response is malformed")
        if raw["success"] or self.protocol.profile != "ldm_matched_v1":
            # Validation completes before an observation is published, but never enters the callback result.
            validation = self.gateway.evaluate(candidate, phase="validation", position=position)
            if type(validation.get("success")) is not bool or not isinstance(validation.get("metrics"), dict):
                raise EvaluationPaused("native private validation response is malformed")
        metrics = {key: value for key, value in raw["metrics"].items()
                   if type(value) in (int, float) and math.isfinite(value)}
        prefix = "mock_" if self.gateway.mock else ""
        observation = Observation(candidate, EvaluationResult(candidate.candidate_id,
            "succeeded" if raw["success"] else "invalid", metrics={prefix + key: value for key, value in metrics.items()},
            error=raw.get("error", ""), resource_usage={"benchmark_jobs": len(raw["jobs"])}),
            metadata={"native_batch": identity, "index": index, "submitted": submitted})

        self.gateway.host.call(lambda: self.runtime.record("native_observation", observation.to_dict(),
            candidate_id=candidate.candidate_id, event_key="native_observation:" + digest(logical)))
        return {"name": candidate.payload["name"], "expression": candidate.payload["expression"],
                "success": raw["success"], "metrics": metrics, "error": raw.get("error", ""),
                "cached": raw.get("cached", False)}

    def settle(self):
        """Finish every paid search attempt; unknown dispatched requests still pause."""
        self.settling = True
        try:
            for event in self.runtime.events():
                if event["event_type"] != "native_batch_reserved":
                    continue
                batch = event["payload"]
                for index, payload in enumerate(batch["candidates"]):
                    self._measure(batch["batch"], batch["submitted"], index, Candidate(**payload))
        finally:
            self.settling = False
        if len(self.observations()) != self.runtime.budget.counters.get("expensive_evaluation_attempts", 0):
            raise EvaluationPaused("native search receipts do not account for every paid attempt")
        return {"stopped": self.stopped, "reserved_but_unstarted": []}

    def observations(self):
        rows = [event["payload"] for event in self.runtime.events() if event["event_type"] == "native_observation"]
        rows.sort(key=lambda row: (row["metadata"]["submitted"], row["metadata"]["index"]))
        return LDMEngineState.from_checkpoint({"observations": rows}).observations
