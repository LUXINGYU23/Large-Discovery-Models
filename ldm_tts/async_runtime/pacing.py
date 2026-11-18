"""Pacing (design 5.4): frozen parameters, the read-only campaign view and the pure rules D and R.

``PacingConfig`` is resolved (presets expanded to numbers) and frozen under ``CampaignConfig.extra["pacing"]``.
``StreamingPacing`` implements dispatch rule D and research rule R with the ``dispatch_during_research``
hold; ``BarrierPacing`` is the synchronous baseline (D and R only at O = 0, one batch per generation).
Both are pure functions of a ``CampaignView``; the controller enforces budget, short queue and dedup.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any

from .ledger import StateView
from .reservoir import ReservoirView
from .selection import PacingDecision

MODES = ("streaming", "barrier")
HOLDS = ("auto", "hold", "continue")
RESERVOIRS = ("resident_decay", "generation_replacement")


@dataclass(frozen=True)
class PacingConfig:
    """Every pacing parameter that shapes the trajectory; all of it is frozen with the campaign."""

    reservoir_size: int  # R, the nominal reservoir size (capacity in distinct candidates)
    half_life: float  # h, in booked results
    selection_batch_size: int = 1  # number chosen in one immutable barrier plan
    mode: str = "streaming"
    preset: str = "custom"
    theta: float = 0.5
    dispatch_during_research: str = "auto"
    kappa: float = 0.1
    reservoir: str = "resident_decay"
    min_weight: float = 1 / 16
    max_empty_reservoir_rounds: int = 3
    research_budget: int | None = None  # generations; None = unlimited (refused with request_research hooks)
    research_timeout_seconds: float = 3600.0  # wall-clock bound of one research round
    runtime_window: int = 25  # completed experiments in the rolling median D~
    dispatch_when_lane_blocked: bool = False
    breaker_window: int = 10
    breaker_failure_rate: float = 0.8  # trip when the failed share of the last window terminals exceeds this
    breaker_unstarted_failures: int = 3  # or when this many consecutive experiments failed before starting
    max_message_chars: int = 2000
    max_messages_per_round: int = 5

    def __post_init__(self):
        def integer(name, low):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"pacing {name} must be an integer >= {low}")

        for name, low in (("reservoir_size", 1), ("selection_batch_size", 1),
                          ("max_empty_reservoir_rounds", 1), ("runtime_window", 1),
                          ("breaker_window", 1), ("breaker_unstarted_failures", 1), ("max_message_chars", 1),
                          ("max_messages_per_round", 1)):
            integer(name, low)
        if self.research_budget is not None:
            integer("research_budget", 0)
        if self.mode not in MODES or self.dispatch_during_research not in HOLDS or self.reservoir not in RESERVOIRS:
            raise ValueError("unknown pacing mode, dispatch_during_research or reservoir maintenance")
        if self.selection_batch_size > self.reservoir_size:
            raise ValueError("selection_batch_size cannot exceed reservoir_size")
        if self.mode == "streaming" and self.selection_batch_size != 1:
            raise ValueError("streaming pacing requires selection_batch_size=1")
        if self.mode == "barrier" and self.reservoir != "generation_replacement":
            raise ValueError("barrier pacing requires generation replacement")
        for name in ("half_life", "theta", "kappa", "research_timeout_seconds", "breaker_failure_rate", "min_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"pacing {name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.half_life <= 0 or not 0 < self.theta <= 1 or self.kappa < 0 or self.research_timeout_seconds <= 0:
            raise ValueError("pacing requires half_life > 0, 0 < theta <= 1, kappa >= 0 and a positive timeout")
        if not 0 <= self.breaker_failure_rate < 1 or not 0 <= self.min_weight < 1:
            raise ValueError("breaker_failure_rate and min_weight must lie in [0, 1)")

    @classmethod
    def balanced(cls, *, concurrency: int, reservoir_size: int, **overrides) -> "PacingConfig":
        """Default preset: theta 0.5, h = concurrency, auto hold with kappa 0.1 (d = 0 is CampaignConfig's)."""
        return cls(**{"reservoir_size": reservoir_size, "half_life": float(concurrency), "preset": "balanced",
                      "theta": 0.5, "dispatch_during_research": "auto", "kappa": 0.1, **overrides})

    @classmethod
    def immediate(cls, *, concurrency: int, reservoir_size: int, **overrides) -> "PacingConfig":
        """Every result or selection triggers research; slots wait for the generation (hold)."""
        return cls(**{"reservoir_size": reservoir_size, "half_life": float(concurrency), "preset": "immediate",
                      "theta": 1.0, "dispatch_during_research": "hold", **overrides})

    @classmethod
    def barrier(cls, *, reservoir_size: int, **overrides) -> "PacingConfig":
        """Synchronous baseline: barrier pacing with generation replacement (reservoir order is the round's)."""
        return cls(**{"reservoir_size": reservoir_size, "half_life": 1.0, "preset": "barrier", "mode": "barrier",
                      "reservoir": "generation_replacement", "dispatch_during_research": "hold", **overrides})

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PacingConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown pacing parameter(s): {sorted(unknown)}")
        return cls(**dict(data))

    def maintenance(self):
        from .reservoir import GenerationReservoir, ResidentDecayReservoir
        if self.reservoir == "generation_replacement":
            return GenerationReservoir(capacity=self.reservoir_size)
        return ResidentDecayReservoir(half_life=self.half_life, capacity=self.reservoir_size,
                                      min_weight=self.min_weight, stale_fraction=self.theta)


def campaign_extra(pacing: PacingConfig, **other) -> dict:
    """``CampaignConfig.extra`` with the resolved pacing parameters (they freeze with the campaign)."""
    return {**other, "pacing": pacing.to_json()}


@dataclass(frozen=True)
class LaneStatus:
    state: str = "idle"  # idle | running | blocked
    generation: int | None = None
    started_at: float | None = None
    deadline_at: float | None = None
    trigger: str = ""


@dataclass(frozen=True)
class CampaignView:
    """What pacing, selectors and task hooks see: the core ``StateView`` plus reservoir, lane and pacing
    counters. Attribute access falls through to the ``StateView`` (``view.dedup_keys``, ``view.budget``...)."""

    state: StateView
    reservoir: ReservoirView
    stale: bool
    lane: LaneStatus
    delta: int  # results booked after the last ended generation's (minimum final) watermark
    selected_since_ready: int  # K
    runtime_median: float | None  # D~ (ES timestamps only); None with fewer than 3 completed experiments
    observed_running: int
    empty_rounds: int
    research_used: int
    research_requested: Mapping[str, Any] | None
    draining: Mapping[str, Any] | None
    current_generation: int  # newest ready generation (0: none)
    batch_generation: int | None  # barrier: generation of the newest batch plan
    open_plan: Mapping[str, Any] | None
    pacing: PacingConfig
    has_research: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __getattr__(self, name):
        if name == "state":
            raise AttributeError(name)
        return getattr(self.state, name)

    @property
    def eligible(self) -> int:
        """E: reservoir candidates outside the dedup set."""
        return len(self.reservoir.candidates)

    @property
    def fresh_stock(self) -> float:
        return self.reservoir.fresh_stock

    @property
    def research_budget_left(self) -> bool:
        budget = self.pacing.research_budget
        return budget is None or self.research_used < budget

    def hold(self) -> tuple[bool, float | None]:
        """``dispatch_during_research`` (5.4): whether a slot waits for the running generation, and until when."""
        lane, mode = self.lane, self.pacing.dispatch_during_research
        if lane.state != "running" or mode == "continue":
            return False, None
        if mode == "hold":
            return True, lane.deadline_at
        if self.runtime_median is None:
            return True, lane.deadline_at  # no D~ yet: bounded by the research round's wall-clock limit
        until = lane.started_at + self.pacing.kappa * self.runtime_median
        return self.state.now < until, until

    def summary(self) -> dict:
        held, until = self.hold()
        return {"eligible": self.eligible, "fresh_stock": self.fresh_stock, "stale": self.stale,
                "occurrences": self.reservoir.occurrence_count, "results_booked": self.reservoir.results_booked,
                "lane": asdict(self.lane), "delta": self.delta, "selected_since_ready": self.selected_since_ready,
                "runtime_median": self.runtime_median, "observed_running": self.observed_running,
                "empty_rounds": self.empty_rounds, "research_used": self.research_used, "hold": held,
                "hold_until": until, "current_generation": self.current_generation, "slot": self.state.slot_available}


def _gate(view: CampaignView) -> str | None:
    """Conditions shared by D and R: lifecycle, pause, drain (budget/queue are the core's)."""
    if view.state.status != "active":
        return f"campaign_{view.state.status}"
    if view.state.collect_only:
        return "collect_only"
    if view.state.paused:
        return "paused"
    if view.draining is not None:
        return "draining"
    return None


def research_guards(view: CampaignView) -> str | None:
    """Rule R clauses a and b; also what a task hook's ``request_research`` must pass (5.9)."""
    reason = _gate(view)
    if reason:
        return reason
    if not view.has_research:
        return "no_research_source"
    if view.lane.state == "running":
        return "lane_busy"
    if view.lane.state == "blocked":
        return "lane_blocked"
    if not view.research_budget_left:
        return "research_budget"
    if view.state.budget.free < 1:
        return "budget"
    return None


class StreamingPacing:
    """Framework streaming policy: rules D and R of design 5.4."""

    name = "streaming"
    version = "1"

    def decide(self, view: CampaignView) -> PacingDecision:
        reason = _gate(view)
        if reason:
            return PacingDecision("wait", reason)
        if view.eligible == 0:
            return PacingDecision("wait", "reservoir_exhausted")
        if view.state.budget.free < 1:
            return PacingDecision("wait", "budget")
        if not view.state.slot_available:
            return PacingDecision("wait", "no_slot")
        if view.state.backoff_until is not None:
            return PacingDecision("wait", "es_backoff")
        if view.lane.state == "blocked" and not view.pacing.dispatch_when_lane_blocked:
            return PacingDecision("wait", "lane_blocked")
        held, _ = view.hold()
        if held:
            return PacingDecision("wait", "research_hold")
        return PacingDecision("dispatch")

    def research(self, view: CampaignView) -> PacingDecision:
        reason = research_guards(view)
        if reason:
            return PacingDecision("wait", reason)
        if not view.stale:
            return PacingDecision("wait", "reservoir_fresh")
        if view.delta >= 1 or view.selected_since_ready >= 1 or (view.eligible == 0 and view.state.outstanding == 0):
            return PacingDecision("research")
        return PacingDecision("wait", "no_new_context")


class BarrierPacing:
    """Synchronous baseline (5.4): D only at O = 0 and once per generation (the controller then draws a batch of
    ``selection_batch_size`` in one stepwise selection); R only once that generation's batch is settled."""

    name = "barrier"
    version = "3"

    def decide(self, view: CampaignView) -> PacingDecision:
        reason = _gate(view)
        if reason:
            return PacingDecision("wait", reason)
        if view.state.outstanding:
            return PacingDecision("wait", "barrier")
        if view.lane.state != "idle" and view.has_research:
            return PacingDecision("wait", f"lane_{view.lane.state}")
        if view.has_research and view.batch_generation is not None and view.batch_generation >= view.current_generation:
            return PacingDecision("wait", "batch_done")
        if view.eligible == 0 or view.state.budget.free < 1:
            return PacingDecision("wait", "reservoir_exhausted" if view.eligible == 0 else "budget")
        if view.state.backoff_until is not None:
            return PacingDecision("wait", "es_backoff")
        return PacingDecision("dispatch")

    def research(self, view: CampaignView) -> PacingDecision:
        reason = research_guards(view)
        if reason:
            return PacingDecision("wait", reason)
        if view.state.outstanding or view.open_plan is not None:
            return PacingDecision("wait", "barrier")
        fresh = view.current_generation > 0 and (view.batch_generation is None
                                                 or view.batch_generation < view.current_generation)
        if fresh and view.eligible > 0:
            return PacingDecision("wait", "batch_pending")
        return PacingDecision("research")


def pacing_for(config: PacingConfig):
    return BarrierPacing() if config.mode == "barrier" else StreamingPacing()


# ---- pure helpers over experiment rows -------------------------------------------------------------

def parse_time(value: str | None) -> float | None:
    return None if not value else datetime.fromisoformat(value).timestamp()


def runtime_median(rows, window: int) -> float | None:
    """D~: median run time of the last ``window`` completed experiments (ES timestamps); None below 3."""
    durations = [parse_time(r["es_finished_at"]) - parse_time(r["es_started_at"]) for r in rows][:window]
    if len(durations) < 3:
        return None
    durations.sort()
    middle = len(durations) // 2
    return durations[middle] if len(durations) % 2 else (durations[middle - 1] + durations[middle]) / 2


def breaker_reason(terminals, config: PacingConfig) -> str | None:
    """Failure breaker (5.4) over terminal rows newest first (``es_state``, ``es_started_at``); CANCELED excluded."""
    finished = [r for r in terminals if r["es_state"] in ("SUCCEEDED", "FAILED")]
    window = finished[:config.breaker_window]
    if len(window) >= config.breaker_window:
        rate = sum(r["es_state"] == "FAILED" for r in window) / len(window)
        if rate > config.breaker_failure_rate:
            return f"failure_rate:{rate:.3f}"
    streak = 0
    for r in finished:
        if r["es_state"] != "FAILED" or r["es_started_at"]:
            break
        streak += 1
    if streak >= config.breaker_unstarted_failures:
        return f"unstarted_failures:{streak}"
    return None


__all__ = ["BarrierPacing", "CampaignView", "LaneStatus", "PacingConfig", "StreamingPacing", "breaker_reason",
           "campaign_extra", "pacing_for", "research_guards", "runtime_median"]
