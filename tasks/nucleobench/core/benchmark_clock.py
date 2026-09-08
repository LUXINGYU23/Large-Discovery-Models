"""The task's wall-time window, shared by research messages and reporting."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkClock:
    max_seconds: int
    elapsed_before_start: float = 0.0
    started_at_unix: float | None = field(default=None, init=False)
    _started_monotonic: float | None = field(default=None, init=False)

    @classmethod
    def resume(cls, max_seconds: int, events: Sequence[Mapping[str, Any]]) -> BenchmarkClock:
        segments = clock_segments(events, max_seconds)
        if not segments:
            raise ValueError("Cannot resume a wall-time campaign without its benchmark clock")
        last = segments[-1]
        stops = [float(event["timestamp_unix"]) for event in events
                 if event.get("event_type") in {"campaign_failed", "campaign_finished", "campaign_paused"}
                 and float(event["timestamp_unix"]) >= last["started_at_unix"]]
        if not stops:
            raise ValueError("Stop the previous campaign before resuming its benchmark clock")
        return cls(max_seconds, last["elapsed_before_start"] + stops[0] - last["started_at_unix"])

    @property
    def remaining_before_start(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed_before_start)

    def start(self) -> dict[str, float | int]:
        if self.started_at_unix is not None:
            raise RuntimeError("The benchmark clock cannot be restarted")
        if self.max_seconds < 1:
            raise ValueError("The benchmark time budget must be positive")
        self.started_at_unix = time.time()
        self._started_monotonic = time.monotonic()
        return {"started_at_unix": self.started_at_unix, "max_seconds": self.max_seconds,
                "elapsed_before_start": self.elapsed_before_start}

    def snapshot(self) -> dict[str, float | int]:
        if self._started_monotonic is None:
            raise RuntimeError("The benchmark clock has not started")
        elapsed = self.elapsed_before_start + max(0.0, time.monotonic() - self._started_monotonic)
        return {
            "max_seconds": self.max_seconds,
            "elapsed_seconds": elapsed,
            "remaining_seconds": max(0.0, self.max_seconds - elapsed),
        }


def clock_segments(events: Sequence[Mapping[str, Any]], max_seconds: int) -> list[dict[str, float]]:
    segments = []
    for event in events:
        kind = event.get("event_type")
        if kind not in {"benchmark_clock_started", "benchmark_clock_resumed"}:
            continue
        payload = event["payload"]
        if payload["max_seconds"] != max_seconds or (kind == "benchmark_clock_started" and segments):
            raise ValueError("Benchmark clock identity changed")
        if kind == "benchmark_clock_resumed" and not segments:
            raise ValueError("Benchmark resume is missing its original clock")
        segments.append({
            "started_at_unix": float(payload["started_at_unix"]),
            "elapsed_before_start": 0.0 if kind == "benchmark_clock_started" else float(payload["elapsed_before_start"]),
        })
    return segments


def elapsed_at(timestamp: float, segments: Sequence[Mapping[str, float]]) -> float:
    for segment in reversed(segments):
        if timestamp >= segment["started_at_unix"]:
            return segment["elapsed_before_start"] + timestamp - segment["started_at_unix"]
    return 0.0
