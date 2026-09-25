"""Durable campaign wall clock shared by proposals, evaluation, and resume.

Each process run is one segment. Heartbeats bound the time lost when a process
is killed without a stop record; a resumed campaign never regains elapsed time.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write

HEARTBEAT_SECONDS = 30.0


class CampaignClock:
    def __init__(self, path: Path, max_seconds: float | None) -> None:
        self.path = Path(path)
        self.max_seconds = max_seconds
        self.segments: list[dict[str, float]] = []
        self._last_write = 0.0

    @classmethod
    def open(cls, path: Path, max_seconds: float | None, *, resume: bool) -> "CampaignClock":
        clock = cls(path, max_seconds)
        if resume and clock.path.exists():
            saved = json.loads(clock.path.read_text())
            if saved["max_seconds"] != max_seconds:
                raise ValueError("campaign wall-clock limit changed on resume")
            for segment in saved["segments"]:
                end = segment.get("stopped_at", segment["last_seen"])
                clock.segments.append({"started_at": segment["started_at"], "last_seen": end, "stopped_at": end})
        elif resume:
            raise ValueError("cannot resume a campaign without its wall-clock record")
        return clock

    def start(self) -> None:
        now = time.time()
        self.segments.append({"started_at": now, "last_seen": now})
        self._write(force=True)

    def heartbeat(self) -> None:
        if self.segments and "stopped_at" not in self.segments[-1]:
            self.segments[-1]["last_seen"] = time.time()
            self._write()

    def stop(self) -> None:
        if self.segments and "stopped_at" not in self.segments[-1]:
            now = time.time()
            self.segments[-1].update(last_seen=now, stopped_at=now)
            self._write(force=True)

    def elapsed(self) -> float:
        total = 0.0
        now = time.time()
        for segment in self.segments:
            total += segment.get("stopped_at", now) - segment["started_at"]
        return max(0.0, total)

    def remaining(self) -> float:
        if self.max_seconds is None:
            return math.inf
        return max(0.0, self.max_seconds - self.elapsed())

    def snapshot(self) -> dict[str, float | None]:
        remaining = self.remaining()
        return {"max_seconds": self.max_seconds, "elapsed_seconds": self.elapsed(),
                "remaining_seconds": None if math.isinf(remaining) else remaining}

    def _write(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_write >= HEARTBEAT_SECONDS:
            atomic_json_write(self.path, {"max_seconds": self.max_seconds, "segments": self.segments})
            self._last_write = now


class WallClockExhausted(RuntimeError):
    """The campaign's total wall-time allowance is spent."""
