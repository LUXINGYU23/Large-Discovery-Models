"""Injectable clocks: every controller timestamp, backoff and due time goes through ``now()``."""
from __future__ import annotations

import threading
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class RealClock:
    def now(self) -> float:
        return time.time()


class VirtualClock:
    """Moves only when told to. Any object with ``now()`` works (e.g. the ES test clock)."""

    def __init__(self, start: float = 0.0):
        self._now = float(start)
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("a virtual clock cannot move backwards")
        with self._lock:
            self._now += float(seconds)
            return self._now

    def advance_to(self, when: float) -> float:
        with self._lock:
            if when < self._now:
                raise ValueError("a virtual clock cannot move backwards")
            self._now = float(when)
            return self._now
