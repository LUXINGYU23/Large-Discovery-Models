"""Campaign configuration: frozen scientific parameters vs. deployment options (design 9.1).

``CampaignConfig`` (plus the plugin identities) is written once into the store and verified on
every restart; ``RuntimeOptions`` (ES address, timeouts, retry and polling cadence) are not.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .store import Store
from .wire import CAMPAIGN_PATTERN, canonical, digest


class CampaignConfigMismatch(ValueError):
    pass


@dataclass(frozen=True)
class CampaignConfig:
    campaign_id: str
    budget: int
    concurrency: int = 1
    buffer_depth: int = 0
    max_attempts_per_key: int = 1
    seed: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)  # frozen parameters owned by later packages (pacing)

    def __post_init__(self):
        if not isinstance(self.campaign_id, str) or not CAMPAIGN_PATTERN.fullmatch(self.campaign_id):
            raise ValueError("campaign_id must match [A-Za-z0-9][A-Za-z0-9._~-]{0,150}")
        for name, low in (("budget", 0), ("concurrency", 1), ("buffer_depth", 0), ("max_attempts_per_key", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "extra", dict(self.extra))
        canonical(self.extra)

    @property
    def max_outstanding(self) -> int:
        return self.concurrency + self.buffer_depth

    @property
    def max_unstarted(self) -> int:
        return max(self.buffer_depth, 1)

    def to_json(self) -> dict:
        return {"campaign_id": self.campaign_id, "budget": self.budget, "concurrency": self.concurrency,
                "buffer_depth": self.buffer_depth, "max_attempts_per_key": self.max_attempts_per_key,
                "seed": self.seed, "extra": dict(self.extra)}


@dataclass(frozen=True)
class RuntimeOptions:
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 60.0
    anti_entropy_interval: float = 300.0
    page_limit: int = 100
    max_pages_per_step: int = 50
    max_intents_per_step: int = 16
    unreachable_alert_seconds: float = 600.0
    max_hook_attempts: int = 3
    action_hook_failure: str = "pause"  # or "skip"
    hook_timeout_seconds: float | None = None  # task hooks run on a worker thread when set; overrun = failure
    poll_interval: float = 2.0
    long_poll_seconds: float = 20.0

    def backoff(self, attempts: int) -> float:
        return min(self.retry_max_seconds, self.retry_base_seconds * (2 ** max(0, attempts - 1)))


def freeze_or_verify(store: Store, frozen: Mapping[str, Any], now: float, ignore: tuple[str, ...] = ()) -> None:
    """First open writes the frozen configuration; later opens must present the same one
    (dotted paths in ``ignore`` are not compared, e.g. the selector in collect-only mode)."""
    text = canonical(dict(frozen))
    with store.tx("freeze") as db:
        row = db.execute("SELECT config_json FROM campaign WHERE singleton=1").fetchone()
        if row is None:
            db.execute("INSERT INTO campaign (singleton, campaign_id, config_json, config_digest, created_at) "
                       "VALUES (1,?,?,?,?)", (frozen["campaign"]["campaign_id"], text, digest(dict(frozen)), now))
            return
    if row[0] != text:
        changed = [path for path in _diff(json.loads(row[0]), json.loads(text)) if path not in ignore]
        if changed:
            raise CampaignConfigMismatch("frozen campaign configuration differs: " + ", ".join(sorted(changed)))


def _diff(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            yield from _diff(left.get(key), right.get(key), f"{prefix}{key}.")
    elif canonical(left) != canonical(right):
        yield prefix.rstrip(".")
