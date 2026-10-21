"""core/ldm/orchestrator/decision_log.py — append-only JSON file writer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ldm_tts.engine.run_store import AtomicJsonLog, utc_timestamp


class DecisionLog:
    """Append orchestrator decisions through the shared atomic JSON log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._log = AtomicJsonLog(
            self.path,
            {"experiment_id": self.path.stem, "config_snapshot": {}, "decisions": []},
        )

    def update_config_snapshot(self, config: dict[str, Any]) -> None:
        """Update the file-level config_snapshot (idempotent)."""
        self._log.update(lambda data: data.__setitem__("config_snapshot", config))

    def append(self, entry: dict) -> None:
        """Append a single decision entry."""
        payload = dict(entry)

        def append_entry(data: dict[str, Any]) -> None:
            payload.setdefault("timestamp", utc_timestamp())
            data["decisions"].append(payload)

        self._log.update(append_entry)
