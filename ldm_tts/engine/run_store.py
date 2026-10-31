"""Durable campaign budgets and machine-readable status heartbeats."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_EVENT_LOCK = threading.Lock()


class BudgetExceededError(RuntimeError):
    """Raised before a campaign consumes more than a declared budget."""


@dataclass
class BudgetLedger:
    """Named limits and counters persisted atomically after every mutation."""

    limits: dict[str, int | float]
    counters: dict[str, int | float] = field(default_factory=dict)
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.limits = _validated_numbers(self.limits, "limits")
        self.counters = _validated_numbers(self.counters, "counters")
        if self.path is not None:
            self.path = Path(self.path)

    @classmethod
    def load(cls, path: Path) -> "BudgetLedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            limits=dict(payload.get("limits", {})),
            counters=dict(payload.get("counters", {})),
            path=Path(path),
            metadata=dict(payload.get("metadata", {})),
        )

    def consume(self, name: str, amount: int | float = 1) -> int | float:
        return self.consume_many({name: amount})[name]

    def consume_many(
        self, amounts: Mapping[str, int | float], *, usage_key: str | None = None
    ) -> dict[str, int | float]:
        """Persist increments, or cumulative usage for an idempotent operation key."""

        return self.consume_groups({usage_key: amounts})

    def consume_groups(
        self, groups: Mapping[str | None, Mapping[str, int | float]]
    ) -> dict[str, int | float]:
        """Reserve a whole batch atomically, retaining each operation's replay key."""

        usage = dict(self.metadata.get("cumulative_usage", {}))
        updates: dict[str, int | float] = {}
        for usage_key, amounts in groups.items():
            if usage_key is not None and not usage_key.strip():
                raise ValueError("usage_key must not be empty")
            previous = usage.get(usage_key, {}) if usage_key is not None else {}
            totals = dict(previous)
            for name, raw_amount in amounts.items():
                amount = _validated_number(raw_amount, f"amount for {name}")
                if usage_key is not None:
                    totals[name] = max(previous.get(name, 0), amount)
                    amount = totals[name] - previous.get(name, 0)
                current = updates.get(name, self.counters.get(name, 0))
                updated = current + amount
                limit = self.limits.get(name)
                if limit is not None and updated > limit:
                    raise BudgetExceededError(
                        f"Budget {name!r} would be exceeded: "
                        f"{current} + {amount} > {limit}"
                    )
                updates[name] = updated
            if usage_key is not None:
                usage[usage_key] = totals
        self.counters.update(updates)
        if any(key is not None for key in groups):
            self.metadata["cumulative_usage"] = usage
        self.write()
        return updates

    def remaining(self, name: str) -> int | float | None:
        limit = self.limits.get(name)
        if limit is None:
            return None
        return max(0, limit - self.counters.get(name, 0))

    def snapshot(self) -> dict[str, Any]:
        names = sorted(set(self.limits) | set(self.counters))
        limits = {
            name: _serialized_number(value)
            for name, value in self.limits.items()
        }
        counters = {
            name: _serialized_number(self.counters.get(name, 0))
            for name in names
        }
        return {
            "limits": limits,
            "counters": counters,
            "remaining": {
                name: (
                    None
                    if self.remaining(name) is None
                    else _serialized_number(self.remaining(name))
                )
                for name in names
            },
            "metadata": dict(self.metadata),
        }

    def write(self) -> None:
        if self.path is not None:
            atomic_json_write(self.path, self.snapshot())


@dataclass
class CampaignStatus:
    """Atomic status writer suitable for polling detached campaigns."""

    path: Path
    task: str
    run_id: str
    contract_sha256: str = ""
    contract_profile: str = ""
    started_at: float = field(default_factory=time.time)

    def update(
        self,
        status: str,
        *,
        phase: str = "",
        iteration: int | None = None,
        message: str = "",
        budget: BudgetLedger | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "task": self.task,
            "run_id": self.run_id,
            "status": str(status),
            "phase": str(phase),
            "iteration": iteration,
            "message": str(message),
            "started_at_unix": self.started_at,
            "updated_at_unix": time.time(),
            "contract_sha256": self.contract_sha256,
            "contract_profile": self.contract_profile,
        }
        if budget is not None:
            payload["budget"] = budget.snapshot()
        if details:
            payload["details"] = dict(details)
        atomic_json_write(self.path, payload)
        return payload


@dataclass(frozen=True)
class CampaignEvent:
    """One append-only event in a campaign's durable history."""

    sequence: int
    event_type: str
    task: str
    run_id: str
    timestamp_unix: float
    iteration: int | None = None
    candidate_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    event_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "task": self.task,
            "run_id": self.run_id,
            "timestamp_unix": self.timestamp_unix,
            "iteration": self.iteration,
            "candidate_id": self.candidate_id,
            "payload": dict(self.payload),
            "event_key": self.event_key,
        }


@dataclass
class CampaignRuntime:
    """Resumable run artifacts, budgets, events, checkpoints, and status.

    Use :meth:`open` rather than constructing this class directly. A fresh run
    refuses to overwrite an existing campaign manifest; a resumed run verifies
    task and run identity before loading its budget and event sequence.
    """

    run_dir: Path
    task: str
    run_id: str
    budget: BudgetLedger
    status: CampaignStatus
    _next_event_sequence: int = 0
    _keyed_events: dict[str, CampaignEvent] = field(default_factory=dict)

    @classmethod
    def open(
        cls,
        run_dir: Path,
        *,
        task: str,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        task_spec: Any = None,
        budget_limits: Mapping[str, int | float] | None = None,
        contract_snapshot: Mapping[str, Any] | None = None,
        contract_sha256: str = "",
        contract_profile: str = "",
        resume: bool = False,
    ) -> "CampaignRuntime":
        path = Path(run_dir)
        manifest_path = path / "campaign.json"
        requested_run_id = str(run_id or path.name)
        if not task.strip():
            raise ValueError("campaign task must not be empty")
        if not requested_run_id.strip():
            raise ValueError("campaign run_id must not be empty")

        existing_manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not resume:
                raise FileExistsError(
                    f"campaign already exists at {path}; pass resume=True to continue it"
                )
            if existing_manifest.get("task") != task:
                raise ValueError(
                    f"cannot resume task {task!r} from campaign for "
                    f"{existing_manifest.get('task')!r}"
                )
            existing_run_id = str(existing_manifest.get("run_id", ""))
            if run_id is not None and existing_run_id != requested_run_id:
                raise ValueError(
                    f"cannot resume run_id {requested_run_id!r} from {existing_run_id!r}"
                )
            requested_run_id = existing_run_id
            contract_sha256 = str(existing_manifest.get("contract_sha256", contract_sha256))
            contract_profile = str(existing_manifest.get("contract_profile", contract_profile))
        elif resume:
            raise FileNotFoundError(f"campaign manifest does not exist: {manifest_path}")

        path.mkdir(parents=True, exist_ok=True)
        budget_path = path / "budget.json"
        if resume and budget_path.exists():
            budget = BudgetLedger.load(budget_path)
            if budget_limits is not None and dict(budget_limits) != budget.limits:
                raise ValueError("resume budget limits do not match the persisted campaign")
        else:
            budget = BudgetLedger(
                limits=dict(budget_limits or {}),
                path=budget_path,
                metadata={"task": task, "run_id": requested_run_id},
            )
            budget.write()

        status_path = path / "status.json"
        started_at = time.time()
        if resume and status_path.exists():
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            started_at = float(status_payload.get("started_at_unix", started_at))
        status = CampaignStatus(
            path=status_path,
            task=task,
            run_id=requested_run_id,
            contract_sha256=contract_sha256,
            contract_profile=contract_profile,
            started_at=started_at,
        )

        event_path = path / "events.jsonl"
        events = load_jsonl(event_path, repair_incomplete_tail=True) if resume else []
        runtime = cls(
            run_dir=path,
            task=task,
            run_id=requested_run_id,
            budget=budget,
            status=status,
            _next_event_sequence=(
                max((int(item.get("sequence", -1)) for item in events), default=-1) + 1
            ),
        )
        for item in events:
            if item.get("event_key"):
                event = CampaignEvent(**{key: value for key, value in item.items() if key != "schema_version"})
                if event.event_key in runtime._keyed_events:
                    raise ValueError("duplicate durable event key")
                runtime._keyed_events[event.event_key] = event

        if existing_manifest is None:
            manifest = {
                "schema_version": 1,
                "task": task,
                "run_id": requested_run_id,
                "created_at_unix": started_at,
                "contract_sha256": contract_sha256,
                "contract_profile": contract_profile,
            }
            atomic_json_write(manifest_path, manifest)
            if config is not None:
                atomic_json_write(path / "config.json", dict(config))
            if task_spec is not None:
                payload = task_spec.to_dict() if hasattr(task_spec, "to_dict") else task_spec
                atomic_json_write(path / "ldm_task_spec.json", payload)
            if contract_snapshot is not None:
                atomic_json_write(path / "experiment_contract.json", dict(contract_snapshot))
            runtime.status.update("running", phase="initializing", budget=budget)
            runtime.record("campaign_started")
        else:
            runtime.status.update("running", phase="resumed", budget=budget)
            runtime.record("campaign_resumed")
        return runtime

    @property
    def event_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    def consume(self, name: str, amount: int | float = 1) -> int | float:
        return self.consume_many({name: amount})[name]

    def consume_many(
        self, amounts: Mapping[str, int | float], *, usage_key: str | None = None
    ) -> dict[str, int | float]:
        return self.consume_groups({usage_key: amounts})

    def consume_groups(
        self, groups: Mapping[str | None, Mapping[str, int | float]]
    ) -> dict[str, int | float]:
        values = self.budget.consume_groups(groups)
        self.status.update("running", phase="budget_updated", budget=self.budget)
        return values

    def record(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        iteration: int | None = None,
        candidate_id: str = "",
        event_key: str = "",
    ) -> CampaignEvent:
        if not event_type.strip():
            raise ValueError("campaign event_type must not be empty")
        if iteration is not None and iteration < 0:
            raise ValueError("campaign event iteration must be non-negative")
        payload_json = json.dumps(dict(payload or {}), sort_keys=True)
        with _EVENT_LOCK:
            if event_key in self._keyed_events:
                event = self._keyed_events[event_key]
                if ((event.event_type, event.iteration, event.candidate_id) !=
                        (event_type, iteration, str(candidate_id)) or
                        json.dumps(event.payload, sort_keys=True) != payload_json):
                    raise ValueError(f"event identity conflict: {event_key}")
                return event
            event = CampaignEvent(
                sequence=self._next_event_sequence,
                event_type=event_type,
                task=self.task,
                run_id=self.run_id,
                timestamp_unix=time.time(),
                iteration=iteration,
                candidate_id=str(candidate_id),
                payload=json.loads(payload_json),
                event_key=event_key,
            )
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._next_event_sequence += 1
            if event_key:
                self._keyed_events[event_key] = event
        return event

    def events(self) -> list[dict[str, Any]]:
        return load_jsonl(self.event_path)

    def checkpoint(self, state: Mapping[str, Any]) -> Path:
        path = self.run_dir / "checkpoint.json"
        atomic_json_write(
            path,
            {
                "schema_version": 1,
                "task": self.task,
                "run_id": self.run_id,
                "event_sequence": self._next_event_sequence,
                "state": dict(state),
                "updated_at_unix": time.time(),
            },
        )
        self.record("checkpoint_written")
        return path

    def load_checkpoint(self) -> dict[str, Any] | None:
        path = self.run_dir / "checkpoint.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task") != self.task or payload.get("run_id") != self.run_id:
            raise ValueError("checkpoint identity does not match the active campaign")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError("campaign checkpoint state must be a JSON object")
        return state

    def finish(self, summary: Mapping[str, Any], *, status: str = "completed") -> Path:
        if status not in {"completed", "stopped"}:
            raise ValueError("terminal campaign status must be 'completed' or 'stopped'")
        summary_path = self.run_dir / "summary.json"
        atomic_json_write(summary_path, dict(summary))
        self.record("campaign_finished", {"status": status},
                    event_key=f"campaign:finished:{status}:{summary.get('next_round')}:{summary.get('observation_count')}")
        self.status.update(status, phase="finished", budget=self.budget, details=summary)
        return summary_path

    def fail(self, error: BaseException | str) -> None:
        message = str(error)
        self.record("campaign_failed", {"error": message})
        self.status.update(
            "failed",
            phase="failed",
            message=message,
            budget=self.budget,
        )

    def pause(
        self,
        status: str,
        *,
        phase: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a non-terminal, resumable campaign interruption."""

        if not status.startswith("paused_"):
            raise ValueError("paused campaign status must start with 'paused_'")
        payload = {"status": status, "phase": phase, "message": message}
        if details:
            payload["details"] = dict(details)
        self.record("campaign_paused", payload)
        self.status.update(
            status,
            phase=phase,
            message=message,
            budget=self.budget,
            details=details,
        )


def atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def unique_run_dir(path: Path) -> Path:
    """Return ``path`` or a numbered sibling without overwriting a campaign."""

    requested = Path(path)
    if not requested.exists():
        return requested
    for index in range(2, 100_000):
        candidate = requested.with_name(f"{requested.name}_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a unique campaign directory beside {requested}")


def load_jsonl(path: str | Path, *, repair_incomplete_tail: bool = False) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records, offset = [], 0
    for index, line in enumerate(lines):
        try:
            if line.strip():
                records.append(json.loads(line.decode("utf-8")))
        except ValueError:
            if not repair_incomplete_tail or index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            # Preserve the interrupted append before repairing the writable log.
            backup = path.with_name(path.name + ".incomplete-" + hashlib.sha256(line).hexdigest() + ".json")
            atomic_json_write(backup, {"offset": offset, "tail_hex": line.hex()})
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            return records
        offset += len(line)
    if repair_incomplete_tail and data and not data.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return records


def _validated_numbers(
    values: Mapping[str, int | float], field_name: str
) -> dict[str, int | float]:
    return {
        str(name): _validated_number(value, f"{field_name}.{name}")
        for name, value in values.items()
    }


def _validated_number(value: int | float, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return value


def _serialized_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def utc_timestamp() -> str:
    """Return a stable UTC timestamp for persisted records."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


class JsonlTrajectoryRecorder:
    """Compatibility recorder for task-owned round artifacts."""

    def __init__(
        self,
        trajectory_dir: str | Path | None,
        *,
        config_snapshot: dict[str, Any] | None = None,
        existing_rounds: Sequence[dict[str, Any]] | None = None,
        rounds_filename: str = "rounds.jsonl",
        reset_rounds_file: bool = False,
        sort_keys: bool = False,
    ) -> None:
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.rounds_filename = rounds_filename
        self.sort_keys = sort_keys
        self.rounds: list[dict[str, Any]] = list(existing_rounds or [])
        if self.trajectory_dir:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            if reset_rounds_file:
                (self.trajectory_dir / self.rounds_filename).write_text("", encoding="utf-8")
            if config_snapshot is not None:
                self.write_json("config.json", config_snapshot)

    def append_round(self, record: dict[str, Any]) -> None:
        self.rounds.append(record)
        if self.trajectory_dir:
            with (self.trajectory_dir / self.rounds_filename).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=self.sort_keys) + "\n"
                )

    def write_json(self, filename: str, payload: Any, *, indent: int = 2) -> Path | None:
        if self.trajectory_dir is None:
            return None
        path = self.trajectory_dir / filename
        path.write_text(
            json.dumps(payload, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


_DOCUMENT_LOCK = threading.Lock()


class AtomicJsonLog:
    """Compatibility atomic JSON document used by decision logs."""

    def __init__(self, path: str | Path, default_payload: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.write(default_payload)

    def read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write(self, payload: dict[str, Any]) -> None:
        atomic_json_write(self.path, payload)

    def update(self, mutator: Any) -> dict[str, Any]:
        with _DOCUMENT_LOCK:
            payload = self.read()
            mutator(payload)
            self.write(payload)
            return payload


__all__ = [
    "BudgetExceededError",
    "BudgetLedger",
    "AtomicJsonLog",
    "CampaignEvent",
    "CampaignRuntime",
    "CampaignStatus",
    "JsonlTrajectoryRecorder",
    "atomic_json_write",
    "load_jsonl",
    "unique_run_dir",
    "utc_timestamp",
]
