"""Task manifests (``async.executors`` in ``task.json``) and deployment configuration.

The ES reads ``task.json`` itself and imports the hooks it names; it never goes through
``ldm_tts`` (design 2.5).
"""
from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .clock import RealClock
from .executors import is_executor
from .service import ExperimentService

CONFIG_KEYS = {"database", "artifact_dir", "host", "port", "token", "pools", "tasks", "executor_options",
               "python_path", "max_queued_per_campaign", "max_running_per_campaign", "max_infra_retries",
               "retry_after_seconds", "poll_interval_seconds"}


@dataclass(frozen=True)
class ExecutorContext:
    """What an executor factory receives."""
    kind: str
    clock: Any
    options: Mapping[str, Any] = field(default_factory=dict)
    task_dir: Path | None = None


def load_object(reference: str):
    module, _, attribute = str(reference).partition(":")
    if not module or not attribute:
        raise ValueError(f"hook {reference!r} must have the form 'module:object'")
    value = importlib.import_module(module)
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def manifest_executors(task_json) -> dict[str, str]:
    """Parse ``async.executors`` (kind -> "module:object") from a task manifest."""
    path = Path(task_json)
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    section = manifest.get("async") or {}
    if not isinstance(section, dict):
        raise ValueError(f"{path}: 'async' must be an object")
    executors = section.get("executors") or {}
    if not isinstance(executors, dict) or not all(isinstance(k, str) and k and isinstance(v, str)
                                                   for k, v in executors.items()):
        raise ValueError(f"{path}: async.executors must map kinds to 'module:object' hooks")
    return dict(executors)


def load_task_executors(task_json, *, clock, options: Mapping[str, Mapping] | None = None) -> list:
    """Build the executors a task declares. A hook names an executor instance or a factory
    called with an ``ExecutorContext``; the result's ``kind`` must equal the manifest key."""
    path, built = Path(task_json), []
    for kind, reference in manifest_executors(path).items():
        target = load_object(reference)
        context = ExecutorContext(kind, clock, dict((options or {}).get(kind, {})), path.parent)
        executor = target if is_executor(target) else target(context)
        if not is_executor(executor):
            raise TypeError(f"{reference} did not produce an executor")
        if executor.kind != kind:
            raise ValueError(f"{reference} produced kind {executor.kind!r}, manifest says {kind!r}")
        built.append(executor)
    return built


def load_config(path) -> dict:
    """Read a deployment config; relative paths resolve against the config file's directory."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("deployment config must be a JSON object")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown deployment config keys: {sorted(unknown)}")
    if not config.get("database"):
        raise ValueError("deployment config needs 'database'")
    base = path.parent

    def resolve(value):
        return str((base / value).resolve())

    resolved = {"host": "127.0.0.1", "port": 8765, "token": None, "pools": {"default": 1}, "tasks": [],
                "executor_options": {}, "python_path": [], "poll_interval_seconds": 1.0, **config}
    resolved["database"] = resolve(config["database"])
    if config.get("artifact_dir"):
        resolved["artifact_dir"] = resolve(config["artifact_dir"])
    resolved["tasks"] = [resolve(t) for t in resolved["tasks"]]
    resolved["python_path"] = [resolve(p) for p in resolved["python_path"]]
    return resolved


def build_service(config: Mapping, *, clock=None) -> ExperimentService:
    for entry in reversed(config.get("python_path", [])):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    clock = clock or RealClock()
    executors = [e for task in config.get("tasks", [])
                 for e in load_task_executors(task, clock=clock, options=config.get("executor_options"))]
    options = {k: config[k] for k in ("artifact_dir", "max_queued_per_campaign", "max_running_per_campaign",
                                      "max_infra_retries", "retry_after_seconds") if k in config}
    return ExperimentService(config["database"], executors, pools=config.get("pools"), clock=clock, **options)
