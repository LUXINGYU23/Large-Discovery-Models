"""Optional ``async`` section of a task manifest and on-demand hook loading.

The experiment service parses the same JSON on its own, so the schema stays flat
and strict (see ``docs/impl-notes/contracts.md``). Parsing never imports task code;
``load_hook`` does, when a caller asks for a specific plugin.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


HOOK_REF_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
EXECUTOR_KIND_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
HOOK_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
# Dotted lowercase segments; the last segment may be "*" (e.g. "anomaly.*"); "*" alone matches all.
EVENT_NAME_PATTERN = re.compile(r"^(\*|[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*(\.\*)?)$")
HOOK_ACTIONS = frozenset({"request_research", "notify_agents", "request_pause"})
ASYNC_SECTION_KEYS = frozenset(
    {"experiment_codec", "executors", "history_projection", "selector", "pacing", "reservoir", "hooks"}
)
HOOK_DECLARATION_KEYS = frozenset({"name", "hook", "subscribe", "actions", "version"})
_OPTIONAL_REF_FIELDS = ("history_projection", "selector", "pacing", "reservoir")


class AsyncManifestError(ValueError):
    """Raised when the ``async`` manifest section is malformed."""


class HookLoadError(RuntimeError):
    """Raised when a declared ``module:object`` hook cannot be imported."""


@dataclass(frozen=True)
class AsyncHookDeclaration:
    """One task hook: which events it observes and which actions it may request."""

    name: str
    hook: str
    subscribe: tuple[str, ...]
    actions: tuple[str, ...] = ()
    version: str = "1"

    @property
    def requests_research(self) -> bool:
        return "request_research" in self.actions

    @property
    def is_action_hook(self) -> bool:
        return bool(self.actions)


@dataclass(frozen=True)
class AsyncManifest:
    """Validated ``async`` section; all values are unimported ``module:object`` references."""

    experiment_codec: str
    executors: Mapping[str, str]
    history_projection: str | None = None
    selector: str | None = None
    pacing: str | None = None
    reservoir: str | None = None
    hooks: tuple[AsyncHookDeclaration, ...] = field(default_factory=tuple)

    @property
    def requests_research(self) -> bool:
        """True when any hook may request research (a finite research budget is then required)."""

        return any(hook.requests_research for hook in self.hooks)

    def hook_declaration(self, name: str) -> AsyncHookDeclaration:
        for hook in self.hooks:
            if hook.name == name:
                return hook
        raise KeyError(f"unknown task hook {name!r}")

    def load_experiment_codec(self) -> Any:
        return load_hook(self.experiment_codec)

    def load_executor_factory(self, kind: str) -> Any:
        try:
            ref = self.executors[kind]
        except KeyError as exc:
            raise KeyError(
                f"unknown executor kind {kind!r}; declared kinds: {sorted(self.executors)}"
            ) from exc
        return load_hook(ref)

    def load_history_projection(self) -> Any | None:
        return None if self.history_projection is None else load_hook(self.history_projection)

    def load_selector(self) -> Any | None:
        return None if self.selector is None else load_hook(self.selector)

    def load_pacing(self) -> Any | None:
        return None if self.pacing is None else load_hook(self.pacing)

    def load_reservoir(self) -> Any | None:
        return None if self.reservoir is None else load_hook(self.reservoir)

    def load_task_hook(self, name: str) -> Any:
        return load_hook(self.hook_declaration(name).hook)


def load_hook(reference: str) -> Any:
    """Import ``module:object`` on demand and return the object."""

    if not isinstance(reference, str) or not HOOK_REF_PATTERN.fullmatch(reference):
        raise HookLoadError(f"invalid hook reference {reference!r}; expected 'python.module:object'")
    module_name, _, attribute = reference.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise HookLoadError(f"cannot import module for hook {reference!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise HookLoadError(f"module {module_name!r} has no object {attribute!r}") from exc


def _hook_ref(value: Any, where: str) -> str:
    if not isinstance(value, str) or not HOOK_REF_PATTERN.fullmatch(value):
        raise AsyncManifestError(f"{where} must be 'python.module:object', got {value!r}")
    return value


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise AsyncManifestError(f"{where} must be a list of strings")
    if len(set(value)) != len(value):
        raise AsyncManifestError(f"{where} must not contain duplicates")
    return tuple(value)


def _parse_hook(raw: Any, index: int) -> AsyncHookDeclaration:
    where = f"async.hooks[{index}]"
    if not isinstance(raw, dict):
        raise AsyncManifestError(f"{where} must be an object")
    unknown = sorted(str(key) for key in raw if key not in HOOK_DECLARATION_KEYS)
    if unknown:
        raise AsyncManifestError(f"unknown field(s) in {where}: {', '.join(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not HOOK_NAME_PATTERN.fullmatch(name):
        raise AsyncManifestError(f"{where}.name must be a lowercase identifier, got {name!r}")
    ref = _hook_ref(raw.get("hook"), f"{where}.hook")
    subscribe = _string_list(raw.get("subscribe"), f"{where}.subscribe")
    if not subscribe:
        raise AsyncManifestError(f"{where}.subscribe must list at least one event")
    for event in subscribe:
        if not EVENT_NAME_PATTERN.fullmatch(event):
            raise AsyncManifestError(
                f"{where}.subscribe has invalid event name {event!r}; expected dotted "
                "lowercase segments such as 'observation.ingested' or 'anomaly.*'"
            )
    actions = _string_list(raw.get("actions", []), f"{where}.actions")
    unsupported = sorted(set(actions) - HOOK_ACTIONS)
    if unsupported:
        raise AsyncManifestError(
            f"{where}.actions has unsupported action(s) {unsupported}; "
            f"allowed: {sorted(HOOK_ACTIONS)}"
        )
    version = raw.get("version", "1")
    if not isinstance(version, str) or not version.strip():
        raise AsyncManifestError(f"{where}.version must be a non-empty string")
    return AsyncHookDeclaration(name, ref, subscribe, actions, version)


def parse_async_manifest(raw: Any) -> AsyncManifest:
    """Validate the manifest's ``async`` object; does not import any task code."""

    if not isinstance(raw, dict):
        raise AsyncManifestError("async section must be a JSON object")
    unknown = sorted(str(key) for key in raw if key not in ASYNC_SECTION_KEYS)
    if unknown:
        raise AsyncManifestError(f"unknown async field(s): {', '.join(unknown)}")
    codec = _hook_ref(raw.get("experiment_codec"), "async.experiment_codec")
    raw_executors = raw.get("executors")
    if not isinstance(raw_executors, dict) or not raw_executors:
        raise AsyncManifestError("async.executors must map at least one kind to a factory hook")
    executors: dict[str, str] = {}
    for kind, ref in raw_executors.items():
        if not isinstance(kind, str) or not EXECUTOR_KIND_PATTERN.fullmatch(kind):
            raise AsyncManifestError(f"invalid executor kind {kind!r}")
        executors[kind] = _hook_ref(ref, f"async.executors[{kind!r}]")
    optional = {
        name: None if raw.get(name) is None else _hook_ref(raw[name], f"async.{name}")
        for name in _OPTIONAL_REF_FIELDS
    }
    raw_hooks = raw.get("hooks", [])
    if raw_hooks is None:
        raw_hooks = []
    if not isinstance(raw_hooks, list):
        raise AsyncManifestError("async.hooks must be a list")
    hooks = tuple(_parse_hook(item, index) for index, item in enumerate(raw_hooks))
    names = [hook.name for hook in hooks]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise AsyncManifestError(f"duplicate async hook name(s): {', '.join(duplicated)}")
    return AsyncManifest(codec, executors, hooks=hooks, **optional)


__all__ = [
    "ASYNC_SECTION_KEYS",
    "AsyncHookDeclaration",
    "AsyncManifest",
    "AsyncManifestError",
    "EVENT_NAME_PATTERN",
    "HOOK_ACTIONS",
    "HookLoadError",
    "load_hook",
    "parse_async_manifest",
]
