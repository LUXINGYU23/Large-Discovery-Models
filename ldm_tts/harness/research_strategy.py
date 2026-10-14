"""Configuration entry point for persistent sessions and research-tree forks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ldm_tts.harness.research_tree import (
    ArchivePolicy, CoverageForks, DiverseArchive, ForkPolicy, LatestMainForks,
)


@dataclass(frozen=True)
class ResearchStrategy:
    """Execution lifecycle plus replaceable pure selection rules.

    A null fork policy means independent persistent identities. A fork policy
    means a persistent coordinator and disposable children of selected snapshots.
    Custom factories can return this object with their own rule implementations.
    """

    fork_policy: ForkPolicy | None = None
    archive_policy: ArchivePolicy = DiverseArchive()

    def __post_init__(self) -> None:
        for obj, method in ((self.fork_policy, "plan"), (self.archive_policy, "retain")):
            if obj is None and method == "plan":
                continue
            if not callable(getattr(obj, method, None)) or not callable(getattr(obj, "identity", None)):
                raise TypeError(f"research policy requires {method}() and identity()")

    @property
    def requires_main(self) -> bool:
        return self.fork_policy is not None

    def identity(self) -> dict:
        return {"lifecycle": "forked" if self.requires_main else "persistent",
                "fork_policy": self.fork_policy.identity() if self.fork_policy else None,
                "archive_policy": self.archive_policy.identity() if self.requires_main else None}


def _factory(config: dict[str, Any]):
    if set(config) - {"factory", "options"}:
        raise ValueError("custom rule accepts only factory and options")
    module, separator, name = config["factory"].partition(":")
    if not separator or not module or not name:
        raise ValueError("factory must be an importable module:callable")
    return getattr(import_module(module), name)(**config.get("options", {}))


def load_research_strategy(config: dict[str, Any]) -> ResearchStrategy:
    """Resolve built-ins or explicit user-supplied Python factories.

    This is trusted application configuration, never model-generated content.
    A custom policy must include its semantic version/parameters in identity().
    """
    if not isinstance(config, dict):
        raise ValueError("research strategy configuration must be a JSON object")
    if "factory" in config:
        strategy = _factory(config)
        if not isinstance(strategy, ResearchStrategy):
            raise TypeError("strategy factory must return ResearchStrategy")
        return strategy
    if set(config) - {"name", "fork_policy", "archive_policy"}:
        raise ValueError("unknown research strategy configuration fields")
    name = config.get("name", "independent")
    if name == "independent":
        if set(config) - {"name"}:
            raise ValueError("independent sessions do not use fork/archive policies")
        return ResearchStrategy()
    if name != "main_forks":
        raise ValueError(f"unknown research strategy: {name}")
    fork = config.get("fork_policy", {"name": "latest"})
    archive = config.get("archive_policy", {"name": "diverse"})
    if not isinstance(fork, dict) or not isinstance(archive, dict):
        raise ValueError("fork_policy and archive_policy must be JSON objects")
    if "factory" in fork:
        policy = _factory(fork)
    elif fork == {"name": "latest"}:
        policy = LatestMainForks()
    elif fork == {"name": "coverage"}:
        policy = CoverageForks()
    else:
        raise ValueError("unknown fork policy configuration")
    if "factory" in archive:
        archive_policy = _factory(archive)
    elif archive.get("name") == "diverse" and not set(archive) - {"name", "capacity", "overlap_threshold"}:
        archive_policy = DiverseArchive(**{k: v for k, v in archive.items() if k != "name"})
    else:
        raise ValueError("unknown archive policy configuration")
    return ResearchStrategy(policy, archive_policy)
