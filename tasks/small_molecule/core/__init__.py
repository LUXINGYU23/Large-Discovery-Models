"""Public re-exports for the LDM-TTS small-molecule adapter.

Exports are resolved lazily so lightweight imports, such as
``tasks.small_molecule.core.llm_advisor`` or ``tasks.small_molecule.core.rng``, do not immediately require the
full GP / Torch / docking stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_REF": ("tasks.small_molecule.core.scorer", "DEFAULT_REF"),
    "GPConfig": ("tasks.small_molecule.core.gp", "GPConfig"),
    "GPSurrogate": ("tasks.small_molecule.core.gp", "GPSurrogate"),
    "NNScorer": ("tasks.small_molecule.core.objective_nn", "NNScorer"),
    "NNScorerConfig": ("tasks.small_molecule.core.objective_nn", "NNScorerConfig"),
    "RNG": ("tasks.small_molecule.core.rng", "RNG"),
    "ReasynConfig": ("tasks.small_molecule.core.analog", "ReasynConfig"),
    "Scorer": ("tasks.small_molecule.core.scorer", "Scorer"),
    "Scorers": ("tasks.small_molecule.core.scorer", "Scorers"),
    "VinaScorer": ("tasks.small_molecule.core.objective_vina", "VinaScorer"),
    "VinaScorerConfig": ("tasks.small_molecule.core.objective_vina", "VinaScorerConfig"),
    "as_rng": ("tasks.small_molecule.core.rng", "as_rng"),
    "as_scorer_tuple": ("tasks.small_molecule.core.scorer", "as_scorer_tuple"),
    "generate_analogs": ("tasks.small_molecule.core.analog", "generate_analogs"),
    "register_ref": ("tasks.small_molecule.core.scorer", "register_ref"),
    "resolve_ref_point": ("tasks.small_molecule.core.scorer", "resolve_ref_point"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
