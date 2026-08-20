from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ldm_tts.contracts import (
    AcquisitionSpec,
    Candidate,
    CandidateDomainSpec,
    LDMTaskSpec,
    ObjectiveSpec,
    ReservoirExpansionSpec,
    ReservoirSpec,
    ResponseSpaceSpec,
    SurrogateSpaceSpec,
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _task_spec_kwargs() -> dict[str, object]:
    return {
        "task": "interface_test",
        "objectives": (ObjectiveSpec("score", "maximize"),),
        "response_spaces": (ResponseSpaceSpec("items", "json"),),
        "acquisition": AcquisitionSpec(
            "reservoir_order",
            ("score",),
            "maximize",
            "first",
        ),
        "reservoir": ReservoirSpec(
            "items",
            (
                ReservoirExpansionSpec(
                    "emit_items",
                    "emit_candidate",
                    "items",
                    True,
                ),
            ),
            "validate item",
            "item identity",
        ),
        "surrogate": SurrogateSpaceSpec("none", "not used", "none"),
    }


def test_package_root_import_does_not_load_optional_dependencies() -> None:
    script = """
import builtins
import sys

blocked = {"numpy", "openai", "torch"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"optional dependency imported: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import ldm_tts

assert not blocked.intersection(sys.modules)
assert ldm_tts.__all__ == ()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_contracts_interface_is_dependency_light() -> None:
    script = """
import builtins

blocked = {"numpy", "openai", "torch"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"optional dependency imported: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from ldm_tts.contracts import Candidate, LDMTaskSpec
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_historical_root_contract_exports_are_lazy() -> None:
    script = """
import builtins
import sys

blocked = {"numpy", "openai", "torch"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"optional dependency imported: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from ldm_tts import CampaignRuntime, Candidate, LDMTaskSpec

assert not blocked.intersection(sys.modules)
assert Candidate.__module__ == "ldm_tts.contracts.candidate"
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_canonical_task_contract_uses_candidate_domain_only() -> None:
    domain = CandidateDomainSpec("items", "structured", None)
    task = LDMTaskSpec(candidate_domain=domain, **_task_spec_kwargs())

    assert task.candidate_domain is domain
    assert "candidate_domain" in task.to_dict()
    assert "candidate_space" not in task.to_dict()


def test_historical_candidate_space_name_resolves_to_canonical_contract() -> None:
    from ldm_tts import CandidateSpaceSpec

    assert CandidateSpaceSpec is CandidateDomainSpec


def test_nanogpt_internals_are_not_reexported_from_package_root() -> None:
    import ldm_tts

    for name in ("OperationSchema", "OperationParameter", "operation_feature_dim"):
        with pytest.raises(AttributeError):
            getattr(ldm_tts, name)


def test_package_root_import_does_not_load_task_modules() -> None:
    script = """
import builtins
import sys

blocked = {"tasks", "tasks.nanogpt", "numpy", "torch", "openai"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError(f"task/optional dependency imported: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import ldm_tts
from ldm_tts import CampaignRuntime, Candidate, LDMTaskSpec

assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
