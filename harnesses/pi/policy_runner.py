#!/usr/bin/env python3
"""Inspect and execute untrusted LDM optimization-policy artifacts."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


POLICY_API_VERSION = 1
MAX_ARTIFACT_BYTES = 128 * 1024
MAX_STAGE_LENGTH = 64
_CAPABILITY_EXPORTS = {
    "prior_mean": "compute_prior_mean",
    "ldm_weights": "choose_ldm_weights",
}
_ALLOWED_IMPORTS = {
    "collections",
    "dataclasses",
    "functools",
    "itertools",
    "math",
    "numpy",
    "operator",
    "statistics",
    "typing",
}
_FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "open"}
_FORBIDDEN_ATTRIBUTES = {
    "Popen",
    "compile",
    "connect",
    "eval",
    "exec",
    "open",
    "popen",
    "run",
    "socket",
    "system",
}


@dataclass(frozen=True)
class RunnerError(Exception):
    code: str
    message: str
    path: str = "/artifact_path"
    hint: str = "Repair the policy artifact and run validation again."

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError("invalid_input", f"Could not read JSON input {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError("invalid_input", f"JSON input {path.name} must be an object")
    return value


def _capabilities(contract: dict[str, Any]) -> dict[str, int]:
    raw = contract.get("enabled_capabilities")
    if not isinstance(raw, list) or not raw:
        raise RunnerError("invalid_contract", "enabled_capabilities must be a non-empty array")
    capabilities: dict[str, int] = {}
    for value in raw:
        if not isinstance(value, str) or "@" not in value:
            raise RunnerError("invalid_contract", "capabilities must use name@version identifiers")
        name, raw_version = value.rsplit("@", 1)
        if name not in _CAPABILITY_EXPORTS or not raw_version.isdigit():
            raise RunnerError("unsupported_capability", f"Unsupported policy capability: {value}")
        version = int(raw_version)
        if version != 1 or name in capabilities:
            raise RunnerError("unsupported_capability", f"Unsupported policy capability: {value}")
        capabilities[name] = version
    return capabilities


def inspect_artifact(artifact: Path, contract: dict[str, Any]) -> dict[str, Any]:
    try:
        body = artifact.read_bytes()
    except OSError as exc:
        raise RunnerError("artifact_not_found", f"Could not read policy artifact: {exc}") from exc
    if len(body) > MAX_ARTIFACT_BYTES:
        raise RunnerError(
            "artifact_too_large",
            f"Policy artifact exceeds {MAX_ARTIFACT_BYTES} bytes",
        )
    try:
        source = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunnerError("invalid_source", "Policy artifact must be UTF-8 Python source") from exc
    try:
        tree = ast.parse(source, filename=artifact.name)
    except SyntaxError as exc:
        raise RunnerError(
            "syntax_error",
            f"Python syntax error at line {exc.lineno}: {exc.msg}",
        ) from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".", 1)[0] for alias in node.names]
            if any(module not in _ALLOWED_IMPORTS for module in modules):
                raise RunnerError("forbidden_import", f"Import is not allowed: {modules}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if node.level or module not in _ALLOWED_IMPORTS:
                raise RunnerError("forbidden_import", f"Import is not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise RunnerError("forbidden_operation", f"Call is not allowed: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_ATTRIBUTES:
                raise RunnerError("forbidden_operation", f"Call is not allowed: .{node.func.attr}")

    assignments: dict[str, Any] = {}
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"POLICY_API_VERSION", "CAPABILITIES"}:
            try:
                assignments[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise RunnerError(
                    "invalid_declaration",
                    f"{target.id} must be a literal value",
                    path=f"/{target.id}",
                ) from exc

    if assignments.get("POLICY_API_VERSION") != POLICY_API_VERSION:
        raise RunnerError(
            "unsupported_api_version",
            f"POLICY_API_VERSION must equal {POLICY_API_VERSION}",
            path="/POLICY_API_VERSION",
        )
    expected = _capabilities(contract)
    if assignments.get("CAPABILITIES") != expected:
        raise RunnerError(
            "capability_mismatch",
            f"CAPABILITIES must exactly equal {expected}",
            path="/CAPABILITIES",
        )
    for capability, export in _CAPABILITY_EXPORTS.items():
        if capability in expected and export not in functions:
            raise RunnerError(
                "missing_export",
                f"Required function is missing: {export}",
                path=f"/exports/{export}",
            )
    return {
        "policy_api_version": POLICY_API_VERSION,
        "capabilities": expected,
        "exports": sorted(functions & set(_CAPABILITY_EXPORTS.values())),
        "artifact_size_bytes": len(body),
    }


def _load_module(artifact: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("ldm_generated_policy", artifact)
    if spec is None or spec.loader is None:
        raise RunnerError("module_load_failed", "Could not create a module loader")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RunnerError(
            "module_load_failed",
            f"Policy module failed to load: {type(exc).__name__}: {exc}",
        ) from exc
    return module


def _matrix(arrays: Any, name: str, columns: int | None = None) -> np.ndarray:
    if name not in arrays:
        raise RunnerError("invalid_input", f"Missing array: {name}")
    value = np.asarray(arrays[name], dtype=np.float64)
    if value.ndim != 2 or (columns is not None and value.shape[1] != columns):
        raise RunnerError("invalid_input", f"{name} must be a two-dimensional feature matrix")
    if not np.isfinite(value).all():
        raise RunnerError("invalid_input", f"{name} contains non-finite values")
    return value


def _vector(arrays: Any, name: str, length: int) -> np.ndarray:
    if name not in arrays:
        raise RunnerError("invalid_input", f"Missing array: {name}")
    value = np.asarray(arrays[name], dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise RunnerError("invalid_input", f"{name} must be a finite vector with length {length}")
    return value


def _prior(
    function: Any,
    history_features: np.ndarray,
    history_utilities: np.ndarray,
    query_features: np.ndarray,
    context: dict[str, Any],
) -> np.ndarray:
    try:
        value = function(
            history_features.copy(),
            history_utilities.copy(),
            query_features.copy(),
            json.loads(json.dumps(context)),
        )
    except Exception as exc:
        raise RunnerError(
            "prior_mean_failed",
            f"compute_prior_mean failed: {type(exc).__name__}: {exc}",
            path="/exports/compute_prior_mean",
        ) from exc
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (len(query_features),):
        raise RunnerError(
            "invalid_prior_shape",
            f"compute_prior_mean returned shape {result.shape}; expected {(len(query_features),)}",
            path="/exports/compute_prior_mean",
        )
    if not np.isfinite(result).all():
        index = int(np.flatnonzero(~np.isfinite(result))[0])
        raise RunnerError(
            "non_finite_output",
            f"compute_prior_mean returned a non-finite value at index {index}",
            path=f"/outputs/prior_mean/{index}",
            hint="Check normalization, zero-variance columns, and numerical overflow.",
        )
    return result


def _weights(function: Any, context: dict[str, Any]) -> dict[str, Any]:
    try:
        value = function(json.loads(json.dumps(context)))
    except Exception as exc:
        raise RunnerError(
            "ldm_weights_failed",
            f"choose_ldm_weights failed: {type(exc).__name__}: {exc}",
            path="/exports/choose_ldm_weights",
        ) from exc
    if not isinstance(value, dict) or set(value) != {"stage", "alpha", "eta"}:
        raise RunnerError(
            "invalid_ldm_weights",
            "choose_ldm_weights must return exactly stage, alpha, and eta",
            path="/outputs/ldm_weights",
        )
    stage = value["stage"]
    if not isinstance(stage, str) or not stage.strip() or len(stage) > MAX_STAGE_LENGTH:
        raise RunnerError(
            "invalid_stage",
            f"stage must be a non-empty string of at most {MAX_STAGE_LENGTH} characters",
            path="/outputs/ldm_weights/stage",
        )
    numbers: dict[str, float] = {}
    for name in ("alpha", "eta"):
        raw = value[name]
        if isinstance(raw, bool):
            raise RunnerError(
                "invalid_ldm_weight",
                f"{name} must be a finite non-negative number",
                path=f"/outputs/ldm_weights/{name}",
            )
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise RunnerError(
                "invalid_ldm_weight",
                f"{name} must be a finite non-negative number",
                path=f"/outputs/ldm_weights/{name}",
            ) from exc
        if not np.isfinite(number) or number < 0:
            raise RunnerError(
                "invalid_ldm_weight",
                f"{name} must be a finite non-negative number",
                path=f"/outputs/ldm_weights/{name}",
            )
        numbers[name] = number
    return {"stage": stage.strip(), **numbers}


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "count": len(values),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def execute_artifact(
    artifact: Path,
    input_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    contract = _load_json(input_directory / "contract.json")
    inspection = inspect_artifact(artifact, contract)
    capabilities = inspection["capabilities"]
    input_data = _load_json(input_directory / "input.json")
    try:
        arrays = np.load(input_directory / "arrays.npz", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RunnerError("invalid_input", f"Could not load arrays.npz: {exc}") from exc
    try:
        history_features = _matrix(arrays, "history_features")
        query_features = _matrix(arrays, "query_features", history_features.shape[1])
        history_utilities = _vector(arrays, "history_utilities", len(history_features))
    finally:
        arrays.close()
    if len(query_features) == 0:
        raise RunnerError("invalid_input", "query_features must contain at least one row")
    execution_context = input_data.get("execution_context")
    if not isinstance(execution_context, dict):
        raise RunnerError("invalid_input", "execution_context must be an object")
    mean_context = execution_context.get("mean_context", {})
    weight_context = execution_context.get("weight_context", {})
    if not isinstance(mean_context, dict) or not isinstance(weight_context, dict):
        raise RunnerError("invalid_input", "mean_context and weight_context must be objects")

    np.random.seed(0)
    module = _load_module(artifact)
    if "prior_mean" in capabilities:
        function = getattr(module, _CAPABILITY_EXPORTS["prior_mean"], None)
        if not callable(function):
            raise RunnerError(
                "missing_export",
                "compute_prior_mean is not callable",
                path="/exports/compute_prior_mean",
            )
        history_prior = _prior(
            function,
            history_features,
            history_utilities,
            history_features,
            mean_context,
        )
        query_prior = _prior(
            function,
            history_features,
            history_utilities,
            query_features,
            mean_context,
        )
        repeated = _prior(
            function,
            history_features,
            history_utilities,
            query_features,
            mean_context,
        )
        if not np.array_equal(query_prior, repeated):
            raise RunnerError(
                "non_deterministic_output",
                "compute_prior_mean changed across identical calls",
                path="/exports/compute_prior_mean",
            )
        permutation = np.arange(len(query_features) - 1, -1, -1)
        permuted = _prior(
            function,
            history_features,
            history_utilities,
            query_features[permutation],
            mean_context,
        )
        if not np.allclose(permuted, query_prior[permutation], rtol=0.0, atol=1e-12):
            raise RunnerError(
                "query_order_dependent",
                "compute_prior_mean is not query-permutation equivariant",
                path="/exports/compute_prior_mean",
            )
        singleton = np.concatenate([
            _prior(function, history_features, history_utilities, row[None, :], mean_context)
            for row in query_features
        ])
        if not np.allclose(singleton, query_prior, rtol=0.0, atol=1e-12):
            raise RunnerError(
                "batch_dependent_output",
                "singleton and batch prior means disagree",
                path="/exports/compute_prior_mean",
            )
        _prior(
            function,
            history_features[:0],
            history_utilities[:0],
            query_features[:1],
            mean_context,
        )
        if len(history_features):
            _prior(
                function,
                history_features[:1],
                history_utilities[:1],
                query_features[:1],
                mean_context,
            )
    else:
        history_prior = np.zeros(len(history_features), dtype=np.float64)
        query_prior = np.zeros(len(query_features), dtype=np.float64)

    if "ldm_weights" in capabilities:
        function = getattr(module, _CAPABILITY_EXPORTS["ldm_weights"], None)
        if not callable(function):
            raise RunnerError(
                "missing_export",
                "choose_ldm_weights is not callable",
                path="/exports/choose_ldm_weights",
            )
        weights = _weights(function, weight_context)
        if weights != _weights(function, weight_context):
            raise RunnerError(
                "non_deterministic_output",
                "choose_ldm_weights changed across identical calls",
                path="/exports/choose_ldm_weights",
            )
    else:
        weights = {
            "stage": "default",
            "alpha": float(contract["default_alpha"]),
            "eta": float(contract["default_eta"]),
        }

    mean_clip = float(contract["mean_clip"])
    if not np.isfinite(mean_clip) or mean_clip <= 0:
        raise RunnerError("invalid_contract", "mean_clip must be finite and positive")
    clip_count = int(
        np.count_nonzero(np.abs(history_prior) > mean_clip)
        + np.count_nonzero(np.abs(query_prior) > mean_clip)
    )
    history_prior = np.clip(history_prior, -mean_clip, mean_clip)
    query_prior = np.clip(query_prior, -mean_clip, mean_clip)

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "arrays.npz",
        history_prior_mean=history_prior,
        query_prior_mean=query_prior,
    )
    result = {
        "status": "ok",
        "stage": weights["stage"],
        "alpha": weights["alpha"],
        "eta": weights["eta"],
        "prior_clip_count": clip_count,
        "prior_summary": {
            "history": _summary(history_prior),
            "query": _summary(query_prior),
        },
        "inspection": inspection,
    }
    (output_directory / "result.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--artifact", type=Path, required=True)
    inspect_parser.add_argument("--contract", type=Path, required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--artifact", type=Path, required=True)
    execute_parser.add_argument("--input", type=Path, required=True)
    execute_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = {
                "status": "ok",
                "inspection": inspect_artifact(
                    args.artifact,
                    _load_json(args.contract),
                ),
            }
        else:
            result = execute_artifact(args.artifact, args.input, args.output)
    except RunnerError as exc:
        result = {"status": "error", "errors": [exc.to_dict()]}
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 2
    except Exception as exc:
        result = {
            "status": "error",
            "errors": [{
                "path": "/artifact_path",
                "code": "runtime_exception",
                "message": f"Policy runner failed: {type(exc).__name__}: {exc}",
                "hint": "Inspect the artifact and input contract, then repair the policy.",
            }],
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
