from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).parents[1] / "harnesses" / "pi" / "policy_runner.py"


def _round_input(path: Path) -> None:
    path.mkdir()
    (path / "contract.json").write_text(
        json.dumps({
            "task_id": "fixture",
            "api_version": 1,
            "enabled_capabilities": ["prior_mean@1", "ldm_weights@1"],
            "feature_names": ["a", "b"],
            "feature_groups": {"all": [0, 2]},
            "mean_clip": 5.0,
            "default_alpha": 1.0,
            "default_eta": 1.0,
        }),
        encoding="utf-8",
    )
    (path / "input.json").write_text(
        json.dumps({
            "execution_context": {
                "mean_context": {
                    "round_index": 2,
                    "target_location": 2.0,
                    "target_scale": 1.0,
                },
                "weight_context": {"round_index": 2},
                "validation_folds": [{
                    "prefix": "diagnostic_fold_0_", "round_index": 1,
                    "train_indices": [0], "test_indices": [1],
                }],
            }
        }),
        encoding="utf-8",
    )
    np.savez_compressed(
        path / "arrays.npz",
        history_features=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        history_utilities=np.asarray([3.0, 1.0]),
        query_features=np.asarray([[1.0, 1.0], [2.0, -1.0]]),
        diagnostic_fold_0_weights=np.asarray([[0.0]]),
        diagnostic_fold_0_location_scale=np.asarray([3.0, 1.0]),
    )


def _run(*args: str) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed, json.loads(completed.stdout)


def test_policy_runner_executes_valid_numpy_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "optimization_policy.py"
    artifact.write_text(
        """\
import numpy as np

POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}

def compute_prior_mean(history_features, history_utilities, query_features, context):
    if len(history_features) == 0:
        return np.zeros(len(query_features))
    standardized = (
        history_utilities - context["target_location"]
    ) / context["target_scale"]
    coefficients = np.linalg.pinv(history_features) @ standardized
    return query_features @ coefficients

def choose_ldm_weights(context):
    return {"stage": "focused", "alpha": 0.8, "eta": 1.4}
""",
        encoding="utf-8",
    )
    round_input = tmp_path / "round"
    output = tmp_path / "output"
    _round_input(round_input)

    completed, result = _run(
        "execute",
        "--artifact",
        str(artifact),
        "--input",
        str(round_input),
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert result["stage"] == "focused"
    assert result["alpha"] == 0.8
    diagnostics = result["draft_diagnostics"]
    assert diagnostics["scope"] == "chronological_measured_history_fixed_gp"
    assert diagnostics["held_out_count"] == 1
    # The perfect full-history fit has no predictive gain on the unseen feature.
    assert np.isclose(diagnostics["baseline_gp_rmse"], 2.0)
    assert np.isclose(diagnostics["draft_gp_rmse"], 2.0)
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        assert arrays["history_prior_mean"].shape == (2,)
        assert arrays["query_prior_mean"].shape == (2,)


def test_policy_runner_rejects_forbidden_imports(tmp_path: Path) -> None:
    artifact = tmp_path / "optimization_policy.py"
    artifact.write_text(
        """\
import os
POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}
def compute_prior_mean(history_features, history_utilities, query_features, context):
    return [0.0] * len(query_features)
def choose_ldm_weights(context):
    return {"stage": "default", "alpha": 1.0, "eta": 1.0}
""",
        encoding="utf-8",
    )
    round_input = tmp_path / "round"
    _round_input(round_input)

    completed, result = _run(
        "inspect",
        "--artifact",
        str(artifact),
        "--contract",
        str(round_input / "contract.json"),
    )

    assert completed.returncode == 2
    assert result["errors"][0]["code"] == "forbidden_import"


def test_policy_runner_rejects_batch_dependent_prior(tmp_path: Path) -> None:
    artifact = tmp_path / "optimization_policy.py"
    artifact.write_text(
        """\
import numpy as np
POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}
def compute_prior_mean(history_features, history_utilities, query_features, context):
    return np.full(len(query_features), np.mean(query_features))
def choose_ldm_weights(context):
    return {"stage": "default", "alpha": 1.0, "eta": 1.0}
""",
        encoding="utf-8",
    )
    round_input = tmp_path / "round"
    output = tmp_path / "output"
    _round_input(round_input)

    completed, result = _run(
        "execute",
        "--artifact",
        str(artifact),
        "--input",
        str(round_input),
        "--output",
        str(output),
    )

    assert completed.returncode == 2
    assert result["errors"][0]["code"] == "batch_dependent_output"
