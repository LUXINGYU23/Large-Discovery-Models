from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ldm_tts.harness.protocol import file_sha256


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


@pytest.mark.parametrize("task", ["iron_mind", "synthonbench", "nucleobench"])
def test_policy_runner_executes_valid_numpy_artifact(tmp_path: Path, task) -> None:
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
    diagnostics_path = RUNNER.parents[2] / "tasks" / task / "resources/harness/policy_diagnostics.py"

    completed, result = _run(
        "execute",
        "--artifact",
        str(artifact),
        "--input",
        str(round_input),
        "--output",
        str(output),
        "--diagnostics", str(diagnostics_path),
        "--diagnostics-sha256", file_sha256(diagnostics_path),
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


@pytest.mark.parametrize("task", ["iron_mind", "synthonbench", "nucleobench"])
@pytest.mark.parametrize("capability,example_index", [("prior_mean@1", 0), ("ldm_weights@1", 1)])
def test_task_skill_single_capability_artifacts_execute(tmp_path, task, capability, example_index):
    resources = RUNNER.parents[2] / "tasks" / task / "resources/harness"
    skill = (resources / "skills/compile-ldm-policy/SKILL.md").read_text()
    artifact = tmp_path / "optimization_policy.py"
    artifact.write_text(re.findall(r"```python\n(.*?)```", skill, re.DOTALL)[example_index])
    round_input = tmp_path / "round"
    _round_input(round_input)
    contract_path = round_input / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["enabled_capabilities"] = [capability]
    contract_path.write_text(json.dumps(contract))
    input_path = round_input / "input.json"
    inputs = json.loads(input_path.read_text())
    inputs["execution_context"]["weight_context"].update(default_alpha=1.0, default_eta=1.0)
    input_path.write_text(json.dumps(inputs))
    output = tmp_path / "output"
    hook = resources / "policy_diagnostics.py"
    completed, result = _run(
        "execute", "--artifact", str(artifact), "--input", str(round_input), "--output", str(output),
        "--diagnostics", str(hook), "--diagnostics-sha256", file_sha256(hook),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert result["alpha"] == result["eta"] == 1.0
    assert result["draft_diagnostics"]["held_out_count"] == 1
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["history_prior_mean"], np.zeros(2))
        np.testing.assert_array_equal(arrays["query_prior_mean"], np.zeros(2))


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


@pytest.mark.parametrize("scalarize", [False, True])
def test_runner_preserves_multiple_objectives_and_loads_dataclasses(tmp_path: Path, scalarize) -> None:
    round_input = tmp_path / "round"
    output = tmp_path / "output"
    _round_input(round_input)
    np.savez_compressed(
        round_input / "arrays.npz",
        history_features=np.eye(2),
        history_utilities=np.asarray([[3.0, -1.0], [1.0, 4.0]]),
        query_features=np.asarray([[1.0, 1.0], [2.0, -1.0]]),
    )
    artifact = tmp_path / "optimization_policy.py"
    artifact.write_text(
        """\
from dataclasses import dataclass
import numpy as np
POLICY_API_VERSION = 1
CAPABILITIES = {"prior_mean": 1, "ldm_weights": 1}
@dataclass
class Stage:
    name: str = "multiobjective"
def compute_prior_mean(history_features, history_utilities, query_features, context):
    result = np.tile(np.asarray([1.5, -2.0]), (len(query_features), 1))
    return result[:, 0] if SCALARIZE else result
def choose_ldm_weights(context):
    return {"stage": Stage().name, "alpha": 1.0, "eta": 0.25}
""".replace("SCALARIZE", repr(scalarize)), encoding="utf-8",
    )
    completed, result = _run(
        "execute", "--artifact", str(artifact), "--input", str(round_input), "--output", str(output),
    )
    if scalarize:
        assert completed.returncode == 2
        assert result["errors"][0]["code"] == "invalid_prior_shape"
    else:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "draft_diagnostics" not in result
        assert [item["mean"] for item in result["prior_summary"]["query"]["objectives"]] == [1.5, -2.0]
        with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
            assert arrays["history_prior_mean"].tolist() == [[1.5, -2.0]] * 2
            assert arrays["query_prior_mean"].tolist() == [[1.5, -2.0]] * 2


def test_runner_rejects_changed_task_diagnostics_before_loading(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics.py"
    diagnostics.write_text("raise AssertionError('must not execute')", encoding="utf-8")
    completed, result = _run(
        "execute", "--artifact", str(tmp_path / "unused.py"),
        "--input", str(tmp_path / "unused"), "--output", str(tmp_path / "output"),
        "--diagnostics", str(diagnostics), "--diagnostics-sha256", "0" * 64,
    )
    assert completed.returncode == 2
    assert result["errors"][0]["code"] == "diagnostics_digest_mismatch"
