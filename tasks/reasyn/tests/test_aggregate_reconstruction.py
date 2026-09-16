"""Synthetic receipt fixtures test full-grid guards, never neural-model scores."""
from copy import deepcopy
import hashlib
import json
import statistics

import pytest

from tasks.reasyn.scripts import aggregate_reconstruction_results as report


@pytest.fixture
def suite(monkeypatch):
    # This fixture tests grid shape only; these identifiers are not molecules.
    monkeypatch.setattr(report, "canonicalize", lambda value, **kwargs: value)
    return {
        "benchmark": "reconstruction", "seeds": [0, 1, 2],
        "targets": {d: [f"fixture-{d}-{i}" for i in range(1000)] for d in report.DATASETS},
        "assets": {d: {"fixture": "synthetic-only"} for d in report.DATASETS},
        "methods": [{"id": "original", "label": "Synthetic test only", "search_method": "baseline",
                     "proposal_mode": "baseline", "model": None, "trials_per_target": 1}],
        "cases": [{"method": "original", "dataset": d, "seed": s, "path": f"{d}-{s}"}
                  for d in report.DATASETS for s in report.SEEDS],
    }


def test_requires_full_three_dataset_three_seed_grid(suite):
    assert set(report.validate_suite(suite)) == {"original"}
    for mutated in [dict(suite, seeds=[0]), dict(suite, cases=suite["cases"][:-1]),
                    dict(suite, cases=suite["cases"] + [suite["cases"][0]])]:
        with pytest.raises(ValueError):
            report.validate_suite(mutated)


def test_explicit_single_seed_keeps_full_dataset_grid(suite):
    suite.update(replication_protocol="single_seed", seeds=[42])
    suite["cases"] = [{"method": "original", "dataset": d, "seed": 42, "path": f"{d}-42"}
                      for d in report.DATASETS]
    assert set(report.validate_suite(suite)) == {"original"}
    mixed = deepcopy(suite)
    mixed["cases"][0]["seed"] = 0
    with pytest.raises(ValueError, match="cases"):
        report.validate_suite(mixed)
    for seeds in ([], [42, 42], [True], [-1]):
        with pytest.raises(ValueError):
            report.validate_suite(dict(suite, seeds=seeds))


def test_single_seed_markdown_omits_uncertainty():
    row = {"label": "Fixture only", "datasets": {d: {m: {"value": 0.5}
           for m in report.METRICS} for d in report.DATASETS}}
    rendered = report.markdown({"rows": [row], "seeds": [42]})
    assert "seed=42 only" in rendered
    assert "±" not in rendered
    assert rendered.count("50.0") == 3
    assert rendered.count("0.500") == 9


def test_single_seed_full_grid_visits_3000_targets(suite, monkeypatch, tmp_path):
    suite.update(replication_protocol="single_seed", seeds=[42])
    suite["cases"] = [{"method": "original", "dataset": d, "seed": 42, "path": f"{d}-42"}
                      for d in report.DATASETS]
    visited = []
    def read(path):
        if path.name == "suite.json":
            return suite
        dataset = path.parent.name.rsplit("-", 1)[0]
        if path.name == "targets_manifest.json":
            return {"mock": False, "targets": suite["targets"][dataset]}
        return {"mock": False, "benchmark": "reconstruction", "target_count": 1000,
                "requested_campaigns": 1000, "completed_campaigns": 1000,
                **{m: 0.5 for m in report.METRICS}}
    def verify(folder, **kwargs):
        assert kwargs["seed"] == 42
        visited.append((kwargs["dataset"], folder.name))
        return {"metrics": {m: 0.5 for m in report.METRICS}, "budget": {"projection_targets": 1},
                "policy_actions": {}, "result_sha256": "fixture"}
    monkeypatch.setattr(report, "read", read)
    monkeypatch.setattr(report, "verify_target", verify)
    monkeypatch.setattr(report, "digest", lambda path: "fixture")
    monkeypatch.setattr(report, "target_directories", lambda folder: {f"target-{i:05d}" for i in range(1000)})
    result = report.aggregate(tmp_path / "suite.json")
    assert len(visited) == len(set(visited)) == 3000
    assert result["seeds"] == [42] and "std_ddof" not in result
    for d in report.DATASETS:
        for m in report.METRICS:
            assert result["rows"][0]["datasets"][d][m] == {"value": 0.5}


def test_rejects_tiny_or_duplicate_targets_and_reused_runs(suite):
    tiny = deepcopy(suite)
    tiny["targets"]["enamine"] = tiny["targets"]["enamine"][:1]
    with pytest.raises(ValueError, match="1000"):
        report.validate_suite(tiny)
    duplicate = deepcopy(suite)
    duplicate["targets"]["enamine"][-1] = duplicate["targets"]["enamine"][0]
    with pytest.raises(ValueError, match="1000"):
        report.validate_suite(duplicate)
    reused = deepcopy(suite)
    reused["cases"][1]["path"] = reused["cases"][0]["path"]
    with pytest.raises(ValueError, match="directory"):
        report.validate_suite(reused)


def test_search_settings_must_be_frozen(suite):
    method = suite["methods"][0]
    method.update(search_method="ldm_harness_compiled", proposal_mode="harness", trials_per_target=4)
    with pytest.raises(ValueError, match="hyperparameters"):
        report.validate_suite(suite)


@pytest.fixture
def target_receipts(tmp_path, monkeypatch):
    # Exercise validation without loading RDKit/TDC or model weights.
    metrics = {metric: 0.0 for metric in report.METRICS}
    monkeypatch.setattr(report, "reconstruction_metrics", lambda targets, rows: dict(metrics))
    assets = {"test-only-asset": "synthetic"}
    settings = {"search_width": 8, "exhaustiveness": 4, "num_editflow_samples": 100,
                "num_cycles": 12, "max_results": 100}
    config = {**settings, "mock": False, "dataset": "enamine", "seed": 0,
              "search_method": "baseline", "proposal_mode": "baseline", "iterations": 1,
              "evaluations_per_round": 1}
    request = {"asset_digests": assets, "targets": ["CCO"],
               "settings": {**settings, "exact_break": True}, "sampling_seed": 0}
    files = {
        "config.json": config,
        "scientific_identity.json": {**settings, "original_target": "CCO", "asset_digests": assets},
        "status.json": {"status": "completed"},
        "result.json": {"mock": False, "benchmark": "reconstruction", "successful_evaluations": 1,
                        "reconstruction_rows": [], "metrics": metrics},
        "budget.json": {"counters": {"successful_evaluations": 1, "projection_targets": 1},
                        "limits": {"projection_targets": 3}},
        "projections/trial/request.json": request,
        "projections/trial/result.json": {"complete": True, "parameter_count": 341397459, "rows": [],
            "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()},
    }
    def save(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    for name, data in files.items():
        save(name, data)
    options = {"target": "CCO", "dataset": "enamine", "seed": 0,
               "method": {"search_method": "baseline", "proposal_mode": "baseline", "trials_per_target": 1},
               "assets": assets}
    return tmp_path, files, save, options


def test_completed_empty_projection_counts_as_measured_zero(target_receipts):
    folder, _, _, options = target_receipts
    result = report.verify_target(folder, **options)
    assert result["metrics"] == {metric: 0 for metric in report.METRICS}
    assert result["budget"]["projection_targets"] == 1


@pytest.mark.parametrize("change", ["mock", "paused", "wrong_seed", "tiny_cycles", "tiny_eb", "tampered_request", "incomplete_projection", "invented_pathway", "changed_assets", "changed_metric"])
def test_target_validation_rejects_invalid_receipts(target_receipts, change):
    folder, files, save, options = target_receipts
    if change == "mock":
        files["result.json"]["mock"] = True
    elif change == "paused":
        files["status.json"]["status"] = "paused_projection"
    elif change == "wrong_seed":
        files["config.json"]["seed"] = 1
    elif change == "tiny_cycles":
        files["config.json"]["num_cycles"] = 1
    elif change == "tiny_eb":
        files["projections/trial/request.json"]["settings"]["num_editflow_samples"] = 4
    elif change == "tampered_request":
        files["projections/trial/request.json"]["sampling_seed"] = 99
    elif change == "incomplete_projection":
        files["projections/trial/result.json"]["complete"] = False
    elif change == "invented_pathway":
        files["result.json"]["reconstruction_rows"] = [{"target": "CCO", "smiles": "CCO"}]
    elif change == "changed_assets":
        files["scientific_identity.json"]["asset_digests"] = {"different": "asset"}
    else:
        files["result.json"]["metrics"] = {metric: 1.0 for metric in report.METRICS}
    for name, data in files.items():
        save(name, data)
    with pytest.raises(ValueError):
        report.verify_target(folder, **options)


def test_markdown_contains_all_twelve_mean_std_cells():
    row = {"label": "Fixture only", "datasets": {d: {m: {"mean": 0.5, "std": 0.1}
           for m in report.METRICS} for d in report.DATASETS}}
    rendered = report.markdown({"rows": [row]})
    assert rendered.count("50.0 ± 10.0") == 3
    assert rendered.count("0.500 ± 0.100") == 9


def test_complete_grid_checks_every_target_and_aggregates_seed_means(suite, monkeypatch, tmp_path):
    # In-memory receipts exercise the entire 9000-target traversal and averaging;
    # target-level physical receipt checks have independent fixtures above.
    values = [0.2, 0.4, 0.6]
    visited = []
    def read(path):
        if path.name == "suite.json":
            return suite
        dataset, seed = path.parent.name.rsplit("-", 1)
        if path.name == "targets_manifest.json":
            return {"mock": False, "targets": suite["targets"][dataset]}
        assert path.name == "benchmark_result.json"
        return {"mock": False, "benchmark": "reconstruction", "target_count": 1000,
                "requested_campaigns": 1000, "completed_campaigns": 1000,
                **{metric: values[int(seed)] for metric in report.METRICS}}
    def verify(folder, **kwargs):
        visited.append((kwargs["dataset"], kwargs["seed"], folder.name))
        return {"metrics": {metric: values[kwargs["seed"]] for metric in report.METRICS},
                "budget": {"projection_targets": 1}, "policy_actions": {}, "result_sha256": "fixture"}
    monkeypatch.setattr(report, "read", read)
    monkeypatch.setattr(report, "verify_target", verify)
    monkeypatch.setattr(report, "digest", lambda path: "fixture-only")
    monkeypatch.setattr(report, "target_directories", lambda folder: {f"target-{i:05d}" for i in range(1000)})
    result = report.aggregate(tmp_path / "suite.json")
    assert len(visited) == len(set(visited)) == 9000
    assert len(result["case_receipts"]) == 9
    assert all(case["budget"]["projection_targets"] == 1000 for case in result["case_receipts"])
    for dataset in report.DATASETS:
        for metric in report.METRICS:
            item = result["rows"][0]["datasets"][dataset][metric]
            assert item["mean"] == pytest.approx(0.4)
            assert item["std"] == pytest.approx(statistics.pstdev(values))
    monkeypatch.setattr(report, "target_directories", lambda folder: {"target-00000"})
    with pytest.raises(ValueError, match="directories"):
        report.aggregate(tmp_path / "suite.json")
