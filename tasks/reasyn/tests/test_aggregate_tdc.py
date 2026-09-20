import json
import statistics

import pytest

from tasks.reasyn.core.metrics import top_auc, top_mean
from tasks.reasyn.scripts import aggregate_tdc_results as report


def write(path, payload):
    path.write_text(json.dumps(payload) + "\n")


def tdc_identity(**overrides):
    identity = {
        "benchmark": "tdc",
        "mock": False,
        "proposal_mode": "openai",
        "search_method": "ldm",
        "max_oracle_calls": report.MAX_ORACLE_CALLS,
        "reservoir_size": 32,
        "evaluations_per_round": 16,
        "proposal_batch_size": 2,
        "bo_pool_size": 32,
        "max_replenishment_batches": 4,
        "num_cycles": 1,
        "search_width": 2,
        "exhaustiveness": 4,
        "num_editflow_samples": 4,
        "max_results": 100,
        "llm_model": "fixture-model",
        "llm_max_tokens": 4096,
        "gp_history_limit": 256,
        "acquisition_beta": 1.0,
        "acquisition_alpha": 1.0,
        "acquisition_eta": 1.0,
        "acquisition_z_clip": 5.0,
        "initialization_mode": "none",
        "original_target": "",
        "source_archive_digest": "source-fixture",
        "asset_digests": {"frozen": "0" * 64},
    }
    identity.update(overrides)
    return identity


def make_run(tmp_path, *, oracle="jnk3", seed=0, shared_identity=None):
    folder = tmp_path / f"{oracle}-{seed}"
    folder.mkdir()
    shared_identity = dict(shared_identity or tdc_identity())
    scores = [((index * 17 + seed) % 1000) / 1000 for index in range(report.MAX_ORACLE_CALLS)]
    auc = top_auc(scores, max_calls=report.MAX_ORACLE_CALLS)
    entries = [
        {
            "smiles": f"candidate-{index:05d}",
            "call_index": index + 1,
            "status": "completed",
            "score": score,
        }
        for index, score in enumerate(scores)
    ]
    ledger = {
        "limits": {
            "successful_evaluations": report.MAX_ORACLE_CALLS,
            "oracle_calls": report.MAX_ORACLE_CALLS,
            "expensive_evaluation_attempts": report.MAX_ORACLE_CALLS,
        },
        "counters": {
            "successful_evaluations": report.MAX_ORACLE_CALLS,
            "oracle_calls": report.MAX_ORACLE_CALLS,
            "expensive_evaluation_attempts": report.MAX_ORACLE_CALLS,
        },
        "remaining": {
            "successful_evaluations": 0,
            "oracle_calls": 0,
            "expensive_evaluation_attempts": 0,
        },
        "metadata": {"task": "reasyn", "run_id": folder.name},
    }
    config = {
        key: value
        for key, value in shared_identity.items()
        if key not in {"asset_digests", "source_archive_digest", "original_target"}
    }
    config.update({
        "oracle": oracle,
        "seed": seed,
    })
    write(folder / "config.json", config)
    write(folder / "scientific_identity.json", {
        **shared_identity,
        "oracle": oracle,
        "seed": seed,
        "original_target": "",
    })
    write(folder / "status.json", {"status": "completed"})
    write(folder / "budget.json", ledger)
    write(folder / "oracle_cache.json", {"oracle": oracle, "mock": False, "entries": entries})
    write(folder / "result.json", {
        "task": "reasyn",
        "benchmark": "tdc",
        "mock": False,
        "successful_evaluations": report.MAX_ORACLE_CALLS,
        "oracle": oracle,
        "oracle_calls": report.MAX_ORACLE_CALLS,
        "completed_oracle_calls": report.MAX_ORACLE_CALLS,
        "oracle_budget_exhausted": True,
        "official_budget": True,
        "metrics": {
            "top10": top_mean(scores),
            "auc_top10_observed": auc,
            "auc_top10": auc,
        },
        "budget": ledger,
    })
    return folder / "result.json", auc


def load(path):
    return json.loads(path.read_text())


def test_complete_tdc_grid_is_recomputed_from_oracle_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ORACLES", ("jnk3",))
    runs = [make_run(tmp_path, seed=seed) for seed in report.SEEDS]

    result = report.aggregate([path for path, _auc in runs], expected_identity=tdc_identity())

    expected = [auc for _path, auc in runs]
    assert result["oracles"]["jnk3"]["mean"] == pytest.approx(statistics.mean(expected))
    assert result["mean_auc_top10"] == pytest.approx(statistics.mean(expected))
    assert result["configuration_identity"]["llm_model"] == "fixture-model"
    assert result["asset_digests_sha256"] == report.json_digest({"frozen": "0" * 64})


def test_tdc_aggregation_requires_predeclared_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ORACLES", ("jnk3",))
    runs = [make_run(tmp_path, seed=seed) for seed in report.SEEDS]

    with pytest.raises(ValueError, match="predeclared configuration identity"):
        report.aggregate([path for path, _auc in runs])


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("llm_model", "other-model", "scientific configuration identity mismatch"),
        ("search_method", "harness", "scientific configuration identity mismatch"),
        ("asset_digests", {"frozen": "1" * 64}, "frozen asset identity differs"),
    ],
)
def test_tdc_aggregation_rejects_mixed_scientific_identities(
    tmp_path, monkeypatch, field, value, match
):
    monkeypatch.setattr(report, "ORACLES", ("jnk3",))
    runs = [make_run(tmp_path, seed=seed) for seed in report.SEEDS]
    identity_path = runs[1][0].parent / "scientific_identity.json"
    identity = load(identity_path)
    identity[field] = value
    write(identity_path, identity)
    config_path = runs[1][0].parent / "config.json"
    config = load(config_path)
    if field in config:
        config[field] = value
        write(config_path, config)

    with pytest.raises(ValueError, match=match):
        report.aggregate([path for path, _auc in runs], expected_identity=tdc_identity())


def test_tdc_aggregation_accepts_predeclared_oracle_specific_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ORACLES", ("drd2", "jnk3"))
    shared = tdc_identity()
    shared.pop("search_width")
    expected = {
        "configuration_identity": shared,
        "oracle_overrides": {"drd2": {"search_width": 1}, "jnk3": {"search_width": 2}},
    }
    runs = []
    for oracle, width in (("drd2", 1), ("jnk3", 2)):
        for seed in report.SEEDS:
            runs.append(
                make_run(
                    tmp_path,
                    oracle=oracle,
                    seed=seed,
                    shared_identity=tdc_identity(search_width=width),
                )
            )

    result = report.aggregate([path for path, _auc in runs], expected_identity=expected)

    assert set(result["oracles"]) == {"drd2", "jnk3"}
    assert result["oracle_overrides"] == expected["oracle_overrides"]


@pytest.mark.parametrize(
    "filename, mutate, match",
    [
        ("status.json", lambda payload: payload.update(status="paused_endpoint_unavailable"), "unfinished"),
        ("scientific_identity.json", lambda payload: payload.update(oracle="drd2"), "oracle identity mismatch"),
        ("oracle_cache.json", lambda payload: payload["entries"].pop(), "exactly 10000 entries"),
    ],
)
def test_tdc_result_verifier_rejects_missing_or_mismatched_receipts(
    tmp_path, filename, mutate, match
):
    path, _auc = make_run(tmp_path)
    target = path.parent / filename
    payload = load(target)
    mutate(payload)
    write(target, payload)

    with pytest.raises(ValueError, match=match):
        report.verify_result(path)


def test_tdc_result_verifier_rejects_edited_budget_ledger(tmp_path):
    path, _auc = make_run(tmp_path)
    ledger = load(path.parent / "budget.json")
    ledger["counters"]["oracle_calls"] = report.MAX_ORACLE_CALLS - 1
    write(path.parent / "budget.json", ledger)
    result = load(path)
    result["budget"] = ledger
    write(path, result)

    with pytest.raises(ValueError, match="oracle_calls budget ledger"):
        report.verify_result(path)


def test_tdc_result_verifier_rejects_edited_auc_summary(tmp_path):
    path, _auc = make_run(tmp_path)
    result = load(path)
    result["metrics"]["auc_top10"] += 0.01
    write(path, result)

    with pytest.raises(ValueError, match="auc_top10: recomputed metric differs"):
        report.verify_result(path)
