"""Run in the pinned Qlib environment; these are synthetic market fixtures."""

import os
from pathlib import Path

import numpy as np
import pytest

qlib = pytest.importorskip("qlib")
import pandas as pd
from qlib.data.storage.file_storage import FileFeatureStorage
from ldm_tts.contracts.evaluation import EvaluationPaused

from tasks.alphabench.core.oracle_worker import load_source, qlib_evaluate
from tasks.alphabench.core.grammar import parse_expression
from tasks.alphabench.core.protocol import T3Protocol


@pytest.fixture
def source():
    source = Path(os.environ.get("ALPHABENCH_SOURCE_ROOT", "/mnt/data1/Large-Discovery-Models/data/alphabench/AlphaBench"))
    if not (source / "ffo/utils/utils.py").exists():
        pytest.skip("pinned AlphaBench source checkout is required")
    return source


def write_market(root, days, stocks, members=None):
    for directory in ("calendars", "instruments"):
        (root / directory).mkdir(parents=True)
    for name in ("day", "day_future"):
        (root / f"calendars/{name}.txt").write_text("\n".join(days.strftime("%Y-%m-%d")) + "\n")
    for market in ("csi300", "all"):
        (root / f"instruments/{market}.txt").write_text("".join(
            f"{symbol}\t{days[0]:%Y-%m-%d}\t{days[-1]:%Y-%m-%d}\n" for symbol in (members or stocks)))
    qlib.init(provider_uri=str(root), region="cn", expression_cache=None, dataset_cache=None, kernels=1)
    for symbol, fields in stocks.items():
        (root / "features" / symbol.lower()).mkdir(parents=True)
        for field, array in fields.items():
            FileFeatureStorage(symbol, field, "day", provider_uri={"day": str(root)}).write(array, index=0)


def test_actual_qlib_nan_filter_and_paper_finite_time_boundaries(tmp_path, source):
    days = pd.bdate_range("2016-01-01", periods=101)
    cases = [("close", np.nan, 1, True, True), ("open", np.nan, 2, False, False),
             ("high", np.inf, 1, True, True), ("low", np.inf, 2, True, False),
             ("volume", np.nan, 100, False, False)]
    arrays = {}
    for field, value, count, _, _ in cases:
        array = np.full(100, 20., dtype=np.float32)
        array[:count] = value
        arrays[field] = array
    write_market(tmp_path, days, {"S00": arrays})
    protocol = T3Protocol(filter_profile="qlib_code_filter_v1")
    request = {"protocol": protocol.to_dict(), "operation": "check", "start": str(days[0].date()),
               "end": str(days[-1].date()), "fast": True}
    config = {"upstream_root": str(source), "data_root": str(tmp_path)}
    source_utils = load_source("t3_filter_reference", source / "ffo/utils/utils.py")
    for field, _, _, source_pass, paper_pass in cases:
        raw = qlib_evaluate(request | {"expression": "$" + field}, config)
        expected = source_utils._check_single_column("$" + field, pd.Series(arrays[field]))
        assert all(raw[key] == value for key, value in expected.items())
        assert raw["success"] is source_pass
        assert raw["nan_ratio"] == np.isnan(arrays[field]).mean()
        assert raw["non_finite_ratio"] == (~np.isfinite(arrays[field])).mean()
        assert protocol.check_passed(raw | {"elapsed_seconds": 30}) is source_pass
        assert protocol.check_passed(raw | {"elapsed_seconds": 30}, paper=True) is paper_pass
        assert protocol.check_passed(raw | {"elapsed_seconds": 30.001}, paper=True) is False
        assert raw["metrics"] == {} and raw["daily"] == [] and raw["scores"] == [] and raw["portfolio"] is None


def test_qlib_executes_unary_factor_negation_without_changing_candidate_identity(tmp_path, source):
    days = pd.bdate_range("2020-01-01", periods=24)
    stocks = {}
    for i in range(3):
        close = np.linspace(10 + i, 20 + i, 24).astype(np.float32)
        stocks[f"S{i:02d}"] = {"close": close, "open": close * (.98 + .01 * np.sin(np.arange(24) + i)),
                              "high": close * 1.05, "low": close * .9}
    write_market(tmp_path, days, stocks)
    expression = "(-((((2*$close)-$high)-$low)/$open))"
    assert parse_expression(expression).canonical.startswith("(-")
    assert parse_expression(expression, qlib_execution=True).canonical.startswith("Mul(-1,")
    request = {"protocol": T3Protocol().to_dict(), "operation": "check", "expression": expression,
               "start": str(days[0].date()), "end": str(days[-1].date()), "fast": True}
    config = {"upstream_root": str(source), "data_root": str(tmp_path)}
    actual = qlib_evaluate(request, config)
    equivalent = qlib_evaluate(request | {"expression": "Mul(-1,Div(Sub(Sub(Mul(2,$close),$high),$low),$open))"}, config)
    assert actual["success"] and actual["nan_ratio"] == equivalent["nan_ratio"]
    assert actual["non_finite_ratio"] == equivalent["non_finite_ratio"]


@pytest.mark.parametrize("profile", ["ldm_matched_v1", "upstream_searcher_v1", "upstream_benchmark_v1"])
def test_qlib_endpoint_and_forward_labels_follow_the_selected_profile(tmp_path, source, profile):
    days = pd.bdate_range("2020-01-01", periods=45)
    t, stock = np.arange(45)[:, None], np.arange(12)[None, :]
    close = (20 + stock + .03 * t + np.sin(t * .7 + stock)).astype(np.float32)
    write_market(tmp_path, days, {f"S{i:02d}": {"close": close[:, i]} for i in range(12)})
    protocol = T3Protocol(profile=profile, forward_n=3)
    request = {"protocol": protocol.to_dict(), "operation": "evaluate", "expression": "Mean($close,3)",
               "start": str(days[20].date()), "end": str(days[30].date()), "fast": True}
    config = {"upstream_root": str(source), "data_root": str(tmp_path)}
    raw = qlib_evaluate(request, config)
    last = 30 if protocol.end_inclusive else 26
    assert [row["date"] for row in raw["daily"]] == list(days[20:last + 1].strftime("%Y-%m-%d"))
    assert raw["label"]["read_end"] == str(days[last + 3].date())
    assert raw["label"]["purged_dates"] == ([] if protocol.end_inclusive else list(days[27:30].strftime("%Y-%m-%d")))
    assert raw["interval"]["end_inclusive"] is protocol.end_inclusive
    for index, row in enumerate(raw["daily"], start=20):
        factor = close[index - 2:index + 1].astype(float).mean(axis=0)
        expected = [np.corrcoef(factor, close[index + h] / close[index + h - 1] - 1)[0, 1] for h in (1, 2, 3)]
        assert row["ic"] == pytest.approx(np.mean(expected), abs=1e-6)
        assert row["n_samples_by_horizon"] == {"1": 12, "2": 12, "3": 12}
    check = qlib_evaluate(request | {"operation": "check"}, config)
    assert check["actual_end"] == str(days[30 if protocol.end_inclusive else 29].date())
    assert "label" not in check and check["daily"] == []
    if protocol.end_inclusive:
        with pytest.raises(EvaluationPaused, match="future sessions"):
            qlib_evaluate(request | {"end": str(days[-1].date())}, config)
    with pytest.raises(EvaluationPaused, match="complete requested interval"):
        qlib_evaluate(request | {"end": "2025-01-01"}, config)


@pytest.mark.parametrize("profile,last", [("ldm_matched_v1", 27), ("upstream_searcher_v1", 30)])
def test_full_qlib_portfolio_uses_the_same_retained_interval(tmp_path, source, profile, last):
    days = pd.bdate_range("2020-01-01", periods=45)
    t, stock = np.arange(45)[:, None], np.arange(6)[None, :]
    close = (20 + stock + .03 * t + np.sin(t * .7 + stock)).astype(np.float32)
    stocks = {f"S{i:02d}": {"close": close[:, i], "open": close[:, i] * .99,
        "high": close[:, i] * 1.02, "low": close[:, i] * .98, "volume": np.full(45, 1e8),
        "factor": np.ones(45), "change": np.r_[0, close[1:, i] / close[:-1, i] - 1]} for i in range(6)}
    members = list(stocks)
    stocks["SH000300"] = {"close": (100 + np.arange(45)).astype(np.float32)}
    write_market(tmp_path, days, stocks, members)
    protocol = T3Protocol(profile=profile, forward_n=2, stock_topk=3, stock_n_drop=1)
    request = {"protocol": protocol.to_dict(), "operation": "evaluate", "expression": "Mean($close,3)",
               "start": str(days[20].date()), "end": str(days[30].date()), "fast": False}
    result = qlib_evaluate(request, {"upstream_root": str(source), "data_root": str(tmp_path), "benchmark": "SH000300"})
    expected_dates = list(days[20:last + 1].strftime("%Y-%m-%d"))
    assert [row["date"] for row in result["daily"]] == expected_dates
    portfolio = result["portfolio"]
    assert [row["date"] for row in portfolio["daily"]] == expected_dates
    assert list(portfolio["holdings"]) == expected_dates and portfolio["actions"]
    assert all(row["amount_delta"] / 100 == pytest.approx(round(row["amount_delta"] / 100)) for row in portfolio["actions"])
    assert any(row["cost"] > 0 for row in portfolio["daily"])
    assert set(portfolio["analysis"]) == {"benchmark", "pure_return_without_cost", "pure_return_with_cost",
        "excess_return_without_cost", "excess_return_with_cost"}
