"""Run with the pinned Assay environment; all data here are explicit fixtures."""

import datetime as dt
import json

import numpy as np
import pytest

pytest.importorskip("assay")
import polars as pl
from assay.portfolio.config import PortfolioBacktestConfig

from tasks.alphabench.core.assay_adapter import assay_evaluate, prepare_panel
from tasks.alphabench.core.data import sha256
from tasks.alphabench.core.oracle_service import OracleService
from tasks.alphabench.core.oracle_worker import clean
from tasks.alphabench.core.protocol import T3Protocol, digest
from tasks.alphabench.core.grammar import REGISTRY


@pytest.fixture
def snapshot(tmp_path):
    days = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    symbols = [f"S{i:02d}" for i in range(12)]
    prices, membership, execution, groups = [], [], [], []
    for i, symbol in enumerate(symbols):
        groups.append({"symbol": symbol, "effective_date": days[0], "as_of_date": days[0], "sector": "a" if i < 6 else "b"})
        groups.append({"symbol": symbol, "effective_date": days[23], "as_of_date": days[23], "sector": "a" if i % 2 else "b"})
        for t, day in enumerate(days):
            close = 15 + i + .07 * t + np.sin(t * .8 + i)
            prices.append({"date": day, "symbol": symbol, "open": close * .99, "close": close,
                           "high": close * 1.02, "low": close * .98, "volume": 1000. + t + i,
                           "vwap": close * 1.001, "as_of_date": day})
            if i != 11 or t >= 22:
                membership.append({"date": day, "symbol": symbol, "as_of_date": day})
            execution.append({"date": day, "symbol": symbol, "tradable": True, "as_of_date": day})
    frames = {"prices": pl.DataFrame(prices), "membership": pl.DataFrame(membership), "execution": pl.DataFrame(execution),
              "groups": pl.DataFrame(groups), "benchmark": pl.DataFrame({"date": days, "close": [100 + i for i in range(40)]}),
              "events": pl.DataFrame(schema={"event_id": pl.String, "symbol": pl.String, "ex_date": pl.Date,
                                             "as_of_date": pl.Date, "split_ratio": pl.Float64, "dividend_cash": pl.Float64})}
    assets = {}
    for name, frame in frames.items():
        path = tmp_path / (name + ".parquet")
        frame.write_parquet(path)
        assets[name] = {"path": path.name, "sha256": sha256(path)}
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("\n".join(map(str, days + [days[-1] + dt.timedelta(days=1)])))
    assets["calendar"] = {"path": calendar.name, "sha256": sha256(calendar)}
    manifest = {"backend": "assay", "market": "nasdaq100", "qualification": "fixture",
                "start": str(days[0]), "end": str(days[-1]), "adjustment": "split", "benchmark": "TEST_INDEX", "assets": assets}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    portfolio = PortfolioBacktestConfig.preset("US", period_start=str(days[20]), period_end=str(days[-1]),
        universe="NASDAQ100", benchmark="custom", benchmark_symbol="TEST_INDEX", rebalance_type="daily", min_rebalance_interval=1,
        execution_price="next_open", signal_transform="rank", weight_method="signal_prop", max_single_weight=.3,
        slippage_model="zero", save_position_log=True).to_dict()
    protocol = T3Protocol(backend="assay", market="nasdaq100", data_digest=digest(manifest), environment_digest="fixture",
                          assay_portfolio=portfolio)
    request = {"request_id": "a"*64, "protocol": protocol.to_dict(), "operation": "evaluate", "expression": "ts_mean(close,5)",
               "start": str(days[20]), "end": str(days[-1] + dt.timedelta(days=1)), "fast": True, "job_permits": 2}
    return request, {"data_manifest": str(path)}, frames, days


def replace_asset(snapshot, name, frame):
    request, config, _, _ = snapshot
    from pathlib import Path
    path = Path(config["data_manifest"])
    manifest = json.loads(path.read_text())
    asset = manifest["assets"][name]
    frame.write_parquet(path.parent / asset["path"])
    asset["sha256"] = sha256(path.parent / asset["path"])
    path.write_text(json.dumps(manifest))
    request["protocol"]["data_digest"] = digest(manifest)


def test_historical_cross_sections_preserve_pre_entry_warmup_and_daily_groups(snapshot):
    request, config, _, days = snapshot
    engine, _, _, _, _, _ = prepare_panel(request, T3Protocol(**request["protocol"]), config)
    mean = engine.evaluate("ts_mean(close,5)").values
    close = engine.field_matrix("close")
    assert mean[22, 11] == pytest.approx(close[18:23, 11].mean())
    rank = engine.evaluate("cs_rank(ts_mean(close,5))").values
    assert np.isnan(rank[21, 11]) and np.isfinite(rank[22, 11])
    assert rank[21, :11].min() == 0 and rank[21, :11].max() == 1
    neutral = engine.evaluate("cs_neutralize(close,'sector')").values
    for date in (22, 23):
        labels = engine.daily_groups["sector"][date]
        for group in ("a", "b"):
            indices = labels == group
            np.testing.assert_allclose(neutral[date, indices], close[date, indices] - close[date, indices].mean())
    a = assay_evaluate(request, config)
    b = assay_evaluate(dict(request, expression="Mean($close,5)"), config)
    assert clean(a) == clean(b)
    assert a["label"]["purged_dates"] == [str(days[-1])]
    assert a["metrics"]["ic"] == pytest.approx(np.nanmean([day["ic"] for day in a["daily"]]))
    assert not any(item["instrument"] == "S11" and item["date"] < str(days[22]) for item in a["scores"])


def test_open_label_purges_horizon_plus_entry_lag_and_check_exposes_no_returns(snapshot):
    request, config, frames, days = snapshot
    request["protocol"].update(label="open_return", forward_n=3)
    result = assay_evaluate(request, config)
    assert result["label"]["purged_dates"] == list(map(str, days[-4:]))
    assert result["label"]["horizons"] == [3]
    close = frames["prices"].sort("date", "symbol")["close"].to_numpy().reshape(40, 12)
    expected_factor = close[16:21, :11].mean(axis=0)
    expected_label = close[24, :11] / close[21, :11] - 1
    assert result["daily"][0]["ic"] == pytest.approx(np.corrcoef(expected_factor, expected_label)[0, 1])
    assert result["daily"][0]["n_samples"] == 11
    check = assay_evaluate(dict(request, operation="check"), config)
    assert check["metrics"] == {} and check["daily"] == [] and check["scores"] == []
    assert "raw_factor_report" not in check and check["portfolio"] is None
    assert check["n_dates"] == 20 and check["nan_ratio"] == 0
    with pytest.raises(ValueError, match="warmup"):
        assay_evaluate(dict(request, expression="ts_mean(close,100)"), config)
    with pytest.raises(ValueError, match="complete requested interval"):
        assay_evaluate(dict(request, end="2021-01-01"), config)


def test_actual_portfolio_and_combination_keep_source_reports_and_benchmark(snapshot):
    request, config, _, _ = snapshot
    result = assay_evaluate(dict(request, fast=False), config)
    portfolio = result["portfolio"]
    assert portfolio["actions"] and portfolio["holdings"] and len(portfolio["daily"]) == len(result["daily"])
    assert any(row["cost"] > 0 for row in portfolio["actions"])
    np.testing.assert_allclose(portfolio["raw"]["benchmark_series"], np.arange(120, 139) / 120)
    assert portfolio["execution"]["qlib_topk_drop_equivalence"] is False
    combined = assay_evaluate(dict(request, operation="combine", expressions=["close", "adv5"], fast=False), config)
    assert combined["portfolio"]["actions"]
    assert combined["portfolio"]["raw"]["factor_id"] != portfolio["raw"]["factor_id"]
    check = assay_evaluate(dict(request, operation="check", expression="safe_div(close,volume,fill=0)"), config)
    assert check["success"]


def test_hash_missing_data_and_unsupported_controls_fail_explicitly(snapshot):
    request, config, frames, _ = snapshot
    from pathlib import Path
    path = Path(config["data_manifest"]).parent / "benchmark.parquet"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="content mismatch"):
        assay_evaluate(request, config)
    path.write_bytes(original)
    request["protocol"]["assay_portfolio"]["benchmark"] = "index"
    with pytest.raises(ValueError, match="actual custom index"):
        assay_evaluate(dict(request, fast=False), config)
    replace_asset(snapshot, "benchmark", frames["benchmark"].slice(1))
    with pytest.raises(ValueError, match="benchmark is missing"):
        assay_evaluate(request, config)


def test_missing_groups_and_duplicate_prices_are_not_silently_repaired(snapshot):
    request, config, frames, _ = snapshot
    replace_asset(snapshot, "groups", frames["groups"].filter(pl.col("symbol") != "S03"))
    with pytest.raises(ValueError, match="missing point-in-time labels"):
        assay_evaluate(dict(request, expression="cs_group_rank(close,'sector')"), config)
    replace_asset(snapshot, "prices", pl.concat([frames["prices"], frames["prices"].head(1)]))
    with pytest.raises(ValueError, match="duplicate keys"):
        assay_evaluate(request, config)


def test_bounded_worker_serializes_real_assay_outputs_and_invalid_results(snapshot, tmp_path):
    request, config, _, _ = snapshot
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config | {key: request["protocol"][key]
        for key in ("backend", "market", "data_digest", "environment_digest")}))
    service = OracleService(config_path, tmp_path / "oracle")
    full_request = dict(request, fast=False)
    result = service.execute(full_request)
    assert result["success"] and result["request_id"] == request["request_id"]
    assert len(result["jobs"]) == 1 and result["portfolio"]["actions"]
    saved = {str(file): file.read_bytes() for file in (tmp_path / "oracle").rglob("*.json")}
    assert service.execute(full_request) == result
    assert saved == {str(file): file.read_bytes() for file in (tmp_path / "oracle").rglob("*.json")}
    invalid = service.execute(dict(request, request_id="b"*64, expression="close-close"))
    assert not invalid["success"] and "correlations" in invalid["error"]
    assert service.health()["market"] == "nasdaq100"
    wrong_market = dict(request, protocol=request["protocol"] | {"market": "sp500"})
    with pytest.raises(ValueError, match="market"):
        service.execute(wrong_market)


def test_all_guide_operators_execute_through_pinned_assay_parser_and_engine(snapshot):
    request, config, _, _ = snapshot
    for dialect in ("qlib", "assay"):
        values = {"value": "$close" if dialect == "qlib" else "close", "number": "1",
                  "window": "5", "lag": "1", "quantile": ".1", "group": "'sector'"}
        for name, signature in REGISTRY[dialect].items():
            expression = name + "(" + ",".join(values[role.removesuffix('?').split(':')[-1]] for role in signature) + ")"
            result = assay_evaluate(dict(request, expression=expression, operation="check"), config)
            assert result["success"], expression


def test_guide_extensions_and_split_adjusted_vwap_have_expected_values(snapshot):
    request, config, frames, days = snapshot
    split = pl.DataFrame({"event_id": ["split-1"], "symbol": ["S03"], "ex_date": [days[25]],
                          "as_of_date": [days[24]], "split_ratio": [2.], "dividend_cash": [0.]})
    replace_asset(snapshot, "events", split)
    engine, _, _, _, _, _ = prepare_panel(request, T3Protocol(**request["protocol"]), config)
    original = frames["prices"].filter(pl.col("symbol") == "S03")["vwap"].to_numpy()
    expected = original.copy(); expected[:25] /= 2
    np.testing.assert_allclose(engine.field_matrix("vwap")[:, 3], expected)
    close = engine.field_matrix("close")
    np.testing.assert_allclose(engine.evaluate("Tanh($close / 20)").values, np.tanh(close / 20))
    mask = engine.evaluate("Mask(Gt($close,20),$close)").values
    np.testing.assert_allclose(mask, np.where(close > 20, close, np.nan), equal_nan=True)


def test_cn_portfolio_uses_actual_price_bands_and_full_effective_costs(snapshot):
    from pathlib import Path
    request, config, frames, days = snapshot
    path = Path(config["data_manifest"])
    manifest = json.loads(path.read_text())
    manifest.update(market="csi300", adjustment="total")
    path.write_text(json.dumps(manifest))
    request["protocol"].update(market="csi300", data_digest=digest(manifest), assay_portfolio=PortfolioBacktestConfig.preset("A",
        universe="CSI300", period_start=str(days[20]), period_end=str(days[-1]), benchmark="custom", benchmark_symbol="TEST_INDEX",
        rebalance_type="daily", min_rebalance_interval=1, slippage_model="zero", st_filter=False,
        new_listing_lockout_days=0, ipo_lockout_days=0, rebalance_around_index=False, save_position_log=True).to_dict())
    execution = frames["execution"].join(frames["prices"].select("date", "symbol", "close"), on=["date", "symbol"])
    execution = execution.with_columns((pl.col("close") * 1.1).alias("up_limit"), (pl.col("close") * .9).alias("down_limit"))
    replace_asset(snapshot, "execution", execution)
    raw = assay_evaluate(dict(request, fast=False), config)["portfolio"]["raw"]
    assert raw["a_share_metrics"] is not None and raw["config"]["stamp_duty_rate"] == .0005
    assert raw["config"]["market"] == "A" and raw["trade_log"]
    replace_asset(snapshot, "execution", execution.with_columns(pl.lit(None, dtype=pl.Float64).alias("up_limit")))
    with pytest.raises(ValueError, match="actual daily price-limit bands"):
        assay_evaluate(dict(request, fast=False), config)
