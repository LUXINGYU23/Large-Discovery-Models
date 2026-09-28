"""Run with the pinned Assay environment; all data here are explicit fixtures."""

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("assay")
import polars as pl
from assay.portfolio.config import PortfolioBacktestConfig
from ldm_tts.contracts.evaluation import EvaluationPaused

from tasks.alphabench.core.assay_adapter import assay_evaluate, prepare_panel
from tasks.alphabench.core.data import sha256
from tasks.alphabench.core.oracle_service import OracleService
from tasks.alphabench.core.oracle_identity import oracle_environment_digest
from tasks.alphabench.core.oracle_worker import clean
from tasks.alphabench.core.protocol import T3Protocol, digest
from tasks.alphabench.core.grammar import REGISTRY


def test_source_lint_runs_without_market_assets_and_retains_native_diagnostics():
    from assay.engine.diagnostics import lint

    protocol = T3Protocol(backend="assay", market="nasdaq100", filter_profile="assay_code_filter_v1")
    request = {"protocol": protocol.to_dict(), "operation": "check"}
    for expression, valid in [("ts_mean(close,5)", True), ("Mean($close,5)", True), ("close", False), ("1", False)]:
        result = assay_evaluate(request | {"expression": expression}, {})
        assert result["diagnostics"] == lint(expression).to_dict()
        assert result["success"] is valid and result["check_kind"] == "lint"
        assert result["nan_ratio"] is None and result["non_finite_ratio"] is None
        assert result["metrics"] == {} and result["daily"] == [] and result["scores"] == [] and result["portfolio"] is None
        assert protocol.check_passed(result | {"elapsed_seconds": .01}) is valid
        assert protocol.check_passed(result | {"elapsed_seconds": .01}, paper=True) is None


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
    protocol = T3Protocol(backend="assay", market="nasdaq100", data_digest=digest(manifest), environment_digest=oracle_environment_digest({"backend": "assay"}),
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
    with pytest.raises(EvaluationPaused, match="complete requested interval"):
        assay_evaluate(dict(request, end="2021-01-01"), config)


@pytest.mark.parametrize("label,horizon,tail", [("close_return", 3, 3), ("open_return", 3, 4)])
@pytest.mark.parametrize("profile", ["ldm_matched_v1", "upstream_searcher_v1", "upstream_benchmark_v1"])
def test_assay_source_keeps_inclusive_panel_tail_without_future_reads(snapshot, label, horizon, tail, profile):
    request, config, _, days = snapshot
    request["protocol"].update(profile=profile, label=label, forward_n=horizon)
    request["end"] = str(days[30])
    protocol = T3Protocol(**request["protocol"])
    raw = assay_evaluate(request, config)
    end = 30 if protocol.end_inclusive else 29
    last = end if protocol.end_inclusive else end - tail
    assert raw["actual_end"] == str(days[end]) and raw["label"]["read_end"] == str(days[end])
    assert raw["interval"]["end_inclusive"] is protocol.end_inclusive
    assert [row["date"] for row in raw["daily"]] == list(map(str, days[20:last + 1]))
    if protocol.end_inclusive:
        assert raw["label"]["purged_dates"] == []
        assert raw["label"]["unavailable_tail_dates"] == list(map(str, days[end - tail + 1:end + 1]))
        assert all(row["n_samples"] == 0 and np.isnan(row["ic"]) for row in raw["daily"][-tail:])
    else:
        assert raw["label"]["unavailable_tail_dates"] == []
        assert raw["label"]["purged_dates"] == list(map(str, days[end - tail + 1:end + 1]))
        assert all(row["n_samples"] > 0 for row in raw["daily"])
    assert raw["metrics"]["ic"] == pytest.approx(np.nanmean([row["ic"] for row in raw["daily"]]))
    # A later split must not alter source-window factors or returns.
    future_event = pl.DataFrame({"event_id": ["future"], "symbol": ["S03"], "ex_date": [days[31]],
        "as_of_date": [days[31]], "split_ratio": [2.], "dividend_cash": [0.]})
    replace_asset(snapshot, "events", future_event)
    unchanged = assay_evaluate(request, config)
    assert clean(raw["daily"]) == clean(unchanged["daily"])
    assert clean(raw["scores"]) == clean(unchanged["scores"])


def test_actual_portfolio_and_combination_keep_source_reports_and_benchmark(snapshot):
    request, config, _, _ = snapshot
    from assay.service import AssayService
    from tasks.alphabench.core.assay_adapter import PreparedPortfolioBacktester

    previous_key, previous_service = os.environ.get("ASSAY_API_KEYS"), AssayService._instance
    result = assay_evaluate(dict(request, fast=False), config)
    portfolio = result["portfolio"]
    assert os.environ.get("ASSAY_API_KEYS") == previous_key and AssayService._instance is previous_service
    assert portfolio["execution"]["route"] == "POST /v1/portfolio/backtest"
    assert portfolio["actions"] and portfolio["holdings"] and len(portfolio["daily"]) == len(result["daily"])
    assert any(row["cost"] > 0 for row in portfolio["actions"])
    np.testing.assert_allclose(portfolio["raw"]["benchmark_series"], np.arange(120, 139) / 120)
    assert portfolio["execution"]["qlib_topk_drop_equivalence"] is False
    protocol = T3Protocol(**request["protocol"])
    engine, days, tradable, execution, benchmark, _ = prepare_panel(request, protocol, config)
    indices = np.array([i for i, day in enumerate(days) if day >= dt.date.fromisoformat(request["start"])][:-1])
    factor = np.where(engine.membership, engine.evaluate(request["expression"]).values, np.nan)[indices]
    identity = json.dumps(portfolio["signal_contract"], sort_keys=True)
    native = PreparedPortfolioBacktester(engine, factor, indices, execution).run(identity,
        PortfolioBacktestConfig.from_dict(portfolio["raw"]["config"]),
        as_of=portfolio["raw"]["config"]["as_of_date"],
        tradable_mask=tradable[indices], benchmark=benchmark[indices]).to_dict()
    assert clean(portfolio["raw"]) == clean(native)
    combined = assay_evaluate(dict(request, operation="combine", expressions=["close", "adv5"], fast=False), config)
    assert combined["portfolio"]["actions"]
    assert combined["portfolio"]["raw"]["factor_id"] != portfolio["raw"]["factor_id"]
    check = assay_evaluate(dict(request, operation="check", expression="safe_div(close,volume,fill=0)"), config)
    assert check["success"]


def test_source_assay_portfolio_retains_the_inclusive_end_even_with_undefined_tail_ic(snapshot):
    request, config, _, days = snapshot
    request["protocol"].update(profile="upstream_searcher_v1", forward_n=3, label="open_return")
    request.update(end=str(days[30]), fast=False)
    result = assay_evaluate(request, config)
    expected_dates = list(map(str, days[20:31]))
    assert [row["date"] for row in result["daily"]] == expected_dates
    assert [row["date"] for row in result["portfolio"]["daily"]] == expected_dates
    assert result["portfolio"]["actions"] and result["portfolio"]["holdings"]
    assert all(np.isnan(row["ic"]) for row in result["daily"][-4:])
    assert result["portfolio"]["execution"]["config"]["period_end"] == str(days[30])


@pytest.mark.parametrize("market", ["nasdaq100", "csi300"])
def test_source_panel_and_metrics_match_actual_assay_service(snapshot, tmp_path, monkeypatch, market):
    from pathlib import Path
    from assay.config import AssayConfig
    from assay.data.schemas import price_partition_path, universe_snapshots_path, adj_events_path
    from assay.service import AssayService

    request, config, frames, days = snapshot
    manifest_path = Path(config["data_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest.update(market=market, adjustment="split")
    manifest_path.write_text(json.dumps(manifest))
    request["protocol"].update(profile="upstream_searcher_v1", market=market,
                               label="open_return", forward_n=2, data_digest=digest(manifest))
    request.update(start=str(days[20]), end=str(days[30]), expression="cs_rank(ts_mean(close,3))")
    # The requested end has a new snapshot but no trading row.
    prices = frames["prices"].filter(pl.col("date") != days[30])
    members = frames["membership"].filter((pl.col("symbol") != "S10") | (pl.col("date") >= days[30]))
    groups = frames["groups"].with_columns(
        pl.when(pl.col("effective_date") == days[23]).then(pl.lit(days[30])).otherwise(pl.col("effective_date")).alias("effective_date"))
    groups = groups.with_columns(pl.col("effective_date").alias("as_of_date"))
    for name, frame in [("prices", prices), ("membership", members), ("groups", groups)]:
        replace_asset(snapshot, name, frame)
    manifest = json.loads(manifest_path.read_text())
    calendar = manifest_path.parent / manifest["assets"]["calendar"]["path"]
    calendar.write_text("\n".join(day for day in calendar.read_text().splitlines() if day != str(days[30])))
    manifest["assets"]["calendar"]["sha256"] = sha256(calendar)
    manifest_path.write_text(json.dumps(manifest))
    request["protocol"]["data_digest"] = digest(manifest)
    events = pl.DataFrame({"event_id": ["split"], "symbol": ["S03"], "ex_date": [days[25]],
        "as_of_date": [days[24]], "split_ratio": [2.], "dividend_cash": [0.]})
    replace_asset(snapshot, "events", events)
    native_root = tmp_path / "native-store"
    native_market = "CN" if market.startswith("csi") else "US"
    for (year, month), frame in prices.with_columns(
        pl.col("date").dt.year().alias("year"), pl.col("date").dt.month().alias("month")).partition_by(
            "year", "month", as_dict=True).items():
        path = price_partition_path(native_root, native_market, year, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.drop("year", "month").write_parquet(path)
    snapshots = members.group_by("date", maintain_order=True).agg(pl.col("symbol").alias("symbols"))
    snapshots = snapshots.rename({"date": "effective_date"}).with_columns(
        pl.lit(market.upper()).alias("index_id"), pl.col("effective_date").alias("as_of_date"))
    for path, frame in [(universe_snapshots_path(native_root, native_market), snapshots),
                         (adj_events_path(native_root, native_market), events)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    monkeypatch.setattr(AssayService, "apply_system_config", lambda _: {})
    service = AssayService(AssayConfig.for_tests(native_root, market=native_market))
    original = service.evaluate(request["expression"], universe=market.upper(),
        period=(request["start"], request["end"]), horizons=[2], save=False)
    result = assay_evaluate(request, config)
    assert original.failure_mode is None and result["success"]
    assert original.execution == "next_open" and result["label"]["execution"] == original.execution
    assert result["adjustment"] == "split" and result["history_origin"] == request["start"]
    assert result["panel_policy"] == "source_period_end_universe"
    for name in ["ic", "rank_ic", "icir", "rank_icir"]:
        assert result["metrics"][name] == pytest.approx(getattr(original, name))
    np.testing.assert_allclose([row["ic"] for row in result["daily"]],
                               [np.nan if value is None else value for value in original.ic_series], equal_nan=True)
    # S11 enters during the interval; the source uses the end snapshot for every row.
    assert any(row["instrument"] == "S11" and row["date"] == request["start"] for row in result["scores"])
    assert any(row["instrument"] == "S10" and row["date"] == request["start"] for row in result["scores"])
    assert result["actual_end"] == str(days[29])
    assert all(np.isnan(row["score"]) for row in result["scores"] if row["date"] == request["start"])
    if native_market == "US":
        portfolio = assay_evaluate(dict(request, fast=False), config)["portfolio"]
        assert portfolio["execution"]["config"]["period_end"] == request["end"]
        assert portfolio["execution"]["config"]["as_of_date"] == request["end"]
        assert portfolio["daily"][-1]["date"] == str(days[29])
    expression = "cs_neutralize(ts_mean(close,3),'sector')"
    original_groups = {"sector": {f"S{i:02d}": "a" if i % 2 else "b" for i in range(12)}}
    grouped_native = service.evaluate(expression, universe=market.upper(), group_data=original_groups,
        period=(request["start"], request["end"]), horizons=[2], save=False)
    grouped_task = assay_evaluate(dict(request, expression=expression), config)
    assert grouped_native.failure_mode is None and grouped_task["success"]
    np.testing.assert_allclose([row["ic"] for row in grouped_task["daily"]],
        [np.nan if value is None else value for value in grouped_native.ic_series], equal_nan=True)
    for expression in ["close", "close-close", "ts_mean(close,100)"]:
        failed = assay_evaluate(dict(request, expression=expression), config)
        native = service.evaluate(expression, universe=market.upper(),
            period=(request["start"], request["end"]), horizons=[2], save=False)
        assert not failed["success"] and native.failure_mode
        assert failed["diagnostics"][0]["failure_mode"] == native.failure_mode


def test_hash_missing_data_and_unsupported_controls_fail_explicitly(snapshot):
    request, config, frames, _ = snapshot
    from pathlib import Path
    path = Path(config["data_manifest"]).parent / "benchmark.parquet"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(EvaluationPaused, match="content mismatch") as exc:
        assay_evaluate(request, config)
    assert exc.value.status == "paused_data_integrity"
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
    missing_calendar = service.execute(dict(request, request_id="c"*64, end="2021-01-01"))
    assert missing_calendar["pause_status"] == "paused_data_coverage"
    assert len(missing_calendar["jobs"]) == 1 and not missing_calendar["metrics"]
    assert service.health()["market"] == "nasdaq100"
    wrong_market = dict(request, protocol=request["protocol"] | {"market": "sp500"})
    with pytest.raises(ValueError, match="market"):
        service.execute(wrong_market)
    changed_config = json.loads(config_path.read_text())
    changed_config["data_digest"] = "0" * 64
    config_path.write_text(json.dumps(changed_config))
    with pytest.raises(ValueError, match="data manifest identity mismatch"):
        OracleService(config_path, tmp_path / "different-oracle")


@pytest.mark.parametrize("point,exit_code", [("after_spawn", 94), ("after_output", 95),
                                           ("worker_lost", 94)])
def test_oracle_service_process_death_reconciles_one_real_worker(snapshot, tmp_path, point, exit_code):
    request, config, _, _ = snapshot
    full_request = dict(request, fast=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config | {key: request["protocol"][key]
        for key in ("backend", "market", "data_digest", "environment_digest")}))
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(full_request))
    root = tmp_path / "oracle"
    script = r'''
import json, os, signal, sys
from pathlib import Path
from ldm_tts.contracts.evaluation import EvaluationPaused
from tasks.alphabench.core import oracle_service as module

config, root, request_path, point = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
marker = root / "crashed.marker"
original_spawn = module.subprocess.Popen

def spawn(*args, **kwargs):
    worker = original_spawn(*args, **kwargs)
    with (root / "launches.txt").open("a") as stream:
        stream.write(str(worker.pid) + "\n")
    if point == "worker_lost" and not marker.exists():
        os.kill(worker.pid, signal.SIGSTOP)
    if point in {"after_spawn", "worker_lost"} and not marker.exists():
        marker.write_text(point)
        os._exit(94)
    return worker

module.subprocess.Popen = spawn
service = module.OracleService(config, root)
original_execute = service.receipts.execute

def execute(identity, request, *, reserve, operation):
    if point == "after_output" and not marker.exists():
        original_operation = operation
        def interrupted():
            answer = original_operation()
            marker.write_text(point)
            os._exit(95)
        operation = interrupted
    return original_execute(identity, request, reserve=reserve, operation=operation)

service.receipts.execute = execute
try:
    print(json.dumps(service.execute(json.loads(request_path.read_text()))))
except EvaluationPaused as exc:
    print(json.dumps({"status": exc.status, "error": str(exc)}))
'''
    command = [sys.executable, "-c", script, str(config_path), str(root), str(request_path), point]
    killed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert killed.returncode == exit_code, killed.stderr
    if point == "worker_lost":
        worker_pid = int((root / "launches.txt").read_text().splitlines()[0])
        os.kill(worker_pid, signal.SIGKILL)
        resumed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert resumed.returncode == 0, resumed.stderr
        assert json.loads(resumed.stdout)["status"] == "paused_outcome_unknown"
        assert len((root / "launches.txt").read_text().splitlines()) == 1
        receipt = json.loads(next((root / "requests").glob("*.json")).read_text())
        assert receipt["state"] == "dispatch_intent"
        assert not (root / "jobs" / request["request_id"] / "response.json").exists()
        return
    output = root / "jobs" / request["request_id"] / "response.json"
    deadline = time.monotonic() + 30
    while not output.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    assert output.exists()
    worker_response = json.loads(output.read_text())
    resumed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout) == worker_response
    assert len((root / "launches.txt").read_text().splitlines()) == 1
    receipt = json.loads(next((root / "requests").glob("*.json")).read_text())
    assert receipt["state"] == "completed" and receipt["response_digest"] == digest(worker_response)


def test_lost_assay_rest_response_pauses_without_reposting(snapshot, tmp_path, monkeypatch):
    request, config, _, _ = snapshot
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config | {key: request["protocol"][key]
        for key in ("backend", "market", "data_digest", "environment_digest")}))
    marker = tmp_path / "rest-posts.txt"
    (tmp_path / "sitecustomize.py").write_text('''
import os
import httpx

original_post = httpx.Client.post

def post(self, url, *args, **kwargs):
    response = original_post(self, url, *args, **kwargs)
    if str(url).endswith("/v1/portfolio/backtest"):
        with open(os.environ["T3_REST_POST_MARKER"], "a") as output:
            output.write(str(response.status_code) + "\\n")
        raise httpx.ReadTimeout("response lost after portfolio completion")
    return response

httpx.Client.post = post
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("T3_REST_POST_MARKER", str(marker))
    service = OracleService(config_path, tmp_path / "oracle")
    full_request = dict(request, fast=False)
    with pytest.raises(EvaluationPaused, match="worker exit requires reconciliation"):
        service.execute(full_request)
    assert marker.read_text().splitlines() == ["200"]
    assert service.receipts.load(request["request_id"])["state"] == "dispatch_intent"
    assert not (tmp_path / "oracle" / "jobs" / request["request_id"] / "response.json").exists()
    with pytest.raises(EvaluationPaused, match="physical request requires reconciliation"):
        service.execute(full_request)
    assert marker.read_text().splitlines() == ["200"]


def test_all_guide_operators_execute_through_pinned_assay_parser_and_engine(snapshot):
    request, config, _, _ = snapshot
    for dialect in ("qlib", "assay"):
        values = {"value": "$close" if dialect == "qlib" else "close", "number": "1",
                  "window": "5", "lag": "1", "quantile": ".1", "group": "'sector'"}
        for name, signature in REGISTRY[dialect].items():
            expression = name + "(" + ",".join(values[role.removesuffix('?').split(':')[-1]] for role in signature) + ")"
            result = assay_evaluate(dict(request, expression=expression, operation="check"), config)
            assert result["success"], expression


@pytest.mark.parametrize("profile", ["ldm_matched_v1", "upstream_searcher_v1"])
def test_guide_extensions_and_split_adjusted_vwap_have_expected_values(snapshot, profile):
    request, config, frames, days = snapshot
    request["protocol"]["profile"] = profile
    request["end"] = str(days[-1])
    split = pl.DataFrame({"event_id": ["split-1"], "symbol": ["S03"], "ex_date": [days[25]],
                          "as_of_date": [days[24]], "split_ratio": [2.], "dividend_cash": [0.]})
    replace_asset(snapshot, "events", split)
    engine, actual_days, _, _, _, _ = prepare_panel(request, T3Protocol(**request["protocol"]), config)
    original = frames["prices"].filter(pl.col("symbol") == "S03")["vwap"].to_numpy()
    expected = original.copy(); expected[:25] /= 2
    np.testing.assert_allclose(engine.field_matrix("vwap")[:, 3], expected[[days.index(day) for day in actual_days]])
    close = engine.field_matrix("close")
    np.testing.assert_allclose(engine.evaluate("Tanh($close / 20)").values, np.tanh(close / 20))
    mask = engine.evaluate("Mask(Gt($close,20),$close)").values
    np.testing.assert_allclose(mask, np.where(close > 20, close, np.nan), equal_nan=True)


@pytest.mark.parametrize("profile", ["ldm_matched_v1", "upstream_searcher_v1"])
def test_cn_portfolio_uses_actual_price_bands_and_full_effective_costs(snapshot, profile):
    from pathlib import Path
    request, config, frames, days = snapshot
    path = Path(config["data_manifest"])
    manifest = json.loads(path.read_text())
    manifest.update(market="csi300", adjustment="total" if profile == "ldm_matched_v1" else "split")
    path.write_text(json.dumps(manifest))
    request["end"] = str(days[-1])
    request["protocol"].update(market="csi300", profile=profile, label="open_return", data_digest=digest(manifest), assay_portfolio=PortfolioBacktestConfig.preset("A",
        universe="CSI300", period_start=str(days[20]), period_end=str(days[-1]), benchmark="custom", benchmark_symbol="TEST_INDEX",
        rebalance_type="daily", min_rebalance_interval=1, slippage_model="zero", st_filter=False,
        new_listing_lockout_days=0, ipo_lockout_days=0, rebalance_around_index=False, save_position_log=True).to_dict())
    if profile == "upstream_searcher_v1":
        events = pl.DataFrame({"event_id": ["split-dividend"], "symbol": ["S03"], "ex_date": [days[25]],
            "as_of_date": [days[24]], "split_ratio": [2.], "dividend_cash": [.5]})
        replace_asset(snapshot, "events", events)
        source_engine, _, _, _, _, _ = prepare_panel(request, T3Protocol(**request["protocol"]), config)
        portfolio_engine, _, _, _, _, _ = prepare_panel(request, T3Protocol(**request["protocol"]), config, portfolio=True)
        assert portfolio_engine.field_matrix("close")[0, 3] < source_engine.field_matrix("close")[0, 3]
    execution = frames["execution"].join(frames["prices"].select("date", "symbol", "close"), on=["date", "symbol"])
    execution = execution.with_columns((pl.col("close") * 1.1).alias("up_limit"), (pl.col("close") * .9).alias("down_limit"))
    replace_asset(snapshot, "execution", execution)
    result = assay_evaluate(dict(request, fast=False), config)
    raw = result["portfolio"]["raw"]
    assert result["portfolio"]["signal_contract"]["factor_adjustment"] == "total"
    assert raw["lineage"]["adj_version"] == "total"
    if profile == "upstream_searcher_v1":
        from assay.data.store.datastore import DataStore
        from assay.portfolio.backtester import PortfolioBacktester

        class SourceStore(DataStore):
            def __init__(self):
                pass

            def get_universe(self, universe, date, as_of_date):
                return sorted(frames["membership"]["symbol"].unique().to_list())

            def _read_prices(self, symbols, start, end, as_of_date):
                return frames["prices"].filter(pl.col("date").is_between(start, end) & pl.col("symbol").is_in(symbols))

            def _read_adj_events(self, symbols, end, as_of_date):
                return events.filter((pl.col("ex_date") <= end) & pl.col("symbol").is_in(symbols))

            def get_trade_status(self, symbols, start, end, as_of_date):
                return execution.filter(pl.col("date").is_between(start, end) & pl.col("symbol").is_in(symbols))

        benchmark = frames["benchmark"].filter(pl.col("date").is_between(days[20], days[-1]))["close"].to_numpy()
        native = PortfolioBacktester(SourceStore()).run(request["expression"],
            PortfolioBacktestConfig.from_dict(raw["config"]), as_of=request["end"],
            tradable_mask=np.ones((20, 12), dtype=bool), benchmark=benchmark).to_dict()
        np.testing.assert_allclose(raw["nav_series"], native["nav_series"])
        np.testing.assert_allclose(raw["benchmark_series"], native["benchmark_series"])
        assert raw["trade_log"] == native["trade_log"]
        assert result["adjustment"] == "split"
    assert raw["a_share_metrics"] is not None and raw["config"]["stamp_duty_rate"] == .0005
    assert raw["config"]["market"] == "A" and raw["trade_log"]
    replace_asset(snapshot, "execution", execution.with_columns(pl.lit(None, dtype=pl.Float64).alias("up_limit")))
    with pytest.raises(ValueError, match="actual daily price-limit bands"):
        assay_evaluate(dict(request, fast=False), config)
