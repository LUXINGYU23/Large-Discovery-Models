"""Offline Assay execution with historical cross-sections and explicit assets."""

import datetime as dt
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import numpy as np
import polars as pl

from assay.data.store.adjust import forward_adjust
from assay.engine import FactorEngine
from assay.engine.ast import OpNode
from assay.engine.engine import EvalContext
from assay.engine import operators
from assay.engine.diagnostics import lint
from assay.engine.parsing import parse
from assay.evaluator.forward_returns import forward_returns
from assay.evaluator.metrics import evaluate_ic
from assay.evaluator.turnover import rank_autocorr
from assay.portfolio.backtester import PortfolioBacktester
from assay.portfolio.config import PortfolioBacktestConfig
from assay.service import AssayService
from ldm_tts.contracts.evaluation import EvaluationPaused

from .data import ASSAY_ASSETS, sha256
from .assay_contract import portfolio_config
from .grammar import parse_expression
from .protocol import T3Protocol, digest


# The pinned registry lacks these guide operators. They share the exact guide
# semantics of the task's Qlib extensions, without rewriting either dialect.
operators.register("Tanh", np.tanh, 1, 1)
operators.register("Mask", lambda condition, value: np.where(condition, value, np.nan), 2, 2)


def read_assets(config, protocol):
    path = Path(config["data_manifest"]).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if digest(manifest) != protocol.data_digest or (manifest["backend"], manifest["market"]) != ("assay", protocol.market):
        raise EvaluationPaused("Assay data manifest differs from the frozen protocol", status="paused_data_integrity")
    inventory = manifest["assets"]
    if set(inventory) != ASSAY_ASSETS:
        raise ValueError("Assay requires all seven offline data assets")
    paths = {}
    for name, record in inventory.items():
        declared = path.parent / record["path"]
        file = declared.resolve()
        if (not file.is_relative_to(path.parent) or declared.is_symlink()
                or not file.is_file() or sha256(file) != record["sha256"]):
            raise EvaluationPaused("Assay asset path or content mismatch: " + name, status="paused_data_integrity")
        paths[name] = file
    calendar = [dt.date.fromisoformat(line) for line in paths.pop("calendar").read_text().splitlines()]
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("calendar must contain distinct ordered sessions")
    return manifest, calendar, {name: pl.read_parquet(file) for name, file in paths.items()}


def unique(frame, keys, name):
    if frame.select(keys).null_count().sum_horizontal().item() or frame.select(keys).is_duplicated().any():
        raise ValueError(name + " contains null or duplicate keys")


def known_by(frame, effective, name):
    if frame[effective].null_count() or frame["as_of_date"].null_count() or frame.filter(pl.col("as_of_date") > pl.col(effective)).height:
        raise ValueError(name + " was not known on its effective date")


class HistoricalFactorEngine(FactorEngine):
    def __init__(self, panel, membership, groups):
        super().__init__(panel)
        self.membership = membership
        self.daily_groups = groups
        if membership.shape != self._shape or any(value.shape != self._shape for value in groups.values()):
            raise ValueError("historical masks must align with the complete panel axes")

    def _eval(self, node, ctx):
        if not isinstance(node, OpNode) or not node.op.startswith("cs_"):
            return super()._eval(node, ctx)
        spec = operators.get(node.op)
        args = [self._eval(child, ctx) for child in node.children]
        args[0] = np.where(self.membership, args[0], np.nan)
        if not spec.needs_ctx:
            return np.where(self.membership, spec.fn(*args), np.nan)
        key = args[1]
        if key not in self.daily_groups:
            raise ValueError("missing point-in-time group: " + key)
        labels = self.daily_groups[key]
        if any(label is None for label in labels[self.membership]):
            raise ValueError("missing point-in-time labels for active constituents: " + key)
        result = np.full(self._shape, np.nan)
        for index, date in enumerate(ctx.dates):
            # The source group kernels accept one label vector, so execute each dated cross-section separately.
            eligible = self.membership[index]
            day_labels = np.where(eligible, labels[index], "__outside_universe__")
            day_ctx = EvalContext([date], ctx.symbols, {}, {key: day_labels})
            result[index] = spec.fn(args[0][index:index+1], *args[1:], ctx=day_ctx)[0]
        return np.where(self.membership, result, np.nan)


def prepare_panel(request, protocol, config, *, portfolio=False):
    manifest, calendar, assets = read_assets(config, protocol)
    start, end = dt.date.fromisoformat(request["start"]), dt.date.fromisoformat(request["end"])
    if calendar[0] > start or calendar[-1] < end:
        raise EvaluationPaused("calendar does not bracket the complete requested interval", status="paused_data_coverage")
    source = protocol.profile != "ldm_matched_v1"
    days = [day for day in calendar if (start <= day <= end if source else day < end)]
    if not days or not any(day >= start for day in days):
        raise ValueError("no trading sessions in the requested interval")
    members = assets["membership"]
    unique(members, ["date", "symbol"], "membership")
    known_by(members, "date", "membership")
    if source:
        members = members.filter(pl.col("date") <= end)
        members = members.filter(pl.col("date") == members["date"].max())
    else:
        members = members.filter(pl.col("date").is_in(days))
    symbols = sorted(members["symbol"].unique().to_list())
    if not symbols:
        raise ValueError("empty historical universe")
    grid = pl.DataFrame({"date": days}).join(pl.DataFrame({"symbol": symbols}), how="cross")
    if source:
        membership = np.ones((len(days), len(symbols)), dtype=bool)
    else:
        membership = grid.join(members.select("date", "symbol").with_columns(pl.lit(True).alias("member")),
            on=["date", "symbol"], how="left")["member"].fill_null(False).to_numpy().reshape(len(days), -1)
    if not membership.sum(axis=1).all():
        raise ValueError("historical universe is empty on a calendar session")
    prices = assets["prices"]
    unique(prices, ["date", "symbol"], "prices")
    known_by(prices, "date", "price")
    prices = prices.filter(pl.col("date").is_in(days) & pl.col("symbol").is_in(symbols))
    required = ["open", "high", "low", "close", "volume", "vwap"]
    if not set(required) <= set(prices.columns):
        raise ValueError("raw OHLCV and actual VWAP are required")
    events = assets["events"]
    unique(events, ["event_id"], "corporate actions")
    events = events.filter((pl.col("ex_date") <= (end if source else days[-1])) & pl.col("symbol").is_in(symbols))
    known_by(events, "ex_date", "corporate action")
    factor_basis = "total" if not source and protocol.market.startswith("csi") else "split"
    if manifest["adjustment"] != factor_basis:
        raise ValueError("Assay adjustment differs from the fixed market basis")
    basis = "total" if portfolio and protocol.market.startswith("csi") else factor_basis
    adjusted = forward_adjust(prices, events, mode=basis)
    # Assay adjusts OHLCV but passes extra fields through; VWAP needs the same price factor.
    adjusted = adjusted.join(prices.select("date", "symbol", pl.col("close").alias("raw_close")),
                             on=["date", "symbol"], how="left")
    adjusted = adjusted.with_columns((pl.col("vwap") * pl.col("close") / pl.col("raw_close")).alias("vwap"))
    panel = grid.join(adjusted.select("date", "symbol", *required), on=["date", "symbol"], how="left")
    groups = assets["groups"]
    unique(groups, ["effective_date", "symbol"], "groups")
    group_keys = set(groups.columns) - {"effective_date", "as_of_date", "symbol"}
    known_by(groups, "effective_date", "group label")
    if source:
        labels = grid.join(groups.filter(pl.col("effective_date") <= end).sort("effective_date").unique(
            subset="symbol", keep="last").select("symbol", *sorted(group_keys)), on="symbol", how="left")
    else:
        labels = grid.sort("date").join_asof(groups.sort("effective_date"), left_on="date", right_on="effective_date",
            by="symbol", strategy="backward", check_sortedness=False).sort("date", "symbol")
    daily_groups = {key: labels[key].to_numpy().reshape(len(days), -1) for key in group_keys}
    engine = HistoricalFactorEngine(panel, membership, daily_groups)
    execution = assets["execution"]
    unique(execution, ["date", "symbol"], "execution")
    known_by(execution, "date", "execution status")
    execution = grid.join(execution, on=["date", "symbol"], how="left")
    tradable_values = execution["tradable"].to_numpy().reshape(engine._shape)
    if any(value is None for value in tradable_values.flat):
        raise ValueError("missing tradability evidence for the historical union")
    tradable = np.equal(tradable_values, True)
    close = engine.field_matrix("close")
    if np.any(tradable & ~np.isfinite(close)):
        raise ValueError("missing price on a reported trading session")
    benchmark = assets["benchmark"]
    unique(benchmark, ["date"], "benchmark")
    benchmark = pl.DataFrame({"date": days}).join(benchmark, on="date", how="left")["close"].to_numpy()
    if not (np.isfinite(benchmark) & (benchmark > 0)).all():
        raise ValueError("actual index benchmark is missing or invalid")
    return engine, days, tradable, execution, benchmark, manifest


def lookback(node):
    if not isinstance(node, OpNode):
        return 0
    previous = max((lookback(child) for child in node.children), default=0)
    if not node.op.startswith("ts_"):
        return previous
    window = int(node.children[1 if node.op == "ts_quantile" else -1].value)
    lag = node.op in {"ts_delay", "ts_delta", "ts_returns", "ts_log_returns"}
    return previous + window - (not lag)


def combine_scores(arrays):
    stacked = np.stack([operators.get("cs_zscore").fn(value) for value in arrays])
    count = np.isfinite(stacked).sum(axis=0)
    return np.divide(np.nansum(stacked, axis=0), count,
                     out=np.full(arrays[0].shape, np.nan), where=count > 0)


class PreparedPortfolioBacktester(PortfolioBacktester):
    def __init__(self, engine, factor, indices, execution):
        super().__init__()
        self.engine, self.factor, self.indices, self.execution = engine, factor, indices, execution

    def _build_matrices(self, expr, config, as_of):
        engine, indices = self.engine, self.indices
        return (engine, self.factor, engine.field_matrix("close")[indices], engine.field_matrix("open")[indices],
                list(engine.dates[indices]), list(engine.symbols))

    def _cn_limit_matrices(self, config, close_adj, dates, symbols, as_of):
        if config.market != "A":
            return None, None
        outputs = []
        for name in ("up_limit", "down_limit"):
            values = self.execution[name].to_numpy().reshape(self.engine._shape)[self.indices]
            raw = self.execution["close"].to_numpy().reshape(self.engine._shape)[self.indices]
            if np.any(np.isfinite(close_adj) & (~np.isfinite(values) | ~np.isfinite(raw) | (raw <= 0))):
                raise ValueError("CN portfolio requires actual daily price-limit bands")
            outputs.append(values * close_adj / raw)
        return tuple(outputs)


def portfolio_result(protocol, engine, factor, indices, execution, tradable, benchmark, expressions, requested_start, requested_end):
    config = portfolio_config(protocol, PortfolioBacktestConfig)
    config.period_start = requested_start if protocol.end_inclusive else str(engine.dates[indices[0]])[:10]
    config.period_end = requested_end if protocol.end_inclusive else str(engine.dates[indices[-1]])[:10]
    config.as_of_date = config.period_end
    backtester = PreparedPortfolioBacktester(engine, factor * protocol.direction, indices, execution)
    signal_contract = {"expressions": expressions, "direction": protocol.direction,
                       "factor_adjustment": "total" if protocol.market.startswith("csi") else "split",
                       "combination": "single" if len(expressions) == 1 else "daily_zscore_equal_weight_mean_finite"}
    identity = json.dumps(signal_contract, sort_keys=True)
    raw = portfolio_over_http(backtester, identity, config, tradable[indices], benchmark[indices],
                              protocol.worker_timeout)
    if not raw["nav_series"] or raw["n_trading_days"] != len(indices) or raw["warnings"]:
        raise ValueError("Assay portfolio did not produce a complete daily report: " + str(raw.get("attribution")))
    return {"raw": raw, "signal_contract": signal_contract,
            "daily": [{"date": day, "nav": nav, "benchmark_nav": bench}
            for day, nav, bench in zip(raw["nav_dates"], raw["nav_series"], raw["benchmark_series"])],
            "holdings": raw["position_log"], "actions": raw["trade_log"],
            "units": "returns and drawdown in fractions; source annualization 252 sessions",
            "execution": {"backend": "Assay PortfolioBacktester", "route": "POST /v1/portfolio/backtest",
                          "config": raw["config"],
                          "qlib_topk_drop_equivalence": False}}


def portfolio_over_http(backtester, identity, config, tradable, benchmark, timeout):
    from assay.api.routes import portfolio as route

    class PreparedService:
        used = False

        def backtest_portfolio(self, expr, received, *, as_of):
            if self.used or expr != identity or received.to_dict() != config.to_dict() or as_of != config.as_of_date:
                raise ValueError("Assay portfolio REST request differs from the prepared signal and config")
            self.used = True
            return backtester.run(expr, received, as_of=as_of, tradable_mask=tradable, benchmark=benchmark)

    # The pinned route resolves AssayService's process singleton. This bounded worker
    # supplies the verified panel for exactly one authenticated loopback request.
    app = FastAPI()
    app.include_router(route.router, prefix="/v1/portfolio")

    @app.exception_handler(ValueError)
    async def invalid_portfolio(_, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False, lifespan="off"))
    token = secrets.token_hex(32)
    previous_key = os.environ.get("ASSAY_API_KEYS")
    previous_service = AssayService._instance
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        os.environ["ASSAY_API_KEYS"] = token
        AssayService._instance = PreparedService()
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        try:
            thread.start()
            deadline = time.monotonic() + 10
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError("Assay portfolio REST service did not start")
                time.sleep(.01)
            with httpx.Client(trust_env=False, timeout=timeout) as client:
                response = client.post(f"http://127.0.0.1:{listener.getsockname()[1]}/v1/portfolio/backtest",
                    json={"expr": identity, "config": config.to_dict(), "as_of": config.as_of_date},
                    headers={"X-API-Key": token})
            if response.status_code == 422:
                raise ValueError(response.json()["detail"])
            if response.status_code != 200:
                raise RuntimeError(f"Assay portfolio REST returned {response.status_code}")
            return response.json()
        finally:
            server.should_exit = True
            if thread.ident is not None:
                thread.join(timeout=5)
            AssayService._instance = previous_service
            if previous_key is None:
                os.environ.pop("ASSAY_API_KEYS", None)
            else:
                os.environ["ASSAY_API_KEYS"] = previous_key


def assay_evaluate(request, config):
    protocol = T3Protocol(**request["protocol"])
    expressions = request.get("expressions", [request.get("expression")])
    if not expressions or request["operation"] not in {"check", "evaluate", "combine"}:
        raise ValueError("a supported operation with at least one expression is required")
    canonical = [parse_expression(value, backend="assay").canonical for value in expressions]
    if request["operation"] == "check" and protocol.check_kind == "lint":
        diagnostics = lint(canonical[0]).to_dict()
        return {"success": diagnostics["ok"], "check_kind": "lint", "diagnostics": diagnostics,
                "error": "; ".join(row["message"] for row in diagnostics["errors"]) if not diagnostics["ok"] else None,
                "nan_ratio": None, "non_finite_ratio": None,
                "metrics": {}, "daily": [], "scores": [], "portfolio": None}
    engine, days, tradable, execution, benchmark, manifest = prepare_panel(request, protocol, config)
    start = dt.date.fromisoformat(request["start"])
    indices = np.array([i for i, day in enumerate(days) if day >= start])
    source = protocol.profile != "ldm_matched_v1"
    if not source and indices[0] < max(lookback(parse(value)) for value in canonical):
        raise ValueError("calendar lacks the required expression warmup")
    diagnostics, arrays = [], []
    for value in canonical:
        if source and request["operation"] != "check":
            checked = engine.diagnose(value)
            diagnostics.append(checked.to_dict())
            if not checked.ok or checked.failure_mode:
                messages = [row["message"] for row in diagnostics[-1]["errors"] + diagnostics[-1]["warnings"]]
                return {"success": False, "diagnostics": diagnostics,
                    "error": "; ".join(messages) or checked.failure_mode,
                    "metrics": {}, "daily": [], "scores": [], "portfolio": None}
            result = checked.result
        else:
            result = engine.evaluate(value)
        arrays.append(np.where(engine.membership, result.values, np.nan))
    if request["operation"] == "combine":
        factor = combine_scores(arrays)
    else:
        factor = arrays[0]
    active = engine.membership[indices]
    finite = np.isfinite(factor[indices])
    result = {"success": bool((active & finite).any()), "nan_ratio": float(np.isnan(factor[indices])[active].mean()),
              "non_finite_ratio": float((~finite[active]).mean()),
              "actual_start": str(days[indices[0]]), "actual_end": str(days[indices[-1]]),
              "interval": {"start": request["start"], "end": request["end"], "end_inclusive": protocol.end_inclusive},
              "metrics": {}, "daily": [], "scores": [], "portfolio": None,
              "data_digest": digest(manifest), "history_origin": str(days[0]),
              "panel_policy": "source_period_end_universe" if source else "historical_daily_universe",
              "adjustment": manifest["adjustment"], "diagnostics": diagnostics,
              "n_dates": len(indices), "n_observations": int(active.sum()), "n_finite": int((active & finite).sum())}
    if request["operation"] == "check":
        result["check_kind"] = "dynamic"
        return result
    label_execution = "next_open" if protocol.label == "open_return" else "next_close"
    farthest = protocol.forward_n + (label_execution == "next_open")
    unavailable_tail = indices[-farthest:]
    purge = [] if protocol.end_inclusive else unavailable_tail
    if not protocol.end_inclusive:
        indices = indices[:-farthest]
    if not len(indices):
        raise ValueError("no sessions remain after forward-label purge")
    labels = forward_returns(engine.field_matrix("close"), engine.field_matrix("open"),
                             [protocol.forward_n], execution=label_execution)[protocol.forward_n]
    factor = factor[indices]
    ic = evaluate_ic(factor, {protocol.forward_n: labels[indices]})
    if not np.isfinite(ic["ic"]):
        raise ValueError("no finite daily cross-sectional correlations")
    turnover = 1 - rank_autocorr(factor)
    result["metrics"] = {name: ic[name] for name in ("ic", "icir", "rank_ic", "rank_icir")}
    result["metrics"]["turnover"] = float(np.nanmean(turnover)) if np.isfinite(turnover).any() else None
    result["daily"] = [{"date": str(days[index]), "ic": ic["ic_series"][row], "rank_ic": ic["rank_ic_series"][row],
                        "n_samples": int((np.isfinite(factor[row]) & np.isfinite(labels[index])).sum()),
                        "raw_turnover": turnover[row]} for row, index in enumerate(indices)]
    result["scores"] = [{"date": str(days[index]), "instrument": symbol, "score": factor[row, column]}
                        for row, index in enumerate(indices) for column, symbol in enumerate(engine.symbols)
                        if engine.membership[index, column]]
    result["raw_factor_report"] = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in ic.items()}
    result["label"] = {"horizons": [protocol.forward_n], "execution": label_execution,
                       "semantics": "single horizon; native Assay forward_returns", "read_end": str(days[-1]),
                       "boundary": "source_panel_tail" if protocol.end_inclusive else "purged_at_split",
                       "unavailable_tail_dates": [str(days[i]) for i in unavailable_tail] if protocol.end_inclusive else [],
                       "purged_dates": [str(days[i]) for i in purge]}
    if not request["fast"]:
        if protocol.assay_portfolio.get("benchmark_symbol") != manifest["benchmark"]:
            raise ValueError("portfolio benchmark differs from the frozen data asset")
        if source and protocol.market.startswith("csi"):
            engine, _, tradable, execution, benchmark, _ = prepare_panel(request, protocol, config, portfolio=True)
            portfolio_arrays = [np.where(engine.membership, engine.evaluate(value).values, np.nan) for value in canonical]
            factor = combine_scores(portfolio_arrays)[indices] if request["operation"] == "combine" else portfolio_arrays[0][indices]
        result["portfolio"] = portfolio_result(protocol, engine, factor, indices, execution, tradable, benchmark,
                                                canonical, request["start"], request["end"])
    return result
