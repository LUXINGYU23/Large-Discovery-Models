"""Offline Assay execution with historical cross-sections and explicit assets."""

from dataclasses import fields
import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from assay.data.store.adjust import forward_adjust
from assay.engine import FactorEngine
from assay.engine.ast import OpNode
from assay.engine.engine import EvalContext
from assay.engine import operators
from assay.engine.parsing import parse
from assay.evaluator.forward_returns import forward_returns
from assay.evaluator.metrics import evaluate_ic
from assay.evaluator.turnover import rank_autocorr
from assay.portfolio.backtester import PortfolioBacktester
from assay.portfolio.config import PortfolioBacktestConfig

from .data import sha256
from .grammar import parse_expression
from .protocol import T3Protocol, digest


ASSETS = {"calendar", "prices", "events", "membership", "groups", "execution", "benchmark"}

# The pinned registry lacks these guide operators. They share the exact guide
# semantics of the task's Qlib extensions, without rewriting either dialect.
operators.register("Tanh", np.tanh, 1, 1)
operators.register("Mask", lambda condition, value: np.where(condition, value, np.nan), 2, 2)


def read_assets(config, protocol):
    path = Path(config["data_manifest"]).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if digest(manifest) != protocol.data_digest or (manifest["backend"], manifest["market"]) != ("assay", protocol.market):
        raise ValueError("Assay data manifest differs from the frozen protocol")
    inventory = manifest["assets"]
    if set(inventory) != ASSETS:
        raise ValueError("Assay requires all seven offline data assets")
    paths = {}
    for name, record in inventory.items():
        declared = path.parent / record["path"]
        file = declared.resolve()
        if not file.is_relative_to(path.parent) or declared.is_symlink() or sha256(file) != record["sha256"]:
            raise ValueError("Assay asset path or content mismatch: " + name)
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


def prepare_panel(request, protocol, config):
    manifest, calendar, assets = read_assets(config, protocol)
    start, end = dt.date.fromisoformat(request["start"]), dt.date.fromisoformat(request["end"])
    if calendar[0] > start or calendar[-1] < end:
        raise ValueError("calendar does not bracket the complete requested interval")
    days = [day for day in calendar if day < end]
    if not days or not any(day >= start for day in days):
        raise ValueError("no trading sessions in the requested interval")
    members = assets["membership"]
    unique(members, ["date", "symbol"], "membership")
    known_by(members, "date", "membership")
    members = members.filter(pl.col("date").is_in(days))
    symbols = sorted(members["symbol"].unique().to_list())
    if not symbols:
        raise ValueError("empty historical universe")
    grid = pl.DataFrame({"date": days}).join(pl.DataFrame({"symbol": symbols}), how="cross")
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
    events = events.filter((pl.col("ex_date") <= days[-1]) & pl.col("symbol").is_in(symbols))
    known_by(events, "ex_date", "corporate action")
    basis = "total" if protocol.market.startswith("csi") else "split"
    if manifest["adjustment"] != basis:
        raise ValueError("Assay adjustment differs from the fixed market basis")
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


def portfolio_result(protocol, engine, factor, indices, execution, tradable, benchmark, expressions):
    settings = protocol.assay_portfolio
    if set(settings) != {item.name for item in fields(PortfolioBacktestConfig)}:
        raise ValueError("freeze the full independent PortfolioBacktestConfig in assay_portfolio")
    settings = dict(settings)
    settings.update(period_start=str(engine.dates[indices[0]])[:10], period_end=str(engine.dates[indices[-1]])[:10],
                    as_of_date=str(engine.dates[indices[-1]])[:10])
    config = PortfolioBacktestConfig(**settings)
    market = "A" if protocol.market.startswith("csi") else "US"
    if config.market != market or config.universe != protocol.market.upper():
        raise ValueError("Assay portfolio market/universe differs from factor evaluation")
    if config.benchmark != "custom" or not config.benchmark_symbol:
        raise ValueError("Assay portfolio requires an actual custom index benchmark")
    if config.execution_price not in {"next_open", "next_close"}:
        raise ValueError("only actual open/close execution is supported by the pinned portfolio engine")
    if config.warmup_days or not config.save_trade_log or not config.save_position_log or config.output_frequency != "daily":
        raise ValueError("portfolio must retain all daily NAV, trades and positions after factor warmup")
    if config.sector_neutral or config.include_bid_ask or config.northbound_flow_filter or config.sz_sh_connect_only or config.inclusion_anticipation:
        raise ValueError("portfolio requests controls without an implemented data input")
    if config.slippage_model != "zero" or config.new_listing_lockout_days or config.ipo_lockout_days or config.rebalance_around_index or config.st_filter:
        raise ValueError("portfolio requests impact or filters without an implemented data input")
    backtester = PreparedPortfolioBacktester(engine, factor * protocol.direction, indices, execution)
    signal_contract = {"expressions": expressions, "direction": protocol.direction,
                       "combination": "single" if len(expressions) == 1 else "daily_zscore_equal_weight_mean_finite"}
    identity = json.dumps(signal_contract, sort_keys=True)
    raw = backtester.run(identity, config, as_of=config.as_of_date,
                         tradable_mask=tradable[indices], benchmark=benchmark[indices]).to_dict()
    if not raw["nav_series"] or raw["n_trading_days"] != len(indices) or raw["warnings"]:
        raise ValueError("Assay portfolio did not produce a complete daily report: " + str(raw.get("attribution")))
    return {"raw": raw, "signal_contract": signal_contract,
            "daily": [{"date": day, "nav": nav, "benchmark_nav": bench}
            for day, nav, bench in zip(raw["nav_dates"], raw["nav_series"], raw["benchmark_series"])],
            "holdings": raw["position_log"], "actions": raw["trade_log"],
            "units": "returns and drawdown in fractions; source annualization 252 sessions",
            "execution": {"backend": "Assay PortfolioBacktester", "config": raw["config"],
                          "qlib_topk_drop_equivalence": False}}


def assay_evaluate(request, config):
    protocol = T3Protocol(**request["protocol"])
    expressions = request.get("expressions", [request.get("expression")])
    if not expressions or request["operation"] not in {"check", "evaluate", "combine"}:
        raise ValueError("a supported operation with at least one expression is required")
    canonical = [parse_expression(value, backend="assay").canonical for value in expressions]
    engine, days, tradable, execution, benchmark, manifest = prepare_panel(request, protocol, config)
    start = dt.date.fromisoformat(request["start"])
    indices = np.array([i for i, day in enumerate(days) if day >= start])
    if indices[0] < max(lookback(parse(value)) for value in canonical):
        raise ValueError("calendar lacks the required expression warmup")
    arrays = [np.where(engine.membership, engine.evaluate(value).values, np.nan) for value in canonical]
    if request["operation"] == "combine":
        arrays = [operators.get("cs_zscore").fn(value) for value in arrays]
        stacked = np.stack(arrays)
        count = np.isfinite(stacked).sum(axis=0)
        factor = np.divide(np.nansum(stacked, axis=0), count, out=np.full(engine._shape, np.nan), where=count > 0)
    else:
        factor = arrays[0]
    active = engine.membership[indices]
    finite = np.isfinite(factor[indices])
    result = {"success": bool((active & finite).any()), "nan_ratio": float(1 - finite[active].mean()),
              "actual_start": str(days[indices[0]]), "actual_end": str(days[indices[-1]]),
              "metrics": {}, "daily": [], "scores": [], "portfolio": None,
              "data_digest": digest(manifest), "history_origin": str(days[0]),
              "n_dates": len(indices), "n_observations": int(active.sum()), "n_finite": int((active & finite).sum())}
    if request["operation"] == "check":
        return result
    label_execution = "next_open" if protocol.label == "open_return" else "next_close"
    farthest = protocol.forward_n + (label_execution == "next_open")
    purge = indices[-farthest:]
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
                       "semantics": "single horizon; native Assay forward_returns", "end_exclusive": request["end"],
                       "purged_dates": [str(days[i]) for i in purge]}
    if not request["fast"]:
        if protocol.assay_portfolio.get("benchmark_symbol") != manifest["benchmark"]:
            raise ValueError("portfolio benchmark differs from the frozen data asset")
        result["portfolio"] = portfolio_result(protocol, engine, factor, indices, execution, tradable, benchmark, canonical)
    return result
