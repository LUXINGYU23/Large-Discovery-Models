"""One bounded physical oracle worker; outputs are committed before service reply."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.contracts.evaluation import EvaluationPaused
from .data import verify_data_files
from .grammar import parse_expression
from .protocol import T3Protocol, digest


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean(value):
    import numpy as np
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def qlib_evaluate(request, config):
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.data import D
    from .qlib_operators import CUSTOM_OPS

    protocol = T3Protocol(**request["protocol"])
    root = Path(config["upstream_root"])
    utils = load_source("t3_ffo_utils", root / "ffo/utils/utils.py")
    qlib.init(provider_uri=config["data_root"], region="cn" if protocol.market.startswith("csi") else "us",
              custom_ops=CUSTOM_OPS, expression_cache=None, dataset_cache=None, kernels=1)
    calendar = list(D.calendar())
    if not calendar or str(calendar[0])[:10] > request["start"] or str(calendar[-1])[:10] < request["end"]:
        raise EvaluationPaused("calendar does not bracket the complete requested interval", status="paused_data_coverage")
    dates = [day for day in calendar if str(day)[:10] >= request["start"]
             and (str(day)[:10] <= request["end"] if protocol.end_inclusive else str(day)[:10] < request["end"])]
    if not dates:
        raise ValueError("no trading sessions in the requested interval")
    expressions = request.get("expressions", [request.get("expression")])
    for expression in expressions:
        parse_expression(expression, backend="qlib")
    instruments = D.instruments(protocol.market)
    features = D.features(instruments, expressions,
                          start_time=dates[0], end_time=dates[-1])
    if features.empty:
        raise ValueError("no observations for the historical universe")
    scores = features.iloc[:, :len(expressions)]
    if request["operation"] == "combine":
        scores = scores.groupby(level="datetime").transform(lambda frame: (frame - frame.mean()) / frame.std(ddof=1))
        factor = scores.mean(axis=1)
    else:
        factor = scores.iloc[:, 0]
    finite = np.isfinite(factor)
    result = {"success": bool(finite.any()), "nan_ratio": float(factor.isna().mean()),
              "non_finite_ratio": float((~finite).mean()),
              "actual_start": str(dates[0])[:10], "actual_end": str(dates[-1])[:10],
              "interval": {"start": request["start"], "end": request["end"], "end_inclusive": protocol.end_inclusive},
              "metrics": {}, "daily": [], "scores": [], "portfolio": None}
    if request["operation"] == "check":
        result.update(utils._check_single_column(expressions[0], factor))
        result["check_kind"] = "dynamic"
        if not result["success"]:
            result["error"] = result["error_message"]
        return result
    kept_dates = dates if protocol.end_inclusive else dates[:-protocol.forward_n]
    if not kept_dates:
        raise ValueError("no sessions remain after forward-label purge")
    purge = [] if protocol.end_inclusive else [str(day)[:10] for day in dates[-protocol.forward_n:]]
    label_end = calendar.index(kept_dates[-1]) + protocol.forward_n
    if label_end >= len(calendar):
        raise EvaluationPaused("calendar lacks future sessions required by the source forward labels", status="paused_data_coverage")
    labels = [item[1] for item in utils._build_forward_label_exprs(protocol.forward_n)]
    # Query only retained label rows; matched labels never read beyond their split.
    label_values = D.features(instruments, labels, start_time=kept_dates[0], end_time=kept_dates[-1])
    kept = features.index.get_level_values("datetime").isin(kept_dates)
    factor = factor[kept].replace([np.inf, -np.inf], np.nan)
    label_values = label_values.reindex(factor.index)
    ic, rank_ic, counts = [], [], []
    for offset in range(len(labels)):
        label = label_values.iloc[:, offset]
        a, b = utils._daily_ic_rankic(factor, label)
        ic.append(a); rank_ic.append(b)
        counts.append((np.isfinite(factor) & np.isfinite(label)).groupby(level="datetime").sum())
    ic_daily = pd.concat(ic, axis=1).mean(axis=1)
    rank_daily = pd.concat(rank_ic, axis=1).mean(axis=1)
    if not ic_daily.notna().any():
        raise ValueError("no finite daily cross-sectional correlations")
    result["metrics"] = utils.summarize_ic_rankic(ic_daily, rank_daily)
    turnover = utils._daily_turnover(factor)
    result["metrics"]["turnover"] = float(turnover.mean())
    result["daily"] = [{"date": str(day)[:10], "ic": value, "rank_ic": rank_daily.get(day),
                        "n_samples_by_horizon": {str(k + 1): int(count.get(day, 0)) for k, count in enumerate(counts)},
                        "raw_turnover": turnover.get(day)} for day, value in ic_daily.items()]
    result["scores"] = [{"instrument": str(instrument), "date": str(day)[:10], "score": value}
                        for (instrument, day), value in factor.items()]
    result["label"] = {"expressions": labels, "semantics": "forward daily correlations averaged across horizons",
                       "purged_dates": purge, "read_end": str(calendar[label_end])[:10],
                       "boundary": "source_forward_reads" if protocol.end_inclusive else "purged_at_split"}
    if not request["fast"]:
        portfolio_module = load_source("t3_portfolio", root / "ffo/backtest/qlib/single_alpha_backtest.py")
        cn = protocol.market.startswith("csi")
        exchange = {"limit_threshold": .095 if cn else None, "deal_price": "close",
                    "open_cost": .0005 if cn else .0001, "close_cost": .0015 if cn else .0001, "min_cost": 5. if cn else 0.}
        analysis, report, positions = portfolio_module.backtest_by_scores(factor * protocol.direction,
            topk=protocol.stock_topk, n_drop=protocol.stock_n_drop, start_time=str(kept_dates[0])[:10],
            end_time=str(kept_dates[-1])[:10], data_path=config["data_root"], region="cn" if cn else "us",
            BENCH=config["benchmark"].upper(), exchange_kwargs=exchange)
        daily = [{"date": str(day)[:10], **row.to_dict()} for day, row in report.iterrows()]
        holdings, actions, previous = {}, [], {}
        for day, position in sorted(positions.items()):
            current = {stock: float(position.get_stock_amount(stock)) for stock in position.get_stock_list()}
            weights = position.get_stock_weight_dict(only_stock=False)
            holdings[str(day)[:10]] = [{"instrument": stock, "amount": amount, "weight": weights[stock]}
                                      for stock, amount in sorted(current.items())]
            for stock in sorted(set(previous) | set(current)):
                delta = current.get(stock, 0) - previous.get(stock, 0)
                if delta:
                    actions.append({"date": str(day)[:10], "instrument": stock, "amount_delta": delta})
            previous = current
        result["portfolio"] = {"daily": daily, "holdings": holdings, "actions": actions,
            "analysis": {name: frame["risk"].to_dict() for name, frame in analysis.groupby(level=0)
                         for frame in [frame.droplevel(0)]}, "units": "fraction; Qlib annualization",
            "execution": {"strategy": "TopkDropoutStrategy", "topk": protocol.stock_topk,
                          "n_drop": protocol.stock_n_drop, "exchange": exchange}}
    return clean(result)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-digest", required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    started = time.monotonic()
    try:
        protocol = T3Protocol(**request["protocol"])
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if digest(config) != args.config_digest:
                raise ValueError("oracle configuration changed after service startup")
            if any(config[key] != getattr(protocol, key) for key in
                   ("backend", "market", "data_digest", "environment_digest")):
                raise ValueError("oracle worker config differs from the request")
            manifest_path = Path(config["data_manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (digest(manifest) != protocol.data_digest
                    or (manifest.get("backend"), manifest.get("market")) != (protocol.backend, protocol.market)):
                raise ValueError("oracle worker data manifest identity mismatch")
            if protocol.backend == "qlib":
                if (Path(config["data_root"]).resolve() != Path(manifest["data_root"]).resolve()
                        or config["benchmark"].lower() != manifest["benchmark"].lower()):
                    raise ValueError("Qlib worker data root or benchmark differs from the manifest")
                verify_data_files(manifest_path, manifest)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise EvaluationPaused("oracle data/config integrity failure: " + str(exc),
                                   status="paused_data_integrity") from exc
        if protocol.backend == "assay":
            from .assay_adapter import assay_evaluate
            result = assay_evaluate(request, config)
        else:
            result = qlib_evaluate(request, config)
    except EvaluationPaused as exc:
        result = {"success": False, "pause_status": exc.status, "error": str(exc),
                  "metrics": {}, "daily": [], "scores": [], "portfolio": None}
    except (ValueError, SyntaxError, FloatingPointError) as exc:
        result = {"success": False, "error": str(exc), "metrics": {}, "daily": [], "scores": [], "portfolio": None}
    if request["operation"] == "check":
        result["check_kind"] = T3Protocol(**request["protocol"]).check_kind
    result.update(request_id=request["request_id"], elapsed_seconds=time.monotonic() - started,
                  jobs=[{"job_id": request["request_id"] + ":0", "operation": request["operation"]}])
    atomic_json_write(args.output, clean(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
