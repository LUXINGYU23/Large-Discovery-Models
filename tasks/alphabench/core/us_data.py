"""Offline coverage audit of the pinned US prices and corporate-action responses."""

import bisect
import json
import math

from ldm_tts.engine.run_store import atomic_json_write
from .data import SOURCES, fetch_plan_page, sha256
from .protocol import digest


def audit_us(root, sessions):
    dates = [str(day) for day in sessions]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("US trading calendar must be nonempty, sorted and unique")
    plan_path = root / "manifests/us-download-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["source"] != SOURCES["us"] or plan["period"] != SOURCES["period"]:
        raise ValueError("US download plan differs from the frozen source/period")
    members = {market: json.loads((root / "raw" / (market + "-membership.json")).read_text(encoding="utf-8"))
               for market in ("sp500", "nasdaq100")}
    if plan["membership_digests"] != {market: digest(value) for market, value in members.items()}:
        raise ValueError("US historical membership differs from the download plan")
    expected_symbols = set().union(*(set(value["symbols"]) for value in members.values()))
    observed = {symbol: set() for symbol in expected_symbols}
    pages, metadata, actions = {}, {}, {"split": {}, "dividend": {}}
    issues, invalid, non_session = [], [], []
    date_set = set(dates)
    price_rows = 0
    for item in plan["requests"]:
        rows, hashes = fetch_plan_page(root, plan, item, offline=True)
        if set(pages).intersection(hashes):
            raise ValueError("duplicate SQL request in US plan")
        pages.update(hashes)
        for row in rows:
            symbol = row["act_symbol"]
            if symbol not in observed:
                raise ValueError("US price or event is outside historical membership")
            table = item["table"]
            if table == "ohlcv":
                date = row["date"]
                if date in observed[symbol]:
                    raise ValueError("duplicate security/date price row")
                observed[symbol].add(date)
                price_rows += 1
                if date not in date_set:
                    non_session.append({"symbol": symbol, "date": date})
                values = {field: float(row[field]) for field in ("open", "high", "low", "close", "volume")}
                if (any(not math.isfinite(v) for v in values.values()) or values["volume"] < 0
                    or min(values[k] for k in ("open", "high", "low", "close")) <= 0
                    or values["high"] < max(values["open"], values["close"], values["low"])
                    or values["low"] > min(values["open"], values["close"])):
                    invalid.append({"symbol": symbol, "date": date, "values": values})
            elif table == "symbol":
                if symbol in metadata:
                    raise ValueError("duplicate security metadata")
                metadata[symbol] = row
            elif table in actions:
                date = row["ex_date"]
                if not plan["period"]["start"] <= date < plan["period"]["end_exclusive"]:
                    raise ValueError("corporate action outside the frozen period")
                key = symbol + ":" + date
                if key in actions[table]:
                    raise ValueError("duplicate corporate action identity")
                actions[table][key] = row
            else:
                raise ValueError("unexpected US data table")
    if invalid:
        issues.append({"code": "invalid_ohlcv", "rows": len(invalid)})
    if non_session:
        issues.append({"code": "prices_outside_exchange_sessions", "rows": len(non_session)})
    missing_metadata = sorted(expected_symbols - metadata.keys())
    if missing_metadata:
        issues.append({"code": "missing_security_metadata", "symbols": missing_metadata})
    # These assets are not present in this source; an empty action result is not proof of no actions.
    issues.extend({"code": code} for code in ("missing_vwap", "missing_actual_index_benchmarks",
        "unverified_security_lineage", "unverified_corporate_action_completeness_and_knowledge_dates",
        "missing_historical_industry_classification"))
    markets = {}
    for market, membership in members.items():
        snapshots = membership["snapshots"]
        effective = [row["effective_date"] for row in snapshots]
        if effective != sorted(set(effective)):
            raise ValueError("membership effective dates must be sorted and unique")
        missing = {}
        for day in dates:
            index = bisect.bisect_right(effective, day) - 1
            if index < 0 or not snapshots[index]["symbols"]:
                raise ValueError("missing historical universe snapshot")
            for symbol in snapshots[index]["symbols"]:
                if day not in observed[symbol]:
                    missing.setdefault(symbol, []).append(day)
        market_issues = list(issues)
        if missing:
            market_issues.append({"code": "unresolved_price_gaps", "symbols": len(missing),
                                  "sessions": sum(map(len, missing.values()))})
        report = {"source": SOURCES["us"], "market": market, "qualification": "blocked",
            "plan_sha256": sha256(plan_path), "membership_digest": digest(membership),
            "calendar_digest": digest(dates), "calendar": dates,
            "response_hashes_digest": digest(pages), "pages": len(pages), "price_rows": price_rows,
            "action_rows": {table: len(rows) for table, rows in actions.items()},
            "issues": market_issues, "gaps": missing, "invalid_prices": invalid, "non_session_prices": non_session}
        atomic_json_write(root / "manifests" / ("us_" + market + ".json"), report)
        markets[market] = {"qualification": "blocked", "issues": market_issues}
    atomic_json_write(root / "manifests/us_responses.files.json", pages)
    return {"price_rows": price_rows, "pages": len(pages), "markets": markets}
