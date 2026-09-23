"""Capture missing CN trading-status evidence locally and audit it offline."""

import datetime as dt
import importlib.metadata
import json
import re
import socket

from ldm_tts.engine.run_store import atomic_json_write
from .data import SOURCES, BENCHMARKS, audit_qlib, instrument_bounds, load_cn_status, sha256
from .protocol import digest

SOURCE = SOURCES["cn_baostock"]
DIRECTORY = "raw/cn-baostock-" + SOURCE["capture"]
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"


def plan_baostock(root):
    path = root / DIRECTORY / "plan.json"
    if path.exists():
        plan = read_plan(root)
    else:
        gaps, inputs = {}, {}
        for market in BENCHMARKS:
            audit_path = root / "manifests" / f"qlib_{market}.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit["archive_sha256"] != SOURCES["cn"]["archive_sha256"]:
                raise ValueError("CN gap audit belongs to a different archive")
            inputs[market] = sha256(audit_path)
            for gap in audit["gaps"]:
                gaps.setdefault(gap["symbol"], set()).update(gap["dates"])
        requests = []
        for symbol, dates in sorted(gaps.items()):
            if not re.fullmatch(r"(?:sh|sz)[0-9]{6}", symbol):
                raise ValueError("invalid BaoStock security identity")
            code = symbol[:2] + "." + symbol[2:]
            requests.append({"operation": "basic", "code": code})
            # Each request is smaller than the client's 2,000-row pagination boundary.
            for year in sorted({day[:4] for day in dates}):
                days = sorted(day for day in dates if day[:4] == year)
                requests.append({"operation": "history", "code": code, "fields": FIELDS,
                    "start_date": days[0], "end_date": days[-1], "frequency": "d", "adjustflag": "3"})
        plan = {"source": SOURCE, "input_audit_hashes": inputs, "requests": requests}
        atomic_json_write(path, plan)
    return {"plan": str(path), "requests": len(plan["requests"])}


def read_plan(root):
    plan = json.loads((root / DIRECTORY / "plan.json").read_text(encoding="utf-8"))
    if plan["source"] != SOURCE or len({digest(item) for item in plan["requests"]}) != len(plan["requests"]):
        raise ValueError("BaoStock plan source or request identity mismatch")
    for item in plan["requests"]:
        if not re.fullmatch(r"(?:sh|sz)\.[0-9]{6}", item["code"]):
            raise ValueError("invalid BaoStock security identity")
        if item["operation"] == "history":
            start, end = [dt.date.fromisoformat(item[key]) for key in ("start_date", "end_date")]
            if (start > end or start.year != end.year or item["fields"] != FIELDS
                or item["frequency"] != "d" or item["adjustflag"] != "3"):
                raise ValueError("invalid BaoStock history request")
        elif item["operation"] != "basic":
            raise ValueError("unsupported BaoStock request")
    return plan


def response_rows(packet, request):
    if packet["source"] != SOURCE or packet["request"] != request or packet["error_code"] != "0":
        raise ValueError("BaoStock response failed or differs from its request")
    fields = packet["fields"]
    if len(set(fields)) != len(fields) or any(len(row) != len(fields) for row in packet["rows"]):
        raise ValueError("malformed BaoStock rows")
    rows = [dict(zip(fields, row)) for row in packet["rows"]]
    if any(row["code"] != request["code"] for row in rows):
        raise ValueError("BaoStock response has a different security")
    if request["operation"] == "history":
        if fields != FIELDS.split(",") or len(rows) > 366:
            raise ValueError("unexpected BaoStock history fields or row count")
        dates = [row["date"] for row in rows]
        if dates != sorted(set(dates)) or any(not request["start_date"] <= day <= request["end_date"] for day in dates):
            raise ValueError("BaoStock response has duplicate or out-of-range dates")
        if any(row["tradestatus"] not in {"0", "1"} or row["adjustflag"] != "3" for row in rows):
            raise ValueError("unknown BaoStock status or price basis")
    elif len(rows) > 1 or not {"code", "ipoDate", "outDate"}.issubset(fields):
        raise ValueError("unexpected BaoStock security metadata")
    return rows


def acquire_baostock(root):
    import baostock as bs
    if importlib.metadata.version("baostock") != SOURCE["client_version"]:
        raise ValueError("BaoStock client differs from the pinned version")
    plan = read_plan(root)
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(20)
    logged_in = False
    completed = 0
    try:
        for request in plan["requests"]:
            path = root / DIRECTORY / (digest(request) + ".json")
            if path.exists():
                response_rows(json.loads(path.read_text(encoding="utf-8")), request)
            else:
                if not logged_in:
                    login = bs.login()
                    if login.error_code != "0":
                        raise ConnectionError("BaoStock login failed: " + login.error_msg)
                    logged_in = True
                params = {key: value for key, value in request.items() if key != "operation"}
                result = (bs.query_stock_basic(**params) if request["operation"] == "basic"
                          else bs.query_history_k_data_plus(**params))
                rows = []
                while result.error_code == "0" and result.next():
                    rows.append(result.get_row_data())
                packet = {"source": SOURCE, "request": request, "error_code": result.error_code,
                    "error_msg": result.error_msg, "fields": result.fields, "rows": rows,
                    "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
                try:
                    response_rows(packet, request)
                except (ValueError, KeyError):
                    atomic_json_write(path.with_name(path.stem + ".failed-" + digest(packet) + ".json"), packet)
                    raise
                atomic_json_write(path, packet)
            completed += 1
            if completed % 20 == 0:
                print(json.dumps({"completed": completed, "planned": len(plan["requests"])}), flush=True)
    finally:
        if logged_in:
            bs.logout()
        socket.setdefaulttimeout(previous_timeout)
    return {"completed": completed, "planned": len(plan["requests"]), "qualification": "unverified"}


def prepare_baostock(root):
    plan = read_plan(root)
    dolt = load_cn_status(root)
    bounds = instrument_bounds(root / "qlib" / ("cn-" + SOURCES["cn"]["release"]) / "instruments/all.txt")
    suspended = {(row["symbol"], row["tradedate"]) for row in dolt["rows"] if str(row["tradestatus"]) == "0"}
    initial = len(suspended)
    pages, metadata, history = {}, {}, {}
    for request in plan["requests"]:
        path = root / DIRECTORY / (digest(request) + ".json")
        rows = response_rows(json.loads(path.read_text(encoding="utf-8")), request)
        pages[path.relative_to(root).as_posix()] = sha256(path)
        for row in rows:
            symbol = row["code"].replace(".", "").upper()
            if request["operation"] == "basic":
                metadata[symbol] = row
            else:
                identity = symbol, row["date"]
                if identity in history:
                    raise ValueError("overlapping BaoStock history responses")
                history[identity] = row
                if row["tradestatus"] == "0":
                    suspended.add(identity)
    plan_hash = sha256(root / DIRECTORY / "plan.json")
    sources = [{**SOURCES["cn_status"], "response_hashes_digest": digest(dolt["pages"])},
               {**SOURCE, "plan_sha256": plan_hash, "response_hashes_digest": digest(pages)}]
    result, unresolved = {}, []
    for market in BENCHMARKS:
        audit, files = audit_qlib(root / "qlib" / ("cn-" + SOURCES["cn"]["release"]), market,
            source=SOURCES["cn"], archive_hash=SOURCES["cn"]["archive_sha256"], suspended=suspended,
            suspension_sources=sources)
        for gap in audit["gaps"]:
            symbol = gap["symbol"].upper()
            out_date = metadata.get(symbol, {}).get("outDate", "")
            archived = bounds.get(symbol)
            for day in gap["dates"]:
                row = history.get((symbol, day))
                unresolved.append({"market": market, "symbol": symbol, "date": day,
                    "source_status": row["tradestatus"] if row else None,
                    "reported_out_date": out_date or None,
                    "archive_instrument_start": archived[0] if archived else None,
                    "archive_instrument_end": archived[1] if archived else None,
                    "reason": "membership_before_archive_instrument_interval" if archived and day < archived[0]
                              else "membership_at_or_after_reported_delisting" if out_date and day >= out_date
                              else "membership_after_archive_instrument_interval" if archived and day > archived[1]
                              else "missing_archive_price_on_reported_trading_day" if row else "no_status_evidence"})
        atomic_json_write(root / "manifests" / f"qlib_{market}.json", audit)
        atomic_json_write(root / "manifests" / f"qlib_{market}.files.json", files)
        result[market] = {"qualification": audit["qualification"], "gap_symbols": len(audit["gaps"]),
                          "unresolved_sessions": sum(gap["missing_sessions"] for gap in audit["gaps"])}
    evidence = {"source": SOURCE, "plan_sha256": plan_hash, "pages": pages,
        "dolt_response_hashes_digest": digest(dolt["pages"]), "additional_suspensions": len(suspended) - initial,
        "unresolved": unresolved, "markets": result}
    atomic_json_write(root / "manifests/cn_baostock_status.json", evidence)
    return {"pages": len(pages), "additional_suspensions": len(suspended) - initial, "markets": result}
