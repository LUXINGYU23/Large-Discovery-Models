"""Reproducible public-data acquisition and explicit full-period coverage audits.

Acquire on the local host; install and audit offline on the remote host.
Raw downloads and audit reports are retained even when qualification fails.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import http.client
import importlib.util
import json
import re
from pathlib import Path
import shutil
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

from ldm_tts.engine.run_store import atomic_json_write
from .protocol import T3Protocol, digest

SOURCES = json.loads((Path(__file__).resolve().parents[1] / "resources/data_sources.json").read_text(encoding="utf-8"))
BENCHMARKS = {"csi300": "sh000300", "csi500": "sh000905", "csi1000": "sh000852"}
ASSAY_ASSETS = {"calendar", "prices", "events", "membership", "groups", "execution", "benchmark"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_data_manifest(path: Path, protocol: T3Protocol):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("qualification") != "qualified":
        raise ValueError("data qualification is incomplete; inspect the recorded coverage issues")
    for key in ("source", "archive_sha256", "files_sha256", "calendar_sha256", "universe_sha256",
                "adjustment", "fields", "start", "end", "benchmark", "historical_universe"):
        if not manifest.get(key):
            raise ValueError(f"data manifest lacks {key}")
    if manifest["historical_universe"] is not True:
        raise ValueError("current constituents cannot replace a historical universe")
    if manifest.get("market") != protocol.market or manifest.get("backend") != protocol.backend:
        raise ValueError("data manifest backend/market mismatch")
    if manifest["start"] > "2015-01-01" or manifest["end"] < "2025-01-01":
        raise ValueError("data does not cover lookback, all splits, and forward labels")
    if digest(manifest) != protocol.data_digest:
        raise ValueError("data manifest does not match the frozen protocol")
    if protocol.backend == "qlib" and any(not manifest.get(key) for key in ("data_root", "all_instruments_sha256")):
        raise ValueError("Qlib data manifest lacks its data root or all-instruments hash")

    def check_file(root, relative, expected):
        declared = root / relative_path(relative)
        target = declared.resolve()
        if declared.is_symlink() or not target.is_relative_to(root) or not target.is_file() or sha256(target) != expected:
            raise ValueError("data asset content mismatch: " + relative)

    if protocol.backend == "qlib":
        root = Path(manifest["data_root"]).resolve()
        files = json.loads(path.with_suffix(".files.json").read_text(encoding="utf-8"))
        if not isinstance(files, dict) or not files or digest(files) != manifest["files_sha256"]:
            raise ValueError("Qlib file inventory differs from the frozen data manifest")
        required = {"calendars/day.txt": manifest["calendar_sha256"],
                    f"instruments/{protocol.market}.txt": manifest["universe_sha256"],
                    "instruments/all.txt": manifest["all_instruments_sha256"]}
        for relative, expected in (files | required).items():
            check_file(root, relative, expected)
    else:
        assets = manifest.get("assets")
        if not isinstance(assets, dict) or set(assets) != ASSAY_ASSETS:
            raise ValueError("Assay requires all seven frozen data assets")
        files = {record["path"]: record["sha256"] for record in assets.values()}
        if (len(files) != len(assets) or digest(files) != manifest["files_sha256"]
                or assets["calendar"]["sha256"] != manifest["calendar_sha256"]
                or assets["membership"]["sha256"] != manifest["universe_sha256"]):
            raise ValueError("Assay file inventory differs from the frozen data manifest")
        for relative, expected in files.items():
            check_file(path.parent, relative, expected)
    return manifest


def fetch_file(request, destination, *, content_range=None):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
                if content_range is not None and (response.status != 206 or response.headers.get("Content-Range") != content_range):
                    raise ValueError("server did not honor the byte range")
                shutil.copyfileobj(response, output, length=1024 * 1024)
                expected_size = response.headers.get("Content-Length")
                if expected_size is not None and output.tell() != int(expected_size):
                    raise ConnectionError("response body is shorter than Content-Length")
            return
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead,
                http.client.RemoteDisconnected, ConnectionError, ssl.SSLError) as exc:
            if attempt == 2 or isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                raise
            time.sleep(attempt + 1)


def download(url, path, expected=None, *, size=None, connections=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = sha256(path)
        if expected and actual != expected:
            raise ValueError(f"existing download hash mismatch: {path}")
        return actual
    partial = path.with_suffix(path.suffix + ".part")
    if size and connections > 1:
        pieces = path.parent / (path.name + ".chunks")
        pieces.mkdir(exist_ok=True)
        chunk_size = 4 * 1024 * 1024
        ranges = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]

        def get_chunk(bounds):
            start, end = bounds
            target = pieces / str(start)
            if target.exists() and target.stat().st_size == end - start + 1:
                return target
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            temporary = target.with_suffix(".part")
            fetch_file(request, temporary, content_range=f"bytes {start}-{end}/{size}")
            if temporary.stat().st_size != end - start + 1:
                raise ValueError("truncated range download")
            temporary.replace(target)
            return target

        with ThreadPoolExecutor(max_workers=connections) as pool:
            completed = list(pool.map(get_chunk, ranges))
        with partial.open("wb") as output:
            for item in completed:
                with item.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
    if not (partial.exists() and expected and sha256(partial) == expected):
        request = urllib.request.Request(url, headers={"User-Agent": "AlphaBench-T3-data/1.0"})
        fetch_file(request, partial)
    actual = sha256(partial)
    if expected and actual != expected:
        raise ValueError(f"download hash mismatch: {path}")
    partial.replace(path)
    return actual


def extract_archive(archive, destination, *, root_marker="calendars/day.txt"):
    """Reject links, devices, absolute paths, and traversal before extraction."""
    destination = Path(destination)
    marker = destination / ".archive_sha256"
    archive_hash = sha256(archive)
    if marker.exists():
        if marker.read_text(encoding="utf-8").strip() != archive_hash:
            raise ValueError("dataset directory belongs to another archive")
        return
    if destination.exists():
        raise ValueError("incomplete extraction: retain it for inspection and choose a new directory")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for member in members:
            relative_path(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsafe archive member: {member.name}")
        markers = sorted([m for m in members if m.name.endswith(root_marker)], key=lambda m: len(Path(m.name).parts))
        if not markers or (len(markers) > 1 and len(Path(markers[0].name).parts) == len(Path(markers[1].name).parts)):
            raise ValueError("expected one unambiguous archive root marker")
        prefix = markers[0].name.removesuffix(root_marker)
        destination.mkdir(parents=True)
        for member in members:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix):]
            if not relative:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as src, target.open("wb") as output:
                shutil.copyfileobj(src, output)
    marker.write_text(archive_hash + "\n")


def binary_series(path, length):
    import numpy as np
    result = np.full(length, np.nan)
    if not path.exists():
        return result
    raw = np.fromfile(path, dtype="<f4")
    if len(raw) < 1 or not np.isfinite(raw[0]) or int(raw[0]) != raw[0] or raw[0] < 0:
        raise ValueError(f"malformed Qlib binary: {path}")
    start = int(raw[0])
    if start + len(raw) - 1 > length:
        raise ValueError(f"Qlib binary exceeds calendar: {path}")
    result[start:start + len(raw) - 1] = raw[1:]
    return result


def instrument_bounds(path):
    bounds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol, start, end = line.split()
        symbol = symbol.upper()
        if symbol in bounds or start > end:
            raise ValueError("duplicate or reversed archive instrument interval")
        bounds[symbol] = (start, end)
    return bounds


def audit_qlib(root, market, *, source, archive_hash, suspended=None, suspension_sources=()):
    import numpy as np
    root = Path(root)
    calendar_path = root / "calendars/day.txt"
    dates = calendar_path.read_text(encoding="utf-8").splitlines()
    if dates != sorted(set(dates)):
        raise ValueError("calendar must be sorted and unique")
    period = np.array(["2015-01-01" <= day < "2025-02-01" for day in dates])
    sessions = [day for day, included in zip(dates, period) if included]
    universe_path = root / "instruments" / (market + ".txt")
    all_path = root / "instruments/all.txt"
    issues, membership = [], []
    bounds = instrument_bounds(all_path) if all_path.exists() else {}
    if not bounds:
        issues.append({"code": "missing_all_instruments", "path": str(all_path)})
    if universe_path.exists():
        for line in universe_path.read_text(encoding="utf-8").splitlines():
            symbol, start, end = line.split()
            if start > end:
                raise ValueError("reversed membership interval")
            membership.append((symbol.lower(), start, end))
    else:
        issues.append({"code": "missing_historical_universe", "path": str(universe_path)})
    interval_conflicts = []
    for symbol, start, end in membership:
        first, after_last = bisect_left(sessions, start), bisect_right(sessions, end)
        if first == after_last or not bounds:
            continue
        archived = bounds.get(symbol.upper())
        if archived is None or sessions[first] < archived[0] or sessions[after_last - 1] > archived[1]:
            interval_conflicts.append({"symbol": symbol, "member_start": start, "member_end": end,
                                       "first_session": sessions[first], "last_session": sessions[after_last - 1],
                                       "archive_start": archived[0] if archived else None,
                                       "archive_end": archived[1] if archived else None})
    if interval_conflicts:
        issues.append({"code": "membership_outside_archive_instrument_interval", "intervals": len(interval_conflicts)})
    if not any(period) or dates[0] > "2015-01-05" or dates[-1] < "2025-01-27":
        issues.append({"code": "insufficient_calendar", "first": dates[0], "last": dates[-1]})
    expected_members = np.zeros(len(dates), dtype=int)
    files, gaps = {}, []
    required = ("open", "high", "low", "close", "volume", "vwap", "factor")
    symbols = sorted({row[0] for row in membership})
    benchmark = BENCHMARKS[market]
    for symbol in symbols + [benchmark]:
        active = np.zeros(len(dates), dtype=bool)
        for member, start, end in membership:
            if member == symbol:
                active |= np.array([start <= day <= end for day in dates])
        if symbol == benchmark:
            active = period.copy()
        active &= period
        values = {}
        for field in required:
            path = root / "features" / symbol / (field + ".day.bin")
            values[field] = binary_series(path, len(dates))
            if path.exists():
                files[path.relative_to(root).as_posix()] = sha256(path)
            elif np.any(active):
                issues.append({"code": "missing_field", "symbol": symbol, "field": field})
        valid = np.isfinite(values["close"]) & (values["close"] > 0)
        for field in required:
            invalid = active & valid & (~np.isfinite(values[field]) | (values[field] < 0 if field == "volume" else values[field] <= 0))
            if np.any(invalid):
                issues.append({"code": "invalid_field_on_trading_session", "symbol": symbol,
                               "field": field, "dates": [dates[int(i)] for i in np.flatnonzero(invalid)]})
        missing = active & ~valid
        if suspended:
            missing &= np.array([(symbol.upper(), day) not in suspended for day in dates])
        if np.any(missing):
            gaps.append({"symbol": symbol, "missing_sessions": int(missing.sum()),
                         "dates": [dates[int(i)] for i in np.flatnonzero(missing)],
                         "first": dates[int(np.flatnonzero(missing)[0])],
                         "last": dates[int(np.flatnonzero(missing)[-1])],
                         "reason": "unresolved; suspension/delisting provenance required"})
        if symbol != benchmark:
            expected_members += active
    search_period = np.array(["2016-01-01" <= day < "2025-01-01" for day in dates])
    empty_days = [dates[i] for i in np.flatnonzero(search_period & (expected_members == 0))]
    if empty_days:
        issues.append({"code": "empty_historical_universe", "days": empty_days})
    if gaps:
        issues.append({"code": "unresolved_price_gaps", "symbols": len(gaps)})
    # Public archive lacks a PIT classification/corporate-action/trading-status audit.
    # Keep these qualifications separate from usable Qlib data for evaluator probes.
    audit = {"schema_version": 1, "backend": "qlib", "market": market,
             "source": source, "archive_sha256": archive_hash, "files_sha256": digest(files),
             "calendar_sha256": sha256(calendar_path),
             "universe_sha256": sha256(universe_path) if universe_path.exists() else None,
             "all_instruments_sha256": sha256(all_path) if all_path.exists() else None,
             "instrument_interval_conflicts": interval_conflicts,
             "fields": required, "start": dates[0], "end": dates[-1],
             "benchmark": benchmark, "historical_universe": bool(membership),
             "adjustment": "investment_data normalized adjusted prices and inverse-adjusted volume; factor retained",
             "symbols": len(symbols), "issues": issues, "gaps": gaps,
             "qualification": "blocked" if issues else "coverage_verified",
             "data_root": str(root.resolve()),
             "suspension_sources": list(suspension_sources)}
    return audit, files


def acquire_cn(root, connections=1):
    source = SOURCES["cn"]
    raw = root / "raw" / ("cn-" + source["release"])
    archive = raw / "qlib_bin.tar.gz"
    manifest = raw / "qlib_bin.manifest.json"
    download(source["manifest"], manifest, source["manifest_sha256"])
    download(source["archive"], archive, source["archive_sha256"], size=source["archive_bytes"], connections=connections)
    return {"archive": str(archive), "sha256": sha256(archive)}


def prepare_cn(root):
    source = SOURCES["cn"]
    raw = root / "raw" / ("cn-" + source["release"])
    archive, manifest = raw / "qlib_bin.tar.gz", raw / "qlib_bin.manifest.json"
    if sha256(archive) != source["archive_sha256"] or sha256(manifest) != source["manifest_sha256"]:
        raise ValueError("offline CN archive or provenance hash mismatch")
    provenance = json.loads(manifest.read_text(encoding="utf-8"))
    if provenance["release_tag"] != source["release"] or provenance["archive_sha256"] != "sha256:" + source["archive_sha256"]:
        raise ValueError("archive provenance mismatch")
    if archive.stat().st_size != source["archive_bytes"]:
        raise ValueError("archive size mismatch")
    dataset = root / "qlib" / ("cn-" + source["release"])
    extract_archive(archive, dataset)
    results = {}
    for market in BENCHMARKS:
        audit, files = audit_qlib(dataset, market, source=source, archive_hash=source["archive_sha256"])
        atomic_json_write(root / "manifests" / f"qlib_{market}.json", audit)
        atomic_json_write(root / "manifests" / f"qlib_{market}.files.json", files)
        results[market] = {"qualification": audit["qualification"], "symbols": audit["symbols"], "issues": audit["issues"]}
    return results


def acquire_sources(root):
    result = {}
    for key, directory in (("assay", "Assay"), ("alphabench", "AlphaBench")):
        source = SOURCES[key]
        archive = root / "raw" / f"{directory}-{source['commit']}.tar.gz"
        download(source["archive"], archive, source["archive_sha256"])
        result[key] = {"archive": str(archive), "sha256": sha256(archive)}
    return result


def prepare_sources(root):
    result = {}
    for key, directory in (("assay", "Assay"), ("alphabench", "AlphaBench")):
        source = SOURCES[key]
        archive = root / "raw" / f"{directory}-{source['commit']}.tar.gz"
        if sha256(archive) != source["archive_sha256"]:
            raise ValueError("offline source archive hash mismatch")
        destination = root / directory
        extract_archive(archive, destination, root_marker="pyproject.toml")
        result[key] = {"path": str(destination), "commit": source["commit"], "archive_sha256": sha256(archive)}
    atomic_json_write(root / "source_manifest.json", result)
    return result


def plan_cn_status(root):
    source = SOURCES["cn_status"]
    dataset = root / "qlib" / ("cn-" + SOURCES["cn"]["release"])
    gaps = {}
    for market in BENCHMARKS:
        audit, _ = audit_qlib(dataset, market, source=SOURCES["cn"], archive_hash=SOURCES["cn"]["archive_sha256"])
        for gap in audit["gaps"]:
            gaps.setdefault(gap["symbol"].upper(), set()).update(day for day in gap["dates"] if day <= source["last_date"])
    batches = [(symbol, sorted(days)[offset:offset+50]) for symbol, days in sorted(gaps.items())
               for offset in range(0, len(days), 50)]

    requests = []
    for batch in batches:
        symbol, days = batch
        if not re.fullmatch(r"[A-Z]{2}[0-9]{6}", symbol) or any(not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day) for day in days):
            raise ValueError("invalid suspension lookup identity")
        literals = ",".join("'" + day + "'" for day in days)
        query = (f"SELECT tradedate,symbol,tradestatus FROM {source['table']} AS OF '{source['commit']}' "
                 f"WHERE symbol='{symbol}' AND tradedate IN ({literals}) ORDER BY tradedate LIMIT 50")
        requests.append({"query": query, "symbols": [symbol], "dates": days, "table": source["table"], "limit": 51})
    plan = {"source": source, "raw_directory": "raw/cn-status-" + source["commit"], "requests": requests}
    atomic_json_write(root / "manifests/cn-status-plan.json", plan)
    return {"planned_pages": len(requests), "plan": str(root / "manifests/cn-status-plan.json")}


def load_cn_status(root):
    source = SOURCES["cn_status"]
    plan = json.loads((root / "manifests/cn-status-plan.json").read_text(encoding="utf-8"))
    if plan["source"] != source:
        raise ValueError("CN status plan differs from its frozen source")
    rows, pages = [], {}
    for item in plan["requests"]:
        found, hashes = fetch_plan_page(root, plan, item, offline=True)
        rows.extend(found); pages.update(hashes)
    return {"source": source, "pages": pages, "rows": rows}


def prepare_cn_status(root):
    source = SOURCES["cn_status"]
    dataset = root / "qlib" / ("cn-" + SOURCES["cn"]["release"])
    evidence = load_cn_status(root)
    rows = evidence["rows"]
    suspended = {(row["symbol"], row["tradedate"]) for row in rows if str(row["tradestatus"]) == "0"}
    atomic_json_write(root / "manifests/cn_suspensions.json", evidence)
    result = {}
    for market in BENCHMARKS:
        audit, files = audit_qlib(dataset, market, source=SOURCES["cn"], archive_hash=SOURCES["cn"]["archive_sha256"],
                                  suspended=suspended, suspension_sources=[{**source, "response_hashes_digest": digest(evidence["pages"])}])
        atomic_json_write(root / "manifests" / f"qlib_{market}.json", audit)
        atomic_json_write(root / "manifests" / f"qlib_{market}.files.json", files)
        result[market] = {"qualification": audit["qualification"], "issues": audit["issues"]}
    return {"suspension_rows": len(rows), "source_last_date": source["last_date"], "markets": result}


def membership(upstream, market):
    name = "sp500" if market == "sp500" else "nasdaq100"
    path = Path(upstream) / "src/assay/data/universe" / (name + ".py")
    spec = importlib.util.spec_from_file_location("t3_universe_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    start, end = dt.date(2015, 1, 1), dt.date(2025, 1, 31)
    bounds = module.coverage_bounds()
    if bounds[0] > start or bounds[1] < end:
        raise ValueError(f"{market} membership source does not cover the full data period: {bounds}")
    snapshots = [{"effective_date": date.isoformat(), "symbols": sorted(symbols)}
                 for date, symbols in module.membership_snapshots(start, end)]
    files = {p.name: sha256(p) for p in sorted(path.parent.joinpath("data").iterdir())
             if (p.name == "sp500-historical.json" if market == "sp500" else p.name.startswith("n100-ticker-changes-"))}
    return {"market": market, "source": SOURCES["assay"], "implementation_sha256": sha256(path),
            "resource_sha256": files, "start": start.isoformat(), "end": end.isoformat(),
            "snapshots": snapshots, "symbols": sorted(set().union(*(set(row["symbols"]) for row in snapshots)))}


def sql_page(root, source, query, *, offline=False):
    """Only complete, query-matching responses enter the reusable raw cache."""
    path = root / (digest(query) + ".json")
    url = source["url"] + "?" + urllib.parse.urlencode({"q": query})
    response_hash = sha256(path) if offline else download(url, path)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        path.replace(path.with_name(path.stem + ".failed-" + response_hash + ".json"))
        raise
    if result.get("query_execution_status") != "Success" or result.get("sql_query") != query:
        path.replace(path.with_name(path.stem + ".failed-" + response_hash + ".json"))
        raise ValueError("incomplete SQL response: " + result.get("query_execution_message", "identity mismatch"))
    return result["rows"], {path.name: response_hash}


def plan_us(root, upstream):
    source, period = SOURCES["us"], SOURCES["period"]
    members = {market: membership(upstream, market) for market in ("sp500", "nasdaq100")}
    for market, item in members.items():
        atomic_json_write(root / "raw" / (market + "-membership.json"), item)
    symbols = sorted(set().union(*(set(item["symbols"]) for item in members.values())))
    if any(not re.fullmatch(r"[A-Z0-9.\-]+", symbol) for symbol in symbols):
        raise ValueError("unrecognized historic security identifier")
    dates = []
    day, end = [dt.date.fromisoformat(period[key]) for key in ("start", "end_exclusive")]
    while day < end:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += dt.timedelta(days=1)
    requests = []
    if len(symbols) >= 1000:
        raise ValueError("historic universe exceeds the source's 1000-row response limit")
    literals = ",".join("'" + symbol + "'" for symbol in symbols)
    for date in dates:
        requests.append({"table": "ohlcv", "date": date, "symbols": symbols, "limit": len(symbols)+1, "query":
            f"SELECT * FROM ohlcv AS OF '{source['commit']}' WHERE date='{date}' "
            f"AND act_symbol IN ({literals}) ORDER BY act_symbol LIMIT {len(symbols)+1}"})
    for offset in range(0, len(symbols), 5):
        group = symbols[offset:offset+5]
        literals = ",".join("'" + symbol + "'" for symbol in group)
        for table in ("split", "dividend", "symbol"):
            predicate = "" if table == "symbol" else f" AND ex_date >= '{period['start']}' AND ex_date < '{period['end_exclusive']}'"
            requests.append({"table": table, "symbols": group, "limit": 1000, "query":
                f"SELECT * FROM {table} AS OF '{source['commit']}' WHERE act_symbol IN ({literals}){predicate} LIMIT 1000"})
    plan_path = root / "manifests/us-download-plan.json"
    atomic_json_write(plan_path, {"source": source, "membership_digests": {key: digest(value) for key, value in members.items()},
                                 "period": period, "raw_directory": "raw/us-dolt-" + source["commit"], "requests": requests})
    return {"planned_pages": len(requests), "symbols": len(symbols), "plan": str(plan_path)}


def relative_path(value):
    path = Path(value)
    if not value or value.startswith("/") or "\\" in value or ":" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe relative data path")
    return path


def fetch_plan_page(root, plan, item, *, offline=False):
    if plan["source"] not in (SOURCES["us"], SOURCES["cn_status"]):
        raise ValueError("download plan source is not pinned in the task")
    if f"AS OF '{plan['source']['commit']}'" not in item["query"]:
        raise ValueError("SQL query must bind its immutable snapshot")
    rows, hashes = sql_page(root / relative_path(plan["raw_directory"]), plan["source"], item["query"], offline=offline)
    if len(rows) >= item["limit"]:
        raise ValueError("SQL page reached its limit; refuse a truncated dataset")
    symbol_key = "symbol" if item["table"] == SOURCES["cn_status"]["table"] else "act_symbol"
    if any(row[symbol_key] not in item["symbols"] or
           (item["table"] == "ohlcv" and row["date"] != item["date"]) or
           (symbol_key == "symbol" and row["tradedate"] not in item["dates"]) for row in rows):
        raise ValueError("SQL response is outside the requested security/date identity")
    return rows, hashes


def acquire_plan(root, plan_path, connections=4):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    requests = plan["requests"]
    def fetch(item):
        rows, hashes = fetch_plan_page(root, plan, item)
        return {"table": item["table"], "rows": len(rows), "files": hashes}
    completed, failures = [], []
    consecutive_failures, pause_reason = 0, None
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = {pool.submit(fetch, item): item for item in requests}
        for processed, future in enumerate(as_completed(futures), 1):
            item, access_denied = futures[future], False
            try:
                completed.append(future.result())
                consecutive_failures = 0
            except (OSError, ValueError, KeyError) as exc:
                failures.append({"query_digest": digest(item["query"]), "error": str(exc)})
                consecutive_failures += 1
                access_denied = isinstance(exc, urllib.error.HTTPError) and exc.code in {401, 403, 429}
            if access_denied or consecutive_failures >= 8:
                pause_reason = "access_denied_or_rate_limited" if access_denied else "eight_consecutive_fetch_failures"
            if processed % connections == 0 or processed == len(requests) or pause_reason:
                report = {"source": plan["source"], "plan_sha256": sha256(plan_path),
                          "planned": len(requests), "completed": len(completed), "pages": completed,
                          "failures": failures, "pause_reason": pause_reason, "qualification": "blocked"}
                atomic_json_write(root / "manifests" / (plan_path.stem + "-progress.json"), report)
            if pause_reason:
                for pending in futures:
                    pending.cancel()
                break
            if processed % 100 == 0:
                print(json.dumps({"downloaded_pages": len(completed), "planned_pages": len(requests)}), flush=True)
    return {"downloaded_pages": len(completed), "planned_pages": len(requests), "failure": bool(failures),
            "failed_pages": len(failures), "progress": str(root / "manifests" / (plan_path.stem + "-progress.json")),
            "pause_reason": pause_reason, "qualification": "blocked"}


def pack_bundle(root, destination):
    files = {path.relative_to(root).as_posix(): sha256(path)
             for directory in (root / "raw",) if directory.exists()
             for path in sorted(directory.rglob("*"))
             if path.is_file() and not path.is_symlink() and ".chunks" not in str(path) and path.suffix != ".part"}
    manifest = {"schema_version": 1, "source_manifest_digest": digest(SOURCES), "files": files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    import io
    with tarfile.open(temporary, "w:gz") as archive:
        for relative in files:
            archive.add(root / relative, arcname=relative, recursive=False)
        body = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("bundle-manifest.json"); info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
    temporary.replace(destination)
    return {"bundle": str(destination), "files": len(files), "sha256": sha256(destination)}


def install_bundle(archive_path, root, expected):
    if sha256(archive_path) != expected:
        raise ValueError("transfer bundle SHA256 mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len({member.name for member in members}) != len(members):
            raise ValueError("duplicate archive entries")
        for member in members:
            relative_path(member.name)
            if not member.isfile():
                raise ValueError("bundle must contain regular files only")
        manifest = json.load(archive.extractfile("bundle-manifest.json"))
        if manifest["source_manifest_digest"] != digest(SOURCES):
            raise ValueError("bundle belongs to different pinned data sources")
        if set(manifest["files"]) != {member.name for member in members} - {"bundle-manifest.json"}:
            raise ValueError("bundle inventory mismatch")
        for relative, expected_hash in manifest["files"].items():
            target = root / relative_path(relative)
            if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("bundle destination escapes data root")
            with archive.extractfile(relative) as stream:
                value = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != expected_hash:
                raise ValueError("bundle member hash mismatch: " + relative)
            if target.exists() and sha256(target) != expected_hash:
                raise ValueError("existing destination differs: " + relative)
        for relative in manifest["files"]:
            target = root / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.extractfile(relative) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)
    return {"installed_files": len(manifest["files"]), "root": str(root), "network_required": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("acquire-sources", "sources", "acquire-cn", "cn", "plan-cn-status", "cn-status",
        "plan-cn-baostock", "acquire-cn-baostock", "cn-baostock", "plan-us", "us", "acquire-plan", "membership", "pack", "install"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assay-root", type=Path)
    parser.add_argument("--market", choices=("sp500", "nasdaq100"), default="sp500")
    parser.add_argument("--connections", type=int, choices=range(1, 33), default=8)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args(argv)
    if args.operation == "cn":
        result = prepare_cn(args.root)
    elif args.operation == "acquire-cn":
        result = acquire_cn(args.root, args.connections)
    elif args.operation == "plan-cn-status":
        result = plan_cn_status(args.root)
    elif args.operation == "acquire-plan":
        if args.plan is None: parser.error("acquire-plan requires --plan")
        result = acquire_plan(args.root, args.plan, min(args.connections, 8))
    elif args.operation in {"pack", "install"}:
        if args.bundle is None: parser.error("pack/install requires --bundle")
        if args.operation == "install" and args.sha256 is None: parser.error("install requires --sha256")
        result = pack_bundle(args.root, args.bundle) if args.operation == "pack" else install_bundle(args.bundle, args.root, args.sha256)
    elif args.operation == "sources":
        result = prepare_sources(args.root)
    elif args.operation == "acquire-sources":
        result = acquire_sources(args.root)
    elif args.operation == "cn-status":
        result = prepare_cn_status(args.root)
    elif args.operation in {"plan-cn-baostock", "acquire-cn-baostock", "cn-baostock"}:
        from .cn_status import plan_baostock, acquire_baostock, prepare_baostock
        operation = {"plan-cn-baostock": plan_baostock, "acquire-cn-baostock": acquire_baostock,
                     "cn-baostock": prepare_baostock}[args.operation]
        result = operation(args.root)
    elif args.operation == "us":
        from assay.data.calendar import trading_days
        from .us_data import audit_us
        start = dt.date.fromisoformat(SOURCES["period"]["start"])
        end = dt.date.fromisoformat(SOURCES["period"]["end_exclusive"]) - dt.timedelta(days=1)
        result = audit_us(args.root, trading_days(start, end))
    else:
        if args.assay_root is None:
            parser.error("US preparation requires --assay-root")
        if args.operation == "membership":
            result = membership(args.assay_root, args.market)
            atomic_json_write(args.root / "raw" / (args.market + "-membership.json"), result)
            result = {"market": args.market, "symbols": len(result["symbols"]), "digest": digest(result)}
        else:
            result = plan_us(args.root, args.assay_root)
    print(json.dumps(result, indent=2))
    return 2 if result.get("failure") else 0


if __name__ == "__main__":
    raise SystemExit(main())
