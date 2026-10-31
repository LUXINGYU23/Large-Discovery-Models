"""Capture historical CN index snapshots for archive-conflict review."""

import datetime as dt
import importlib.metadata
import json
import re
import socket

from ldm_tts.engine.run_store import atomic_json_write
from .data import SOURCES, sha256
from .protocol import digest

SOURCE = SOURCES["cn_baostock"]
DIRECTORY = "raw/cn-membership-" + SOURCE["capture"]
COUNTS = {"csi300": 300, "csi500": 500}
FIELDS = ["updateDate", "code", "code_name"]


def audits(root):
    result, hashes = {}, {}
    for market in COUNTS:
        path = root / "manifests" / f"qlib_{market}.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit["market"] != market or audit["archive_sha256"] != SOURCES["cn"]["archive_sha256"]:
            raise ValueError("CN membership audit belongs to a different archive")
        result[market], hashes[market] = audit, sha256(path)
    return result, hashes


def requests_for(audit):
    return [{"market": market, "date": day} for market in COUNTS
            for day in sorted({row[key] for row in audit[market]["instrument_interval_conflicts"]
                               for key in ("first_session", "last_session")})]


def read_plan(root):
    plan = json.loads((root / DIRECTORY / "plan.json").read_text(encoding="utf-8"))
    audit, hashes = audits(root)
    if (plan["source"] != SOURCE or plan["input_audit_hashes"] != hashes
            or plan["requests"] != requests_for(audit)):
        raise ValueError("CN membership plan differs from the frozen archive audits")
    return plan


def plan_membership(root):
    path = root / DIRECTORY / "plan.json"
    if path.exists():
        plan = read_plan(root)
    else:
        audit, hashes = audits(root)
        plan = {"source": SOURCE, "input_audit_hashes": hashes, "requests": requests_for(audit)}
        atomic_json_write(path, plan)
    return {"plan": str(path), "requests": len(plan["requests"])}


def response_rows(packet, request):
    if (packet["source"] != SOURCE or packet["request"] != request or packet["error_code"] != "0"
            or packet["response_date"] != request["date"] or packet["fields"] != FIELDS):
        raise ValueError("BaoStock membership response differs from its request")
    rows = packet["rows"]
    if len(rows) > COUNTS[request["market"]] or any(len(row) != len(FIELDS) for row in rows):
        raise ValueError("malformed BaoStock membership rows")
    if (len({row[1] for row in rows}) != len(rows)
            or any(not re.fullmatch(r"(?:sh|sz)\.[0-9]{6}", row[1]) for row in rows)):
        raise ValueError("duplicate or invalid BaoStock membership security")
    for row in rows:
        if dt.date.fromisoformat(row[0]) > dt.date.fromisoformat(request["date"]):
            raise ValueError("BaoStock membership update is after the requested date")
    if len({row[0] for row in rows}) > 1:
        raise ValueError("BaoStock membership snapshot has mixed update dates")
    return rows


def acquire_membership(root):
    import baostock as bs

    if importlib.metadata.version("baostock") != SOURCE["client_version"]:
        raise ValueError("BaoStock client differs from the pinned version")
    plan = read_plan(root)
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(20)
    logged_in = False
    try:
        for index, request in enumerate(plan["requests"], 1):
            path = root / DIRECTORY / (digest(request) + ".json")
            if path.exists():
                response_rows(json.loads(path.read_text(encoding="utf-8")), request)
                continue
            if not logged_in:
                login = bs.login()
                if login.error_code != "0":
                    raise ConnectionError("BaoStock login failed: " + login.error_msg)
                logged_in = True
            query = bs.query_hs300_stocks if request["market"] == "csi300" else bs.query_zz500_stocks
            result = query(date=request["date"])
            rows = []
            while result.error_code == "0" and result.next():
                rows.append(result.get_row_data())
            packet = {"source": SOURCE, "request": request, "error_code": result.error_code,
                      "error_msg": result.error_msg, "response_date": result.date,
                      "fields": result.fields, "rows": rows,
                      "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            try:
                response_rows(packet, request)
            except (ValueError, KeyError):
                atomic_json_write(path.with_name(path.stem + ".failed-" + digest(packet) + ".json"), packet)
                raise
            atomic_json_write(path, packet)
            if index % 10 == 0:
                print(json.dumps({"completed": index, "planned": len(plan["requests"])}), flush=True)
    finally:
        if logged_in:
            bs.logout()
        socket.setdefaulttimeout(previous_timeout)
    return {"completed": len(plan["requests"]), "qualification": "evidence_only"}


def audit_membership(root):
    plan = read_plan(root)
    audit, _ = audits(root)
    snapshots, pages = {}, {}
    for request in plan["requests"]:
        path = root / DIRECTORY / (digest(request) + ".json")
        rows = response_rows(json.loads(path.read_text(encoding="utf-8")), request)
        pages[path.relative_to(root).as_posix()] = sha256(path)
        snapshots[request["market"], request["date"]] = {
            "response_sha256": pages[path.relative_to(root).as_posix()],
            "update_date": rows[0][0] if rows else None,
            "rows": len(rows), "complete": len(rows) == COUNTS[request["market"]],
            "symbols": {row[1].replace(".", "").lower() for row in rows}}
    conflicts = []
    for market in COUNTS:
        for conflict in audit[market]["instrument_interval_conflicts"]:
            observations = {}
            for edge in ("first_session", "last_session"):
                snapshot = snapshots[market, conflict[edge]]
                observations[edge] = {key: snapshot[key] for key in
                    ("response_sha256", "update_date", "rows", "complete")}
                observations[edge]["present"] = conflict["symbol"] in snapshot["symbols"]
            conflicts.append({"market": market, **conflict, "observations": observations})
    report = {"source": SOURCE, "plan_sha256": sha256(root / DIRECTORY / "plan.json"),
              "pages": pages, "snapshots": len(snapshots),
              "incomplete_snapshots": sum(not item["complete"] for item in snapshots.values()),
              "conflicts": conflicts, "qualification": "evidence_only"}
    atomic_json_write(root / "manifests/cn_membership_probe.json", report)
    return {"snapshots": report["snapshots"], "incomplete_snapshots": report["incomplete_snapshots"],
            "conflicts": len(conflicts), "qualification": report["qualification"]}
