import copy
import json

import pytest

from tasks.alphabench.core import cn_status, data


def history_response():
    request = {"operation": "history", "code": "sh.600027", "fields": cn_status.FIELDS,
               "start_date": "2024-07-18", "end_date": "2024-07-19", "frequency": "d", "adjustflag": "3"}
    fields = cn_status.FIELDS.split(",")
    row = dict.fromkeys(fields, "")
    row.update(date="2024-07-19", code=request["code"], tradestatus="0", adjustflag="3")
    packet = {"source": cn_status.SOURCE, "request": request, "error_code": "0",
              "fields": fields, "rows": [[row[key] for key in fields]]}
    return request, packet


def test_status_evidence_rejects_failed_mismatched_and_duplicate_responses():
    request, packet = history_response()
    assert cn_status.response_rows(packet, request)[0]["tradestatus"] == "0"
    empty = {**packet, "rows": []}
    assert cn_status.response_rows(empty, request) == []
    mutations = [lambda p: p.update(error_code="10001001"),
                 lambda p: p["rows"][0].__setitem__(0, "2024-07-20"),
                 lambda p: p["rows"][0].__setitem__(1, "sh.600000"),
                 lambda p: p["rows"].append(list(p["rows"][0]))]
    for mutate in mutations:
        bad = copy.deepcopy(packet)
        mutate(bad)
        with pytest.raises(ValueError):
            cn_status.response_rows(bad, request)


def test_offline_status_audit_preserves_unexplained_gaps(tmp_path, monkeypatch):
    request, packet = history_response()
    folder = tmp_path / cn_status.DIRECTORY
    folder.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps({"source": cn_status.SOURCE, "requests": [request]}))
    (folder / (cn_status.digest(request) + ".json")).write_text(json.dumps(packet))
    listing = tmp_path / "qlib" / ("cn-" + data.SOURCES["cn"]["release"]) / "instruments/all.txt"
    listing.parent.mkdir(parents=True)
    listing.write_text("SH600027\t2000-01-01\t2026-09-22\n")
    monkeypatch.setattr(cn_status, "load_cn_status", lambda root: {"rows": [], "pages": {}})
    seen = []
    def audit(root, market, **kwargs):
        seen.append(kwargs["suspended"])
        return {"qualification": "blocked", "gaps": [{"symbol": "sh600027", "missing_sessions": 1,
                "dates": ["2024-07-18"]}]}, {}
    monkeypatch.setattr(cn_status, "audit_qlib", audit)
    result = cn_status.prepare_baostock(tmp_path)
    assert result["additional_suspensions"] == 1
    assert all(value == {("SH600027", "2024-07-19")} for value in seen)
    report = json.loads((tmp_path / "manifests/cn_baostock_status.json").read_text())
    assert all(row["reason"] == "no_status_evidence" for row in report["unresolved"])
    assert all(row["qualification"] == "blocked" for row in result["markets"].values())
    listing.write_text("SH600027\t2024-07-19\t2026-09-22\n")
    cn_status.prepare_baostock(tmp_path)
    report = json.loads((tmp_path / "manifests/cn_baostock_status.json").read_text())
    assert all(row["reason"] == "membership_before_archive_instrument_interval" for row in report["unresolved"])
    (folder / (cn_status.digest(request) + ".json")).unlink()
    with pytest.raises(FileNotFoundError):
        cn_status.prepare_baostock(tmp_path)


def test_qlib_audit_rejects_membership_before_a_security_code_exists(tmp_path):
    instruments = tmp_path / "instruments"
    instruments.mkdir()
    (instruments / "all.txt").write_text("SZ302132\t2025-02-17\t2026-09-22\nSH600027\t2015-01-05\t2026-09-22\n")
    (instruments / "csi1000.txt").write_text("SZ302132\t2022-12-30\t2023-12-28\nSH600027\t2015-01-01\t2015-01-05\n")
    calendar = tmp_path / "calendars/day.txt"
    calendar.parent.mkdir()
    calendar.write_text("2015-01-05\n2022-12-30\n2023-01-03\n2025-01-27\n")
    audit, _ = data.audit_qlib(tmp_path, "csi1000", source=data.SOURCES["cn"], archive_hash="fixture")
    assert audit["qualification"] == "blocked"
    assert audit["instrument_interval_conflicts"] == [{"symbol": "sz302132", "member_start": "2022-12-30",
        "member_end": "2023-12-28", "first_session": "2022-12-30", "last_session": "2023-01-03",
        "archive_start": "2025-02-17", "archive_end": "2026-09-22"}]
    assert {issue["code"] for issue in audit["issues"]} >= {"membership_outside_archive_instrument_interval"}
