import copy
import json

import pytest

from tasks.alphabench.core import cn_status


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
    (folder / (cn_status.digest(request) + ".json")).unlink()
    with pytest.raises(FileNotFoundError):
        cn_status.prepare_baostock(tmp_path)
