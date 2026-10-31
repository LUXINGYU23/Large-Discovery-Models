import copy
import json

import pytest

from tasks.alphabench.core import cn_membership, data


def archive_audits(root):
    directory = root / "manifests"
    directory.mkdir()
    for market in cn_membership.COUNTS:
        (directory / f"qlib_{market}.json").write_text(json.dumps({
            "market": market, "archive_sha256": data.SOURCES["cn"]["archive_sha256"],
            "instrument_interval_conflicts": [{"symbol": "sh600005", "first_session": "2017-03-01",
                                               "last_session": "2017-03-01"}]}))


def response(request, count):
    return {"source": cn_membership.SOURCE, "request": request, "error_code": "0",
            "response_date": request["date"], "fields": cn_membership.FIELDS,
            "rows": [["2017-02-27", f"sh.{index:06d}", "name"] for index in range(1, count + 1)]}


def test_incomplete_historical_snapshot_remains_evidence_only(tmp_path):
    archive_audits(tmp_path)
    assert cn_membership.plan_membership(tmp_path)["requests"] == 2
    for request in cn_membership.read_plan(tmp_path)["requests"]:
        count = cn_membership.COUNTS[request["market"]] - (request["market"] == "csi500")
        path = tmp_path / cn_membership.DIRECTORY / (data.digest(request) + ".json")
        path.write_text(json.dumps(response(request, count)))
    result = cn_membership.audit_membership(tmp_path)
    assert result == {"snapshots": 2, "incomplete_snapshots": 1, "conflicts": 2,
                      "qualification": "evidence_only"}
    report = json.loads((tmp_path / "manifests/cn_membership_probe.json").read_text())
    by_market = {item["market"]: item for item in report["conflicts"]}
    assert by_market["csi300"]["observations"]["first_session"]["complete"] is True
    assert by_market["csi500"]["observations"]["first_session"]["complete"] is False
    assert all(not item["observations"]["first_session"]["present"] for item in report["conflicts"])


def test_membership_probe_rejects_mismatched_responses_and_changed_audit(tmp_path):
    archive_audits(tmp_path)
    cn_membership.plan_membership(tmp_path)
    request = cn_membership.read_plan(tmp_path)["requests"][0]
    packet = response(request, 300)
    mutations = [lambda p: p["rows"].append(p["rows"][0]),
                 lambda p: p["rows"][0].__setitem__(0, "2017-03-02"),
                 lambda p: p.update(response_date="2017-03-02")]
    for mutate in mutations:
        changed = copy.deepcopy(packet)
        mutate(changed)
        with pytest.raises(ValueError):
            cn_membership.response_rows(changed, request)
    path = tmp_path / "manifests/qlib_csi300.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="plan differs"):
        cn_membership.read_plan(tmp_path)
