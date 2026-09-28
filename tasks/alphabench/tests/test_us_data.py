import json

import pytest

from tasks.alphabench.core import data
from tasks.alphabench.core.us_data import audit_us


def cached_us_plan(root):
    members = {"symbols": ["AAA", "BBB"], "snapshots": [
        {"effective_date": "2020-01-01", "symbols": ["AAA"]},
        {"effective_date": "2020-01-03", "symbols": ["BBB"]}]}
    raw = root / "raw"
    cache = raw / ("us-dolt-" + data.SOURCES["us"]["commit"])
    cache.mkdir(parents=True)
    (root / "manifests").mkdir()
    for market in ("sp500", "nasdaq100"):
        (raw / (market + "-membership.json")).write_text(json.dumps(members))
    requests = []
    for date, symbols in (("2020-01-02", ["AAA"]), ("2020-01-03", ["AAA"])):
        query = f"SELECT * FROM ohlcv AS OF '{data.SOURCES['us']['commit']}' WHERE date='{date}'"
        requests.append({"query": query, "date": date, "symbols": ["AAA", "BBB"], "table": "ohlcv", "limit": 3})
        rows = [{"act_symbol": symbol, "date": date, "open": "10", "high": "11", "low": "9", "close": "10", "volume": "100"}
                for symbol in symbols]
        (cache / (data.digest(query) + ".json")).write_text(json.dumps({
            "query_execution_status": "Success", "sql_query": query, "rows": rows}))
    plan = {"source": data.SOURCES["us"], "period": data.SOURCES["period"],
            "raw_directory": cache.relative_to(root).as_posix(), "requests": requests,
            "membership_digests": {m: data.digest(members) for m in ("sp500", "nasdaq100")}}
    (root / "manifests/us-download-plan.json").write_text(json.dumps(plan))
    return cache


def test_us_audit_uses_effective_membership_and_never_qualifies_missing_assets(tmp_path, monkeypatch):
    cached_us_plan(tmp_path)
    monkeypatch.setattr(data, "download", lambda *a, **kw: pytest.fail("offline audit downloaded data"))
    result = audit_us(tmp_path, ["2020-01-02", "2020-01-03"])
    assert result["markets"]["sp500"]["qualification"] == "blocked"
    report = json.loads((tmp_path / "manifests/us_sp500.json").read_text(encoding="utf-8"))
    assert report["gaps"] == {"BBB": ["2020-01-03"]}
    assert {row["code"] for row in report["issues"]} >= {"missing_vwap", "missing_actual_index_benchmarks", "unresolved_price_gaps"}


def test_missing_us_page_cannot_publish_a_complete_audit(tmp_path):
    cache = cached_us_plan(tmp_path)
    next(cache.iterdir()).unlink()
    with pytest.raises(FileNotFoundError):
        audit_us(tmp_path, ["2020-01-02", "2020-01-03"])
    assert not (tmp_path / "manifests/us_sp500.json").exists()
