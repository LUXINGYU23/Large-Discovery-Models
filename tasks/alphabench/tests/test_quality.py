import json

import pytest

from ldm_tts.contracts import RawProposal
from ldm_tts.engine.run_store import CampaignRuntime, atomic_json_write
from tasks.alphabench.core.candidate import FactorDomain
from tasks.alphabench.core.gateway import OracleGateway
from tasks.alphabench.core.protocol import T3Protocol, digest
from tasks.alphabench.core.quality import audit_quality
from tasks.alphabench.core.receipts import Receipts


def quality_fixture(root):
    protocol = T3Protocol()
    runtime = CampaignRuntime.open(root, task="alphabench", budget_limits={"quality_checks": 0})
    gateway = OracleGateway(protocol, runtime, mock=True)
    candidate = FactorDomain().admit(RawProposal({"expression": "Mean($close,5)"}, "model"))
    gateway.evaluate(candidate, phase="check", position=[0, 0])
    identity = gateway.identity("check", [0, 0], candidate)
    occurrences = [
        {"attempt": 0, "index": 0, "payload": candidate.payload, "status": "accepted", "check_receipt": identity},
        {"attempt": 0, "index": 1, "payload": candidate.payload, "status": "duplicate"},
        {"attempt": 0, "index": 2, "payload": {"expression": "Mean($close,8)"}, "status": "unprocessed"},
        {"attempt": 0, "index": 3, "payload": {"expression": "Ref($close,-1)"}, "status": "static_rejected"}]
    Receipts(root / "generation").accept("step", {"attempt_count": 1, "complete": True, "occurrences": occurrences})
    return protocol, runtime, gateway, identity


def test_quality_reuses_checks_and_does_not_report_rates_for_missing_coverage(tmp_path):
    protocol, runtime, gateway, _ = quality_fixture(tmp_path)
    budget = (tmp_path / "budget.json").read_bytes()
    result = audit_quality(protocol, runtime, gateway)
    assert result["coverage"] == .75
    assert result["audited"] == 3
    assert result["static_success_rate"] == .75
    assert result["qlib_dynamic_success_rate"] is None
    assert result["paper_dynamic_success_rate"] is None
    assert result["unique_admitted"] == 1
    assert result["rows"][0]["backend_valid"] is True
    assert result["rows"][1]["backend_valid"] is True
    assert result["rows"][2]["backend_valid"] is None
    assert result["rows"][3]["backend_valid"] is False
    assert (tmp_path / "budget.json").read_bytes() == budget
    assert audit_quality(protocol, runtime, gateway) == result


def test_quality_refuses_a_receipt_from_a_different_interval(tmp_path):
    protocol, runtime, gateway, identity = quality_fixture(tmp_path)
    path = gateway.receipts.path(identity)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["request"]["start"] = "2017-01-01"
    record["request_digest"] = digest(record["request"])
    atomic_json_write(path, record)
    with pytest.raises(ValueError, match="frozen evaluation contract"):
        audit_quality(protocol, runtime, gateway)
