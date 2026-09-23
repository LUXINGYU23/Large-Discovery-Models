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
    assert result["unique_generated"] == 2
    assert result["unique_admission_rate"] == .5
    assert result["rows"][0]["backend_valid"] is True
    assert result["rows"][1]["backend_valid"] is True
    assert result["rows"][2]["backend_valid"] is None
    assert result["rows"][3]["backend_valid"] is False
    assert (tmp_path / "budget.json").read_bytes() == budget
    assert audit_quality(protocol, runtime, gateway) == result


def test_empty_occurrence_set_is_not_complete_quality_evidence(tmp_path):
    protocol = T3Protocol(method="harness")
    runtime = CampaignRuntime.open(tmp_path, task="alphabench")
    result = audit_quality(protocol, runtime, OracleGateway(protocol, runtime, mock=True))
    assert result["raw_occurrences"] == 0
    assert result["complete"] is False
    assert result["unavailable_reason"] == "no_occurrences"


def test_quality_refuses_a_receipt_from_a_different_interval(tmp_path):
    protocol, runtime, gateway, identity = quality_fixture(tmp_path)
    path = gateway.receipts.path(identity)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["request"]["start"] = "2017-01-01"
    record["request_digest"] = digest(record["request"])
    atomic_json_write(path, record)
    with pytest.raises(ValueError, match="frozen evaluation contract"):
        audit_quality(protocol, runtime, gateway)


@pytest.mark.parametrize("backend,profile,expression,rate,paper_rate,paper_coverage", [
    ("assay", "assay_code_filter_v1", "ts_mean(close,5)", "assay_lint_success_rate", None, 0),
    ("qlib", "qlib_code_filter_v1", "Abs(" * 6 + "$close" + ")" * 6, "qlib_dynamic_success_rate", 0, 1),
])
def test_source_checks_do_not_become_unmeasured_paper_success(tmp_path, backend, profile, expression, rate, paper_rate, paper_coverage):
    protocol = T3Protocol(backend=backend, filter_profile=profile)
    runtime = CampaignRuntime.open(tmp_path, task="alphabench")
    gateway = OracleGateway(protocol, runtime, mock=True)
    candidate = FactorDomain(backend, protocol.grammar_depth).admit(RawProposal({"expression": expression}, "model"))
    gateway.evaluate(candidate, phase="check", position=0)
    occurrences = [{"attempt": 0, "index": 0, "payload": candidate.payload, "status": "accepted",
                    "check_receipt": gateway.identity("check", 0, candidate)}]
    Receipts(tmp_path / "generation").accept("step", {"occurrences": occurrences})
    before = (tmp_path / "budget.json").read_bytes()
    report = audit_quality(protocol, runtime, gateway)
    assert report[rate] == 1 and report["coverage"] == 1 and report["unique_valid"] == 1
    assert report["paper_dynamic_success_rate"] == paper_rate and report["paper_coverage"] == paper_coverage
    if backend == "assay":
        assert "assay_dynamic_success_rate" not in report
        assert report["paper_unavailable_reason"] == "lint_only"
    assert (tmp_path / "budget.json").read_bytes() == before
