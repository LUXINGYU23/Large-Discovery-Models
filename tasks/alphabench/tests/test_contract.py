import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from ldm_tts.contracts import Candidate, RawProposal
from ldm_tts.contracts.evaluation import EvaluationPaused
from tasks.alphabench.core.candidate import FactorDomain
from tasks.alphabench.core.data import extract_archive
from tasks.alphabench.core.grammar import ExpressionError, REGISTRY, parse_expression
from tasks.alphabench.core.receipts import Receipts
from tasks.alphabench.core.reporting import daily_metrics, search_metrics, signal_diversity
from tasks.alphabench.core.selection import FactorEncoder


@pytest.mark.parametrize("dialect,operator,signature", [
    (dialect, name, signature) for dialect in ("qlib", "assay") for name, signature in REGISTRY[dialect].items()
])
def test_complete_guide_space_is_admitted_and_encoded(dialect, operator, signature):
    values = {"value": "$close" if dialect == "qlib" else "close", "number": "1",
              "window": "5", "lag": "1", "quantile": ".5", "group": "'sector'"}
    expression = operator + "(" + ",".join(values[role.removesuffix('?').split(':')[-1]] for role in signature) + ")"
    candidate = FactorDomain("assay").admit(RawProposal({"expression": expression}, "test"))
    assert isinstance(candidate, Candidate)
    features = FactorEncoder().encode(candidate)
    assert np.isfinite(features.values).all()


@pytest.mark.parametrize("left,right", [("Mean( $close , 5 )", "Mean($close,5)"),
    ("adv20", "ts_mean(volume,20)"), ("safe_div(close,volume,fill=0)", "safe_div(close,volume,0)")])
def test_canonical_identity_and_features_agree(left, right):
    domain = FactorDomain("assay")
    a, b = [domain.admit(RawProposal({"expression": text}, "test")) for text in (left, right)]
    assert a.candidate_id == b.candidate_id
    assert FactorEncoder().encode(a).values == FactorEncoder().encode(b).values


@pytest.mark.parametrize("expression", ["Ref($close,-1)", "Mean($close,0)", "Mean($close,1.5)",
    "__import__('os').system('id')", "close.__class__", "[x for x in close]", "Mean($close,5", "1e1000",
    "Mean(close,5)", "ts_mean($close,5)", "safe_div(close,volume,fill=0,fill=1)"])
def test_invalid_or_future_expressions_never_reach_backend(expression):
    with pytest.raises(ExpressionError):
        parse_expression(expression, backend="assay")


def test_receipts_replay_reserve_dispatch_and_reconciliation(tmp_path):
    receipts, calls = Receipts(tmp_path), []
    request = {"expression": "Mean($close,5)"}
    def dispatched():
        calls.append("dispatch")
        raise OSError("connection lost after acceptance")
    with pytest.raises(OSError):
        receipts.execute("a", request, reserve=lambda: calls.append("reserve"), operation=dispatched)
    with pytest.raises(EvaluationPaused):
        receipts.execute("a", request, reserve=lambda: calls.append("reserve"), operation=dispatched)
    assert calls == ["reserve", "dispatch"]
    result = receipts.execute("a", request, reserve=lambda: calls.append("reserve"), operation=dispatched,
                              reconcile=lambda: {"success": True})
    assert result == {"success": True}
    assert calls == ["reserve", "dispatch"]
    with pytest.raises(ValueError, match="different request"):
        receipts.execute("a", {"expression": "$close"}, reserve=lambda: None, operation=lambda: None)


def test_metric_boundaries_and_degeneracy():
    report = daily_metrics([{"ic": -1., "rank_ic": 1.}, {"ic": 0., "rank_ic": 1.}, {"ic": 1., "rank_ic": 1.}])
    assert report["ic_derived"] == {"mean": 0., "ir": 0., "winrate": 1/3, "skewness": 0.,
        "skewness_estimator": "central_moment_m3_over_m2_pow_1.5", "reason": None}
    assert report["rank_ic_derived"]["ir"] is None
    assert search_metrics([{"metrics": {"ic": .03}}, {"metrics": {}}, {"metrics": {"ic": .031}}])["search_cost_attempts"] == 3
    assert search_metrics([{"metrics": {"ic": -1.}}])["gain"] == 0
    a = [{"date": "2022-01-03", "instrument": str(i), "score": float(i)} for i in range(3)]
    b = [{**row, "score": -row["score"]} for row in a]
    assert signal_diversity([a, b])["diversity"] == 0


def test_data_archive_rejects_paths_and_links(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"bad"
        member = tarfile.TarInfo("../outside")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe"):
        extract_archive(archive, tmp_path / "data")
    assert not (tmp_path / "outside").exists()
