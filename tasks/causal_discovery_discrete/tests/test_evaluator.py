import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from ldm_tts.contracts import Candidate
from tasks.causal_discovery_discrete.core import evaluator as module


def _fixture_case_worker(connection, prepared, spec):
    Path(prepared["pid_file"]).write_text(str(os.getpid()))
    if spec["behavior"] == "timed_out":
        time.sleep(10)
    if spec["behavior"] == "failed":
        connection.send({"error": "invalid graph"})
    else:
        connection.send({"metrics": {"shd": 1.0}})
    connection.close()


@pytest.mark.parametrize("behavior", ["succeeded", "failed", "timed_out"])
def test_evaluation_process_deadline_and_accounting(tmp_path, monkeypatch, behavior):
    monkeypatch.setattr(module, "validate_upstream_contract", lambda root: None)
    monkeypatch.setattr(module.MLSBenchCausalEvaluator, "_prepare_cases", lambda self: {
        "Cancer": {"pid_file": str(tmp_path / "worker.pid")}})
    monkeypatch.setattr(module, "_case_worker", _fixture_case_worker)
    scored = []

    def score(*args, **kwargs):
        scored.append(kwargs["timeout_seconds"])
        return {"official_score": .5}

    monkeypatch.setattr(module, "score_with_pinned_mlsbench", score)
    evaluator = module.MLSBenchCausalEvaluator(upstream_root=tmp_path, run_dir=tmp_path,
        timeout_seconds=.5 if behavior == "timed_out" else 5)
    children = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    result = evaluator.evaluate(Candidate("candidate", {"spec": {"behavior": behavior}}, "candidate"))
    assert result.status == behavior
    assert time.monotonic() - started < 4
    assert {child.pid for child in multiprocessing.active_children()} == children
    assert result.resource_usage["benchmark_jobs"] == 1
    manifest = json.loads((tmp_path / result.artifacts["evaluation_manifest"]).read_text())
    assert manifest["status"] == behavior
    assert manifest["cases"][0]["status"] == behavior
    if behavior == "succeeded":
        assert scored == [5]
        assert result.metrics["selection_score"] == .5
    else:
        assert not scored
        assert result.error
