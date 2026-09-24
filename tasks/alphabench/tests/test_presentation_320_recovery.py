import json

import pytest

from ldm_tts.engine.run_store import atomic_json_write
from tasks.alphabench import presentation_320
from tasks.alphabench.core.protocol import T3Protocol, digest


@pytest.mark.parametrize("phase,operation", [("check", "check"), ("search", "evaluate")])
def test_failed_qlib_worker_is_settled_without_reexecution(tmp_path, monkeypatch, phase, operation):
    request_id = "a" * 64
    directory = tmp_path / "oracle/jobs" / request_id
    request = {"request_id": request_id, "phase": phase, "operation": operation,
               "protocol": T3Protocol().to_dict()}
    atomic_json_write(directory / "request.json", request)
    atomic_json_write(directory / "permit.json", {"request_digest": digest(request),
                                                       "job_id": request_id + ":0"})
    (directory / "worker.log").write_text(
        "Traceback (most recent call last):\n"
        "AttributeError: 'numpy.float64' object has no attribute 'name'\n")

    class Service:
        def __init__(self, *_):
            pass

        def reconciled(self, identity):
            assert identity == request_id
            return {"state": "completed" if (directory / "response.json").exists() else "dispatch_intent",
                    "request": request}

    monkeypatch.setattr(presentation_320, "STUDY", tmp_path)
    monkeypatch.setattr("tasks.alphabench.core.oracle_service.OracleService", Service)
    monkeypatch.setattr(presentation_320.subprocess, "check_output", lambda *_, **__: "")
    presentation_320.resolve_worker(request_id)
    response = json.loads((directory / "response.json").read_text())
    assert response["success"] is False and response["jobs"][0]["job_id"] == request_id + ":0"
    if phase == "check":
        assert response["check_kind"] == "dynamic"
        assert T3Protocol().check_passed(response) is False
    else:
        assert "check_kind" not in response
    assert response["elapsed_estimated_from_files"] is True
    assert json.loads((directory / "recovery.json").read_text())["response_digest"] == digest(response)
    with pytest.raises(ValueError, match="unresolved expression crash"):
        presentation_320.resolve_worker(request_id)
