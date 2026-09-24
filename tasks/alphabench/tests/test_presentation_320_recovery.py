import json
from urllib.error import URLError

import pytest

from ldm_tts.engine.run_store import atomic_json_write
from tasks.alphabench import presentation_320
from tasks.alphabench.core.protocol import T3Protocol, digest
from tasks.alphabench.core.receipts import Receipts


def test_oracle_restart_reuses_existing_process_registry_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(presentation_320, "STUDY", tmp_path)
    monkeypatch.setattr(presentation_320, "ORACLE_CONFIG", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({"backend": "qlib", "market": "csi300",
        "data_digest": "a" * 64, "environment_digest": "b" * 64}))
    atomic_json_write(tmp_path / "processes.json", {"oracle": {"pid": 999999}})
    calls = []

    def health():
        if not calls:
            raise URLError("oracle unavailable")
        return {"backend": "qlib", "market": "csi300", "data_digest": "a" * 64,
                "environment_digest": "b" * 64, "config_digest": "c" * 64}

    def spawn(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(presentation_320, "health", health)
    monkeypatch.setattr(presentation_320, "spawn", spawn)
    monkeypatch.setattr(presentation_320.time, "sleep", lambda _: None)
    presentation_320.start_oracle()
    assert len(calls) == 1 and calls[0][0][0] == "oracle" and calls[0][1]["resume"] is True


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


def test_model_recovery_authorization_targets_the_existing_receipt(tmp_path, monkeypatch):
    method, seed = "alphabench_ea", 42
    monkeypatch.setattr(presentation_320, "STUDY", tmp_path)
    def dead(*_):
        raise ProcessLookupError
    monkeypatch.setattr(presentation_320.os, "kill", dead)
    atomic_json_write(tmp_path / "processes.json", {f"{method}:seed_{seed}": {"pid": 999999}})
    protocol = presentation_320.protocol(method, seed)
    atomic_json_write(presentation_320.proto_path(method, seed), protocol.to_dict())
    directory = presentation_320.run_dir(method, seed)
    atomic_json_write(directory / "protocol.json", protocol.to_dict())
    atomic_json_write(directory / "status.json", {"status": "paused_outcome_unknown"})
    identity = digest({"run": "campaign", "identity": ["native", "example", 0],
                       "protocol": protocol.identity})
    request = {"messages": [], "protocol": protocol.identity, "logical": ["native", "example", 0]}
    receipt = {"identity": identity, "request": request,
               "request_digest": digest(request), "state": "dispatch_intent"}
    path = Receipts(directory / "private/model").path(identity)
    atomic_json_write(path, receipt)

    presentation_320.resolve_model(method, seed)
    marker = json.loads((directory / "private/model_recovery" / path.name).read_text())
    assert marker == {"action": "discard_unknown_response_and_retry_once",
                      "original_receipt_digest": digest(receipt)}
