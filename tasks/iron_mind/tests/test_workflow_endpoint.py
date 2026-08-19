"""Endpoint failure and strict-response workflow contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ldm_tts.engine.run_store import CampaignRuntime
from ldm_tts.transport import ProposalResponse
from ldm_tts.transport.openai_http import EndpointRequestError

from tasks.iron_mind.core import workflow
from tasks.iron_mind.core.workflow import main


def test_endpoint_preflight_pause_keeps_all_counters_zero_and_redacts_key(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    test_key = "test-workflow-key"

    class OfflineEndpoint:
        def preflight(self):
            raise EndpointRequestError("endpoint offline")

    monkeypatch.setenv("LLM_API_KEY", test_key)
    monkeypatch.setattr(
        workflow,
        "build_openai_reaction_client",
        lambda **_kwargs: OfflineEndpoint(),
        raising=False,
    )
    code = main(_endpoint_args(tmp_path, "paused"))

    output = capsys.readouterr().out
    payload = json.loads(output)
    run_dir = Path(payload["run_dir"])
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    artifacts = [
        output,
        (run_dir / "config.json").read_text(encoding="utf-8"),
        (run_dir / "events.jsonl").read_text(encoding="utf-8"),
        (run_dir / "status.json").read_text(encoding="utf-8"),
    ]
    assert code == 2
    assert status["status"] == "paused_endpoint_unavailable"
    assert status["budget"]["counters"] == _zero_counters()
    assert all(test_key not in artifact for artifact in artifacts)


def test_preflight_programming_error_is_not_silently_paused(tmp_path: Path, monkeypatch) -> None:
    class BrokenEndpoint:
        def preflight(self):
            raise ValueError("broken endpoint fixture")

    monkeypatch.setenv("LLM_API_KEY", "test-workflow-key")
    monkeypatch.setattr(
        workflow,
        "build_openai_reaction_client",
        lambda **_kwargs: BrokenEndpoint(),
        raising=False,
    )

    with pytest.raises(ValueError, match="broken endpoint fixture"):
        main(_endpoint_args(tmp_path, "programming-error"))


def test_missing_endpoint_key_pauses_before_the_engine(tmp_path: Path, monkeypatch, capsys) -> None:
    for name in ("LLM_API_KEY", "TTS_LLM_API_KEY", "LDM_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    code = main(_endpoint_args(tmp_path, "missing-key"))

    payload = json.loads(capsys.readouterr().out)
    status = json.loads(
        (Path(payload["run_dir"]) / "status.json").read_text(encoding="utf-8")
    )
    assert code == 2
    assert status["status"] == "paused_endpoint_unavailable"
    assert status["budget"]["counters"] == _zero_counters()


def test_expansion_endpoint_error_preserves_request_budget_and_can_resume(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class CircuitOpenClient:
        def preflight(self):
            return {"model": "test-model"}

        def propose(self, _request):
            raise EndpointRequestError("endpoint circuit is open")

    monkeypatch.setenv("LLM_API_KEY", "test-workflow-key")
    monkeypatch.setattr(
        workflow,
        "build_openai_reaction_client",
        lambda **_kwargs: CircuitOpenClient(),
        raising=False,
    )
    code = main(_endpoint_args(tmp_path, "circuit-open"))

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    counters = status["budget"]["counters"]
    resumed = CampaignRuntime.open(run_dir, task="iron_mind", resume=True)
    assert code == 2
    assert status["status"] == "paused_endpoint_unavailable"
    assert counters["llm_requests"] == 64
    assert counters["proposal_attempts"] == 0
    assert counters["selected_candidates"] == 0
    assert counters["external_evaluations"] == 0
    assert resumed.run_dir == run_dir


@pytest.mark.parametrize("case", ("invalid", "non_json", "extra"))
def test_invalid_endpoint_responses_never_reach_the_evaluator(
    case: str, tmp_path: Path, monkeypatch, capsys
) -> None:
    class StaticEndpoint:
        def preflight(self):
            return {"model": "test-model"}

        def propose(self, _request):
            return _malformed_response(case)

    monkeypatch.setenv("LLM_API_KEY", "test-workflow-key")
    monkeypatch.setattr(
        workflow,
        "build_openai_reaction_client",
        lambda **_kwargs: StaticEndpoint(),
        raising=False,
    )
    code = main(_endpoint_args(tmp_path, case))

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    counters = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))["counters"]
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert code == 1
    assert payload["engine_summary"]["successful_evaluation_count"] == 0
    assert counters["llm_requests"] == 64
    assert counters["proposal_attempts"] == 64
    assert counters["valid_search_candidates"] == 0
    assert counters["selected_candidates"] == 0
    assert counters["external_evaluations"] == 0
    assert "candidate_evaluated" not in events


def test_duplicate_endpoint_responses_are_deduplicated_by_the_shared_reservoir(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class StaticEndpoint:
        def preflight(self):
            return {"model": "test-model"}

        def propose(self, _request):
            return _valid_response()

    monkeypatch.setenv("LLM_API_KEY", "test-workflow-key")
    monkeypatch.setattr(
        workflow,
        "build_openai_reaction_client",
        lambda **_kwargs: StaticEndpoint(),
        raising=False,
    )
    code = main(_endpoint_args(tmp_path, "duplicates", prompt_policy="baseline_v1"))

    payload = json.loads(capsys.readouterr().out)
    run_dir = Path(payload["run_dir"])
    counters = json.loads((run_dir / "budget.json").read_text(encoding="utf-8"))["counters"]
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert code == 0
    assert payload["engine_summary"]["successful_evaluation_count"] == 1
    assert counters["llm_requests"] == 64
    assert counters["proposal_attempts"] == 64
    assert counters["valid_search_candidates"] == 1
    assert counters["selected_candidates"] == 1
    assert counters["external_evaluations"] == 1
    assert '"duplicate"' in events


def _endpoint_args(
    tmp_path: Path,
    run_name: str,
    *,
    prompt_policy: str = "portfolio_v1",
) -> list[str]:
    return [
        "--mock",
        "--proposal-mode",
        "openai",
        "--llm-url",
        "https://example.invalid/v1",
        "--llm-model-name",
        "test-model",
        "--out-dir",
        str(tmp_path),
        "--run-name",
        run_name,
        "--prompt-policy",
        prompt_policy,
    ]


def _zero_counters() -> dict[str, int]:
    return {
        "benchmark_jobs": 0,
        "expensive_evaluation_attempts": 0,
        "external_evaluations": 0,
        "llm_requests": 0,
        "outer_iterations": 0,
        "proposal_attempts": 0,
        "selected_candidates": 0,
        "successful_evaluations": 0,
        "valid_search_candidates": 0,
    }


def _valid_response() -> ProposalResponse:
    table = workflow._load_mock_table(
        workflow._schema_for("buchwald_hartwig"),
        candidate_count=64,
    )
    row = table.rows[0]
    return ProposalResponse(
        text=json.dumps({"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)})
    )


def _malformed_response(case: str) -> ProposalResponse:
    payload = json.loads(_valid_response().text)
    if case == "invalid":
        payload = {
            "dataset_id": payload["dataset_id"],
            "conditions": {**payload["conditions"], "base": "unknown"},
        }
    elif case == "non_json":
        return ProposalResponse(text="not JSON")
    elif case == "extra":
        payload = {**payload, "extra": True}
    elif case not in {"invalid", "non_json", "extra"}:
        raise AssertionError(f"Unexpected malformed response case: {case}")
    return ProposalResponse(text=json.dumps(payload))
