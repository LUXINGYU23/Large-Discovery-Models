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

    monkeypatch.setenv("LDM_LLM_API_KEY", test_key)
    monkeypatch.setattr(
        workflow,
        "build_deepseek_reaction_client",
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

    monkeypatch.setenv("LDM_LLM_API_KEY", "test-workflow-key")
    monkeypatch.setattr(
        workflow,
        "build_deepseek_reaction_client",
        lambda **_kwargs: BrokenEndpoint(),
        raising=False,
    )

    with pytest.raises(ValueError, match="broken endpoint fixture"):
        main(_endpoint_args(tmp_path, "programming-error"))


def test_missing_endpoint_key_pauses_before_the_engine(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("LDM_LLM_API_KEY", raising=False)

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

    monkeypatch.setenv("LDM_LLM_API_KEY", "test-workflow-key")
    monkeypatch.setattr(
        workflow,
        "build_deepseek_reaction_client",
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
    assert counters["llm_requests"] == 1
    assert counters["proposal_attempts"] == 0
    assert counters["selected_candidates"] == 0
    assert counters["external_evaluations"] == 0
    assert resumed.run_dir == run_dir


@pytest.mark.parametrize("case", ("fewer", "more", "duplicate", "invalid"))
def test_nonconforming_endpoint_responses_never_reach_the_evaluator(
    case: str, tmp_path: Path, monkeypatch, capsys
) -> None:
    class StaticEndpoint:
        def preflight(self):
            return {"model": "test-model"}

        def propose(self, _request):
            return ProposalResponse(text=_malformed_response(case))

    monkeypatch.setenv("LDM_LLM_API_KEY", "test-workflow-key")
    monkeypatch.setattr(
        workflow,
        "build_deepseek_reaction_client",
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
    assert counters["llm_requests"] == 1
    assert counters["proposal_attempts"] == 1
    assert counters["valid_search_candidates"] == 0
    assert counters["selected_candidates"] == 0
    assert counters["external_evaluations"] == 0
    assert "candidate_evaluated" not in events


def _endpoint_args(tmp_path: Path, run_name: str) -> list[str]:
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


def _malformed_response(case: str) -> str:
    table = workflow._load_mock_table(workflow._schema_for("buchwald_hartwig"))
    candidates = [
        {"dataset_id": table.schema.dataset_id, "conditions": dict(row.conditions)}
        for row in table.rows
    ]
    if case == "fewer":
        candidates = candidates[:3]
    elif case == "more":
        candidates = candidates + [candidates[0]]
    elif case == "duplicate":
        candidates[-1] = candidates[0]
    elif case == "invalid":
        candidates[-1] = {
            "dataset_id": table.schema.dataset_id,
            "conditions": {**candidates[-1]["conditions"], "base": "unknown"},
        }
    else:
        raise AssertionError(f"Unexpected malformed response case: {case}")
    return json.dumps({"candidates": candidates})
