from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from ldm_tts.harness import (
    HarnessClient,
    HarnessError,
    HarnessPoolConfig,
    HarnessProfile,
    HarnessSubmissionValidation,
    HarnessTurn,
)


def test_candidate_schema_digest_covers_the_transmitted_json_bytes(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "number", "enum": [1.0, 0.000001]}},
        "required": ["value"],
        "additionalProperties": False,
    }
    config = HarnessPoolConfig(
        artifact_root=tmp_path,
        base_url="https://provider.example/v1",
        model="test-model",
        profiles=(HarnessProfile(
            "chemist",
            Path("/resources/AGENTS.md"),
            1,
            agents_sha256="a" * 64,
        ),),
        campaign_id="campaign-1",
        task_id="fixture",
        case_id="case-1",
        seed=1,
        candidate_schema=schema,
    )

    frame = config.initialize_frame("initialize-1")

    schema_json = frame["candidateSchemaJson"]
    assert isinstance(schema_json, str)
    assert json.loads(schema_json) == schema
    assert frame["candidateSchemaSha256"] == hashlib.sha256(
        schema_json.encode("utf-8")
    ).hexdigest()
    assert "candidateSchema" not in frame
    assert frame["webSearch"] == {
        "providers": ["parallel-mcp", "exa", "duckduckgo"],
        "fallbackOn": [
            "transient",
            "quota",
            "network",
            "invalid-response",
            "unsupported",
        ],
    }
    assert "webProvider" not in frame
    assert frame["mcpServers"] == []


def test_persistent_harness_client_runs_one_profile_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "fake_harness_sidecar.py"
    config = HarnessPoolConfig(
        artifact_root=tmp_path,
        base_url="https://provider.example/v1",
        model="test-model",
        profiles=(HarnessProfile(
            "chemist",
            Path("/resources/AGENTS.md"),
            1,
            agents_sha256="a" * 64,
        ),),
        campaign_id="campaign-1",
        task_id="fixture",
        case_id="case-1",
        seed=1,
        candidate_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    monkeypatch.setenv("HARNESS_TEST_SECRET", "test-secret")
    monkeypatch.setenv("HARNESS_MCP_SECRET", "mcp-secret")
    client = HarnessClient(
        (sys.executable, "-u", str(fixture)),
        api_key="test-secret",
        config=config,
        named_secrets={"mcp.fixture.env.token": "mcp-secret"},
        response_timeout_seconds=5,
    )
    turn = HarnessTurn(
        profile_id="chemist",
        turn_id="round_0_chemist",
        round_index=0,
        history_from_seq=0,
        history_to_seq=0,
        history_digest="c" * 64,
        message="research",
    )

    with client:
        result = client.run_turn(
            (turn,),
            submission_validator=lambda _request: HarnessSubmissionValidation(),
        )

    assert result[0].session_id == "session-chemist"
    assert result[0].input_digest == turn.input_digest
    assert result[0].candidates == ({"value": "chemist"},)
    assert result[0].usage["providerCalls"] == 1


@pytest.mark.parametrize(
    "environment_variable",
    ("HARNESS_TEST_SKIP_VALIDATION", "HARNESS_TEST_CHANGE_AFTER_VALIDATION"),
)
def test_persistent_harness_client_rejects_unvalidated_submission(
    tmp_path: Path,
    monkeypatch,
    environment_variable: str,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "fake_harness_sidecar.py"
    config = HarnessPoolConfig(
        artifact_root=tmp_path,
        base_url="https://provider.example/v1",
        model="test-model",
        profiles=(HarnessProfile(
            "chemist",
            Path("/resources/AGENTS.md"),
            1,
            agents_sha256="a" * 64,
        ),),
        campaign_id="campaign-1",
        task_id="fixture",
        case_id="case-1",
        seed=1,
        candidate_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    monkeypatch.setenv(environment_variable, "1")
    client = HarnessClient(
        (sys.executable, "-u", str(fixture)),
        api_key="test-secret",
        config=config,
        response_timeout_seconds=5,
    )
    turn = HarnessTurn(
        profile_id="chemist",
        turn_id="round_0_chemist",
        round_index=0,
        history_from_seq=0,
        history_to_seq=0,
        history_digest="c" * 64,
        message="research",
    )

    with client, pytest.raises(
        HarnessError,
        match="committed candidates without matching task validation: chemist",
    ):
        client.run_turn(
            (turn,),
            submission_validator=lambda _request: HarnessSubmissionValidation(),
        )
