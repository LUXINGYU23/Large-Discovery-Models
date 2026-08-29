from __future__ import annotations

import sys
from pathlib import Path

from ldm_tts.harness import (
    HarnessClient,
    HarnessPoolConfig,
    HarnessProfile,
    HarnessSubmissionValidation,
    HarnessTurn,
)


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
        candidate_schema_sha256="b" * 64,
    )
    monkeypatch.setenv("HARNESS_TEST_SECRET", "test-secret")
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

    with client:
        result = client.run_turn(
            (turn,),
            submission_validator=lambda _request: HarnessSubmissionValidation(),
        )

    assert result[0].session_id == "session-chemist"
    assert result[0].input_digest == turn.input_digest
    assert result[0].candidates == ({"value": "chemist"},)
    assert result[0].usage["providerCalls"] == 1
