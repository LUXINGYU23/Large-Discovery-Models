from __future__ import annotations

import sys
from pathlib import Path

from ldm_tts.harness import HarnessClient, HarnessPoolConfig, HarnessProfile, HarnessTurn


def test_persistent_harness_client_runs_one_profile_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "fake_harness_sidecar.py"
    config = HarnessPoolConfig(
        artifact_root=tmp_path,
        base_url="https://provider.example/v1",
        model="test-model",
        profiles=(HarnessProfile("chemist", Path("/resources/AGENTS.md"), 1),),
    )
    monkeypatch.setenv("HARNESS_TEST_SECRET", "test-secret")
    client = HarnessClient(
        (sys.executable, "-u", str(fixture)),
        api_key="test-secret",
        config=config,
        response_timeout_seconds=5,
    )
    turn = HarnessTurn("chemist", "round_0_chemist", "research")

    with client:
        result = client.run_turn((turn,))

    assert result[0].session_id == "session-chemist"
    assert result[0].input_digest == turn.input_digest
    assert result[0].candidates == ({"value": "chemist"},)
    assert result[0].usage["providerCalls"] == 1
