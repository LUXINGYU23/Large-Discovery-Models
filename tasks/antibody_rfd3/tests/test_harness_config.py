"""Verify the research harness configuration without a provider or sidecar."""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from ldm_tts.harness import HarnessHistoryConfig
from ldm_tts.harness.history import HistoryLog
from tasks.antibody_rfd3.core import harness
from tasks.antibody_rfd3.core.synchronous.harness import PiAgent


@pytest.mark.parametrize(("effort", "expected"), [("low", "low"), (None, "off")])
def test_reasoning_effort_reaches_pi_configuration(tmp_path, monkeypatch, effort, expected):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1/responses")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    if effort is None:
        monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("LLM_REASONING_EFFORT", effort)

    preflights, clients = [], []

    class FakeProbe:
        def __init__(self, **kwargs):
            pass

        def preflight(self):
            preflights.append(True)

    class FakeClient:
        def __init__(self, command, *, api_key, config, history, response_timeout_seconds):
            self.config, self.history = config, history
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    @contextmanager
    def fake_worker_socket(worker, *, workspaces):
        yield tmp_path

    monkeypatch.setattr(harness, "OpenAICompatibleProposalClient", FakeProbe)
    monkeypatch.setattr(harness, "HarnessClient", FakeClient)
    monkeypatch.setattr(harness, "worker_socket", fake_worker_socket)
    monkeypatch.setattr(harness, "pi_sidecar_command", lambda *args, **kwargs: ("fake-sidecar",))
    service = SimpleNamespace(config=HarnessHistoryConfig(), lineage_of=lambda profile: profile, log=HistoryLog())
    bridge = SimpleNamespace(service=service, add_client=lambda client: None)
    args = SimpleNamespace(pool_size=2, antigen="1ADQ_A", seed=42, sidecar_image="fake-image",
                           guest_cache=None, method="harness")

    with harness.open_agent(args, tmp_path, SimpleNamespace(), bridge, "test-campaign") as agent:
        assert agent.client is clients[0]
        assert agent.client.config.thinking == expected
        assert agent.client.config.initialize_payload()["thinking"] == expected
    assert preflights == [True]


def test_synchronous_pi_reuses_frozen_turn_after_resume(tmp_path):
    runtime = SimpleNamespace(run_dir=tmp_path)
    agent = PiAgent(None, None, runtime)
    original = agent._turn(17, [{"score": 1}], {"remaining": 2})
    root = tmp_path / "harness"
    prior = root / "turns/design-17/turn_committed.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps(original.to_dict()))
    agent.cursor = original.history_to_seq
    history = [{"score": 1}, {"score": 2}]
    interrupted = agent._turn(18, history, {"remaining": 2})
    saved = root / "sessions/cdrh3_research/turns/design-18/input.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps(interrupted.to_dict()))

    resumed = PiAgent(None, None, runtime)
    assert resumed._turn(18, history, {"remaining": 1}) == interrupted
    with pytest.raises(ValueError, match="does not match the campaign checkpoint"):
        resumed._turn(18, [{"score": 3}, {"score": 2}], {"remaining": 1})
