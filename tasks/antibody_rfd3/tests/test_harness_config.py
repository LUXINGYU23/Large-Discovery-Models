"""Verify the research harness configuration without a provider or sidecar."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ldm_tts.harness import HarnessHistoryConfig
from tasks.antibody_rfd3.core import harness


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
    service = SimpleNamespace(config=HarnessHistoryConfig(), lineage_of=lambda profile: profile)
    bridge = SimpleNamespace(service=service, add_client=lambda client: None)
    args = SimpleNamespace(pool_size=2, antigen="1ADQ_A", seed=42, sidecar_image="fake-image",
                           guest_cache=None, method="harness")

    with harness.open_agent(args, tmp_path, object(), bridge, "test-campaign") as agent:
        assert agent.client is clients[0]
        assert agent.client.config.thinking == expected
        assert agent.client.config.initialize_payload()["thinking"] == expected
    assert preflights == [True]
