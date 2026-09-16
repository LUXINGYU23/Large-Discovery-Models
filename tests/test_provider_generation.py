import pytest
import json
import urllib.error
import urllib.request
from io import BytesIO

from ldm_tts.transport import ProposalRequest
from ldm_tts.transport.openai import EndpointRequestError, OpenAICompatibleProposalClient, generation_body


@pytest.mark.parametrize("wire,key,value", [
    ("responses", "reasoning", {"effort": "high"}),
    ("chat_completions", "reasoning_effort", "high"),
])
def test_explicit_reasoning_uses_the_selected_wire_contract(wire, key, value):
    assert generation_body(wire_api=wire, reasoning="high", extra_body={"top_p": 0.8}) == {
        key: value, "top_p": 0.8,
    }


@pytest.mark.parametrize("extra", [
    {"tools": []}, {"model": "other"}, {"temperature": 0},
    {"reasoning_effort": "high"}, {"reasoning": {"effort": "high"}},
    {"provider": {"api_key": "not-a-credential"}},
])
def test_generation_options_cannot_hide_protocol_or_credential_overrides(extra):
    with pytest.raises(ValueError):
        generation_body(wire_api="responses", extra_body=extra)


@pytest.mark.parametrize("wire", ["responses", "chat_completions"])
@pytest.mark.parametrize("reasoning,effort", [("off", "none"), ("none", "none"), ("high", "high")])
def test_actual_direct_preflight_and_proposal_bodies_disable_reasoning_explicitly(monkeypatch, wire, reasoning, effort):
    sent = []
    def urlopen(request, **kwargs):
        sent.append(json.loads(request.data))
        body = ({"model": "fixture", "status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}
                if wire == "responses" else {"model": "fixture", "choices": [{"message": {"content": "OK"}}]})
        return BytesIO(json.dumps(body).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAICompatibleProposalClient(url="http://fixture.invalid/v1", model="fixture",
        wire_api=wire, extra_body=generation_body(wire_api=wire, reasoning=reasoning))
    client.preflight()
    assert client.propose(ProposalRequest(messages=({"role": "user", "content": "OK"},))).text == "OK"
    assert len(sent) == 2
    for body in sent:
        assert (body["reasoning"]["effort"] if wire == "responses" else body["reasoning_effort"]) == effort


def test_provider_rejecting_none_fails_preflight_without_silent_fallback(monkeypatch):
    sent = []
    def urlopen(request, **kwargs):
        sent.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 400, "none unsupported", {}, BytesIO(b"none unsupported"))
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAICompatibleProposalClient(url="http://fixture.invalid/v1", model="fixture",
        wire_api="responses", extra_body=generation_body(wire_api="responses"))
    with pytest.raises(EndpointRequestError):
        client.preflight()
    assert len(sent) == 1 and sent[0]["reasoning"] == {"effort": "none"}
