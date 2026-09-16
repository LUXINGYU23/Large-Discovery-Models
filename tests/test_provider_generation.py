import pytest

from ldm_tts.transport.openai import generation_body


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
