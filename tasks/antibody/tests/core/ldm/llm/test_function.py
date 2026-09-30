from __future__ import annotations

import pytest

from tasks.antibody.core.ldm.llm.client import LLMClient
from tasks.antibody.core.ldm.llm.function import LLMFunction


class ScriptedClient(LLMClient):
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.prompts: list[str] = []

    def call(self, prompt, temperature, timeout_s):
        self.prompts.append(prompt)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class IntegerFunction(LLMFunction):
    def construct_prompt(self, value):
        suffix = f" Previous error: {self.last_error}" if self.last_error else ""
        return f"Return {value}.{suffix}"

    def parse_response(self, raw, value):
        return int(raw)

    def fallback(self, value):
        return -value


def test_base_methods_define_required_subclass_contract():
    function = LLMFunction(ScriptedClient([]))
    with pytest.raises(NotImplementedError):
        function.construct_prompt()
    with pytest.raises(NotImplementedError):
        function.parse_response("")
    with pytest.raises(NotImplementedError, match="has no fallback"):
        function.fallback()


def test_retry_recovers_from_transport_and_parse_errors():
    client = ScriptedClient([RuntimeError("offline"), "not-an-int", "12"])
    function = IntegerFunction(client, max_retries=3, temperature=0.4, timeout_s=7)

    assert function(12) == 12
    assert function.fallback_used is False
    assert function.previous_attempts == [
        (None, "offline"),
        ("not-an-int", "invalid literal for int() with base 10: 'not-an-int'"),
    ]
    assert "offline" in client.prompts[1]
    assert "invalid literal" in client.prompts[2]


def test_retry_exhaustion_uses_fallback_and_resets_between_calls():
    client = ScriptedClient(["bad", "still bad", "8"])
    function = IntegerFunction(client, max_retries=2)

    assert function(4) == -4
    assert function.fallback_used is True
    assert function.last_error is not None

    assert function(8) == 8
    assert function.fallback_used is False
    assert function.previous_attempts == []
