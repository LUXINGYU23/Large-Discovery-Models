"""Tests for portable OpenAI-compatible provider resolution."""

from __future__ import annotations

import pytest

from tasks.iron_mind.core.workflow_args import parse_openai_extra_body_json


def test_provider_extra_body_requires_a_json_object() -> None:
    assert parse_openai_extra_body_json('{"thinking":{"type":"disabled"}}') == {
        "thinking": {"type": "disabled"}
    }

    with pytest.raises(ValueError, match="not valid JSON"):
        parse_openai_extra_body_json("{")
    with pytest.raises(ValueError, match="JSON object"):
        parse_openai_extra_body_json("[]")
