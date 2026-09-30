"""LLM client abstractions.

Public surface:

* :class:`LLMClient` — Protocol for any chat-completion backend.
* :class:`OpenAIChatClient` — production implementation against an
  OpenAI-compatible endpoint. Reads API key / base URL / model from
  :mod:`tasks.small_molecule.core.llm_advisor.config` (which loads ``.env``).
* :class:`MockLLMClient` — deterministic script-driven client for
  tests. Accepts a list of pre-recorded LLM response strings.

Both :class:`OpenAIChatClient` and :class:`MockLLMClient` set
``model_name`` so the orchestrator / trajectory can log which model
produced each response.

The ``MockLLMClient`` serves one pre-recorded response per call.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

from tasks.small_molecule.core.llm_advisor.config import LLMClientConfig, load_env

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Chat-completion backend protocol.

    Implementations must be thread-safe (or document otherwise) and
    must set :attr:`model_name` so the trajectory records which model
    produced each response.
    """

    model_name: str

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Production client
# ---------------------------------------------------------------------------


@dataclass
class OpenAIChatClient:
    """OpenAI-compatible chat-completion client.

    Settings are read from :class:`LLMClientConfig.from_env` when
    ``__post_init__`` runs. Pass overrides explicitly to bypass the
    environment (useful in tests).
    """

    config: LLMClientConfig
    temperature: float = 0.2
    timeout: float = 60.0
    max_tokens: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    extra_body: dict[str, Any] | None = None
    _client: Any = field(default=None, init=False, repr=False)
    model_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.model_name = self.config.model
        # Import lazily so unit tests that don't talk to the network
        # can still construct the dataclass.
        try:
            from openai import OpenAI                       # type: ignore
        except ImportError as exc:                          # pragma: no cover
            raise RuntimeError(
                "openai package not installed; pip install 'openai>=1.99.0'"
            ) from exc
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.timeout,
            max_retries=0,
        )

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
    ) -> str:
        """Send a chat-completion request and return the assistant text."""
        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        if json_mode:
            # OpenAI's structured output. LiteLLM proxies (the
            # project's default endpoint) honor this; if the backend
            # doesn't, the response just won't be JSON and the parser
            # will raise ParseError (which gets retried).
            kwargs["response_format"] = {"type": "json_object"}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        t0 = time.monotonic()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            LOGGER.warning(
                "LLM call failed after %.2fs: %s: %s",
                time.monotonic() - t0, type(exc).__name__, exc,
            )
            raise

        if not resp.choices:
            raise RuntimeError(f"LLM returned no choices (raw: {resp!r})")
        text = resp.choices[0].message.content or ""
        LOGGER.debug(
            "LLM call OK in %.2fs, returned %d chars",
            time.monotonic() - t0, len(text),
        )
        return text


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


@dataclass
class MockLLMClient:
    """Deterministic script-driven client for tests.

    The :meth:`chat` method serves each pre-recorded response in order.
    If no responses are set, it raises :class:`RuntimeError` on the first call.

    The mock records every call (system, user, response) in
    :attr:`call_log` for assertion in tests.
    """

    model_name: str = "mock-llm"
    scripted_responses: Optional[List[str]] = None
    fail_every: int = 0                                # if > 0, fail every Nth call with ParseError-ish text
    fail_text: str = "this is not a json block"

    call_log: List[dict] = field(default_factory=list, init=False, repr=False)
    _idx: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scripted_responses is not None:
            self.scripted_responses = list(self.scripted_responses)

    def _next_response(self) -> str:
        # Optional fail-every-N for retry tests.
        # `fail_every=2` means the 2nd, 4th, 6th, ... calls
        # (1-indexed) return the fail text. Call #1 always uses
        # the scripted response.
        if self.fail_every > 0 and (self._idx + 1) % self.fail_every == 0:
            self._idx += 1
            self.call_log.append({"response": self.fail_text, "forced_fail": True})
            return self.fail_text
        if self.scripted_responses is not None:
            if self._idx >= len(self.scripted_responses):
                raise RuntimeError(
                    f"MockLLMClient exhausted: {self._idx} calls vs "
                    f"{len(self.scripted_responses)} scripted responses"
                )
            text = self.scripted_responses[self._idx]
        else:
            raise RuntimeError("MockLLMClient: scripted_responses is not set")
        self._idx += 1
        return text

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
    ) -> str:
        text = self._next_response()
        self.call_log.append({
            "system": system,
            "user": user,
            "response": text,
            "idx": self._idx - 1,
        })
        return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_default_client_from_env(
    *, model: Optional[str] = None,
    timeout: float = 60.0,
) -> OpenAIChatClient:
    """Construct an :class:`OpenAIChatClient` from the current environment.

    Args:
        model: Optional explicit model name; defaults to the
            hardcoded :data:`tasks.small_molecule.core.llm_advisor.config.DEFAULT_LLM_MODEL`.
    """
    load_env()
    return OpenAIChatClient(
        LLMClientConfig.from_env(model=model),
        timeout=timeout,
        max_tokens=_max_tokens_from_env(),
    )


def _max_tokens_from_env() -> int | None:
    raw = os.environ.get("LDM_LLM_MAX_TOKENS")
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("LDM_LLM_MAX_TOKENS must be a positive integer")
    return value


__all__ = [
    "LLMClient",
    "OpenAIChatClient",
    "MockLLMClient",
    "build_default_client_from_env",
]
