"""Measured provider usage and the API spending stop rule.

Token counts come from provider responses: the direct transport's usage field,
or the Pi sidecar's redacted provider captures for Harness turns. Cost uses
explicitly configured per-million-token prices. Measured cost is always
recorded; the cap is checked before each new request or turn, so at most the
request in flight can overshoot it. Unparseable usage is counted as unknown,
never as zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ApiBudgetExhausted(RuntimeError):
    """Measured API spending reached the configured cap."""


@dataclass(frozen=True)
class ApiPricing:
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None
    budget_usd: float | None = None

    @property
    def priced(self) -> bool:
        return self.input_usd_per_mtok is not None and self.output_usd_per_mtok is not None

    def cost(self, tokens: dict[str, float]) -> float | None:
        if not self.priced:
            return None
        return (tokens.get("input_tokens", 0) * self.input_usd_per_mtok
                + tokens.get("output_tokens", 0) * self.output_usd_per_mtok) / 1_000_000

    def check(self, runtime) -> None:
        if self.budget_usd is None:
            return
        spent = runtime.budget.counters.get("api_cost_usd", 0.0)
        if spent >= self.budget_usd:
            raise ApiBudgetExhausted(f"measured API cost {spent:.4f} USD reached the {self.budget_usd} USD cap")

    def authorize_reservation(self, runtime, *, outstanding_usd: float, request_usd: float) -> None:
        """Admit one provider request without allowing concurrent turn calls to bypass a cap.

        The provider reports exact usage only after a response. Harness callers
        therefore reserve a configured upper bound before opening the request and
        settle it from the sidecar capture afterwards. The reserve must be
        conservative: it is a hard-stop guard, not a price estimate.
        """
        if self.budget_usd is None:
            return
        if request_usd <= 0:
            raise ValueError("provider request reservation must be positive when an API budget is configured")
        if outstanding_usd < 0:
            raise ValueError("outstanding provider reservations cannot be negative")
        spent = float(runtime.budget.counters.get("api_cost_usd", 0.0))
        projected = spent + outstanding_usd + request_usd
        if projected > self.budget_usd:
            raise ApiBudgetExhausted(
                "API budget would be exceeded by the next Harness provider request: "
                f"measured={spent:.4f}, reserved={outstanding_usd:.4f}, "
                f"next={request_usd:.4f}, cap={self.budget_usd:.4f} USD"
            )


def normalize_tokens(usage: Any) -> dict[str, float] | None:
    """Map Chat Completions or Responses usage objects onto input/output tokens."""
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 for v in (input_tokens, output_tokens)):
        return None
    return {"input_tokens": float(input_tokens), "output_tokens": float(output_tokens)}


def account_tokens(runtime, pricing: ApiPricing, prefix: str, tokens: dict[str, float] | None, *, usage_key: str) -> None:
    """Record cumulative tokens and cost for one operation, or mark usage unknown."""
    if tokens is None:
        runtime.consume_many({f"{prefix}_usage_unknown": 1}, usage_key=usage_key)
        return
    amounts = {f"{prefix}_input_tokens": tokens["input_tokens"], f"{prefix}_output_tokens": tokens["output_tokens"]}
    cost = pricing.cost(tokens)
    if cost is not None:
        amounts["api_cost_usd"] = cost
    runtime.consume_many(amounts, usage_key=usage_key)


def harness_turn_tokens(artifact_root: Path, turn_id: str) -> dict[str, float] | None:
    """Sum provider-reported tokens for one turn from the sidecar's redacted captures."""
    totals = {"input_tokens": 0.0, "output_tokens": 0.0}
    exchanges = 0
    for index in Path(artifact_root).rglob("provider_index.jsonl"):
        for line in index.read_text().splitlines():
            record = json.loads(line)
            if record.get("turnId") != turn_id:
                continue
            if record.get("type") == "provider_transport_error":
                return None
            if record.get("type") != "provider_exchange":
                continue
            exchanges += 1
            tokens = _response_tokens(index.parent / record["response"]["artifact"])
            if tokens is None:
                return None
            totals = {name: totals[name] + tokens[name] for name in totals}
    return totals if exchanges else None


def _response_tokens(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    text = path.read_bytes().decode("utf-8", errors="replace")
    found = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = (event.get("response") or {}).get("usage") if event.get("type") == "response.completed" else event.get("usage")
        tokens = normalize_tokens(usage)
        if tokens is not None:
            found = tokens
    return found


def captured_turn_ids(artifact_root: Path) -> list[str]:
    """Turn identities that have provider captures below one Harness artifact root."""
    turns = set()
    for index in Path(artifact_root).rglob("provider_index.jsonl"):
        for line in index.read_text().splitlines():
            turn = json.loads(line).get("turnId")
            if turn:
                turns.add(turn)
    return sorted(turns)


def account_pool_tokens(runtime, pricing: ApiPricing, artifact_root: Path, prefix: str) -> None:
    """Idempotently record cumulative tokens for every captured turn of a pool."""
    for turn in captured_turn_ids(artifact_root):
        account_tokens(runtime, pricing, prefix, harness_turn_tokens(artifact_root, turn), usage_key=f"{prefix}:{turn}")
