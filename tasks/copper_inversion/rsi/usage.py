"""Count actual provider usage from preserved response streams."""

import json
from pathlib import Path


def _usage(path):
    raw = Path(path).read_bytes().decode("utf-8", errors="replace")
    found = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            try:
                event = json.loads(line[6:])
            except ValueError:
                continue
            if event.get("type") == "response.completed":
                found.append(event.get("response", {}).get("usage"))
    if not found:
        try:
            payload = json.loads(raw)
            found = [payload.get("usage")]
        except (ValueError, AttributeError):
            return None
    if len(found) != 1 or not isinstance(found[0], dict):
        return None
    value = found[0]
    input_tokens, output_tokens = value.get("input_tokens"), value.get("output_tokens")
    details = value.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else None
    if any(type(n) is not int or n < 0 for n in (input_tokens, output_tokens, cached)) or cached > input_tokens:
        return None
    return input_tokens, cached, output_tokens


def collect(turn_root):
    turn_root = Path(turn_root)
    response_files = sorted(turn_root.rglob("*.response.bin"))
    totals = {"requests": len(response_files), "input_tokens": 0, "cached_input_tokens": 0,
              "output_tokens": 0, "responses_missing_usage": 0, "compactions": 0}
    for path in response_files:
        counts = _usage(path)
        if counts is None:
            totals["responses_missing_usage"] += 1
        else:
            for key, value in zip(("input_tokens", "cached_input_tokens", "output_tokens"), counts):
                totals[key] += value
    events = turn_root / "history-events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            if json.loads(line).get("type") == "compaction_committed":
                totals["compactions"] += 1
    return totals
