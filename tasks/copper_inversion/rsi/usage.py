"""Count actual provider usage from preserved response streams (contracts §6).

``collect`` is what the agent reports per turn. ``recompute``/``verify`` are a
separately written recount over the whole run directory for AT-17; they share
no parsing code with ``collect`` on purpose.
"""

import argparse
import json
import re
import sys
from pathlib import Path

COUNTS = ("requests", "input_tokens", "cached_input_tokens", "output_tokens", "responses_missing_usage",
          "compactions")


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


def collect(turn_root, provider_calls=None):
    """Usage of one sidecar turn directory.

    ``requests`` is the larger of the sidecar's authorized provider calls and the
    response files; every request without a parseable usage block is counted in
    ``responses_missing_usage`` and never estimated. A count mismatch is recorded
    (``provider_calls``, ``response_files``), not treated as an error.
    """
    turn_root = Path(turn_root)
    response_files = sorted(turn_root.rglob("*.response.bin"))
    totals = {"requests": 0, "input_tokens": 0, "cached_input_tokens": 0,
              "output_tokens": 0, "responses_missing_usage": 0, "compactions": 0}
    parsed = 0
    for path in response_files:
        counts = _usage(path)
        if counts is not None:
            parsed += 1
            for key, value in zip(("input_tokens", "cached_input_tokens", "output_tokens"), counts):
                totals[key] += value
    calls = provider_calls if type(provider_calls) is int and provider_calls >= 0 else None
    totals["requests"] = max(len(response_files), calls or 0)
    totals["responses_missing_usage"] = totals["requests"] - parsed
    totals["provider_calls"], totals["response_files"] = calls, len(response_files)
    events = turn_root / "history-events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            if line.strip() and json.loads(line).get("type") == "compaction_committed":
                totals["compactions"] += 1
    return totals


# Independent recount (AT-17). Sidecar layout: <artifact root>/sessions/<profile>/turns/<turn id>/
# holds provider/<turn id>-provider-<n>.{request.bin,authorized.json,response.bin} and
# history-events.jsonl; <artifact root>/turns/<turn id>/turn_committed.json is the sidecar's log.
TURN_ID = re.compile(r"round_(\d{3})(?:_t(\d+))?")


def _sse_events(body):
    """Server-sent events per the SSE framing: blank-line separated blocks, data lines joined."""
    text = body.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    for block in text.split("\n\n"):
        data = [line[5:][1:] if line[5:].startswith(" ") else line[5:]
                for line in block.split("\n") if line.startswith("data:")]
        if data:
            yield "\n".join(data)


def response_usage(body):
    """(input, cached, output) from one response body, or None when it has no usable usage."""
    blocks = []
    for data in _sse_events(body):
        try:
            blocks.append(json.loads(data))
        except ValueError:
            pass
    if blocks:
        usages = [b["response"].get("usage") for b in blocks if isinstance(b, dict)
                  and b.get("type") == "response.completed" and isinstance(b.get("response"), dict)]
    else:
        try:
            whole = json.loads(body)
        except ValueError:
            return None
        usages = [whole.get("usage")] if isinstance(whole, dict) else []
    if len(usages) != 1 or not isinstance(usages[0], dict):
        return None
    usage = usages[0]
    details = usage.get("input_tokens_details")
    if details is None:
        details = {}
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else -1
    numbers = (usage.get("input_tokens"), cached, usage.get("output_tokens"))
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in numbers):
        return None
    return numbers if numbers[1] <= numbers[0] else None


def recount_turn(turn):
    """Counts of one turn directory from its files alone."""
    provider = Path(turn) / "provider"
    authorized = len(list(provider.glob("*-provider-*.authorized.json")))
    responses = sorted(provider.glob("*-provider-*.response.bin"))
    with_usage = [u for u in (response_usage(p.read_bytes()) for p in responses) if u is not None]
    requests = max(authorized, len(responses))
    compactions = 0
    events = Path(turn) / "history-events.jsonl"
    if events.is_file():
        compactions = sum(1 for line in events.read_text().splitlines()
                          if line.strip() and json.loads(line).get("type") == "compaction_committed")
    return {"requests": requests, "input_tokens": sum(u[0] for u in with_usage),
            "cached_input_tokens": sum(u[1] for u in with_usage), "output_tokens": sum(u[2] for u in with_usage),
            "responses_missing_usage": requests - len(with_usage), "compactions": compactions,
            "authorized": authorized, "response_files": len(responses)}


def recompute(run_dir):
    """{round: {profile: counts}} over every preserved turn below ``harness/``, plus sidecar-log notes."""
    run_dir = Path(run_dir)
    rounds, log = {}, []
    for turn in sorted(run_dir.glob("harness/epoch_*/branch_*/sessions/*/turns/*")):
        match = TURN_ID.fullmatch(turn.name)
        if not turn.is_dir() or match is None:
            continue
        profile, artifact_root = turn.parent.parent.name, turn.parents[3]
        counts = recount_turn(turn)
        slot = rounds.setdefault(int(match[1]), {}).setdefault(profile, dict.fromkeys(COUNTS, 0))
        for key in COUNTS:
            slot[key] += counts[key]
        commit = artifact_root / "turns" / turn.name / "turn_committed.json"
        if commit.is_file():
            calls = json.loads(commit.read_text()).get("usage", {}).get("providerCalls")
            if calls != counts["authorized"]:
                log.append({"turn": str(turn.relative_to(run_dir)), "committed_provider_calls": calls,
                            "authorized_files": counts["authorized"]})
    return rounds, log


def verify(run_dir):
    """Compare usage/round_*.json and result.json with the recount; returns a report, never raises on a diff."""
    run_dir = Path(run_dir)
    counted, log = recompute(run_dir)
    differences, total_input, total_missing, recorded_rounds = [], 0, 0, set()
    for path in sorted((run_dir / "usage").glob("round_*.json")):
        record = json.loads(path.read_text())
        round_index = record["round"]
        recorded_rounds.add(round_index)
        mine = counted.get(round_index, {})
        for profile in sorted(set(record["by_profile"]) | set(mine)):
            reported = record["by_profile"].get(profile, {})
            ours = mine.get(profile, dict.fromkeys(COUNTS, 0))
            for key in COUNTS:
                if reported.get(key) != ours[key]:
                    differences.append({"round": round_index, "profile": profile, "field": key,
                                        "usage_file": reported.get(key), "recount": ours[key]})
            total_input += ours["input_tokens"]
            total_missing += ours["responses_missing_usage"]
        if record.get("run_total_input_tokens") != total_input:
            differences.append({"round": round_index, "field": "run_total_input_tokens",
                                "usage_file": record.get("run_total_input_tokens"), "recount": total_input})
    result = run_dir / "result.json"
    if result.is_file():
        report = json.loads(result.read_text())
        for key, value in (("input_tokens", total_input), ("responses_missing_usage", total_missing)):
            if key in report and report[key] != value:
                differences.append({"field": "result." + key, "usage_file": report[key], "recount": value})
    unrecorded = sorted(set(counted) - recorded_rounds)
    return {"status": "passed" if not differences and not log else "failed", "rounds": sorted(recorded_rounds),
            "unrecorded_rounds": unrecorded, "input_tokens": total_input, "responses_missing_usage": total_missing,
            "differences": differences, "sidecar_log_mismatches": log}


def main(argv=None):
    parser = argparse.ArgumentParser(description="AT-17: recount provider usage of one run directory.")
    parser.add_argument("run", type=Path)
    report = verify(parser.parse_args(argv).run)
    print(json.dumps(report, sort_keys=True, indent=1))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
