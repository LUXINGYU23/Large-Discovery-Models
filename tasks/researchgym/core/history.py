"""Read-only measured-history projection built from engine observations only."""

from __future__ import annotations

import json
from pathlib import Path

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.harness import canonical_sha256


def measured_records(observations, objective: str) -> list[dict]:
    """Every evaluated candidate, including failures, in evaluation order."""
    records = []
    for seq, observation in enumerate(observations):
        evaluation = observation.evaluation
        records.append({
            "seq": seq,
            "candidate_id": observation.candidate.candidate_id,
            "canonical_key": observation.candidate.canonical_key,
            "round_idx": observation.round_idx,
            "status": evaluation.status,
            "objective": evaluation.metrics.get(objective) if evaluation.succeeded else None,
            "metrics": dict(evaluation.metrics),
            "error": evaluation.error[:600],
            "research_annotations": list(observation.candidate.metadata.get("research_annotations", [])),
            "program": observation.candidate.payload["program"],
        })
    return records


def compact(record: dict) -> dict:
    return {key: record[key] for key in ("seq", "candidate_id", "round_idx", "status", "objective")} | {
        "annotation_count": len(record["research_annotations"]), "program_chars": len(record["program"])}


def write_history(path: Path, records: list[dict]) -> str:
    """Persist the projection; authoritative history may only grow."""
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text())["observations"]
        if [r["candidate_id"] for r in records[:len(previous)]] != [r["candidate_id"] for r in previous]:
            raise ValueError("authoritative measured history changed or shrank on resume")
    digest = canonical_sha256(records)
    atomic_json_write(path, {"observations": records, "sha256": digest})
    return digest


def best_records(records: list[dict], limit: int) -> list[dict]:
    succeeded = [r for r in records if r["objective"] is not None]
    return sorted(succeeded, key=lambda r: (-r["objective"], r["seq"]))[:limit]
