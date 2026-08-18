"""Run-artifact validation for one completed Iron Mind tiny campaign."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tasks.iron_mind.core.qualification import COLLECTION_ARTIFACTS
from tasks.iron_mind.core.qualification_support import read_json_object
from tasks.iron_mind.core.tiny_campaign_support import (
    TinyCampaignRecordError,
    candidate_ids,
    candidate_key,
    event_payload,
    finite_number,
    one_event,
    reaction_score,
    read_json_lines,
    require_mapping,
    require_sequence,
)


def require_search_and_selection(
    run_dir: Path, events: Sequence[Mapping[str, Any]], seed_key: str
) -> tuple[list[str], str]:
    search = read_json_object(run_dir / "search_manifest.json", "search manifest")
    selection = read_json_object(run_dir / "selection_record.json", "selection record")
    rounds = require_sequence(search.get("rounds"), "search manifest rounds")
    selections = require_sequence(selection.get("selections"), "selection record selections")
    if len(rounds) != 1 or len(selections) != 1:
        raise TinyCampaignRecordError("Tiny campaign requires one reservoir and one selection record.")
    reservoir = event_payload(one_event(events, "reservoir_built"), "reservoir_built")
    manifest_payload = event_payload(require_mapping(rounds[0], "search manifest round"), "search manifest round")
    if manifest_payload != reservoir:
        raise TinyCampaignRecordError("Search manifest does not match the reservoir event.")
    candidate_pairs = candidate_ids(reservoir.get("candidate_ids"), count=4, label="reservoir candidate ids")
    candidate_keys = [key for _, key in candidate_pairs]
    if seed_key in candidate_keys:
        raise TinyCampaignRecordError("Campaign reservoir must not repeat the qualification seed.")
    selection_event = event_payload(one_event(events, "candidates_selected"), "candidates_selected")
    manifest_payload = event_payload(require_mapping(selections[0], "selection record entry"), "selection record entry")
    if manifest_payload != selection_event:
        raise TinyCampaignRecordError("Selection record does not match the selection event.")
    return _require_selection(selection_event, candidate_pairs)


def _require_selection(
    selection_event: Mapping[str, Any], candidate_pairs: list[tuple[str, str]]
) -> tuple[list[str], str]:
    selected_pairs = candidate_ids(selection_event.get("selected_candidate_ids"), count=1, label="selected candidate ids")
    predictions = require_sequence(selection_event.get("predictions"), "predictions")
    prediction_pairs = candidate_ids(
        [require_mapping(item, "selection prediction").get("candidate_id") for item in predictions],
        count=4,
        label="prediction candidate ids",
    )
    candidate_ids_set = {item[0] for item in candidate_pairs}
    if {item[0] for item in prediction_pairs} != candidate_ids_set:
        raise TinyCampaignRecordError("Selection must score the four reservoir candidates.")
    if selected_pairs[0][0] not in candidate_ids_set:
        raise TinyCampaignRecordError("Selected candidate is not in the candidate reservoir.")
    surrogate = require_mapping(
        require_mapping(selection_event.get("metadata"), "selection metadata").get("surrogate"),
        "selection surrogate",
    )
    if surrogate.get("history_size") != 1:
        raise TinyCampaignRecordError("Selection must use exactly one qualification seed prior.")
    return [key for _, key in candidate_pairs], selected_pairs[0][1]


def require_evaluation(
    run_dir: Path,
    events: Sequence[Mapping[str, Any]],
    selected_key: str,
    seed: Mapping[str, Any],
) -> float:
    checkpoint = read_json_object(run_dir / "checkpoint.json", "campaign checkpoint")
    observations = require_sequence(
        require_mapping(checkpoint.get("state"), "checkpoint state").get("observations"),
        "checkpoint observations",
    )
    if len(observations) != 1:
        raise TinyCampaignRecordError("Campaign checkpoint must contain exactly one non-seed observation.")
    observation = require_mapping(observations[0], "checkpoint observation")
    candidate = require_mapping(observation.get("candidate"), "evaluated candidate")
    candidate_id, candidate_key_value = candidate_key(candidate.get("candidate_id"))
    if candidate_key_value != selected_key or candidate_key_value == seed["canonical_key"]:
        raise TinyCampaignRecordError("Campaign checkpoint does not contain the selected non-seed candidate.")
    _require_source_schema(candidate, seed)
    evaluation = require_mapping(observation.get("evaluation"), "checkpoint evaluation")
    score = reaction_score(evaluation, candidate_id)
    if require_mapping(evaluation.get("metadata"), "evaluation metadata").get("schema_sha256") != seed["schema_sha256"]:
        raise TinyCampaignRecordError("Evaluation diagnostics do not match the source-pinned schema.")
    _require_evaluation_event(run_dir, events, candidate_id, score)
    return score


def _require_source_schema(candidate: Mapping[str, Any], seed: Mapping[str, Any]) -> None:
    payload = require_mapping(candidate.get("payload"), "evaluated candidate payload")
    metadata = require_mapping(candidate.get("metadata"), "evaluated candidate metadata")
    if payload.get("dataset_id") != seed["dataset_id"] or metadata.get("schema_sha256") != seed["schema_sha256"]:
        raise TinyCampaignRecordError("Evaluated candidate does not match the source-pinned dataset schema.")


def _require_evaluation_event(
    run_dir: Path, events: Sequence[Mapping[str, Any]], candidate_id: str, score: float
) -> None:
    manifest = read_json_object(run_dir / "evaluation_manifest.json", "evaluation manifest")
    evaluations = require_sequence(manifest.get("evaluations"), "evaluation manifest entries")
    event = event_payload(one_event(events, "candidate_evaluated"), "candidate_evaluated")
    if len(evaluations) != 1 or event_payload(require_mapping(evaluations[0], "evaluation manifest entry"), "evaluation manifest entry") != event:
        raise TinyCampaignRecordError("Evaluation manifest does not match the one evaluation event.")
    if reaction_score(require_mapping(event.get("evaluation"), "evaluation event"), candidate_id) != score:
        raise TinyCampaignRecordError("Evaluation event does not match the checkpoint evaluation.")


def require_reports(run_dir: Path, result: Mapping[str, Any], selected_key: str, score: float) -> None:
    best = require_mapping(result.get("best_candidate"), "result best candidate")
    candidate_id, candidate_key_value = candidate_key(best.get("candidate_id"))
    if candidate_key_value != selected_key or finite_number(best.get("reaction_score"), "result score") != score:
        raise TinyCampaignRecordError("Result does not match the selected frozen-table evaluation.")
    try:
        with (run_dir / "trajectory.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise TinyCampaignRecordError("Could not read campaign trajectory.") from exc
    if len(rows) != 1 or rows[0].get("candidate_id") != candidate_id:
        raise TinyCampaignRecordError("Trajectory does not contain the selected evaluation.")
    if _trajectory_score(rows[0].get("reaction_score")) != score:
        raise TinyCampaignRecordError("Trajectory score does not match the frozen-table evaluation.")


def require_collection(run_dir: Path) -> dict[str, int]:
    collection = {
        "ir_rows": len(read_json_lines(run_dir / COLLECTION_ARTIFACTS[0])),
        "sft_rows": len(read_json_lines(run_dir / COLLECTION_ARTIFACTS[1])),
    }
    if collection != {"ir_rows": 4, "sft_rows": 4}:
        raise TinyCampaignRecordError("Tiny campaign collection must contain exactly four IR and SFT rows.")
    return collection


def _trajectory_score(value: object) -> float:
    if not isinstance(value, str):
        raise TinyCampaignRecordError("Trajectory score must be a CSV numeric string.")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise TinyCampaignRecordError("Trajectory score must be a CSV numeric string.") from exc
    return finite_number(parsed, "trajectory score")


__all__ = [
    "require_collection",
    "require_evaluation",
    "require_reports",
    "require_search_and_selection",
]
