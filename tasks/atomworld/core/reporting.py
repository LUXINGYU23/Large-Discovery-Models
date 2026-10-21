"""Reports for the fixed public sample and revision schedule."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import write_json
from .methods import LDM_METHODS, HARNESS_METHODS
from .selection import summarize_rounds


def project_result(
    run_dir: Path,
    samples: list[dict],
    attempts: int,
    result,
    *,
    mock: bool,
    dataset_manifest: dict,
    proposal_format="cif",
    search_method="llm",
    runtime=None,
) -> dict:
    # Never select engine.best: it is a hidden-answer oracle. Fixed last-submission reporting.
    by_key = {
        observation.canonical_key: observation.evaluation
        for observation in result.state.observations
    }
    rows = []
    ldm_rounds = []
    for sample_index, sample in enumerate(samples):
        details = []
        for attempt in range(attempts):
            ordinal = sample_index * attempts + attempt
            path = run_dir / "attempts" / f"{ordinal:06d}.json"
            raw = json.loads(path.read_text()) if path.exists() else None
            evaluation = by_key.get(raw["canonical_key"]) if raw else None
            if raw and search_method in LDM_METHODS:
                ldm_rounds.append({"round": ordinal, "sample_id": sample["sample_id"], **raw["selection"]})
            if evaluation is not None and not evaluation.succeeded:
                message = (
                    evaluation.error
                    or f"evaluation status {evaluation.status}"
                )
                error = RuntimeError(
                    f"AtomWorld evaluation failed for round {ordinal:06d}: {message}"
                )
                if runtime is not None:
                    runtime.fail(error)
                raise error
            details.append(
                {
                    "attempt": attempt + 1,
                    "submitted": raw is not None,
                    "correct": bool(
                        evaluation and evaluation.metrics.get("correct") == 1
                    ),
                    "candidate_id": evaluation.candidate_id if evaluation else None,
                    "error": None if evaluation else "missing_or_rejected_submission",
                }
            )
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "action_name": sample["action_name"],
                "one_shot_correct": details[0]["correct"],
                "final_correct": details[-1]["correct"],
                "oracle_any_attempt_correct": any(item["correct"] for item in details),
                "attempts": details,
            }
        )
    count = len(rows)
    macro = {}
    for action in sorted({row["action_name"] for row in rows}):
        group = [row for row in rows if row["action_name"] == action]
        macro[action] = {
            "n": len(group),
            "one_shot_accuracy": sum(r["one_shot_correct"] for r in group) / len(group),
            "final_accuracy": sum(r["final_correct"] for r in group) / len(group),
        }
    payload = {
        "task": "atomworld",
        "qualification": "draft",
        "ldm_applicability": "measured_feedback_optimization" if search_method in LDM_METHODS else "blind_baseline",
        "feedback_protocol": "measured_correctness_optimization" if search_method in LDM_METHODS else "blind_refinement",
        "mode": "mock" if mock else "real",
        "proposal_format": proposal_format,
        "search_method": search_method,
        "selection_protocol": "final_submission",
        "dataset_kind": dataset_manifest["dataset_kind"],
        "paper_split_verified": dataset_manifest.get("paper_split_verified", False),
        "sample_count": count,
        "attempts_per_sample": attempts,
        "schedule_completed": result.state.next_round >= count * attempts and all(
            item["error"] is None for row in rows for item in row["attempts"]
        ),
        "one_shot_accuracy": sum(row["one_shot_correct"] for row in rows) / count,
        "extended_final_accuracy": sum(row["final_correct"] for row in rows) / count
        if attempts > 1
        else None,
        "oracle_any_attempt_accuracy_diagnostic": sum(
            row["oracle_any_attempt_correct"] for row in rows
        )
        / count,
        "final_selection": "last scheduled answer; missing/rejected final answers count as incorrect",
        "official_paper_score_claimed": False,
        "by_action": macro,
        "samples": rows,
        "engine_summary": result.summary,
        "warning": "Synthetic fixture only; not an LLM benchmark score"
        if mock
        else "Released local subset; not a verified paper test split",
    }
    if search_method in LDM_METHODS:
        payload["ldm_diagnostics"] = summarize_rounds(ldm_rounds)
    write_json(run_dir / "result.json", payload)
    if not payload["schedule_completed"] and runtime is not None:
        runtime.finish({**result.summary, "schedule_completed": False}, status="stopped")
    if search_method in HARNESS_METHODS:
        kinds = ("harness", "policy_harness") if (
            search_method == "harness_public_audit" or
            (search_method == "ldm_harness_compiled" and attempts > 1)
        ) else ("harness",)
        write_json(run_dir / "harness_provenance.json", {
            "schema_version": 1,
            "pools": {kind: [p.relative_to(run_dir).as_posix()
                for pattern in ("sample_*/manifest.json", "sample_*/main/manifest.json", "sample_*/branches/round_*/manifest.json")
                for p in sorted((run_dir / kind).glob(pattern))] for kind in kinds},
        })
    with (run_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "action_name",
                "round",
                "attempt",
                "correct",
                "submitted",
                "candidate_id",
            ],
        )
        writer.writeheader()
        for sample_index, row in enumerate(rows):
            for item in row["attempts"]:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "action_name": row["action_name"],
                        "round": sample_index * attempts + item["attempt"] - 1,
                        **{
                            key: item[key]
                            for key in (
                                "attempt",
                                "correct",
                                "submitted",
                                "candidate_id",
                            )
                        },
                        "correct": int(item["correct"]),
                    }
                )
    return payload
