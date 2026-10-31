#!/usr/bin/env python3
"""Normalize a pinned ResearchGym official-grade summary into a Table 3 receipt.

No model-supplied score is accepted.  This command reads a task's official JSON
summary, hashes it, and combines it with the LDM-CodeAgent cost/transcript
receipt.  Missing primary cells become explicit zeros only when the official
summary proves that the task grader ran but did not find that cell; their
completion flags remain false.  This is needed to reproduce the paper's
partial-completion accounting without silently dropping failed trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping

from tasks.researchgym.table3_protocol import CMR_PRIMARY_ROWS, TASKS


class GradeError(ValueError):
    """An official summary cannot support a Table 3 task result."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GradeError(f"{label} is not a finite number")
    return float(value)


def _mean_or_zero(values: Any, label: str, *, scale: float = 1.0) -> tuple[float, bool]:
    if not isinstance(values, list) or not values:
        return 0.0, False
    return fmean(_finite(value, label) for value in values) * scale, True


def _cl(summary: Mapping[str, Any], _: str | None) -> tuple[dict[str, float], dict[str, bool]]:
    tables = summary.get("tables")
    if not isinstance(tables, list):
        raise GradeError("CL summary lacks tables")
    for table in tables:
        for dataset in table.get("datasets", []) if isinstance(table, Mapping) else []:
            if dataset.get("dataset") == "CIFAR100" and dataset.get("num_tasks") == 10:
                metrics = dataset.get("metrics") or {}
                return ({"acc": _finite(metrics.get("acc_mean"), "CL acc_mean"),
                         "aaa": _finite(metrics.get("aaa_mean"), "CL aaa_mean")},
                        {"acc": True, "aaa": True})
    raise GradeError("CL summary lacks CIFAR100 N=10 official metrics")


def _mdt(summary: Mapping[str, Any], _: str | None) -> tuple[dict[str, float], dict[str, bool]]:
    classification = summary.get("classification")
    if not isinstance(classification, Mapping):
        raise GradeError("MDT summary lacks classification table")

    def aggregate(metric: str) -> tuple[float, bool]:
        table = classification.get(metric)
        if not isinstance(table, Mapping):
            raise GradeError(f"MDT summary lacks {metric}")
        values, completion = [], []
        for key in ("ner_sofc_test", "ner_matscholar_test"):
            cell = table.get(key)
            raw = cell.get("values") if isinstance(cell, Mapping) else None
            value, present = _mean_or_zero(raw, f"MDT {metric}/{key}", scale=100.0)
            values.append(value)
            completion.append(present)
        return fmean(values), all(completion)

    macro, macro_complete = aggregate("macro")
    micro, micro_complete = aggregate("micro")
    return {"macro_f1": macro, "micro_f1": micro}, {"macro_f1": macro_complete, "micro_f1": micro_complete}


def _cmr(summary: Mapping[str, Any], selector: str | None) -> tuple[dict[str, float], dict[str, bool]]:
    if selector not in {"qgs_clip_base", "qgs_blip_base"}:
        raise GradeError("CMR requires --cmr-row qgs_clip_base or qgs_blip_base fixed before launch")
    rows = summary.get("rows")
    if not isinstance(rows, list):
        raise GradeError("CMR summary lacks rows")
    row = next((item for item in rows if isinstance(item, Mapping) and item.get("row") == selector), None)
    if row is None:
        # Official grade did run; this particular primary row was absent.
        return {"i2tr_at_1": 0.0, "t2ir_at_1": 0.0}, {"i2tr_at_1": False, "t2ir_at_1": False}
    values = {item.get("id"): item.get("value") for item in row.get("columns", []) if isinstance(item, Mapping)}
    output, completion = {}, {}
    for output_key, source_key in (("i2tr_at_1", "flickr:txt_r1"), ("t2ir_at_1", "flickr:img_r1")):
        value = values.get(source_key)
        if value is None:
            output[output_key], completion[output_key] = 0.0, False
        else:
            output[output_key], completion[output_key] = _finite(value, f"CMR {source_key}"), True
    return output, completion


def _tim(summary: Mapping[str, Any], _: str | None) -> tuple[dict[str, float], dict[str, bool]]:
    real = summary.get("real")
    if not isinstance(real, Mapping):
        raise GradeError("TIM summary lacks real-data table")
    pam = real.get("PAM")
    if not isinstance(pam, Mapping):
        return {"cpd_a": 0.0, "cpd_z": 0.0}, {"cpd_a": False, "cpd_z": False}
    result, completion = {}, {}
    for output_key, source_key in (("cpd_a", "Average"), ("cpd_z", "Zeros")):
        cell = pam.get(source_key)
        if not isinstance(cell, Mapping) or cell.get("mean") is None:
            result[output_key], completion[output_key] = 0.0, False
        else:
            result[output_key], completion[output_key] = _finite(cell["mean"], f"TIM PAM {source_key}"), True
    return result, completion


def _irb(summary: Mapping[str, Any], _: str | None) -> tuple[dict[str, float], dict[str, bool]]:
    environments = summary.get("environments")
    if not isinstance(environments, Mapping):
        raise GradeError("IRB summary lacks environments")
    hopper = environments.get("Hopper-v2")
    value = hopper.get("average_return", {}).get("mean") if isinstance(hopper, Mapping) else None
    if value is None:
        return {"return": 0.0}, {"return": False}
    return {"return": _finite(value, "IRB Hopper-v2 average return")}, {"return": True}


EXTRACTORS: dict[str, Callable[[Mapping[str, Any], str | None], tuple[dict[str, float], dict[str, bool]]]] = {
    "continual_learning": _cl,
    "materials_tokenization": _mdt,
    "cross_modal_retrieval": _cmr,
    "time_series_explanation": _tim,
    "improving_replay_buffers": _irb,
}


def normalize(*, manifest: Mapping[str, Any], agent_receipt: Mapping[str, Any], summary_path: Path,
              task_id: str, cmr_row: str | None = None) -> dict[str, Any]:
    if task_id not in TASKS:
        raise GradeError(f"unknown task {task_id}")
    if task_id == "cross_modal_retrieval" and not manifest.get("table3_eligibility", {}).get("eligible"):
        raise GradeError("CMR cannot be normalized as Table 3 without a scoring contract")
    if agent_receipt.get("task_id") != task_id:
        raise GradeError("agent receipt task mismatch")
    for field in ("profile", "protocol", "source_commit", "model", "harness"):
        expected = manifest.get(field)
        actual = agent_receipt.get(field)
        if field == "profile":
            expected = manifest.get("profile", {}).get("name")
        if expected != actual:
            raise GradeError(f"agent receipt {field} mismatch")
    manifest_cmr_row = manifest.get("metric_selectors", {}).get("cmr_primary_row")
    if manifest_cmr_row not in CMR_PRIMARY_ROWS:
        raise GradeError("manifest has no valid immutable CMR primary row")
    if cmr_row is not None and cmr_row != manifest_cmr_row:
        raise GradeError("--cmr-row does not match the immutable manifest selector")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise GradeError("official summary is not an object")
    selector = manifest_cmr_row if task_id == "cross_modal_retrieval" else None
    metrics, completion = EXTRACTORS[task_id](summary, selector)
    return {
        "run_id": agent_receipt["run_id"],
        "status": "completed",
        "agent_status": agent_receipt["status"],
        "task_id": task_id,
        "seed": agent_receipt["seed"],
        "profile": agent_receipt["profile"],
        "protocol": agent_receipt["protocol"],
        "source_commit": agent_receipt["source_commit"],
        "model": agent_receipt["model"],
        "harness": agent_receipt["harness"],
        "wall_clock_hours": _finite(agent_receipt.get("wall_clock_hours"), "wall_clock_hours"),
        "api_cost_usd": _finite(agent_receipt.get("api_cost_usd"), "api_cost_usd"),
        "metrics": metrics,
        "metric_completion": completion,
        "official_grade": {
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
            "extractor": f"{task_id}_official_summary_v1",
            "cmr_row": selector,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--agent-receipt", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--cmr-row", choices=CMR_PRIMARY_ROWS,
                        help="Compatibility check only; must equal the manifest's pre-committed selector.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record = normalize(manifest=json.loads(args.manifest.read_text(encoding="utf-8")),
                           agent_receipt=json.loads(args.agent_receipt.read_text(encoding="utf-8")),
                           summary_path=args.official_summary, task_id=args.task, cmr_row=args.cmr_row)
    except (OSError, json.JSONDecodeError, GradeError) as exc:
        raise SystemExit(f"Table 3 result rejected: {exc}") from exc
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
