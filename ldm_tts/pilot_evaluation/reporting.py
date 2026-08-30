"""Aggregation and integrity evidence for fixed-round pilot evaluations."""

from __future__ import annotations

import csv
import json
import math
import operator
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ldm_tts.engine.run_store import atomic_json_write
from ldm_tts.pilot_evaluation.config import PilotEvaluationSpec


def write_evaluation_reports(spec: PilotEvaluationSpec, manifest: dict[str, Any]) -> None:
    """Validate completed child artifacts and export task-neutral evaluation outputs."""

    runs = _run_records(spec, manifest)
    rows, trajectories = _collect(spec, runs)
    integrity = _integrity(spec, rows, trajectories)
    if not integrity["valid"]:
        _write_outputs(spec, rows, trajectories, {"verdict": "invalid", "integrity": integrity})
        manifest.update(state="invalid", integrity=integrity)
        atomic_json_write(spec.output_root / "evaluation_manifest.json", manifest)
        raise RuntimeError("pilot evaluation integrity validation failed")
    aggregates = _aggregate(rows)
    verdict = _verdict(rows, aggregates, spec.trajectory.direction)
    _write_outputs(spec, rows, trajectories, verdict)
    manifest.update(state="completed", integrity=integrity)
    manifest.pop("error", None)
    manifest["reports"] = {
        "summary": "summary.csv", "summary_json": "summary.json", "trajectories": "trajectories.csv",
        "plot": "best_so_far.png", "verdict": "summary.json",
    }
    atomic_json_write(spec.output_root / "evaluation_manifest.json", manifest)


def _run_records(spec, manifest) -> list[tuple[str, dict[str, Any]]]:
    expected = {f"{case.case_id}/{method}/seed_{seed}" for case in spec.cases for method in spec.methods for seed in spec.seeds}
    actual = set(manifest["runs"])
    if actual != expected:
        raise ValueError("evaluation manifest does not contain the complete case/method/seed matrix")
    return sorted(manifest["runs"].items())


def _collect(spec, records) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for key, record in records:
        if record.get("status") != "completed":
            raise ValueError(f"incomplete child campaign: {key}")
        case, method, seed_text = key.split("/")
        run_dir = spec.output_root / record["run_dir"]
        result = _json_object(run_dir / "result.json")
        observations = _observations(run_dir)
        round_rows = _round_rows(
            spec, run_dir, case, method, int(seed_text.removeprefix("seed_")), observations
        )
        budget = _json_object(run_dir / "budget.json")
        config = _json_object(run_dir / "config.json")
        campaign = _json_object(run_dir / "campaign.json")
        evaluations = int(config["evaluations_per_round"])
        if len(observations) < evaluations:
            raise ValueError(
                f"checkpoint has fewer observations than its initialization batch: {run_dir}"
            )
        initial_candidate_ids = tuple(
            str(item["candidate"]["candidate_id"])
            for item in observations[:evaluations]
        )
        canonical_keys = [
            str(item["candidate"]["canonical_key"])
            for item in observations
        ]
        final_best = round_rows[-1]["best_so_far"]
        row = {
            "case": case, "method": method, "seed": int(seed_text.removeprefix("seed_")),
            "final_best": final_best,
            "round_auc": statistics.fmean(item["best_so_far"] for item in round_rows),
            **{name: _scalar_at_path(result, path) for name, path in spec.result_fields.items()},
            **_budget_fields(budget),
            "wall_time_seconds": _wall_time(run_dir),
            "evaluations_per_round": evaluations,
            "expected_evaluations": spec.iterations * evaluations,
            "evaluation_utilization": len(observations) / (spec.iterations * evaluations),
            "completed_rounds": len(round_rows),
            "proposal_samples": int(config["proposal_samples"]),
            "proposal_candidates_per_request": int(
                config.get("proposal_candidates_per_request", 1)
            ),
            "harness_candidates_per_session": int(config.get("harness_candidates_per_session", 0)),
            "contract_sha256": str(campaign["contract_sha256"]),
            "initial_candidate_ids": initial_candidate_ids,
            "candidate_ids_unique": len(canonical_keys) == len(set(canonical_keys)),
        }
        rows.append(row)
        trajectories.extend(round_rows)
    return rows, trajectories


def _round_rows(spec, run_dir, case, method, seed, observations) -> list[dict[str, Any]]:
    records = _read_trajectory(run_dir / "trajectory.csv")
    if len(records) != len(observations):
        raise ValueError(f"trajectory and checkpoint observation counts differ: {run_dir}")
    groups: dict[int, list[float]] = defaultdict(list)
    for index, (record, observation) in enumerate(zip(records, observations, strict=True)):
        step = int(_finite(record.get(spec.trajectory.step_column), "trajectory step"))
        if spec.trajectory.step_kind == "round":
            round_idx = step
        else:
            round_idx = observation.get("round_idx")
            if isinstance(round_idx, bool) or not isinstance(round_idx, int):
                raise ValueError(f"checkpoint observation {index} has invalid round_idx")
        groups[round_idx].append(_finite(record.get(spec.trajectory.objective_column), "trajectory objective"))
    expected_rounds = list(range(spec.iterations))
    if sorted(groups) != expected_rounds:
        raise ValueError(f"trajectory does not contain exactly {spec.iterations} completed rounds: {run_dir}")
    best = None
    result = []
    official_evaluations = 0
    select_best = max if spec.trajectory.direction == "maximize" else min
    for round_idx in expected_rounds:
        values = groups[round_idx]
        if not values:
            raise ValueError(f"round {round_idx} has no successful evaluations")
        official_evaluations += len(values)
        round_best = select_best(values)
        best = round_best if best is None else select_best(best, round_best)
        result.append({
            "case": case, "method": method, "seed": seed, "round": round_idx,
            "round_evaluations": len(values),
            "official_evaluations": official_evaluations,
            "round_best": round_best, "best_so_far": best,
        })
    return result


def _integrity(spec, rows, trajectories) -> dict[str, Any]:
    errors = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["seed"])].append(row)
    for key, group in grouped.items():
        if {item["method"] for item in group} != set(spec.methods):
            errors.append(f"missing method for {key}")
        initial_ids = {tuple(item["initial_candidate_ids"]) for item in group}
        if len(initial_ids) != 1:
            errors.append(f"shared initialization differs for {key}")
    for row in rows:
        if row["completed_rounds"] != spec.iterations:
            errors.append(f"unexpected completed round count for {row['case']}/{row['method']}/{row['seed']}")
        if row["budget_outer_iterations"] != spec.iterations:
            errors.append(f"unexpected outer iteration count for {row['case']}/{row['method']}/{row['seed']}")
        if not row["candidate_ids_unique"]:
            errors.append(f"duplicate canonical candidate in {row['case']}/{row['method']}/{row['seed']}")
        if row["method"] == "bo" and (row["budget_llm_requests"] != 0 or row["budget_proposal_attempts"] != 0):
            errors.append(f"BO used model proposals: {row['case']}/{row['seed']}")
        if row["method"] in {"ldm", "llm"}:
            actual = row["budget_proposal_attempts"]
            expected = _expected_model_proposal_attempts(
                row, spec.optimization_rounds
            )
            if actual != expected:
                errors.append(f"unexpected proposal count for {row['case']}/{row['method']}/{row['seed']}")
        if row["method"] == "harness":
            per_session = row["harness_candidates_per_session"]
            if per_session < 1 or row["proposal_samples"] % per_session:
                errors.append(f"invalid harness minibatch for {row['case']}/{row['seed']}")
            else:
                turns = spec.optimization_rounds * (row["proposal_samples"] // per_session)
                if row["budget_proposal_attempts"] != turns or row["budget_harness_turns"] != turns:
                    errors.append(f"unexpected harness turn count for {row['case']}/{row['seed']}")
    if len(trajectories) != len(rows) * spec.iterations:
        errors.append("round trajectory count is incomplete")
    return {"valid": not errors, "errors": errors}


def _expected_model_proposal_attempts(row: dict[str, Any], rounds: int) -> int:
    per_round = (
        row["proposal_samples"]
        if row["method"] == "ldm"
        else row["evaluations_per_round"]
    )
    candidates_per_request = row.get("proposal_candidates_per_request", 1)
    return rounds * math.ceil(per_round / candidates_per_request)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["method"])].append(row)
    output = []
    for (case, method), group in sorted(grouped.items()):
        output.append({
            "case": case, "method": method, "seed_count": len(group),
            "mean_final_best": statistics.fmean(item["final_best"] for item in group),
            "std_final_best": _sample_std(group, "final_best"),
            "mean_round_auc": statistics.fmean(item["round_auc"] for item in group),
            "std_round_auc": _sample_std(group, "round_auc"),
        })
    return output


def _verdict(rows, aggregates, direction: str) -> dict[str, Any]:
    cases = []
    better = operator.gt if direction == "maximize" else operator.lt
    for case in sorted({item["case"] for item in rows}):
        summary = {item["method"]: item for item in aggregates if item["case"] == case}
        verdict, wins = _method_verdict(rows, case, summary, "ldm", better)
        result = {"case": case, "verdict": verdict, "ldm_seed_wins": wins}
        if "harness" in summary:
            harness_verdict, harness_wins = _method_verdict(rows, case, summary, "harness", better)
            result.update(harness_verdict=harness_verdict, harness_seed_wins=harness_wins)
        cases.append(result)
    return {"schema_version": 1, "cases": cases, "aggregates": aggregates}


def _method_verdict(rows, case, summary, method, better):
    candidate = summary[method]
    baselines = [summary["bo"], summary["llm"]]
    seeds = sorted({item["seed"] for item in rows if item["case"] == case})
    wins = sum(
        all(
            better(
                _seed_metric(rows, case, seed, method, "round_auc"),
                _seed_metric(rows, case, seed, baseline, "round_auc"),
            )
            for baseline in ("bo", "llm")
        )
        for seed in seeds
    )
    better_auc = all(better(candidate["mean_round_auc"], item["mean_round_auc"]) for item in baselines)
    worse_final = any(better(item["mean_final_best"], candidate["mean_final_best"]) for item in baselines)
    if better_auc and wins >= 2:
        return "promising", wins
    if worse_final and wins == 0:
        return "not_promising", wins
    return "mixed", wins


def _seed_metric(rows, case, seed, method, field):
    return next(item[field] for item in rows if item["case"] == case and item["seed"] == seed and item["method"] == method)


def _write_outputs(spec, rows, trajectories, summary) -> None:
    _write_csv(spec.output_root / "summary.csv", rows)
    _write_csv(spec.output_root / "trajectories.csv", trajectories)
    atomic_json_write(spec.output_root / "summary.json", summary)
    _plot(spec, trajectories)


def _plot(spec, trajectories) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for item in trajectories:
        groups[(item["case"], item["method"], item["round"])].append(item["best_so_far"])
    for case in sorted({item["case"] for item in trajectories}):
        for method in spec.methods:
            points = [(round_idx, statistics.fmean(values)) for (label, name, round_idx), values in groups.items() if label == case and name == method]
            if points:
                points.sort()
                x = [item[0] + 1 for item in points]
                axis.plot(
                    x,
                    [item[1] for item in points],
                    label=f"{case}/{method} ({len(spec.seeds)} seeds)",
                )
                raw = {
                    round_idx: values
                    for (label, name, round_idx), values in groups.items()
                    if label == case and name == method
                }
                axis.fill_between(
                    x,
                    [min(raw[item[0]]) for item in points],
                    [max(raw[item[0]]) for item in points],
                    alpha=0.1,
                )
    axis.set_xlabel("Campaign round (round 1 is shared initialization)")
    axis.set_ylabel(
        f"Best {spec.trajectory.objective_column} so far ({spec.trajectory.direction})"
    )
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(spec.output_root / "best_so_far.png", dpi=160)
    plt.close(figure)


def _read_trajectory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"trajectory is empty: {path}")
    return rows


def _budget_fields(budget: dict[str, Any]) -> dict[str, float]:
    counters = budget.get("counters")
    if not isinstance(counters, dict):
        raise ValueError("budget counters must be an object")
    return {f"budget_{key}": _finite(value, f"budget {key}") for key, value in counters.items()}


def _observations(run_dir: Path) -> list[dict[str, Any]]:
    checkpoint = _json_object(run_dir / "checkpoint.json")
    state = checkpoint.get("state")
    if not isinstance(state, dict) or not isinstance(state.get("observations"), list):
        raise ValueError(f"invalid checkpoint observations: {run_dir}")
    return state["observations"]


def _wall_time(run_dir: Path) -> float:
    status = _json_object(run_dir / "status.json")
    return _finite(status["updated_at_unix"], "updated_at_unix") - _finite(status["started_at_unix"], "started_at_unix")


def _sample_std(rows, field):
    values = [item[field] for item in rows]
    return 0.0 if len(values) < 2 else statistics.stdev(values)


def _scalar_at_path(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"result field is missing: {path}")
        value = value[part]
    return _finite(value, path)


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["write_evaluation_reports"]
