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


METHOD_LABELS = {
    "ldm": "LDM",
    "ldm_harness": "LDM + Research Harness",
    "ldm_harness_compiled": "Harness-Compiled LDM",
    "bo": "Bayesian Optimization",
    "llm": "Direct LLM",
    "harness": "Direct Research Harness",
}
_COMPILED_METHOD = "ldm_harness_compiled"
_REPORT_BUDGET_COUNTERS = (
    "outer_iterations",
    "llm_requests",
    "proposal_attempts",
    "harness_turns",
    "harness_tool_calls",
    "harness_validation_submissions",
    "harness_artifact_bytes",
    "harness_wall_time_seconds",
    "policy_harness_turns",
    "policy_provider_requests",
    "policy_tool_calls",
    "policy_validation_submissions",
    "policy_artifact_bytes",
    "policy_wall_time_seconds",
    "valid_search_candidates",
    "selected_candidates",
    "external_evaluations",
    "expensive_evaluation_attempts",
    "successful_evaluations",
    "benchmark_jobs",
)


def write_evaluation_reports(spec: PilotEvaluationSpec, manifest: dict[str, Any]) -> None:
    """Validate completed child artifacts and export task-neutral evaluation outputs."""
    if spec.selection_protocol == "final_submission":
        from ldm_tts.pilot_evaluation.submission_reporting import write_submission_reports
        return write_submission_reports(spec, manifest)
    runs = _run_records(spec, manifest)
    rows, trajectories, policy_rounds = _collect(spec, runs)
    integrity = _integrity(spec, rows, trajectories)
    if not integrity["valid"]:
        _write_outputs(
            spec,
            rows,
            trajectories,
            policy_rounds,
            {"verdict": "invalid", "integrity": integrity},
        )
        manifest.update(state="invalid", integrity=integrity)
        atomic_json_write(spec.output_root / "evaluation_manifest.json", manifest)
        raise RuntimeError("pilot evaluation integrity validation failed")
    aggregates = _aggregate(rows)
    verdict = _verdict(rows, aggregates, spec.trajectory.direction)
    _write_outputs(spec, rows, trajectories, policy_rounds, verdict)
    manifest.update(state="completed", integrity=integrity)
    manifest.pop("error", None)
    manifest["reports"] = {
        "summary": "summary.csv", "summary_json": "summary.json", "trajectories": "trajectories.csv",
        "plot": "best_so_far.png", "verdict": "summary.json",
    }
    if policy_rounds:
        manifest["reports"]["compiled_policy_rounds"] = "compiled_policy_rounds.csv"
    atomic_json_write(spec.output_root / "evaluation_manifest.json", manifest)


def _run_records(spec, manifest) -> list[tuple[str, dict[str, Any]]]:
    expected = {f"{case.case_id}/{method}/seed_{seed}" for case in spec.cases for method in spec.methods for seed in spec.seeds}
    actual = set(manifest["runs"])
    if actual != expected:
        raise ValueError("evaluation manifest does not contain the complete case/method/seed matrix")
    return sorted(manifest["runs"].items())


def _collect(
    spec,
    records,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    policy_rounds: list[dict[str, Any]] = []
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
        initial_candidate_ids = tuple(
            str(item["candidate"]["candidate_id"])
            for item in observations if item["round_idx"] == 0
        )
        if not initial_candidate_ids:
            raise ValueError(f"checkpoint has no initialization observations: {run_dir}")
        expected_evaluations = len(initial_candidate_ids) + spec.optimization_rounds * evaluations
        canonical_keys = [
            str(item["candidate"]["canonical_key"])
            for item in observations
        ]
        final_best = round_rows[-1]["best_so_far"]
        row = {
            "case": case,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seed": int(seed_text.removeprefix("seed_")),
            "final_best": final_best,
            "round_auc": statistics.fmean(item["best_so_far"] for item in round_rows),
            **{name: _scalar_at_path(result, path) for name, path in spec.result_fields.items()},
            **_budget_fields(budget),
            "wall_time_seconds": _wall_time(run_dir),
            "evaluations_per_round": evaluations,
            "expected_evaluations": expected_evaluations,
            "evaluation_utilization": len(observations) / expected_evaluations,
            "completed_rounds": len(round_rows),
            "proposal_samples": int(config["proposal_samples"]),
            "proposal_candidates_per_request": int(
                config.get("proposal_candidates_per_request", 1)
            ),
            "harness_candidates_per_session": int(config.get("harness_candidates_per_session", 0)),
            "contract_sha256": str(campaign["contract_sha256"]),
            "initial_candidate_ids": initial_candidate_ids,
            "candidate_ids_unique": len(canonical_keys) == len(set(canonical_keys)),
            "proposal_counting": config.get("proposal_counting", "fixed"),
            "max_replenishment_batches": int(config.get("max_replenishment_batches", 0)),
            "harness_sessions": int(config.get("harness_sessions", 1)),
            "mock_proposals": bool(config.get("mock")) and config.get("proposal_mode") == "mock",
        }
        if method == _COMPILED_METHOD:
            compiled = _compiled_policy_records(
                run_dir,
                case,
                int(seed_text.removeprefix("seed_")),
                spec.policy_fields,
            )
            row.update(_compiled_policy_summary(compiled, spec.policy_mean_fields))
            policy_rounds.extend(compiled)
        rows.append(row)
        trajectories.extend(round_rows)
    return rows, trajectories, policy_rounds


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
            "case": case,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seed": seed,
            "round": round_idx,
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
        if row.get("proposal_counting") == "bounded_minibatches":
            _bounded_proposal_integrity(row, spec.optimization_rounds, errors)
            continue
        if row["method"] in {"ldm", "llm"}:
            actual = row["budget_proposal_attempts"]
            expected = _expected_model_proposal_attempts(
                row, spec.optimization_rounds
            )
            if actual != expected:
                errors.append(f"unexpected proposal count for {row['case']}/{row['method']}/{row['seed']}")
        if row["method"] in {"ldm_harness", _COMPILED_METHOD}:
            per_session = row["harness_candidates_per_session"]
            if per_session < 1 or row["proposal_samples"] % per_session:
                errors.append(
                    f"invalid LDM Harness minibatch for {row['case']}/{row['method']}/{row['seed']}"
                )
            else:
                turns = spec.optimization_rounds * (row["proposal_samples"] // per_session)
                if row["budget_proposal_attempts"] != turns or row["budget_harness_turns"] != turns:
                    errors.append(
                        f"unexpected LDM Harness turn count for {row['case']}/{row['method']}/{row['seed']}"
                    )
            if row["method"] == _COMPILED_METHOD:
                if row.get("budget_policy_harness_turns", 0) != spec.optimization_rounds:
                    errors.append(
                        f"unexpected compiled policy attempt count for {row['case']}/{row['seed']}"
                    )
                if row.get("policy_rounds", 0) != spec.optimization_rounds:
                    errors.append(
                        f"unexpected compiled policy result count for {row['case']}/{row['seed']}"
                    )
        if row["method"] == "harness":
            if row["proposal_samples"] != row["evaluations_per_round"]:
                errors.append(
                    f"invalid direct Harness minibatch for {row['case']}/{row['seed']}"
                )
            turns = spec.optimization_rounds
            if (
                row["budget_proposal_attempts"] != turns
                or row["budget_harness_turns"] != turns
            ):
                errors.append(
                    f"unexpected direct Harness turn count for {row['case']}/{row['seed']}"
                )
    if len(trajectories) != len(rows) * spec.iterations:
        errors.append("round trajectory count is incomplete")
    return {"valid": not errors, "errors": errors}


def _bounded_proposal_integrity(row, rounds, errors):
    """Validate independently budgeted replenishment without assuming one turn per round."""
    batch = row["proposal_candidates_per_request"]
    samples = row["proposal_samples"]
    refill = row["max_replenishment_batches"]
    if batch < 1 or samples < row["evaluations_per_round"] or refill < 0:
        errors.append("invalid bounded proposal sampling configuration")
        return
    minimum = rounds * math.ceil(samples / batch)
    maximum = minimum + rounds * refill
    actual = row["budget_proposal_attempts"]
    if row["method"] == "bo" or row.get("mock_proposals"):
        if actual != 0:
            errors.append("local proposal source unexpectedly recorded model attempts")
    elif not minimum <= actual <= maximum:
        errors.append("bounded proposal attempt count is outside the declared sampling allowance")
    if "harness" in row["method"]:
        turns = row["budget_harness_turns"]
        if not actual <= turns <= actual * min(batch, row["harness_sessions"]):
            errors.append("bounded Harness session count does not match proposal minibatches")
    if row["method"] == _COMPILED_METHOD:
        if row.get("policy_rounds") != rounds or row.get("budget_policy_harness_turns") != rounds:
            errors.append("compiled policy did not complete every optimization round")


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
            "case": case,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seed_count": len(group),
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
        result = {"case": case}
        if not {"bo", "llm"} <= summary.keys():
            cases.append(result)
            continue
        for method in ("ldm", "ldm_harness", _COMPILED_METHOD, "harness"):
            if method not in summary:
                continue
            verdict, wins = _method_verdict(rows, case, summary, method, better)
            verdict_key = "verdict" if method == "ldm" else f"{method}_verdict"
            wins_key = "ldm_seed_wins" if method == "ldm" else f"{method}_seed_wins"
            result.update({verdict_key: verdict, wins_key: wins})
        cases.append(result)
    return {"schema_version": 1, "cases": cases, "aggregates": aggregates}


def _method_verdict(rows, case, summary, method, better):
    candidate = summary[method]
    baselines = [summary["bo"], summary["llm"]]
    seed_rows = {
        (item["seed"], item["method"]): item
        for item in rows
        if item["case"] == case
    }
    seeds = sorted({seed for seed, _method in seed_rows})
    wins = sum(
        all(
            better(
                seed_rows[(seed, method)]["round_auc"],
                seed_rows[(seed, baseline)]["round_auc"],
            )
            for baseline in ("bo", "llm")
        )
        for seed in seeds
    )
    better_auc = all(better(candidate["mean_round_auc"], item["mean_round_auc"]) for item in baselines)
    worse_final = any(better(item["mean_final_best"], candidate["mean_final_best"]) for item in baselines)
    if better_auc and wins > len(seeds) / 2:
        return "promising", wins
    if worse_final and wins == 0:
        return "not_promising", wins
    return "mixed", wins


def _compiled_policy_records(
    run_dir: Path,
    case: str,
    seed: int,
    fields: dict[str, str],
) -> list[dict[str, Any]]:
    records = []
    with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    for event in events:
        if not isinstance(event, dict):
            raise ValueError(f"campaign event must be an object: {run_dir}")
        if event.get("event_type") != "candidates_selected":
            continue
        if not isinstance(event.get("payload"), dict):
            raise ValueError(f"compiled policy selection event is invalid: {run_dir}")
        payload = event["payload"]
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            continue
        policy = metadata.get("compiled_policy")
        if policy is None:
            continue
        if not isinstance(policy, dict):
            raise ValueError(f"compiled policy metadata is invalid: {run_dir}")
        round_index = event.get("iteration")
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise ValueError(f"compiled policy round index is invalid: {run_dir}")
        records.append(
            _compiled_policy_record(
                run_dir,
                case,
                seed,
                round_index,
                metadata,
                policy,
                fields,
            )
        )
    return sorted(records, key=lambda item: item["round"])


def _compiled_policy_record(
    run_dir: Path,
    case: str,
    seed: int,
    round_index: int,
    metadata: dict[str, Any],
    policy: dict[str, Any],
    fields: dict[str, str],
) -> dict[str, Any]:
    round_dir = run_dir / "policy_harness" / "rounds" / f"round_{round_index:03d}"
    persisted = _json_object(round_dir / "result.json")
    round_manifest = _json_object(round_dir / "manifest.json")
    if persisted.get("input_sha256") != round_manifest.get("input_sha256"):
        raise ValueError(f"compiled policy input digest mismatch: {round_dir}")
    for field in (
        "epoch_id",
        "artifact_sha256",
        "source",
        "degraded",
        "stage",
        "alpha",
        "eta",
    ):
        if not _equivalent(policy.get(field), persisted.get(field)):
            raise ValueError(
                f"compiled policy selector/result {field} mismatch: {round_dir}"
            )
    turn = policy.get("harness_turn")
    if turn is None:
        if policy.get("status") != "runtime_fallback" or policy.get("degraded") is not True:
            raise ValueError("a policy round without a committed turn must record runtime fallback")
        usage = policy.get("failed_harness_usage")
        if not isinstance(usage, dict) or usage != persisted.get("failed_harness_usage"):
            raise ValueError("compiled policy failed usage does not match the persisted result")
    elif not isinstance(turn, dict) or not isinstance(turn.get("usage"), dict):
        raise ValueError("compiled policy Harness usage is invalid")
    else:
        _validate_policy_turn(run_dir, persisted, turn)
        usage = turn["usage"]
    tool_calls = usage.get("toolCalls")
    if tool_calls is not None and not isinstance(tool_calls, dict):
        raise ValueError("compiled policy toolCalls must be an object")
    validation_errors = policy.get("validation_errors", [])
    if not isinstance(validation_errors, list):
        raise ValueError("compiled policy validation_errors must be an array")
    validation_submissions = usage.get("validationSubmissions")
    validation_failures = (
        max(int(validation_submissions) - int(policy.get("status") == "accepted"), 0)
        if validation_submissions is not None else None
    )
    record = {
        "case": case,
        "method": _COMPILED_METHOD,
        "method_label": METHOD_LABELS[_COMPILED_METHOD],
        "seed": seed,
        "round": round_index,
        "action": str(policy.get("action", "")),
        "status": str(policy.get("status", "")),
        "source": str(policy.get("source", "")),
        "epoch_id": policy.get("epoch_id"),
        "artifact_sha256": policy.get("artifact_sha256"),
        "degraded": bool(policy.get("degraded", False)),
        "turn_committed": turn is not None,
        "usage_complete": all(key in usage for key in (
            "providerCalls", "toolCalls", "artifactBytes", "validationSubmissions",
        )),
        "stage": str(policy.get("stage", "")),
        "alpha": _optional_finite(policy.get("alpha"), "compiled alpha"),
        "eta": _optional_finite(policy.get("eta"), "compiled eta"),
        "prior_clip_count": int(policy.get("prior_clip_count", 0)),
        "validation_error_count": len(validation_errors),
        "validation_submission_count": validation_submissions,
        "validation_failure_count": validation_failures,
        "provider_request_count": usage.get("providerCalls"),
        "tool_call_count": sum(int(value) for value in tool_calls.values()) if tool_calls is not None else None,
        "artifact_bytes": usage.get("artifactBytes"),
    }
    if fields.keys() & record.keys():
        raise ValueError("policy_fields must not replace built-in policy columns")
    for name, path in fields.items():
        value: Any = metadata
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        record[name] = value
    return record


def _validate_policy_turn(
    run_dir: Path,
    persisted: dict[str, Any],
    turn: Any,
) -> None:
    if not isinstance(turn, dict) or not isinstance(turn.get("turn_id"), str):
        raise ValueError(f"compiled policy Harness turn metadata is invalid: {run_dir}")
    committed = _json_object(
        run_dir
        / "policy_harness"
        / "turns"
        / turn["turn_id"]
        / "turn_committed.json"
    )
    expected = {
        "turnId": turn["turn_id"],
        "sessionId": turn.get("session_id"),
        "submissionDigest": persisted.get("submission_sha256"),
    }
    if any(committed.get(field) != value for field, value in expected.items()):
        raise ValueError(
            "compiled policy committed turn does not match selector metadata: "
            f"{run_dir}"
        )


def _compiled_policy_summary(
    records: list[dict[str, Any]], mean_fields: tuple[str, ...],
) -> dict[str, Any]:
    if not records:
        return {
            "policy_rounds": 0,
            "policy_degraded_rounds": 0,
            "policy_fallback_rounds": 0,
            "policy_committed_turns": 0,
            "policy_failed_turns": 0,
            "policy_usage_incomplete_rounds": 0,
            "policy_validation_failures": 0,
            "policy_prior_clip_count": 0,
        }
    last = records[-1]
    return {
        "policy_rounds": len(records),
        "policy_degraded_rounds": sum(bool(item["degraded"]) for item in records),
        "policy_fallback_rounds": sum(item["action"] == "fallback" for item in records),
        "policy_committed_turns": sum(item["turn_committed"] for item in records),
        "policy_failed_turns": sum(not item["turn_committed"] for item in records),
        "policy_usage_incomplete_rounds": sum(not item["usage_complete"] for item in records),
        "policy_validation_failures": sum(
            int(item["validation_failure_count"]) for item in records
            if item["validation_failure_count"] is not None
        ),
        "policy_prior_clip_count": sum(int(item["prior_clip_count"]) for item in records),
        "policy_last_action": last["action"],
        "policy_last_source": last["source"],
        "policy_last_epoch_id": last["epoch_id"],
        "policy_last_stage": last["stage"],
        "policy_last_alpha": last["alpha"],
        "policy_last_eta": last["eta"],
        **{f"policy_mean_{name}": _mean_present(records, name) for name in mean_fields},
    }

def _mean_present(records: list[dict[str, Any]], key: str) -> float | None:
    values = [_finite(item[key], key) for item in records if item.get(key) is not None]
    return statistics.fmean(values) if values else None


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _equivalent(left: Any, right: Any) -> bool:
    if (
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    return left == right


def _write_outputs(spec, rows, trajectories, policy_rounds, summary) -> None:
    _write_csv(spec.output_root / "summary.csv", rows)
    _write_csv(spec.output_root / "trajectories.csv", trajectories)
    if policy_rounds:
        _write_csv(
            spec.output_root / "compiled_policy_rounds.csv",
            policy_rounds,
        )
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
            series = sorted(
                (round_idx, values)
                for (label, name, round_idx), values in groups.items()
                if label == case and name == method
            )
            if series:
                x = [round_idx + 1 for round_idx, _values in series]
                axis.plot(
                    x,
                    [statistics.fmean(values) for _round_idx, values in series],
                    label=(
                        f"{case}/{METHOD_LABELS.get(method, method)} "
                        f"({len(spec.seeds)} seeds)"
                    ),
                )
                axis.fill_between(
                    x,
                    [min(values) for _round_idx, values in series],
                    [max(values) for _round_idx, values in series],
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
    names = set(counters) | set(_REPORT_BUDGET_COUNTERS)
    return {
        f"budget_{key}": _finite(counters.get(key, 0), f"budget {key}")
        for key in sorted(names)
    }


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
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be a numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in rows
        )


__all__ = ["write_evaluation_reports"]
