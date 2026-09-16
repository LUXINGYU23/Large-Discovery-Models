"""Pilot reporting for tasks where the last scheduled answer is authoritative.

Repeated submissions may reuse a measured candidate. Tasks declare which past
feedback is allowed; this reporter never chooses a winner using judge scores.
"""
from collections import defaultdict
import statistics
from ldm_tts.engine.run_store import atomic_json_write
from .reporting import (_run_records, _json_object, _read_trajectory, _finite,
    _observations, _scalar_at_path, _write_csv, _budget_fields, _wall_time)


def submission_trajectory(spec, records, observations, *, run_dir=None, result=None):
    """Validate normalized scheduled rows; task adapters own scientific receipts."""
    from importlib import import_module
    from ldm_tts.registration.registry import get_task_definition

    task = getattr(spec, "task", None)
    hook = get_task_definition(task).pilot_evaluation.get("submission_adapter") if task else None
    if hook:
        module, name = hook.split(":")
        records = getattr(import_module(module), name)(
            spec, records, observations, run_dir=run_dir, result=result,
        )
    steps = [_finite(row.get("step"), "submission step") for row in records]
    if steps != list(range(spec.iterations)):
        raise ValueError("final_submission requires exactly one scheduled answer per step")
    observed = {}
    for observation in observations:
        candidate = observation.get("candidate", {})
        candidate_id = candidate.get("candidate_id")
        if not candidate_id or candidate_id in observed:
            raise ValueError("checkpoint requires distinct nonempty candidate IDs")
        observed[candidate_id] = observation
    values = []
    for row in records:
        if not {"step", "candidate_id", "canonical_key", "objective", "status"} <= row.keys():
            raise ValueError("Scheduled submission is missing normalized fields")
        value = _finite(row["objective"], "submission objective")
        candidate_id = row["candidate_id"]
        if row["status"] == "missing":
            if candidate_id or row["canonical_key"]:
                raise ValueError("Missing submission cannot reference a candidate")
        elif row["status"] == "succeeded":
            if candidate_id not in observed:
                raise ValueError("Scheduled submission references an unmeasured candidate")
            observation = observed[candidate_id]
            if not row["canonical_key"] or observation["candidate"].get("canonical_key") != row["canonical_key"]:
                raise ValueError("Scheduled submission canonical key differs from checkpoint")
            evaluation = observation["evaluation"]
            if evaluation.get("status") != "succeeded":
                raise ValueError("Scheduled submission references an unsuccessful evaluation")
            metric = evaluation["metrics"].get(spec.trajectory.objective_column)
            if _finite(metric, "measured objective") != value:
                raise ValueError("Scheduled submission objective differs from its measured candidate")
        else:
            raise ValueError("Scheduled submission has not been scientifically scored")
        values.append(value)
    return values

def write_submission_reports(spec, manifest):
    rows, trajectories = [], []
    for key, entry in _run_records(spec, manifest):
        if entry.get("status") != "completed":
            raise ValueError("Cannot report an incomplete submission campaign")
        case, method, seed_text = key.split("/")
        root = spec.output_root / entry["run_dir"]
        if _json_object(root / "status.json").get("status") != "completed":
            raise ValueError("Submission campaign has not completed")
        result = _json_object(root / "result.json")
        records = _read_trajectory(root / "trajectory.csv")
        values = submission_trajectory(
            spec,
            records,
            _observations(root),
            run_dir=root,
            result=result,
        )
        budget = _json_object(root / "budget.json")
        if budget["counters"]["outer_iterations"] != spec.iterations:
            raise ValueError("Submission campaign did not complete its scheduled round budget")
        row = {"case": case, "method": method, "seed": int(seed_text.removeprefix("seed_")),
            "first_submission": values[0], "final_submission": values[-1],
            "scheduled_submissions": len(values), "wall_time_seconds": _wall_time(root),
            **{name: _scalar_at_path(result, path) for name, path in spec.result_fields.items()},
            **_budget_fields(budget)}
        rows.append(row)
        trajectories.extend({"case": case, "method": method, "seed": row["seed"],
            "round": i, "submission_value": value} for i, value in enumerate(values))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["method"])].append(row["final_submission"])
    aggregates = [{"case": case, "method": method, "mean_final_submission": statistics.fmean(values),
        "std_final_submission": statistics.stdev(values) if len(values) > 1 else 0.0}
        for (case, method), values in sorted(groups.items())]
    _write_csv(spec.output_root / "summary.csv", rows)
    _write_csv(spec.output_root / "trajectories.csv", trajectories)
    atomic_json_write(spec.output_root / "summary.json", {"schema_version": 1,
        "selection_protocol": "final_submission", "aggregates": aggregates,
        "interpretation": "Final scheduled submission; no best-so-far selection."})
    manifest.update(state="completed", integrity={"valid": True, "errors": []},
        reports={"summary": "summary.csv", "summary_json": "summary.json", "trajectories": "trajectories.csv"})
    manifest.pop("error", None)
    atomic_json_write(spec.output_root / "evaluation_manifest.json", manifest)
