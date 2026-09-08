"""Pilot reporting for tasks where the last scheduled answer is authoritative.

Repeated submissions may reuse a measured candidate. Judges remain offline from
the proposal protocol; this reporter never chooses a winner using their scores.
"""
from collections import defaultdict
import statistics
from ldm_tts.engine.run_store import atomic_json_write
from .reporting import (_run_records, _json_object, _read_trajectory, _finite,
    _observations, _scalar_at_path, _write_csv, _budget_fields, _wall_time)


def submission_trajectory(spec, records, observations):
    if spec.trajectory.step_kind != "round":
        raise ValueError("final_submission requires an explicit scheduled round column")
    steps = [int(_finite(row.get(spec.trajectory.step_column), "submission round")) for row in records]
    if steps != list(range(spec.iterations)):
        raise ValueError("final_submission requires exactly one scheduled answer per round")
    sample_ids = {row.get("sample_id") for row in records if "sample_id" in row}
    if len(sample_ids) > 1:
        raise ValueError("Each final-submission pilot case must describe a single sample")
    observed = {o["candidate"]["candidate_id"]: o for o in observations}
    values = []
    for row in records:
        value = _finite(row.get(spec.trajectory.objective_column), "submission objective")
        candidate_id = row.get("candidate_id")
        if not candidate_id and value != 0:
            raise ValueError("A missing submission must have the protocol's zero score")
        if candidate_id and candidate_id not in observed:
            raise ValueError("Scheduled submission references an unmeasured candidate")
        if candidate_id:
            metrics = observed[candidate_id]["evaluation"]["metrics"]
            metric = metrics.get(spec.trajectory.objective_column)
            if metric is None or _finite(metric, "measured objective") != value:
                raise ValueError("Scheduled submission objective differs from its measured candidate")
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
        values = submission_trajectory(spec, records, _observations(root))
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
        "interpretation": "Final scheduled submission; hidden judge values do not select an answer. No LDM/BO advantage claim."})
    manifest.update(state="completed", integrity={"valid": True, "errors": []},
        reports={"summary": "summary.csv", "summary_json": "summary.json", "trajectories": "trajectories.csv"})
    manifest.pop("error", None)
    atomic_json_write(spec.output_root / "evaluation_manifest.json", manifest)
