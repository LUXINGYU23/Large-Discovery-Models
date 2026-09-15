"""Pilot reporting for tasks where the last scheduled answer is authoritative.

Repeated submissions may reuse a measured candidate. Judges remain offline from
the proposal protocol; this reporter never chooses a winner using their scores.
"""
from collections import defaultdict
import statistics
from ldm_tts.engine.run_store import atomic_json_write
from .reporting import (_run_records, _json_object, _read_trajectory, _finite,
    _observations, _scalar_at_path, _write_csv, _budget_fields, _wall_time)


def submission_trajectory(spec, records, observations, *, run_dir=None, result=None):
    if spec.trajectory.step_kind != "round":
        raise ValueError("final_submission requires an explicit scheduled round column")
    steps = [int(_finite(row.get(spec.trajectory.step_column), "submission round")) for row in records]
    if steps != list(range(spec.iterations)):
        raise ValueError("final_submission requires exactly one scheduled answer per round")
    sample_ids = {row.get("sample_id") for row in records if "sample_id" in row}
    if len(sample_ids) > 1:
        raise ValueError("Each final-submission pilot case must describe a single sample")
    observed = {}
    observed_by_key = {}
    for index, observation in enumerate(observations, start=1):
        candidate = observation.get("candidate")
        evaluation = observation.get("evaluation")
        if not isinstance(candidate, dict) or not isinstance(evaluation, dict):
            raise ValueError(f"checkpoint observation {index} is missing candidate or evaluation")
        candidate_id = candidate.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id:
            if candidate_id in observed:
                raise ValueError("checkpoint contains duplicate candidate_id values")
            observed[candidate_id] = observation
        canonical_key = candidate.get("canonical_key")
        if isinstance(canonical_key, str) and canonical_key:
            if canonical_key in observed_by_key:
                raise ValueError("checkpoint contains duplicate canonical_key values")
            observed_by_key[canonical_key] = observation
    values = []
    for row, round_idx in zip(records, steps, strict=True):
        receipt = _attempt_receipt(run_dir, round_idx) if run_dir is not None else None
        receipt_key = receipt.get("canonical_key") if receipt is not None else ""
        if receipt is not None:
            _check_receipt_row(row, receipt)
        value = _finite(row.get(spec.trajectory.objective_column), "submission objective")
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id and value != 0:
            raise ValueError("A missing submission must have the protocol's zero score")
        if candidate_id and candidate_id not in observed:
            raise ValueError("Scheduled submission references an unmeasured candidate")
        if candidate_id:
            observation = observed[candidate_id]
            evaluation = observation["evaluation"]
            if evaluation.get("status", "succeeded") != "succeeded":
                raise ValueError("Scheduled submission references an unsuccessful evaluation")
            if receipt is not None:
                candidate_key = observation["candidate"].get("canonical_key")
                if not isinstance(candidate_key, str) or not candidate_key:
                    raise ValueError("Measured submission is missing its canonical key")
                if candidate_key != receipt_key:
                    raise ValueError("Scheduled submission does not match its same-round attempt receipt")
            metrics = evaluation["metrics"]
            metric = metrics.get(spec.trajectory.objective_column)
            if metric is None or _finite(metric, "measured objective") != value:
                raise ValueError("Scheduled submission objective differs from its measured candidate")
        elif receipt is not None and receipt_key in observed_by_key:
            raise ValueError("Scheduled submission omits the measured same-round attempt")
        values.append(value)
    if result is not None:
        _check_result_consistency(spec, result, values)
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
        "interpretation": "Final scheduled submission; hidden judge values do not select an answer. No LDM/BO advantage claim."})
    manifest.update(state="completed", integrity={"valid": True, "errors": []},
        reports={"summary": "summary.csv", "summary_json": "summary.json", "trajectories": "trajectories.csv"})
    manifest.pop("error", None)
    atomic_json_write(spec.output_root / "evaluation_manifest.json", manifest)


def _attempt_receipt(run_dir, round_idx):
    path = run_dir / "attempts" / f"{round_idx:06d}.json"
    try:
        receipt = _json_object(path)
    except FileNotFoundError as exc:
        raise ValueError("Scheduled submission attempt receipt is missing") from exc
    actual_round = receipt.get("round_idx")
    if isinstance(actual_round, bool) or actual_round != round_idx:
        raise ValueError("Scheduled submission attempt receipt has the wrong round_idx")
    canonical_key = receipt.get("canonical_key")
    if not isinstance(canonical_key, str) or not canonical_key:
        raise ValueError("Scheduled submission attempt receipt is missing canonical_key")
    return receipt


def _check_receipt_row(row, receipt):
    for field in ("sample_id", "action_name"):
        if row.get(field) and row.get(field) != receipt.get(field):
            raise ValueError(f"Scheduled submission {field} disagrees with its attempt receipt")
    if row.get("attempt"):
        expected = int(_finite(row.get("attempt"), "submission attempt")) - 1
        actual = receipt.get("attempt")
        if isinstance(actual, bool) or actual != expected:
            raise ValueError("Scheduled submission attempt number disagrees with its receipt")


def _check_result_consistency(spec, result, values):
    fields = getattr(spec, "result_fields", {}) or {}
    for name, expected in (
        ("one_shot_accuracy", values[0]),
        ("extended_final_accuracy", values[-1]),
    ):
        path = fields.get(name)
        if path and _scalar_at_path(result, path) != expected:
            raise ValueError(f"result field {name} disagrees with scheduled submissions")
    samples = result.get("samples")
    if samples is None:
        return
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError("final_submission result.json must contain exactly one sample")
    attempts = samples[0].get("attempts") if isinstance(samples[0], dict) else None
    if not isinstance(attempts, list) or len(attempts) != len(values):
        raise ValueError("final_submission result.json attempt count disagrees with trajectory")
    for index, (attempt, expected) in enumerate(zip(attempts, values, strict=True), start=1):
        if not isinstance(attempt, dict):
            raise ValueError("final_submission result.json attempt rows must be objects")
        actual = attempt.get(spec.trajectory.objective_column)
        if isinstance(actual, bool):
            actual = int(actual)
        if _finite(actual, f"result attempt {index}") != expected:
            raise ValueError("final_submission result.json attempts disagree with trajectory")
