"""AtomWorld receipt auditing and normalized scheduled submissions for pilots."""
from ldm_tts.pilot_evaluation.reporting import _json_object, _finite, _scalar_at_path
from .methods import LDM_METHODS


def scheduled_submissions(spec, records, observations, *, run_dir, result):
    if run_dir is None:
        raise ValueError("AtomWorld reporting requires same-round attempt receipts")
    values = _audit_trajectory(spec, records, observations, run_dir=run_dir, result=result)
    by_id = {row["candidate"]["candidate_id"]: row["candidate"] for row in observations}
    return [
        {"step": index, "candidate_id": row.get("candidate_id") or "",
         "canonical_key": by_id[row["candidate_id"]]["canonical_key"] if row.get("candidate_id") else "",
         "objective": value, "status": "succeeded" if row.get("candidate_id") else "missing"}
        for index, (row, value) in enumerate(zip(records, values, strict=True))
    ]


def _audit_trajectory(spec, records, observations, *, run_dir=None, result=None):
    if spec.trajectory.step_kind != "round":
        raise ValueError("final_submission requires an explicit scheduled round column")
    steps = [_finite(row.get(spec.trajectory.step_column), "submission round") for row in records]
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
    ldm_rounds = []
    for row, round_idx in zip(records, steps, strict=True):
        receipt = _attempt_receipt(run_dir, int(round_idx)) if run_dir is not None else None
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
                if result and result.get("search_method") in LDM_METHODS:
                    _audit_ldm_receipt(run_dir, int(round_idx), receipt, observation, result)
                    ldm_rounds.append({"round": int(round_idx), "sample_id": receipt["sample_id"], **receipt["selection"]})
            metrics = evaluation["metrics"]
            metric = metrics.get(spec.trajectory.objective_column)
            if metric is None or _finite(metric, "measured objective") != value:
                raise ValueError("Scheduled submission objective differs from its measured candidate")
        elif receipt is not None and receipt_key in observed_by_key:
            raise ValueError("Scheduled submission omits the measured same-round attempt")
        values.append(value)
    if result is not None:
        _check_result_consistency(spec, result, values)
        if result.get("search_method") in LDM_METHODS:
            from .selection import summarize_rounds

            if result.get("ldm_diagnostics") != summarize_rounds(ldm_rounds):
                raise ValueError("LDM diagnostics differ from scheduled selection receipts")
    return values


def _audit_ldm_receipt(run_dir, round_idx, receipt, observation, result):
    from ldm_tts.harness import file_sha256

    if observation.get("round_idx") != round_idx:
        raise ValueError("LDM scheduled submission must use its own round's evaluation")
    selection = receipt.get("selection", {})
    if selection.get("method") != result["search_method"]:
        raise ValueError("LDM selection method differs from result.json")
    sample = _json_object(run_dir / "schedule.json")
    if sample.get("search_method") != result["search_method"]:
        raise ValueError("LDM method differs from immutable scientific schedule")
    policy = selection.get("compiled_policy")
    if result["search_method"] != "ldm_harness_compiled" or round_idx == 0:
        if policy is not None:
            raise ValueError("Unexpected compiled policy outside measured optimization rounds")
        return
    if not isinstance(policy, dict):
        raise ValueError("Compiled LDM round is missing its policy receipt")
    # A final-submission pilot case is exactly one sample; paths remain task-owned.
    root = run_dir / "policy_harness" / "sample_000000" / "rounds" / f"round_{round_idx:03d}"
    saved = _json_object(root / "result.json")
    manifest = _json_object(root / "manifest.json")
    if saved.get("input_sha256") != manifest.get("input_sha256"):
        raise ValueError("Compiled LDM policy input digest mismatch")
    if file_sha256(root / "compiled.npz") != saved.get("compiled_arrays_sha256"):
        raise ValueError("Compiled LDM prior array receipt mismatch")
    for field in ("epoch_id", "artifact_sha256", "source", "degraded", "stage", "alpha", "eta"):
        if policy.get(field) != saved.get(field):
            raise ValueError(f"Compiled LDM selection differs from policy receipt: {field}")


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
