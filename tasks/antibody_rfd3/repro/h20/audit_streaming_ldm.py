"""Read-only acceptance and timing audit for a completed real streaming HER2 run.

Run on the campaign host after it has stopped. Selector replay uses persisted inputs
and priors; this program neither invokes the models nor contacts the LLM provider.
The optional barrier run is compared only after it finishes the same evaluation
budget. Timing comparisons are descriptive when campaigns share resources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from tasks.antibody_rfd3.repro.h20.audit_full_ldm import (
    Audit, ReadOnlyStore, ROLES, UTILITY, file_sha256, finite, onehot, read_json,
    read_json_value, read_jsonl, read_skills, successful_reads,
)


def _intervals(spans):
    """Validate positive finite intervals and merge their union, including adjacent spans."""
    result = []
    for start, end in sorted(spans):
        if not finite(start) or not finite(end) or end <= start:
            raise ValueError(f"invalid interval: {start!r}, {end!r}")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _length(spans):
    return math.fsum(end - start for start, end in spans)


def _intersection(left, right):
    return [(max(a, c), min(b, d)) for a, b in left for c, d in right
            if min(b, d) > max(a, c)]


def _timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("launcher timestamp has no timezone")
    return stamp.timestamp()


def runner_intervals(root, launch, start, end):
    """Intervals covered by actual launcher cases, including unsuccessful cases."""
    recoveries = launch.get("recoveries", [])
    if recoveries:
        first = read_json(root.parent / "recoveries" / "attempt-001" / "launch-before.json")
        cases = [first["case"], *(entry["case"] for entry in recoveries)]
    else:
        cases = [launch["case"]]
    spans = [(_timestamp(case["started_at"]), _timestamp(case["finished_at"])) for case in cases]
    merged = _intervals(spans)
    if not math.isclose(_length(merged), _length(spans), abs_tol=1e-6):
        raise ValueError("launcher case intervals overlap")
    return _intersection(merged, [(start, end)])


def research_intervals(db, generations, active):
    """Use append-only events so retries do not erase the first research attempt."""
    result = []
    for row in generations:
        number = row["generation"]
        events = db.rows("""SELECT seq,type,payload_json,created_at FROM events WHERE subject_id=?
            AND type IN ('generation.started','generation.blocked') ORDER BY seq""", (str(number),))
        begin = None
        previous = None
        for event in events:
            if event["type"] == "generation.started":
                if begin is not None or (previous is not None and previous != "generation.blocked"):
                    raise ValueError(f"generation {number}: invalid start event sequence")
                begin = event["created_at"]
            else:
                if begin is None:
                    raise ValueError(f"generation {number}: block without a start")
                for a, b in _intersection(_intervals([(begin, event["created_at"])]), active):
                    result.append({"generation": number, "state": row["state"], "started_at": a, "finished_at": b})
                begin = None
            previous = event["type"]
        if begin is None or row["finished_at"] is None:
            raise ValueError(f"generation {number}: missing terminal research interval")
        for a, b in _intersection(_intervals([(begin, row["finished_at"])]), active):
            result.append({"generation": number, "state": row["state"], "started_at": a, "finished_at": b})
    return result


def scoring_intervals(root, attempts):
    """Result file mtime estimates completion; ES end can be recorded after downtime."""
    spans, delayed = [], []
    for attempt in attempts:
        start, end = attempt["started_at"], attempt["ended_at"]
        if not finite(start) or not finite(end) or end <= start:
            raise ValueError("invalid scoring service interval")
        result = root / "absolut-attempts" / attempt["executor_ref"] / "result.json"
        stamp = result.stat().st_mtime
        if not finite(stamp) or stamp <= start:
            raise ValueError(f"invalid scoring result timestamp: {result}")
        # Filesystem timestamp precision may be coarser than the service clock.
        finish = min(end, stamp)
        spans.append(dict(attempt, ended_at=finish))
        if end > stamp:
            delayed.append(end - stamp)
    return spans, delayed


def scoring_attempt_problems(root, service, sequence, energy, attempts, asset_names):
    """Only a proven stopped infrastructure retry may precede one successful score."""
    eid = service["experiment_id"]
    problems = []
    attempts = sorted(attempts, key=lambda item: item["idx"])
    if [item["idx"] for item in attempts] != list(range(1, len(attempts) + 1)) or not attempts:
        problems.append(f"{eid}: missing or noncontiguous scoring attempts")
    successes = 0
    for attempt in attempts:
        reference = hashlib.sha256(json.dumps([eid, attempt["idx"]], separators=(",", ":")).encode()).hexdigest()
        base = root / "absolut-attempts" / reference
        request, result, worker = (read_json(base / name) for name in ("request.json", "result.json", "worker.json"))
        outcome = read_json_value(attempt["outcome"])
        if not (attempt["state"] == "ended" and attempt["executor_ref"] == reference and
                finite(attempt["started_at"]) and finite(attempt["ended_at"]) and
                attempt["ended_at"] > attempt["started_at"]):
            problems.append(f"{eid}: invalid scoring attempt {attempt['idx']}")
        if not (request.get("sequence") == sequence and request.get("antigen") == "1N8Z_C" and
                request.get("oracle") == "absolut" and
                Path(request.get("absolut_path", "")).resolve() == (root.parent / "absolut").resolve() and
                all(worker.get(k) is not None for k in ("pid", "start", "boot_id")) and
                (base / "worker.log").is_file()):
            problems.append(f"{eid}: worker receipt differs for attempt {attempt['idx']}")
        for name in asset_names:
            link = base / "absolut" / name
            if not link.is_symlink() or link.resolve() != (root.parent / "absolut" / name).resolve():
                problems.append(f"{eid}: input link differs: {name}")
        if outcome.get("state") == "succeeded":
            successes += 1
            if not (attempt is attempts[-1] and result.get("state") == "succeeded" and
                    result.get("metrics", {}).get("absolut_energy") == energy):
                problems.append(f"{eid}: successful scoring receipt differs")
        elif not (outcome == {"state": "stopped", "failure_class": "infrastructure",
                              "message": "executor stopped the attempt without a request"} and
                  result.get("state") == "stopped" and service["cancel_requested_at"] is None and
                  all(attempt[k] is None for k in ("cancel_attempted_at", "cancel_acknowledged_at", "timeout_at"))):
            problems.append(f"{eid}: attempt {attempt['idx']} is not a proven infrastructure stop")
    if successes != 1:
        problems.append(f"{eid}: expected exactly one successful scoring attempt")
    return problems


def timing(research_spans, scoring_spans, start, end, workers):
    """Wall-clock union/intersection and worker-seconds, on half-open intervals."""
    if not finite(start) or not finite(end) or end <= start or not isinstance(workers, int) or workers < 1:
        raise ValueError("invalid campaign window or worker count")
    research = _intervals(research_spans)
    scoring = _intervals(scoring_spans)
    if any(a < start or b > end for a, b in research + scoring):
        raise ValueError("research or scoring interval lies outside the campaign window")
    intersections = [(max(a, c), min(b, d)) for a, b in research for c, d in scoring
                     if min(b, d) > max(a, c)]
    overlap = _length(_intervals(intersections))
    elapsed = end - start
    work = _length(scoring_spans)
    capacity = workers * elapsed
    if work > capacity + 1e-7:
        raise ValueError("scoring worker-seconds exceed total worker capacity")
    return {
        "campaign_wall_seconds": elapsed,
        "research_wall_seconds": _length(research),
        "research_lane_seconds_sum": _length(research_spans),
        "scoring_wall_seconds": _length(scoring),
        "scoring_worker_seconds": work,
        "research_scoring_overlap_seconds": overlap,
        "scoring_worker_capacity_seconds": capacity,
        "scoring_worker_idle_seconds": capacity - work,
        "scoring_worker_utilization": work / capacity,
    }


def peak_intervals(rows, *, keyed=False):
    """Peak simultaneously active intervals; keyed=True counts distinct live keys.

    A transaction may remove and add rows at the same instant; apply all changes
    together. Rows born and removed in the same transaction were never live.
    """
    edges = defaultdict(list)
    for item in rows:
        start, end = item["start"], item["end"]
        if not finite(start) or (end is not None and (not finite(end) or end < start)):
            raise ValueError(f"invalid interval: {start!r}, {end!r}")
        if end == start:
            continue
        key = item["key"] if keyed else None
        edges[start].append((key, 1))
        if end is not None:
            edges[end].append((key, -1))
    active = Counter()
    peak = 0
    for point in sorted(edges):
        for key, change in edges[point]:
            active[key] += change
        if any(value < 0 for value in active.values()):
            raise ValueError("interval end has no matching start")
        peak = max(peak, sum(value > 0 for value in active.values()) if keyed else active[None])
    return peak


def decision_timeline(decisions, experiments, observations, generations, attempts):
    """Audit each single choice against all locally booked history and pending IDs."""
    problems, evidence = [], []
    obs_by_seq = {o["ingest_seq"]: o for o in observations}
    if sorted(obs_by_seq) != list(range(1, len(observations) + 1)):
        problems.append("observation sequence is not contiguous")
    experiment_by_decision = {x["decision_id"]: x for x in experiments}
    starts_by_id = {}
    for attempt in attempts:
        eid, stamp = attempt["experiment_id"], attempt["started_at"]
        starts_by_id[eid] = min(stamp, starts_by_id.get(eid, stamp))
    for row in decisions:
        number, stamp = row["decision_id"], row["created_at"]
        snapshot = read_json_value(row["snapshot_json"])
        if snapshot.get("plan") is not None or "inputs" not in snapshot or "record" not in snapshot:
            problems.append(f"decision {number}: not an independent streaming selection")
            continue
        inputs = snapshot["inputs"]
        head = inputs.get("head_ingest_seq")
        if not isinstance(head, int) or isinstance(head, bool) or not 0 <= head <= len(observations):
            problems.append(f"decision {number}: invalid history head")
            continue
        # Equal timestamps alone cannot order transactions; decision IDs and ingest sequence
        # still determine pending and the head must lie between the strict/inclusive bounds.
        before = max((o["ingest_seq"] for o in observations if o["ingested_at"] < stamp), default=0)
        through = max((o["ingest_seq"] for o in observations if o["ingested_at"] <= stamp), default=0)
        if not before <= head <= through or snapshot.get("head_ingest_seq") != head:
            problems.append(f"decision {number}: history head omits or anticipates a booked result")
        settled = {obs_by_seq[seq]["experiment_id"] for seq in range(1, head + 1)}
        expected = {(x["experiment_id"], x["canonical_key"]) for n, x in experiment_by_decision.items()
                    if n < number and x["experiment_id"] not in settled}
        pending_rows = inputs.get("pending", [])
        pending = {tuple(item) for item in pending_rows}
        if pending != expected or len(pending) != len(pending_rows):
            problems.append(f"decision {number}: pending inputs differ from outstanding experiments")
        if set(snapshot.get("pending_keys", [])) != {key for _, key in pending}:
            problems.append(f"decision {number}: pending snapshot differs from selector inputs")
        candidates = inputs.get("candidates", [])
        candidate_keys = [item["canonical_key"] for item in candidates]
        if (inputs.get("count") != 1 or len(candidate_keys) != len(set(candidate_keys)) or
                set(candidate_keys) & {key for _, key in pending}):
            problems.append(f"decision {number}: invalid candidate pool or selection count")
        exp = experiment_by_decision.get(number)
        if exp is None:
            problems.append(f"decision {number}: experiment missing")
            continue
        for generation in generations:
            if generation["state"] != "ready":
                continue
            begin, finish = generation["started_at"], generation["finished_at"]
            if not begin < stamp < finish:
                continue
            overlap = any(max(begin, a["started_at"]) < min(finish, a["ended_at"]) for a in attempts)
            updates = [obs_by_seq[seq] for seq in range(1, head + 1)
                       if begin < obs_by_seq[seq]["ingested_at"] < stamp]
            if overlap and updates:
                latest = updates[-1]
                evidence.append({"generation": generation["generation"], "ingest_seq": latest["ingest_seq"],
                                 "ingested_at": latest["ingested_at"], "decision_id": number,
                                 "decision_at": stamp, "head_ingest_seq": head,
                                 "attempt_started_at": starts_by_id.get(exp["experiment_id"])})
                break
    return problems, evidence


def _usage(sessions):
    """Keep known token sums separate from completeness; omitted fields are not zero."""
    result = {}
    for role, entries in sessions.items():
        calls = [item.get("providerCalls") for item in entries]
        tokens = [item.get("tokens") for item in entries]
        keys = ("input", "cachedInput", "output", "reasoning")
        valid = lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 0
        complete_calls = bool(entries) and all(valid(n) for n in calls)
        complete_tokens = bool(entries) and all(isinstance(t, dict) and all(
            valid(t.get(k)) for k in keys) for t in tokens)
        result[role] = {"turns": len(entries), "provider_calls": sum(calls) if complete_calls else None,
                        "known_provider_calls": sum(n for n in calls if valid(n)),
                        "tokens": {k: sum(t[k] for t in tokens) for k in keys} if complete_tokens else None,
                        "known_tokens": {k: sum(t[k] for t in tokens if isinstance(t, dict) and valid(t.get(k)))
                                         for k in keys},
                        "token_usage_missing_turns": sum(not isinstance(t, dict) or any(
                            not valid(t.get(k)) for k in keys) for t in tokens)}
    return result


def policy_retry_results(root, db, generations):
    """Verify retry archives and return every failed policy result for cost accounting."""
    retries = defaultdict(list)
    for row in generations:
        number = row["generation"]
        path = root / "policy_harness" / "rounds" / f"round_{number:03d}"
        events = db.rows("""SELECT seq,type,payload_json FROM events WHERE subject_id=?
            AND type IN ('generation.started','generation.blocked') ORDER BY seq""", (str(number),))
        starts = [event for event in events if event["type"] == "generation.started"]
        expected = set()
        for event in starts[1:]:
            payload = read_json_value(event["payload_json"])
            previous = next((item for item in reversed(events) if item["seq"] < event["seq"]), None)
            if (not payload.get("policy_retry") or payload.get("resolution") != "retry" or
                    previous is None or previous["type"] != "generation.blocked" or
                    "status=runtime_fallback, degraded=True" not in read_json_value(previous["payload_json"]).get("error", "")):
                raise ValueError(f"generation {number}: undocumented or unsupported research retry")
            expected.add(f"event-{event['seq']:012d}")
            archive = path / "retries" / f"event-{event['seq']:012d}"
            marker = read_json(archive / "prepared.json")
            archived = read_json(archive / "outcome.json")
            result = read_json(archive / "result.json")
            manifest = read_json(path / "manifest.json")
            names = ("outcome.json", "result.json", "compiled.npz")
            if not (marker.get("generation") == number and marker.get("retry_event_seq") == event["seq"] and
                    marker.get("research_input_sha256") == row["input_digest"] == archived.get("input_digest") and
                    marker.get("policy_input_sha256") == result.get("input_sha256") == manifest.get("input_sha256") and
                    marker.get("files") == {name: file_sha256(archive / name) for name in names} and
                    result.get("compiled_arrays_sha256") == file_sha256(archive / "compiled.npz") and
                    result.get("status") == "runtime_fallback" and result.get("degraded") is True and
                    not result.get("harness_turn")):
                raise ValueError(f"generation {number}: invalid failed policy retry archive")
            original = archived["outcome"]
            original_policy = original.get("metadata", {}).get("policy_round", {})
            final = read_json_value(row["outcome_json"])
            # The research panel is reused; only policy metadata and merge fields change.
            def panel(value):
                return {**{k: v for k, v in value.items() if k not in ("metadata", "merge", "eligible")},
                        "metadata": {k: v for k, v in value.get("metadata", {}).items() if k != "policy_round"}}
            if not (original_policy.get("status") == "runtime_fallback" and original_policy.get("degraded") is True
                    and panel(original) == panel(final)):
                raise ValueError(f"generation {number}: failed policy result or preserved panel differs")
            retries[number].append(result)
        archives = {p.name for p in (path / "retries").glob("event-*") if p.is_dir()}
        if archives != expected:
            raise ValueError(f"generation {number}: retry events and archives differ")
    return retries


def cumulative_usage(final, failed):
    """A retried policy turn reports cumulative usage, not independent attempt totals."""
    valid = lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 0
    snapshots = [*failed, final]
    known = {}
    for key in ("providerCalls", "input", "cachedInput", "output", "reasoning"):
        values = []
        for snapshot in snapshots:
            value = (snapshot.get(key) if key == "providerCalls"
                     else (snapshot.get("tokens") or {}).get(key))
            if valid(value):
                values.append(value)
        if values != sorted(values):
            raise ValueError(f"cumulative policy usage decreased for {key}")
        if values:
            if key == "providerCalls":
                known[key] = values[-1]
            else:
                known.setdefault("tokens", {})[key] = values[-1]
    return known


def proposal_matches_source(proposal, issued, receipt, generation):
    """Bind the reservoir lineage and sequence to the real job that issued them."""
    candidate = proposal["candidate"]
    role = receipt["profile_id"]
    return (role in ("explorer", "refiner") and proposal.get("lineage") == role and
            receipt["round"] + 1 == generation and not issued["violations"] and
            issued["design_id"] == candidate["payload"]["design_id"] and
            issued["sequence"] == candidate["payload"]["sequence"] == candidate["canonical_key"])


def decision_uses_gp_on_fresh_head(decision_id, chosen, pending, refreshed_ids):
    """A real pending-aware GP choice after an observed result during overlapping research."""
    metadata = chosen.get("metadata") or {}
    return (decision_id in refreshed_ids and bool(pending) and metadata.get("gp_ready") is True and
            metadata.get("pending_handling") == "kriging_believer")


def efficiency(root: Path, db, es, workers: int):
    """Compute comparable end-to-end and evidenced activity measurements."""
    campaign = db.row("SELECT * FROM campaign WHERE singleton=1")
    stopped = db.row("SELECT created_at FROM events WHERE type='campaign.stopped' ORDER BY seq DESC LIMIT 1")
    if stopped is None:
        raise ValueError("campaign has no completed stop event")
    start, end = campaign["created_at"], stopped["created_at"]
    generations = db.rows("SELECT * FROM generations ORDER BY generation")
    attempts = es.rows("SELECT * FROM attempts ORDER BY started_at")
    launch = read_json(root.parent / "launch.json")
    active = runner_intervals(root, launch, start, end)
    research = research_intervals(db, generations, active)
    spans = [(r["started_at"], r["finished_at"]) for r in research]
    measured, delayed = scoring_intervals(root, attempts)
    scoring = [(r["started_at"], r["ended_at"]) for r in measured]
    metrics = timing(spans, scoring, start, end, workers)
    active_seconds = _length(active)
    active_scoring = _length(_intersection(scoring, active))
    metrics["campaign_runner_active_seconds"] = active_seconds
    metrics["campaign_outside_runner_seconds"] = metrics["campaign_wall_seconds"] - active_seconds
    metrics["scoring_worker_seconds_during_runner"] = active_scoring
    metrics["scoring_worker_seconds_outside_runner"] = metrics["scoring_worker_seconds"] - active_scoring
    metrics["scoring_worker_utilization_during_runner"] = (active_scoring / (workers * active_seconds)
                                                           if active_seconds else None)
    metrics["scoring_service_recorded_worker_seconds"] = _length(
        [(r["started_at"], r["ended_at"]) for r in attempts])
    metrics["scoring_service_recorded_peak_concurrency"] = peak_intervals(
        [{"start": r["started_at"], "end": r["ended_at"]} for r in attempts])
    metrics["scoring_results_recorded_before_service_end"] = len(delayed)
    metrics["scoring_result_to_service_end_seconds_sum"] = math.fsum(delayed)
    metrics["successful_scores"] = db.value("SELECT COUNT(*) FROM observations WHERE status='succeeded'")
    metrics["successful_scores_per_hour"] = 3600 * metrics["successful_scores"] / metrics["campaign_wall_seconds"]
    metrics["launcher_latest_case_wall_seconds"] = launch.get("case", {}).get("elapsed_seconds")
    metrics["launcher_wall_seconds"] = (datetime.fromisoformat(launch["finished_at"]) -
                                         datetime.fromisoformat(launch["started_at"])).total_seconds()
    jobs = [read_json(path) for path in (root / "design_jobs").glob("job-*/receipt.json")]
    metrics["research_generations_started"] = len(generations)
    metrics["research_generations_ready"] = sum(row["state"] == "ready" for row in generations)
    metrics["raw_sequences_requested"] = sum(job["count"] for job in jobs)
    metrics["raw_sequences_generated"] = sum(len(job.get("candidates", [])) for job in jobs)
    metrics["generation_jobs_by_status"] = dict(Counter(job.get("status", "unknown") for job in jobs))
    metrics["research_lane_seconds_per_success"] = (metrics["research_lane_seconds_sum"] /
                                                     metrics["successful_scores"] if metrics["successful_scores"] else None)
    model = {"rfd3": {"executions": 0, "elapsed_seconds_sum": 0.0},
             "proteinmpnn": {"executions": 0, "elapsed_seconds_sum": 0.0}}
    for label, pattern in (("rfd3", "rfd3.execution.json"), ("proteinmpnn", "proteinmpnn-*.execution.json")):
        for path in (root / "design_jobs").glob(f"job-*/output/{pattern}"):
            record = read_json(path)
            value = record.get("elapsed_seconds")
            if not finite(value) or value < 0:
                raise ValueError(f"invalid execution duration: {path}")
            model[label]["executions"] += 1
            model[label]["elapsed_seconds_sum"] += value
    metrics["model_execution_seconds"] = model
    metrics["model_scoring_overlap_seconds"] = None
    sessions = {role: [] for role in (*ROLES, "policy_architect")}
    policy_bounds, retry_usage = [], []
    retries = policy_retry_results(root, db, generations)
    for row in generations:
        outcome = read_json_value(row["outcome_json"] or "{}")
        generation_sessions = outcome.get("metadata", {}).get("sessions") or {}
        for role in ROLES:
            # A missing session/usage is unknown, including for incomplete generations.
            sessions[role].append((generation_sessions.get(role) or {}).get("usage") or {})
        path = root / "policy_harness" / "rounds" / f"round_{row['generation']:03d}" / "result.json"
        usage = {}
        if path.is_file():
            result = read_json(path)
            turn = result.get("harness_turn")
            usage = (turn or {}).get("usage") if turn else result.get("failed_harness_usage")
        usage = usage or {}
        failed = [item.get("failed_harness_usage") or {} for item in retries[row["generation"]]]
        sessions["policy_architect"].append(usage)
        policy_bounds.append(cumulative_usage(usage, failed))
        if failed:
            retry_usage.append({"generation": row["generation"], "failed_cumulative_snapshots": failed,
                                "final_cumulative_usage": usage})
    metrics["llm_usage_by_role"] = _usage(sessions)
    policy_known = _usage({"policy": policy_bounds})["policy"]
    metrics["llm_usage_by_role"]["policy_architect"].update(
        known_provider_calls=policy_known["known_provider_calls"], known_tokens=policy_known["known_tokens"])
    metrics["policy_retry_cumulative_usage"] = retry_usage
    usages = list(metrics["llm_usage_by_role"].values())
    calls = [value["provider_calls"] for value in usages]
    token_values = [value["tokens"] for value in usages]
    total_calls = sum(calls) if all(value is not None for value in calls) else None
    total_tokens = ({key: sum(value[key] for value in token_values) for key in
                     ("input", "cachedInput", "output", "reasoning")}
                    if all(value is not None for value in token_values) else None)
    count = metrics["successful_scores"]
    metrics["llm_usage_total"] = {"provider_calls": total_calls, "tokens": total_tokens,
                                  "known_provider_calls": sum(value["known_provider_calls"] for value in usages),
                                  "known_tokens": {k: sum(value["known_tokens"][k] for value in usages) for k in
                                                   ("input", "cachedInput", "output", "reasoning")},
                                  "provider_calls_per_success": total_calls / count if count and total_calls is not None else None,
                                  "tokens_per_success": {k: v / count for k, v in total_tokens.items()}
                                  if count and total_tokens is not None else None}
    metrics["policy_retry_count"] = sum(len(items) for items in retries.values())
    metrics["timing_scope"] = ("Campaign wall includes all gaps from controller creation through campaign.stopped. "
                               "Runner activity is clipped to recorded launcher case intervals; workers may run outside "
                               "those intervals. Research uses each started-to-blocked or finished event interval, clipped "
                               "to runner activity. Scoring ends at the earlier of the service end and result.json mtime; "
                               "mtime is a proxy for writing the result, not a confirmed process-group exit. Raw service "
                               "attempt durations are retained and can include downtime. Model execution durations have "
                               "no absolute timestamps, so model/scoring overlap is unavailable. Same-turn policy retry "
                               "usage snapshots are cumulative and counted once. LLM counts are source-reported; missing "
                               "receipts are unknown and provider responses without usage may not be counted.")
    return metrics


class StreamingAudit(Audit):
    def __init__(self, directory, evaluations=640, role_candidates=32, pool=64, window=32, workers=4):
        super().__init__(directory, 0, evaluations)
        self.role_candidates, self.pool, self.window, self.workers = role_candidates, pool, window, workers
        self.report.update(expected_evaluations=evaluations, expected_candidates_per_role=role_candidates,
                           expected_live_pool_capacity=pool, expected_evaluation_window=window,
                           expected_scoring_workers=workers)
        self.report.pop("expected_rounds")
        self.report["coverage"].update({"controlled_cas_race": "not_tested_by_receipt_audit",
                                        "controlled_crash_recovery": "not_tested_by_receipt_audit",
                                        "independent_native_recomputation": "not_performed"})

    def configuration(self):
        self.config = c = read_json(self.root / "config.json")
        self.require(c.get("method") == "ldm_harness" and c.get("antigen") == "1N8Z_C" and
                     c.get("oracle") == "absolut" and c.get("mock") is False and c.get("dry_run") is False,
                     "expected real HER2 generation and Absolut scoring")
        self.require(c.get("iterations") == self.minimum_evaluations and c.get("batch_size") == 1 and
                     c.get("pool_size") == self.pool and c.get("research_candidates_per_role") == self.role_candidates and
                     c.get("concurrency") == self.workers,
                     "evaluation budget, streaming selection, role count, pool, or workers differ")
        summary = read_json(self.root / "summary.json")
        self.require(not summary.get("error") and not summary.get("invariant_problems") and
                     summary.get("successful_evaluation_count") == self.minimum_evaluations,
                     "campaign summary reports an error or incomplete evaluations")
        self.report["stop_reason"] = summary.get("stop_reason")
        launch = read_json(self.root.parent / "launch.json")
        source = self.root.parent / "absolut"
        self.require(launch.get("status") == "completed" and launch.get("case", {}).get("status") == "completed" and
                     Path(c["absolut_path"]).resolve() == source.resolve(), "launch or isolated input directory differs")
        manifest = read_json(source / "assets.json")
        self.require(manifest.get("schema_version") == 1 and manifest.get("antigen") == "1N8Z_C" and
                     manifest.get("assets") == launch.get("assets"), "input manifests differ")
        self.asset_names = []
        for item in manifest.get("assets", []):
            name, relative = item["name"], Path(item["name"])
            path = source / relative
            self.require(not relative.is_absolute() and ".." not in relative.parts and not path.is_symlink() and
                         path.is_file() and path.stat().st_size == item["bytes"] and not path.stat().st_mode & 0o222 and
                         file_sha256(path) == item["sha256"], f"invalid immutable input asset: {name}")
            self.asset_names.append(name)
        self.require("src/bin/Absolut" in self.asset_names and len(set(self.asset_names)) == len(self.asset_names),
                     "missing executable or duplicate input assets")
        self.report["configuration"] = {k: c.get(k) for k in
            ("method", "antigen", "oracle", "iterations", "research_rounds", "batch_size", "pool_size", "concurrency")}

    def evaluations(self):
        self.db = ReadOnlyStore(self.root / "controller.sqlite3")
        self.es = ReadOnlyStore(self.root / "experiment-service.sqlite3")
        frozen = read_json_value(self.db.value("SELECT config_json FROM campaign WHERE singleton=1"))
        config = frozen["campaign"]
        pacing = config["extra"]["pacing"]
        self.require(config["budget"] == self.minimum_evaluations and config["concurrency"] == self.workers and
                     config["buffer_depth"] + config["concurrency"] == self.window and
                     pacing["mode"] == "streaming" and pacing["selection_batch_size"] == 1 and
                     pacing["reservoir_size"] == self.pool and pacing["dispatch_during_research"] == "continue" and
                     frozen["plugins"]["reservoir"].get("capacity") == self.pool,
                     "frozen controller configuration differs from the streaming window and pool")
        self.generations = self.db.rows("SELECT * FROM generations ORDER BY generation")
        self.experiments = self.db.rows("SELECT * FROM experiments ORDER BY decision_id")
        self.observation_rows = self.db.rows("SELECT * FROM observations ORDER BY ingest_seq")
        self.decisions = self.db.rows("SELECT * FROM decisions ORDER BY decision_id")
        attempts = self.es.rows("SELECT * FROM attempts ORDER BY started_at")
        remote = {r["experiment_id"]: r for r in self.es.rows("SELECT * FROM experiments ORDER BY seq")}
        self.require(len(self.generations) >= 2 and
                     [r["generation"] for r in self.generations] == list(range(1, len(self.generations) + 1)),
                     "fewer than two or noncontiguous research generations")
        self.require(len(self.experiments) == len(self.decisions) == len(self.observation_rows) ==
                     self.minimum_evaluations, "evaluations, decisions or observations differ")
        self.require({r["experiment_id"] for r in self.experiments} == set(remote),
                     "controller and service experiment IDs differ")
        self.require(len({r["canonical_key"] for r in self.experiments}) == self.minimum_evaluations,
                     "sequences were repeated or evaluations are incomplete")
        self.require(not self.db.value("SELECT COUNT(*) FROM anomalies WHERE resolved_at IS NULL"),
                     "controller has unresolved anomalies")
        self.require(self.db.value("SELECT COUNT(*) FROM plans") == 0, "barrier plans exist in the streaming run")
        self.observations = {r["experiment_id"]: read_json_value(r["observation_json"]) for r in self.observation_rows}
        self.by_candidate = {o["candidate"]["candidate_id"]: o for o in self.observations.values()}
        by_attempt = defaultdict(list)
        for item in attempts:
            by_attempt[item["experiment_id"]].append(item)
        self.require(set(by_attempt) == set(remote), "scoring attempts and service experiment IDs differ")
        for row in self.experiments:
            eid = row["experiment_id"]
            service, observation, linked = remote.get(eid), self.observations.get(eid), by_attempt[eid]
            self.require(row["submit_state"] == "ACKED" and row["es_state"] == "SUCCEEDED" and
                         row["budget_state"] == "CONSUMED" and not row["pending"] and row["disposition"] == "ingested" and
                         service is not None and observation is not None and bool(linked),
                         f"{eid}: unsettled or missing real scoring attempt")
            if service is None or observation is None or not linked:
                continue
            candidate, evaluation = observation["candidate"], observation["evaluation"]
            sequence = candidate["payload"]["sequence"]
            onehot(sequence)
            energy, utility = evaluation["metrics"].get("absolut_energy"), evaluation["metrics"].get(UTILITY)
            self.require(evaluation["status"] == "succeeded" and finite(energy) and finite(utility) and
                         math.isclose(utility, -energy, abs_tol=1e-12) and
                         sequence == row["canonical_key"] == candidate["canonical_key"],
                         f"{eid}: invalid sequence or scored utility")
            spec, result = read_json_value(service["spec"]), read_json_value(service["result"])
            self.require(service["state"] == "SUCCEEDED" and spec ==
                         {"antigen": "1N8Z_C", "oracle": "absolut", "sequence": sequence} and
                         result.get("absolut_energy") == energy and service["result_digest"] == row["es_result_digest"],
                         f"{eid}: service result differs from observation")
            self.problems.extend(scoring_attempt_problems(self.root, service, sequence, energy, linked, self.asset_names))
        measured, _ = scoring_intervals(self.root, attempts)
        spans = [{"start": a["started_at"], "end": a["ended_at"]} for a in measured]
        peak = peak_intervals(spans)
        self.require(peak <= self.workers, "estimated scoring concurrency exceeded the physical worker limit")
        self.report["evaluations"] = {"unique_successful": len(self.observations), "attempts": len(attempts),
                                      "interrupted_attempts": sum(len(items) - 1 for items in by_attempt.values()),
                                      "peak_estimated_scoring_concurrency": peak}

    def designs(self):
        from tasks.antibody.tools.structure_design.common import extract_h3

        self.catalog, self.catalog_jobs = {}, {}
        ready = {r["generation"] for r in self.generations if r["state"] == "ready"}
        usage = defaultdict(lambda: {"backbones": 0, "sequences": 0})
        receipts = sorted((self.root / "design_jobs").glob("job-*/receipt.json"))
        self.require(bool(receipts), "no generation receipts")
        for path in receipts:
            receipt = read_json(path)
            generation, role = receipt["round"] + 1, receipt.get("profile_id")
            self.require(1 <= generation <= len(self.generations) and role in ("explorer", "refiner") and
                         receipt.get("mock_generator") is False,
                         f"{path.parent.name}: invalid generator provenance")
            usage[(generation, role)]["backbones"] += int(receipt["action"] == "design_h3")
            usage[(generation, role)]["sequences"] += receipt["count"]
            if generation not in ready:
                continue  # Partial jobs are reported with the terminal research state below.
            self.require(receipt.get("status") == "succeeded", f"{path.parent.name}: generation failed")
            output = read_json(path.parent / "output/designs.json")
            mpnn = read_json(path.parent / "output/proteinmpnn-0000.execution.json")
            self.require(output.get("status") == "success" and len(output["candidates"]) == receipt["count"] and
                         len(receipt["candidates"]) == receipt["count"] and mpnn.get("returncode") == 0 and
                         mpnn.get("elapsed_seconds", 0) > 0, f"{path.parent.name}: ProteinMPNN output missing")
            if receipt["action"] == "design_h3":
                rfd = read_json(path.parent / "output/rfd3.execution.json")
                self.require(rfd.get("returncode") == 0 and rfd.get("elapsed_seconds", 0) > 0,
                             f"{path.parent.name}: RFD3 output missing")
            for index, issued in enumerate(receipt["candidates"]):
                design_id = issued["design_id"]
                self.require(design_id not in self.catalog and index < len(output["candidates"]) and
                             issued["sequence"] == extract_h3(output["candidates"][index], 11),
                             f"{path.parent.name}: design receipt differs from model output: {design_id}")
                self.catalog[design_id], self.catalog_jobs[design_id] = issued, receipt
        for generation in ready:
            self.require(sum(usage[(generation, role)]["sequences"] for role in ("explorer", "refiner")) <=
                         self.config["sequences_per_round"] and
                         sum(usage[(generation, role)]["backbones"] for role in ("explorer", "refiner")) <=
                         self.config["backbones_per_round"], f"generation {generation}: generation budget exceeded")
            for role in ("explorer", "refiner"):
                self.require(usage[(generation, role)]["sequences"] >= self.role_candidates,
                             f"generation {generation} {role}: insufficient generated sequences")
        self.report["designs"] = {"jobs": len(receipts), "by_generation_and_role":
                                  {f"{g}:{role}": value for (g, role), value in sorted(usage.items())}}

    def research(self):
        from ldm_tts.async_runtime.wire import digest

        visibility = read_jsonl(self.root / "history/visibility.jsonl")
        history = read_jsonl(self.root / "history/log.jsonl")
        self.require([e["seq"] for e in history] == list(range(1, len(history) + 1)),
                     "history event sequence is not contiguous")
        by_turn = defaultdict(list)
        for event in visibility:
            by_turn[(event["profile_id"], event["turn_id"])].append(event)
        launch = read_json(self.root.parent / "launch.json")
        reads = {}
        for role in ROLES:
            root = self.root / "harness" / role
            manifest = read_json(root / "manifest.json")
            profiles = manifest["profiles"]
            self.require(manifest.get("model") == launch.get("model") and len(profiles) == 1 and
                         profiles[0].get("profileId") == role, f"{role}: model or profile differs")
            session = (root / profiles[0]["session"]).resolve()
            self.require(session.is_relative_to(root.resolve()), f"{role}: invalid session path")
            reads[role] = successful_reads(session)
            names = {"cdrh3-design", "cdrh3-evidence-review" if role == "evidence_reviewer"
                     else "cdrh3-experiment-design"}
            self.require(read_skills(reads[role], names) == names, f"{role}: skill reads missing")
        statuses, admitted, cancelled = Counter(), {}, []
        for row in self.generations:
            generation = row["generation"]
            statuses[row["state"]] += 1
            self.require(finite(row["started_at"]) and finite(row["finished_at"]) and
                         row["finished_at"] > row["started_at"], f"generation {generation}: invalid span")
            if row["state"] != "ready":
                # Only an explicit and persisted budget-end cancellation is acceptable.
                events = self.db.rows("SELECT type,payload_json FROM events WHERE subject_id=? AND type LIKE 'generation.%'",
                                      (str(generation),))
                budget_cancel = any("budget_exhausted" in read_json_value(e["payload_json"]).values() and
                                    e["type"] in ("generation.canceled", "generation.cancelled", "generation.abandoned")
                                    for e in events)
                self.require(budget_cancel and self.report["stop_reason"] == "budget_exhausted",
                             f"generation {generation}: research ended unsuccessfully ({row['state']}: {row['error']})")
                cancelled.append({"generation": generation, "state": row["state"], "error": row["error"],
                                  "budget_cancellation_verified": budget_cancel})
                continue
            input_file = self.root / "controller.sqlite3.research" / row["input_path"]
            inputs = read_json(input_file)
            saved = read_json(input_file.with_name(f"g{generation:06d}.outcome.json"))
            outcome = read_json_value(row["outcome_json"])
            unmerged = {k: v for k, v in outcome.items() if k not in ("merge", "eligible")}
            self.require(digest(inputs) == row["input_digest"] == saved.get("input_digest") and
                         inputs["watermarks"]["base_ingest_seq"] == row["base_ingest_seq"] and
                         saved.get("outcome") == unmerged and row["proposal_count"] == self.pool and
                         outcome.get("status") == "committed" and 0 <= row["admitted"] <= self.pool,
                         f"generation {generation}: research input or result differs")
            metadata, proposals = outcome["metadata"], outcome["proposals"]
            self.require(metadata.get("bootstrap") is False and set(metadata.get("sessions", {})) == set(ROLES) and
                         metadata.get("candidate_counts") == {"explorer": self.role_candidates,
                                                               "refiner": self.role_candidates},
                         f"generation {generation}: panel or candidate counts differ")
            role_counts, role_ids, keys = Counter(), defaultdict(set), set()
            for proposal in proposals:
                candidate = proposal["candidate"]
                design_id = candidate["payload"]["design_id"]
                receipt, issued = self.catalog_jobs.get(design_id), self.catalog.get(design_id)
                if receipt is None or issued is None:
                    self.require(False, f"generation {generation}: unissued design {design_id}")
                    continue
                role = receipt["profile_id"]
                role_counts[role] += 1
                role_ids[role].add(design_id)
                key = candidate["canonical_key"]
                self.require(proposal_matches_source(proposal, issued, receipt, generation) and key not in keys,
                             f"generation {generation}: invalid or repeated proposed design {design_id}")
                keys.add(key)
            self.require(len(keys) == self.pool and role_counts == {"explorer": self.role_candidates,
                                                                   "refiner": self.role_candidates},
                         f"generation {generation}: researcher proposals lack independent sources")
            submissions = {}
            for role in ROLES:
                session = metadata["sessions"][role]
                events = by_turn[(role, session["turn_id"])]
                validations = [e for e in events if e["kind"] == "validate"]
                accepts = [e for e in events if e["kind"] == "decision" and e.get("decision") == "accept"]
                queries = [e for e in events if e["kind"] == "query" and not e.get("error")]
                commit = read_json(self.root / "harness" / role / "turns" / session["turn_id"] / "turn_committed.json")
                self.require(session.get("submission_status") == "accepted" and
                             session.get("usage", {}).get("providerCalls", 0) > 0 and validations and accepts and queries and
                             commit.get("submissionStatus") == "accepted" and
                             accepts[-1].get("submission_digest") == commit.get("submissionDigest"),
                             f"generation {generation} {role}: accepted provider submission missing")
                if validations:
                    check = validations[-1]
                    self.require(check.get("history_seq") == check.get("head") == session.get("history_seq") and
                                 (check.get("visible") == check.get("head") or check.get("forced")),
                                 f"generation {generation} {role}: invalid CAS validation")
                submissions[role] = commit["submission"]
            for role in ("explorer", "refiner"):
                ids = submissions[role].get("design_ids", [])
                self.require(len(ids) == self.role_candidates and len(set(ids)) == len(ids) and
                             set(ids) == role_ids[role], f"generation {generation} {role}: role source differs")
            review = submissions["evidence_reviewer"]
            self.require(review == metadata.get("review") and bool(review.get("evidence", "").strip()) and
                         bool(review.get("hypotheses", "").strip()) and
                         set(review.get("design_ids", [])) <= role_ids["explorer"] | role_ids["refiner"],
                         f"generation {generation}: reviewer submission differs")
            catalogs = metadata["peer_catalogs"]
            self.require(set(catalogs) == {"refiner", "evidence_reviewer"},
                         f"generation {generation}: peer catalogs missing")
            for role, info in catalogs.items():
                record = (self.root / info["record"]).resolve()
                workspace = self.root / "harness" / role / "sessions" / role / "workspace"
                guest = workspace / Path(info["path"]).relative_to("/workspace")
                self.require(record.is_relative_to(self.root) and file_sha256(record) == info["sha256"] ==
                             file_sha256(guest) and info["path"] in reads[role],
                             f"generation {generation} {role}: peer catalog digest differs")
                catalog = read_json(record)
                expected = role_ids["explorer"] | (role_ids["refiner"] if role == "evidence_reviewer" else set())
                self.require(catalog.get("generation") == generation and catalog.get("recipient") == role and
                             catalog.get("base_history_seq") == metadata["base_history_seq"] and
                             {item["design_id"] for item in catalog["candidates"]} == expected,
                             f"generation {generation} {role}: peer catalog provenance differs")
            admitted[str(generation)] = row["admitted"]
        stale = [e for e in visibility if e["kind"] == "cas_stale"]
        self.require(all(e["visible"] < e["head"] for e in stale), "invalid stale CAS receipt")
        self.report["coverage"]["cas_stale_in_campaign"] = "exercised" if stale else "not_exercised"
        self.report["research"] = {"states": dict(statuses), "admitted_by_generation": admitted,
                                   "other_terminal_generations": cancelled, "cas_stale_events": len(stale)}

    def streaming(self):
        occurrences = self.db.rows("SELECT canonical_key,created_at,removed_at FROM occurrences")
        peak_pool = peak_intervals([{"key": r["canonical_key"], "start": r["created_at"], "end": r["removed_at"]}
                                    for r in occurrences], keyed=True)
        self.require(peak_pool <= self.pool, "live resident pool exceeded capacity")
        obs = {r["experiment_id"]: r for r in self.observation_rows}
        pending = []
        for x in self.experiments:
            item = obs.get(x["experiment_id"])
            if item is None:
                self.require(False, f"{x['experiment_id']}: no settlement time for pending-window audit")
                continue
            pending.append({"start": x["created_at"], "end": item["ingested_at"]})
        peak_pending = peak_intervals(pending)
        self.require(peak_pending <= self.window, "evaluation window exceeded its maximum")
        attempts = self.es.rows("SELECT * FROM attempts ORDER BY started_at")
        attempts, _ = scoring_intervals(self.root, attempts)
        campaign = self.db.row("SELECT created_at FROM campaign WHERE singleton=1")
        stopped = self.db.row("SELECT created_at FROM events WHERE type='campaign.stopped' ORDER BY seq DESC LIMIT 1")
        if stopped is None:
            raise ValueError("campaign has no completed stop event")
        active = runner_intervals(self.root, read_json(self.root.parent / "launch.json"),
                                  campaign["created_at"], stopped["created_at"])
        generations = research_intervals(self.db, self.generations, active)
        problems, evidence = decision_timeline(self.decisions, self.experiments, self.observation_rows,
                                                generations, attempts)
        self.problems.extend(problems)
        self.require(bool(evidence), "no result was ingested and used by a new decision during overlapping research")
        self._refreshed_decisions = {item["decision_id"] for item in evidence}
        self.report["streaming"] = {"peak_live_pool": peak_pool, "peak_evaluation_window": peak_pending,
                                    "updated_head_decisions_during_research": len(evidence),
                                    "updated_head_examples": evidence[:5]}

    def policy(self):
        from ldm_tts.async_runtime.policy import PriorApplication, check_prior_invariants
        from ldm_tts.async_runtime.selectors import decision_choice

        root = self.root / "policy_harness"
        retries = policy_retry_results(self.root, self.db, self.generations)
        self.report["policy_retry_attempts"] = {str(number): len(items) for number, items in retries.items() if items}
        manifest = read_json(root / "manifest.json")
        profiles, launch = manifest["profiles"], read_json(self.root.parent / "launch.json")
        self.require(manifest.get("model") == launch.get("model") and len(profiles) == 1 and
                     profiles[0].get("profileId") == "policy_architect", "policy model or profile differs")
        session = (root / profiles[0]["session"]).resolve()
        self.require(session.is_relative_to(root.resolve()) and
                     read_skills(successful_reads(session), {"compile-ldm-policy"}) == {"compile-ldm-policy"},
                     "policy skill read missing")
        applications = self.db.rows("SELECT * FROM prior_applications ORDER BY seq")
        by_id = {row["application_id"]: row for row in applications}
        self.require(bool(applications) and not any(row["degraded"] for row in applications),
                     "missing or degraded frozen policy applications")
        self.problems.extend(check_prior_invariants(self.db, UTILITY))
        contexts = []
        for application in applications:
            self.section(f"application {application['application_id']} context",
                         lambda app=application: contexts.append(self.check_application_context(app, decision_choice)))
        epochs, rounds = {}, []
        for generation in self.generations:
            if generation["state"] != "ready":
                continue
            number = generation["generation"]
            saved = read_json_value(generation["outcome_json"])["metadata"]["policy_round"]
            path = root / "rounds" / f"round_{number:03d}"
            policy_manifest, result = read_json(path / "manifest.json"), read_json(path / "result.json")
            self.require(saved.get("status") == result.get("status") == "accepted" and
                         not saved.get("degraded") and not result.get("degraded") and
                         saved.get("epoch_id") == result.get("epoch_id") and
                         saved.get("artifact_sha256") == result.get("artifact_sha256") and
                         result.get("input_sha256") == policy_manifest.get("input_sha256") and
                         result.get("compiled_arrays_sha256") == file_sha256(path / "compiled.npz"),
                         f"generation {number}: policy result or frozen artifact differs")
            for name, expected in policy_manifest["files"].items():
                self.require(file_sha256(path / name) == expected, f"generation {number}: policy input digest differs: {name}")
            turn = result["harness_turn"]
            commit = read_json(root / "turns" / turn["turn_id"] / "turn_committed.json")
            self.require(turn.get("usage", {}).get("providerCalls", 0) > 0 and
                         commit.get("profileId") == "policy_architect" and commit.get("submissionStatus") == "accepted" and
                         commit.get("submissionDigest") == result.get("submission_sha256") and
                         commit.get("submission", {}).get("action") == result.get("action"),
                         f"generation {number}: policy turn receipt differs")
            epoch = result.get("epoch_id")
            if epoch:
                epoch_path = root / "epochs" / epoch
                ep = read_json(epoch_path / "manifest.json")
                artifact = (root / ep["artifact_path"]).resolve()
                self.require(artifact.is_relative_to(root.resolve()) and ep["epoch_id"] == epoch and
                             ep["artifact_sha256"] == result["artifact_sha256"] == file_sha256(artifact),
                             f"generation {number}: policy epoch differs")
                for name, expected in ep["files"].items():
                    self.require(file_sha256(epoch_path / name) == expected,
                                 f"generation {number}: policy epoch digest differs: {name}")
                epochs[epoch] = ep
            rounds.append({"generation": number, "action": result.get("action"), "epoch": epoch})
        by_decision = {row["decision_id"]: row for row in self.experiments}
        used_generations, used_epochs, fitted, pending_decisions, replay = set(), set(), [], 0, []
        updated_gp_pending = []
        for decision in self.decisions:
            snapshot = read_json_value(decision["snapshot_json"])
            record, inputs, step = decision_choice(self.db, decision["decision_id"])
            aid = inputs.get("prior_application_id")
            application = by_id.get(aid)
            self.require(application is not None, f"decision {decision['decision_id']}: missing policy application")
            if application is None:
                continue
            used_generations.add(application["generation"])
            self.require(application["created_at"] <= decision["created_at"] and
                         (application["superseded_at"] is None or decision["created_at"] <= application["superseded_at"]),
                         f"decision {decision['decision_id']}: inactive policy application")
            selection, chosen = record["selection"], record["selection"]["steps"][step]
            index = chosen["candidate_ids"].index(chosen["selected_candidate_id"])
            pending = {key for _, key in inputs["pending"]}
            pending_decisions += bool(pending)
            if decision_uses_gp_on_fresh_head(decision["decision_id"], chosen, pending,
                                              getattr(self, "_refreshed_decisions", set())):
                updated_gp_pending.append(decision["decision_id"])
            self.require(pending <= set(chosen["pending_keys"]) and
                         chosen["canonical_keys"][index] == by_decision[decision["decision_id"]]["canonical_key"] and
                         snapshot["probability"] == chosen["selected_probability"] and
                         math.isclose(float(selection["parameters"]["alpha"]), application["alpha"], abs_tol=1e-12) and
                         math.isclose(float(selection["parameters"]["eta"]), application["eta"], abs_tol=1e-12) and
                         record["selector"].get("objective") == UTILITY and
                         record["selector"].get("feature_version") == "cdrh3-one-hot-v1" and
                         record.get("pending_handling") == "kriging_believer",
                         f"decision {decision['decision_id']}: policy choice or pending handling differs")
            if not application["prior_free"]:
                epoch = application["epoch_id"]
                self.require(epoch in epochs and epochs[epoch]["artifact_sha256"] == application["artifact_sha256"] and
                             {item["canonical_key"] for item in inputs["candidates"]} <=
                             set(read_json_value(application["values_json"])),
                             f"decision {decision['decision_id']}: policy artifact or coverage differs")
                used_epochs.add(epoch)
                if chosen.get("metadata", {}).get("gp_ready") and selection.get("metadata", {}).get("prior_applied"):
                    fitted.append(decision["decision_id"])
            replay.append((decision["decision_id"], aid))
        self.require(len(replay) == len(self.decisions), "one or more decisions lack a replayable policy application")
        self.require(bool(pending_decisions), "no decisions included pending experiments")
        self.require(bool(updated_gp_pending), "no GP-ready, pending-conditioned choice used an updated history "
                     "during overlapping research")
        self.report["coverage"]["fitted_gp_with_custom_prior"] = "exercised" if fitted else "not_exercised"
        self.report["policy"] = {"rounds": rounds, "applications": len(applications),
                                 "used_generations": sorted(used_generations), "used_epochs": sorted(used_epochs),
                                 "fitted_prior_decisions": len(fitted), "pending_decisions": pending_decisions,
                                 "updated_head_gp_pending_decisions": len(updated_gp_pending),
                                 "application_contexts": contexts}
        self.replay_decisions(replay, by_id, PriorApplication)

    def run(self):
        self.section("configuration", self.configuration)
        if not hasattr(self, "asset_names"):
            return self.finish()
        self.section("controller and service", self.evaluations)
        if not hasattr(self, "db"):
            return self.finish()
        try:
            self.section("generation receipts", self.designs)
            self.section("research and CAS", self.research)
            self.section("streaming evidence", self.streaming)
            self.section("policy and selector replay", self.policy)
            self.section("efficiency", lambda: self.report.update(efficiency=efficiency(self.root, self.db, self.es,
                                                                                       self.workers)))
        finally:
            self.db.close()
            if hasattr(self, "es"):
                self.es.close()
        return self.finish()


def comparison(baseline_root: Path, stream_metrics, expected: int, workers: int):
    """Report an equal-score comparison only after the reference run is complete."""
    baseline_root = baseline_root.resolve()
    try:
        launch = read_json(baseline_root.parent / "launch.json")
        config = read_json(baseline_root / "config.json")
        summary = read_json(baseline_root / "summary.json")
        if (launch.get("status") != "completed" or launch.get("case", {}).get("status") != "completed" or
                summary.get("error") or summary.get("invariant_problems") or
                summary.get("successful_evaluation_count") != expected or config.get("iterations") != expected):
            return {"status": "unavailable", "reason": "baseline has not completed the same successful-score budget"}
        if config.get("batch_size") != 32 or config.get("pool_size") != 64 or config.get("concurrency") != workers:
            return {"status": "unavailable", "reason": "baseline settings do not match the requested barrier reference"}
        db = ReadOnlyStore(baseline_root / "controller.sqlite3")
        es = ReadOnlyStore(baseline_root / "experiment-service.sqlite3")
        try:
            baseline = efficiency(baseline_root, db, es, workers)
        finally:
            db.close()
            es.close()
        if baseline["successful_scores"] != expected:
            return {"status": "unavailable", "reason": "baseline persisted observations differ from its summary"}
        ratio = baseline["campaign_wall_seconds"] / stream_metrics["campaign_wall_seconds"]
        return {"status": "available", "baseline_run_dir": str(baseline_root), "baseline": baseline,
                "baseline_wall_seconds_divided_by_streaming": ratio,
                "baseline_cpu_affinity": launch.get("cpu_affinity"), "baseline_gpu": config.get("gpu"),
                "interpretation": ("Descriptive equal-score comparison only: the numbers of research generations, "
                                   "raw sequences and LLM calls/tokens can differ, so the wall-time ratio combines "
                                   "workload and scheduling effects. CPU/GPU placement may be separate, but concurrent "
                                   "runs can still share host I/O and the provider; it is not a causal async speedup. "
                                   "Baseline integrity is not independently re-audited by this comparison.")}
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--evaluations", type=int, default=640)
    parser.add_argument("--per-role", type=int, default=32)
    parser.add_argument("--pool", type=int, default=64)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline", type=Path, help="completed barrier campaign directory for descriptive comparison")
    args = parser.parse_args()
    if min(args.evaluations, args.per_role, args.pool, args.window, args.workers) < 1 or \
            args.pool != 2 * args.per_role or args.window < args.workers:
        parser.error("all limits must be positive, pool twice per-role, and window >= workers")
    report = StreamingAudit(args.campaign_dir, args.evaluations, args.per_role, args.pool,
                            args.window, args.workers).run()
    if args.baseline is not None:
        report["baseline_comparison"] = (comparison(args.baseline, report["efficiency"], args.evaluations, args.workers)
                                         if "efficiency" in report else
                                         {"status": "unavailable", "reason": "streaming efficiency is unavailable"})
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
