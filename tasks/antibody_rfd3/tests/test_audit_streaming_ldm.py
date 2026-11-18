import json
import hashlib
import os
import sqlite3

import pytest

from tasks.antibody_rfd3.repro.h20.audit_full_ldm import ReadOnlyStore
from tasks.antibody_rfd3.repro.h20.audit_streaming_ldm import (
    _usage, comparison, cumulative_usage, decision_timeline, decision_uses_gp_on_fresh_head, efficiency, peak_intervals,
    proposal_matches_source, scoring_attempt_problems, timing,
)


def test_overlap_and_worker_utilization_count_union_and_worker_seconds_separately():
    report = timing([(0, 10), (11, 16)], [(2, 6), (4, 9), (12, 18)], 0, 20, 4)
    assert report["research_wall_seconds"] == 15
    assert report["scoring_wall_seconds"] == 13
    assert report["research_scoring_overlap_seconds"] == 11
    assert report["scoring_worker_seconds"] == 15
    assert report["scoring_worker_capacity_seconds"] == 80
    assert report["scoring_worker_idle_seconds"] == 65
    assert report["scoring_worker_utilization"] == 15 / 80
    with pytest.raises(ValueError, match="invalid interval"):
        timing([(0, 0)], [(1, 2)], 0, 20, 4)
    with pytest.raises(ValueError, match="outside"):
        timing([(0, 10)], [(19, 21)], 0, 20, 4)


def test_pool_counts_distinct_live_keys_and_applies_same_timestamp_atomically():
    rows = [
        {"key": "A", "start": 1, "end": 3},
        {"key": "A", "start": 2, "end": 3},
        {"key": "B", "start": 3, "end": 6},
        {"key": "C", "start": 3, "end": 3},  # rejected before the merge transaction committed
        {"key": "D", "start": 4, "end": None},
    ]
    assert peak_intervals(rows, keyed=True) == 2
    assert peak_intervals([{"start": 1, "end": 3}, {"start": 3, "end": 5}]) == 1
    with pytest.raises(ValueError, match="invalid interval"):
        peak_intervals([{"start": 2, "end": 1}])


def _decision(number, timestamp, head, pending, candidates):
    return {"decision_id": number, "created_at": timestamp,
            "snapshot_json": json.dumps({"head_ingest_seq": head, "pending_keys": [k for _, k in pending],
                                         "inputs": {"head_ingest_seq": head, "pending": pending,
                                                    "candidates": [{"canonical_key": k} for k in candidates],
                                                    "count": 1}, "record": {}})}


def _timeline():
    decisions = [_decision(1, 1, 0, [], ["A", "B", "C"]),
                 _decision(2, 2, 0, [["e1", "A"]], ["B", "C"]),
                 _decision(3, 4, 1, [["e2", "B"]], ["C"])]
    experiments = [{"decision_id": i, "experiment_id": f"e{i}", "canonical_key": key}
                   for i, key in enumerate("ABC", start=1)]
    observations = [{"ingest_seq": i, "experiment_id": f"e{i}", "ingested_at": t}
                    for i, t in enumerate((3, 5.2, 5.5), start=1)]
    generations = [{"generation": 1, "state": "ready", "started_at": 0, "finished_at": 0.5},
                   {"generation": 2, "state": "ready", "started_at": 2.5, "finished_at": 6}]
    attempts = [{"experiment_id": f"e{i}", "started_at": a, "ended_at": b}
                for i, (a, b) in enumerate(((1, 2.9), (2, 5), (4.1, 5.1)), start=1)]
    return decisions, experiments, observations, generations, attempts


def test_result_returned_during_research_is_used_by_later_decision():
    decisions, experiments, observations, generations, attempts = _timeline()
    problems, proof = decision_timeline(decisions, experiments, observations, generations, attempts)
    assert problems == []
    assert len(proof) == 1
    assert proof[0]["generation"] == 2
    assert proof[0]["decision_id"] == 3
    assert proof[0]["ingest_seq"] == proof[0]["head_ingest_seq"] == 1

    decisions[2] = _decision(3, 4, 0, [["e1", "A"], ["e2", "B"]], ["C"])
    problems, proof = decision_timeline(decisions, experiments, observations, generations, attempts)
    assert any("history head omits" in p for p in problems)
    assert proof == []


def test_overlap_alone_or_stale_pending_does_not_prove_streaming():
    decisions, experiments, observations, generations, attempts = _timeline()
    generations[1]["finished_at"] = 3.5  # scoring overlaps and returns, but no new decision during research
    problems, proof = decision_timeline(decisions, experiments, observations, generations, attempts)
    assert problems == []
    assert proof == []
    generations[1]["finished_at"] = 6
    decisions[2] = _decision(3, 4, 1, [], ["C"])
    problems, _ = decision_timeline(decisions, experiments, observations, generations, attempts)
    assert any("pending inputs differ" in p for p in problems)
    decisions[2] = _decision(3, 4, 1, [["e2", "B"]], ["C"])
    snapshot = json.loads(decisions[2]["snapshot_json"])
    snapshot["plan"] = {"id": 1, "step": 0}
    decisions[2]["snapshot_json"] = json.dumps(snapshot)
    problems, proof = decision_timeline(decisions, experiments, observations, generations, attempts)
    assert any("not an independent streaming selection" in p for p in problems)
    assert proof == []


def test_token_usage_does_not_treat_missing_as_zero():
    complete = {"providerCalls": 2, "tokens": {"input": 10, "cachedInput": 3, "output": 4, "reasoning": 1}}
    result = _usage({"complete": [complete], "partial": [complete, {"providerCalls": 1}], "absent": []})
    assert result["complete"]["tokens"]["input"] == 10
    assert result["complete"]["provider_calls"] == 2
    assert result["partial"]["provider_calls"] == 3
    assert result["partial"]["tokens"] is None
    assert result["partial"]["known_tokens"]["input"] == 10
    assert result["partial"]["token_usage_missing_turns"] == 1
    assert result["absent"]["provider_calls"] is None
    assert result["absent"]["tokens"] is None


def test_cumulative_retry_usage_is_not_counted_twice():
    failed = {"providerCalls": 7, "tokens": {"input": 100, "output": 5}}
    completed = {"providerCalls": 9, "tokens": {"input": 130, "output": 6}}
    assert cumulative_usage(completed, [failed]) == completed
    assert cumulative_usage({}, [failed]) == failed
    with pytest.raises(ValueError, match="decreased"):
        cumulative_usage({"providerCalls": 6}, [failed])


def test_scoring_retry_requires_proven_stop_and_one_success(tmp_path):
    root = tmp_path / "campaign"
    eid, sequence = "experiment", "ABCDEFGHIJK"
    service = {"experiment_id": eid, "cancel_requested_at": None}
    attempts = []
    for idx, state in ((1, "stopped"), (2, "succeeded")):
        ref = hashlib.sha256(json.dumps([eid, idx], separators=(",", ":")).encode()).hexdigest()
        base = root / "absolut-attempts" / ref
        base.mkdir(parents=True)
        (base / "request.json").write_text(json.dumps({"sequence": sequence, "antigen": "1N8Z_C", "oracle": "absolut",
                                                       "absolut_path": str(tmp_path / "absolut")}))
        (base / "worker.json").write_text(json.dumps({"pid": 1, "start": 1, "boot_id": "boot"}))
        (base / "worker.log").touch()
        result = {"state": state, **({"metrics": {"absolut_energy": -10}} if idx == 2 else {})}
        (base / "result.json").write_text(json.dumps(result))
        if idx == 1:
            # executor.close writes a cancellation file, even when ES did not request a cancel.
            (base / "cancel.json").write_text('{}')
            outcome = {"state": "stopped", "failure_class": "infrastructure",
                       "message": "executor stopped the attempt without a request"}
        else:
            outcome = {"state": "succeeded"}
        attempts.append({"idx": idx, "state": "ended", "executor_ref": ref, "started_at": idx,
                         "ended_at": idx + 1, "outcome": json.dumps(outcome), "cancel_attempted_at": None,
                         "cancel_acknowledged_at": None, "timeout_at": None})
    assert scoring_attempt_problems(root, service, sequence, -10, attempts, []) == []
    attempts[0]["timeout_at"] = 1.5
    assert any("not a proven infrastructure stop" in p for p in
               scoring_attempt_problems(root, service, sequence, -10, attempts, []))
    attempts[0]["timeout_at"] = None
    service["cancel_requested_at"] = 1.5
    assert any("not a proven infrastructure stop" in p for p in
               scoring_attempt_problems(root, service, sequence, -10, attempts, []))
    service["cancel_requested_at"] = None
    attempts[0]["outcome"] = json.dumps({"state": "failed", "failure_class": "execution_error"})
    assert any("not a proven infrastructure stop" in p for p in
               scoring_attempt_problems(root, service, sequence, -10, attempts, []))
    attempts[0]["outcome"] = json.dumps({"state": "succeeded"})
    assert any("exactly one successful" in p for p in
               scoring_attempt_problems(root, service, sequence, -10, attempts, []))


def test_proposal_is_bound_to_issuing_researcher_and_model_receipt():
    issued = {"design_id": "job-1/0", "sequence": "ABCDEFGHIKL", "violations": []}
    receipt = {"profile_id": "explorer", "round": 1}
    proposal = {"lineage": "explorer", "candidate": {"canonical_key": issued["sequence"],
                "payload": {"design_id": issued["design_id"], "sequence": issued["sequence"]}}}
    assert proposal_matches_source(proposal, issued, receipt, 2)
    proposal["lineage"] = "refiner"
    assert not proposal_matches_source(proposal, issued, receipt, 2)
    proposal["lineage"] = "explorer"
    issued["violations"] = ["invalid"]
    assert not proposal_matches_source(proposal, issued, receipt, 2)


def test_fresh_head_must_be_selected_by_a_fitted_pending_aware_gp():
    choice = {"metadata": {"gp_ready": True, "pending_handling": "kriging_believer"}}
    assert decision_uses_gp_on_fresh_head(3, choice, {"B"}, {3})
    assert not decision_uses_gp_on_fresh_head(2, choice, {"B"}, {3})
    assert not decision_uses_gp_on_fresh_head(3, choice, set(), {3})
    assert not decision_uses_gp_on_fresh_head(3, {"metadata": {"gp_ready": False,
        "pending_handling": "kriging_believer"}}, {"B"}, {3})
    assert not decision_uses_gp_on_fresh_head(3, {"metadata": {"gp_ready": True,
        "pending_handling": "dedup_only"}}, {"B"}, {3})


def test_efficiency_uses_persisted_events_and_complete_token_receipts(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (tmp_path / "launch.json").write_text(json.dumps({"started_at": "1970-01-01T00:00:00+00:00",
                                                     "finished_at": "1970-01-01T00:00:25+00:00",
                                                     "case": {"started_at": "1970-01-01T00:00:00+00:00",
                                                              "finished_at": "1970-01-01T00:00:22+00:00",
                                                              "elapsed_seconds": 22}}))
    usage = {"providerCalls": 1, "tokens": {"input": 10, "cachedInput": 2, "output": 4, "reasoning": 1}}
    outcome = {"metadata": {"sessions": {role: {"usage": usage}
                                      for role in ("explorer", "refiner", "evidence_reviewer")}}}
    connection = sqlite3.connect(root / "controller.sqlite3")
    connection.executescript("""
        CREATE TABLE campaign(singleton INTEGER PRIMARY KEY, created_at REAL);
        INSERT INTO campaign VALUES(1, 0);
        CREATE TABLE events(seq INTEGER, type TEXT, subject_id TEXT, payload_json TEXT, created_at REAL);
        INSERT INTO events VALUES(1, 'generation.started', '1', '{}', 0),
                                 (2, 'generation.started', '2', '{}', 11),
                                 (3, 'campaign.stopped', NULL, '{}', 20);
        CREATE TABLE generations(generation INTEGER, state TEXT, started_at REAL, finished_at REAL, outcome_json TEXT);
        CREATE TABLE observations(status TEXT);
        INSERT INTO observations VALUES('succeeded'),('succeeded'),('succeeded');
    """)
    connection.executemany("INSERT INTO generations VALUES (?, 'ready', ?, ?, ?)",
                           [(1, 0, 10, json.dumps(outcome)), (2, 11, 16, json.dumps(outcome))])
    connection.commit()
    connection.close()
    connection = sqlite3.connect(root / "experiment-service.sqlite3")
    connection.execute("CREATE TABLE attempts(executor_ref TEXT, started_at REAL, ended_at REAL)")
    connection.executemany("INSERT INTO attempts VALUES (?, ?, ?)", [("a", 2, 6), ("b", 4, 9), ("c", 12, 18)])
    connection.commit()
    connection.close()
    for ref, stamp in (("a", 6), ("b", 9), ("c", 18)):
        path = root / "absolut-attempts" / ref
        path.mkdir(parents=True)
        result = path / "result.json"
        result.write_text("{}")
        os.utime(result, (stamp, stamp))
    for number in (1, 2):
        path = root / "policy_harness" / "rounds" / f"round_{number:03d}"
        path.mkdir(parents=True)
        (path / "result.json").write_text(json.dumps({"harness_turn": {"usage": usage}}))
    job = root / "design_jobs" / "job-00001"
    (job / "output").mkdir(parents=True)
    (job / "receipt.json").write_text(json.dumps({"count": 2, "candidates": [{}, {}], "status": "succeeded"}))
    (job / "output" / "rfd3.execution.json").write_text(json.dumps({"elapsed_seconds": 3}))
    (job / "output" / "proteinmpnn-0000.execution.json").write_text(json.dumps({"elapsed_seconds": 5}))
    db, es = ReadOnlyStore(root / "controller.sqlite3"), ReadOnlyStore(root / "experiment-service.sqlite3")
    try:
        report = efficiency(root, db, es, 4)
    finally:
        db.close()
        es.close()
    assert report["research_generations_started"] == report["research_generations_ready"] == 2
    assert report["raw_sequences_generated"] == 2
    assert report["successful_scores_per_hour"] == 540
    assert report["research_scoring_overlap_seconds"] == 11
    assert report["campaign_runner_active_seconds"] == 20
    assert report["campaign_outside_runner_seconds"] == 0
    assert report["model_execution_seconds"]["rfd3"]["elapsed_seconds_sum"] == 3
    assert report["model_scoring_overlap_seconds"] is None
    assert report["llm_usage_total"]["provider_calls"] == 8
    assert report["llm_usage_total"]["tokens"]["input"] == 80
    assert report["llm_usage_total"]["provider_calls_per_success"] == 8 / 3


def test_resumed_efficiency_preserves_failures_and_cumulative_usage(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    case = lambda begin, end: {"started_at": f"1970-01-01T00:00:{begin:02d}+00:00",
                               "finished_at": f"1970-01-01T00:00:{end:02d}+00:00"}
    recovery = tmp_path / "recoveries" / "attempt-001"
    recovery.mkdir(parents=True)
    (recovery / "launch-before.json").write_text(json.dumps({"case": case(0, 9)}))
    launch = {"started_at": "1970-01-01T00:00:00+00:00", "finished_at": "1970-01-01T00:00:22+00:00",
              "case": case(11, 21), "recoveries": [{"case": case(11, 21)}]}
    (tmp_path / "launch.json").write_text(json.dumps(launch))
    usage = {"providerCalls": 1, "tokens": {"input": 10, "cachedInput": 2, "output": 4, "reasoning": 1}}
    failed_usage = {"providerCalls": 2, "tokens": {"input": 20, "cachedInput": 4, "output": 8, "reasoning": 2}}
    final_usage = {"providerCalls": 3, "tokens": {"input": 30, "cachedInput": 6, "output": 12, "reasoning": 3}}
    metadata = {"sessions": {role: {"usage": usage} for role in ("explorer", "refiner", "evidence_reviewer")}}
    final_outcome = {"status": "committed", "proposals": [], "metadata": {**metadata,
                     "policy_round": {"status": "accepted", "degraded": False}}, "merge": {}}
    failed_outcome = {"status": "committed", "proposals": [], "metadata": {**metadata,
                      "policy_round": {"status": "runtime_fallback", "degraded": True}}}
    connection = sqlite3.connect(root / "controller.sqlite3")
    connection.executescript("""
        CREATE TABLE campaign(singleton INTEGER PRIMARY KEY, created_at REAL);
        INSERT INTO campaign VALUES(1, 0);
        CREATE TABLE events(seq INTEGER, type TEXT, subject_id TEXT, payload_json TEXT, created_at REAL);
        CREATE TABLE generations(generation INTEGER, state TEXT, started_at REAL, finished_at REAL,
                                 outcome_json TEXT, input_digest TEXT);
        CREATE TABLE observations(status TEXT);
        INSERT INTO observations VALUES('succeeded');
    """)
    connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", [
        (1, "generation.started", "1", "{}", 0),
        (2, "generation.blocked", "1", json.dumps({"error": "status=runtime_fallback, degraded=True"}), 8),
        (3, "generation.started", "1", json.dumps({"resolution": "retry", "policy_retry": True}), 12),
        (4, "campaign.stopped", None, "{}", 20),
    ])
    connection.execute("INSERT INTO generations VALUES(1, 'ready', 12, 18, ?, ?)",
                       (json.dumps(final_outcome), "sha256:research"))
    connection.commit()
    connection.close()
    connection = sqlite3.connect(root / "experiment-service.sqlite3")
    connection.execute("CREATE TABLE attempts(executor_ref TEXT, started_at REAL, ended_at REAL)")
    connection.executemany("INSERT INTO attempts VALUES (?, ?, ?)", [("a", 2, 12), ("b", 12, 18)])
    connection.commit()
    connection.close()
    for ref, stamp in (("a", 8), ("b", 17)):
        path = root / "absolut-attempts" / ref
        path.mkdir(parents=True)
        result = path / "result.json"
        result.write_text("{}")
        os.utime(result, (stamp, stamp))
    policy = root / "policy_harness" / "rounds" / "round_001"
    archive = policy / "retries" / "event-000000000003"
    archive.mkdir(parents=True)
    (policy / "manifest.json").write_text(json.dumps({"input_sha256": "policy"}))
    (policy / "result.json").write_text(json.dumps({"harness_turn": {"usage": final_usage}}))
    (archive / "outcome.json").write_text(json.dumps({"input_digest": "sha256:research", "outcome": failed_outcome}))
    (archive / "compiled.npz").write_bytes(b"original arrays")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (archive / "result.json").write_text(json.dumps({"status": "runtime_fallback", "degraded": True,
        "input_sha256": "policy", "compiled_arrays_sha256": sha(archive / "compiled.npz"),
        "failed_harness_usage": failed_usage}))
    names = ("outcome.json", "result.json", "compiled.npz")
    (archive / "prepared.json").write_text(json.dumps({"generation": 1, "retry_event_seq": 3,
        "research_input_sha256": "sha256:research", "policy_input_sha256": "policy",
        "files": {name: sha(archive / name) for name in names}}))
    db, es = ReadOnlyStore(root / "controller.sqlite3"), ReadOnlyStore(root / "experiment-service.sqlite3")
    try:
        report = efficiency(root, db, es, 4)
        assert report["campaign_wall_seconds"] == 20
        assert report["campaign_runner_active_seconds"] == 18
        assert report["campaign_outside_runner_seconds"] == 2
        assert report["research_wall_seconds"] == 14
        assert report["scoring_worker_seconds"] == 11
        assert report["scoring_service_recorded_worker_seconds"] == 16
        assert report["policy_retry_count"] == 1
        assert report["llm_usage_total"]["provider_calls"] == 6
        assert report["llm_usage_total"]["tokens"]["input"] == 60
        assert report["policy_retry_cumulative_usage"][0]["failed_cumulative_snapshots"] == [failed_usage]
        (archive / "compiled.npz").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="invalid failed policy retry archive"):
            efficiency(root, db, es, 4)
    finally:
        db.close()
        es.close()


def test_incomplete_baseline_is_not_compared(tmp_path):
    result = comparison(tmp_path / "missing", {"campaign_wall_seconds": 10}, 640, 4)
    assert result["status"] == "unavailable"
