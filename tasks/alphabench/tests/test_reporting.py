import json

import pytest

from tasks.alphabench.aggregate import aggregate
from tasks.alphabench.core.protocol import digest
from tasks.alphabench.core.reporting import aggregate_fraction_success, ea_update_metrics, generation_costs


def _snapshot(round_index, pool, candidates=()):
    return {"kind": "ea.pool", "boundary": f"root/{round_index}", "state": {
        "round": round_index,
        "pool": [{"expression": expression} for expression in pool],
        "candidates": [{"expression": expression} for expression in candidates]}}


def test_ea_update_rate_counts_new_top_pool_entries_per_completed_round():
    states = [_snapshot(0, ["a", "b"]),
              _snapshot(1, ["c", "a"], ["c", "d"]),
              _snapshot(2, ["c", "a"], ["e"])]
    result = ea_update_metrics(states, planned_rounds=2)
    assert [event["U_t"] for event in result["events"]] == [1, 0]
    assert result["events"][0]["entered_expressions"] == ["c"]
    assert result["sum_U_t"] == 1 and result["update_rate"] == .5
    assert result["rounds"] == 2 and result["best_update_events"] == 1
    assert result["reason"] is None
    partial = ea_update_metrics(states[:2], planned_rounds=2)
    assert partial["update_rate"] == 1 and partial["reason"] == "incomplete_rounds"


def test_ea_update_rate_marks_missing_or_noncontiguous_history():
    empty = ea_update_metrics([], planned_rounds=2)
    assert empty["update_rate"] is None and empty["reason"] == "no_ea_pool_snapshots"
    with pytest.raises(ValueError, match="contiguous"):
        ea_update_metrics([_snapshot(0, ["a"]), _snapshot(2, ["b"], ["b"])], planned_rounds=2)
    with pytest.raises(ValueError, match="recorded generation"):
        ea_update_metrics([_snapshot(0, ["a"]), _snapshot(1, ["b"], [])], planned_rounds=1)


def test_harness_provider_usage_missing_is_not_reported_as_zero():
    result = generation_costs([{"attempt_kind": "harness_provider_calls", "attempt_count": None,
                                "submission_attempts": 1, "sidecar_usage": {}, "complete": True}])
    assert result["attempts_per_step"] == [None]
    assert result["total_attempts"] is None and result["mean"] is None
    assert result["unavailable_reason"] == "sidecar_provider_usage_missing"


def test_initialization_and_harness_search_keep_distinct_attempt_units():
    result = generation_costs([
        {"attempt_count": 2, "complete": True},
        {"attempt_kind": "harness_provider_calls", "attempt_count": 3,
         "submission_attempts": 2, "sidecar_usage": {"providerCalls": 3}, "complete": True}])
    assert result["total_attempts"] is None
    assert result["unavailable_reason"] == "different_attempt_units"
    assert result["components"]["direct"]["total_attempts"] == 2
    assert result["components"]["harness"]["total_attempts"] == 3


def test_frac_success_keeps_failed_and_missing_runs_in_registered_denominator():
    roster = [{"run_id": str(seed), "seed": seed, "protocol_digest": "digest" + str(seed)}
              for seed in (11, 12, 13)]
    reports = {
        "11": {"protocol_digest": "digest11", "protocol": {"random_seed": 11},
               "method": "alphabench_cot", "search": {"run_success": True}},
        "12": {"protocol_digest": "digest12", "protocol": {"random_seed": 12},
               "method": "alphabench_cot", "search": {"run_success": False}},
    }
    result = aggregate_fraction_success(roster, reports)
    assert result["frac_success"] == 1 / 3
    assert result["included_runs"] == 3
    assert result["runs"][-1]["reason"] == "missing_result"
    excluded = aggregate_fraction_success(roster, reports,
        invalid_evidence={"13": "receipt indicates provider outage"}, invalid_rule="exclude_with_evidence")
    assert excluded["frac_success"] == .5
    assert excluded["runs"][-1]["included"] is False
    assert aggregate_fraction_success(roster, reports,
        invalid_evidence={"13": "receipt indicates provider outage"})["frac_success"] == 1 / 3
    with pytest.raises(ValueError, match="cannot remove"):
        aggregate_fraction_success(roster, reports, invalid_evidence={"11": "invalid"},
                                   invalid_rule="exclude_with_evidence")


def test_aggregate_reads_frozen_cohort_and_retains_result_hashes(tmp_path):
    runs = []
    for seed in (11, 12, 13):
        directory = tmp_path / str(seed)
        directory.mkdir()
        protocol = {"random_seed": seed, "backend": "qlib", "market": "csi300",
                    "profile": "upstream_searcher_v1"}
        runs.append({"run_id": str(seed), "seed": seed, "protocol_digest": digest(protocol), "run_dir": str(seed)})
        if seed != 13:
            (directory / "result.json").write_text(json.dumps({"protocol_digest": digest(protocol),
                "protocol": protocol, "method": "alphabench_cot",
                "search": {"run_success": seed == 11}}))
            (directory / "status.json").write_text(json.dumps({"status": "completed"}))
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"schema_version": 1, "method": "alphabench_cot", "backend": "qlib",
        "market": "csi300", "profile": "upstream_searcher_v1", "invalid_rule": "exclude_with_evidence",
        "runs": runs}))
    evidence = tmp_path / "outage.txt"
    evidence.write_text("provider outage")
    invalidations = tmp_path / "invalidations.json"
    invalidations.write_text(json.dumps({"13": evidence.name}))
    result = aggregate(roster, invalidations)
    assert result["frac_success"] == .5
    assert result["pre_registered_runs"] == 3 and result["included_runs"] == 2
    assert set(result["result_sha256"]) == {"11", "12"}
    assert result["invalidation_evidence"]["13"]["sha256"]
    assert result["aggregation_qualified"] is False
    assert result["cohort_protocol_digest"]
    (tmp_path / "11/status.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="completed status"):
        aggregate(roster, invalidations)
    (tmp_path / "11/status.json").write_text(json.dumps({"status": "completed"}))
    changed = json.loads((tmp_path / "12/result.json").read_text())
    changed["protocol"]["temperature"] = .9
    changed["protocol_digest"] = digest(changed["protocol"])
    (tmp_path / "12/result.json").write_text(json.dumps(changed))
    updated_roster = json.loads(roster.read_text())
    updated_roster["runs"][1]["protocol_digest"] = changed["protocol_digest"]
    roster.write_text(json.dumps(updated_roster))
    with pytest.raises(ValueError, match="beyond the random seed"):
        aggregate(roster, invalidations)
