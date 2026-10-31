import copy
import math

import pytest

from tasks.researchgym.table3_protocol import (
    CMR_PRIMARY_ROWS,
    PROFILES,
    SEEDS,
    TASKS,
    ProtocolError,
    aggregate,
    protocol_manifest,
)


CMR_CONTRACT = {"source": "benchmark-authors/contract.json", "sha256": "b" * 64}


def _records(manifest):
    records = []
    for task_index, (task_id, task) in enumerate(TASKS.items()):
        for seed in SEEDS:
            # Every metric improves with seed.  This makes it testable that
            # Best@3 takes one seed, instead of independent column maxima.
            records.append({
                "task_id": task_id,
                "seed": seed,
                "run_id": f"{task_id}-{seed}",
                "status": "completed",
                "source_commit": manifest["source_commit"],
                "protocol": manifest["protocol"],
                "profile": manifest["profile"]["name"],
                "wall_clock_hours": manifest["profile"]["wall_clock_hours"],
                "api_cost_usd": manifest["profile"]["api_budget_usd"],
                "metrics": {metric: float(task_index * 10 + seed + position)
                            for position, metric in enumerate(task.metrics)},
                "official_grade": {"summary_path": f"/immutable/{task_id}/{seed}.json",
                                   "summary_sha256": "a" * 64,
                                   "cmr_row": (manifest["metric_selectors"]["cmr_primary_row"]
                                               if task_id == "cross_modal_retrieval" else None)},
            })
    return records


def test_profiles_match_requested_table3_and_scaffold_budgets():
    assert (PROFILES["table3"].api_budget_usd, PROFILES["table3"].wall_clock_hours) == (10.0, 12.0)
    assert (PROFILES["scaffold_comparison"].api_budget_usd,
            PROFILES["scaffold_comparison"].wall_clock_hours) == (20.0, 24.0)
    assert len(TASKS) == 5 and SEEDS == (0, 1, 2)


def test_aggregate_requires_all_five_tasks_and_three_seeds():
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base", cmr_table3_contract=CMR_CONTRACT)
    records = _records(manifest)
    table = aggregate(manifest, records)
    assert table["status"] == "completed"
    assert table["tasks"]["continual_learning"]["best_seed"] == 2
    assert table["tasks"]["continual_learning"]["best_at_3"] == {"acc": 2.0, "aaa": 3.0}
    assert table["tasks"]["continual_learning"]["mean_std"]["acc"]["mean"] == 1.0
    assert math.isclose(table["tasks"]["continual_learning"]["mean_std"]["acc"]["std"], 1.0)

    with pytest.raises(ProtocolError, match="needs seeds"):
        aggregate(manifest, records[:-1])


def test_aggregate_rejects_any_non_agent_or_over_budget_record():
    manifest = protocol_manifest("scaffold_comparison", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base", cmr_table3_contract=CMR_CONTRACT)
    records = _records(manifest)
    invalid = copy.deepcopy(records)
    invalid[0]["protocol"] = "program_search_v0"
    with pytest.raises(ProtocolError, match="agent-trajectory"):
        aggregate(manifest, invalid)

    invalid = copy.deepcopy(records)
    invalid[0]["api_cost_usd"] = 20.00001
    with pytest.raises(ProtocolError, match="API cap"):
        aggregate(manifest, invalid)


def test_manifest_requires_a_cmr_selector_and_record_cannot_change_it():
    with pytest.raises(ProtocolError, match="cmr_primary_row"):
        protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent", cmr_primary_row="")
    assert "qgs_clip_base" in CMR_PRIMARY_ROWS
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base", cmr_table3_contract=CMR_CONTRACT)
    records = _records(manifest)
    cmr = next(item for item in records if item["task_id"] == "cross_modal_retrieval")
    cmr["official_grade"]["cmr_row"] = "qgs_clip_base"
    with pytest.raises(ProtocolError, match="CMR row"):
        aggregate(manifest, records)


def test_aggregate_is_blocked_without_author_cmr_contract():
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base")
    with pytest.raises(ProtocolError, match="CMR scoring contract"):
        aggregate(manifest, _records(manifest))
