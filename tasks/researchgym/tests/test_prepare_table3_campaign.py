import json

import pytest

from tasks.researchgym.scripts.prepare_table3_campaign import prepare_campaign
from tasks.researchgym.table3_protocol import ProtocolError


CMR_CONTRACT = {"source": "benchmark-authors/contract.json", "sha256": "b" * 64}


def test_prepare_campaign_writes_exact_three_seed_matrix_and_caps(tmp_path):
    campaign = tmp_path / "table3"
    plan = prepare_campaign(campaign_dir=campaign, profile_name="table3", model="provider/model",
                            harness="LDM-CodeAgent", cmr_primary_row="qgs_blip_base",
                            external_access="disabled", max_concurrent=8,
                            cmr_table3_contract=CMR_CONTRACT)
    assert plan["totals"] == {
        "jobs": 15, "max_api_spend_usd": 150.0, "max_gpu_hours": 180.0,
        "suggested_waves": 2, "best_case_wall_clock_hours": 24.0,
    }
    assert {(job["task_id"], job["seed"]) for job in plan["jobs"]} == {
        (task_id, seed) for task_id in ("continual_learning", "materials_tokenization",
                                        "cross_modal_retrieval", "time_series_explanation",
                                        "improving_replay_buffers") for seed in (0, 1, 2)
    }
    manifest = json.loads((campaign / "manifest.json").read_text())
    assert manifest["metric_selectors"] == {"cmr_primary_row": "qgs_blip_base"}
    assert manifest["table3_eligibility"]["eligible"] is True
    with pytest.raises(ProtocolError, match="already initialized"):
        prepare_campaign(campaign_dir=campaign, profile_name="table3", model="provider/model",
                         harness="LDM-CodeAgent", cmr_primary_row="qgs_blip_base",
                         external_access="disabled", max_concurrent=8,
                         cmr_table3_contract=CMR_CONTRACT)
