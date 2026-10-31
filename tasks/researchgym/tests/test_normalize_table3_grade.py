import copy
import hashlib
import json

import pytest

from tasks.researchgym.scripts.normalize_table3_grade import GradeError, normalize
from tasks.researchgym.table3_protocol import protocol_manifest


CMR_CONTRACT = {"source": "benchmark-authors/contract.json", "sha256": "b" * 64}


def _receipt(manifest, task, seed=0):
    return {
        "run_id": f"{task}-{seed}", "status": "budget_exhausted", "task_id": task, "seed": seed,
        "profile": manifest["profile"]["name"], "protocol": manifest["protocol"],
        "source_commit": manifest["source_commit"], "model": manifest["model"], "harness": manifest["harness"],
        "wall_clock_hours": 1.0, "api_cost_usd": 1.0,
    }


def _normalize(tmp_path, task, payload, **kwargs):
    manifest = protocol_manifest("table3", model="provider/model", harness="LDM-CodeAgent",
                                 cmr_primary_row="qgs_blip_base", cmr_table3_contract=CMR_CONTRACT)
    summary = tmp_path / f"{task}.json"
    summary.write_text(json.dumps(payload))
    return normalize(manifest=manifest, agent_receipt=_receipt(manifest, task), summary_path=summary,
                     task_id=task, **kwargs)


def test_materials_primary_is_sofc_and_matscholar_test_average_with_explicit_missing_zero(tmp_path):
    payload = {"classification": {
        "macro": {"ner_sofc_test": {"values": [0.8]}, "ner_matscholar_test": {"values": [0.9]}},
        "micro": {"ner_sofc_test": {"values": [0.81]}, "ner_matscholar_test": {"values": []}},
    }}
    record = _normalize(tmp_path, "materials_tokenization", payload)
    assert record["metrics"] == {"macro_f1": 85.0, "micro_f1": 40.5}
    assert record["metric_completion"] == {"macro_f1": True, "micro_f1": False}


def test_cmr_requires_an_immutable_primary_row_and_reads_paired_columns(tmp_path):
    payload = {"rows": [{"row": "qgs_blip_base", "columns": [
        {"id": "flickr:txt_r1", "value": 61.0}, {"id": "flickr:img_r1", "value": 70.0},
    ]}]}
    record = _normalize(tmp_path, "cross_modal_retrieval", payload)
    assert record["metrics"] == {"i2tr_at_1": 61.0, "t2ir_at_1": 70.0}
    with pytest.raises(GradeError, match="does not match"):
        _normalize(tmp_path, "cross_modal_retrieval", payload, cmr_row="qgs_clip_base")


def test_time_and_irb_primary_extractors_preserve_missing_completion(tmp_path):
    tim = _normalize(tmp_path, "time_series_explanation", {"real": {"PAM": {
        "Average": {"mean": 0.5}, "Zeros": {"mean": 0.4},
    }}})
    assert tim["metrics"] == {"cpd_a": 0.5, "cpd_z": 0.4}
    irb = _normalize(tmp_path, "improving_replay_buffers", {"environments": {}})
    assert irb["metrics"] == {"return": 0.0}
    assert irb["metric_completion"] == {"return": False}
