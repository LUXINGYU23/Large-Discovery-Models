"""Replacement batches expose the same candidates to policy research and BO."""
import json

from ldm_tts.async_runtime import CampaignConfig, PacingConfig, campaign_extra
from tasks.antibody_rfd3.tests.test_policy_lifecycle import lifecycle


def test_policy_query_replaces_unselected_old_candidates(lifecycle):
    rig = lifecycle
    rig.pacing = PacingConfig.barrier(reservoir_size=6, selection_batch_size=2, research_budget=4,
                                     research_timeout_seconds=100)
    rig.config = CampaignConfig("policy-lifecycle", budget=8, concurrency=2, seed=42,
                                extra=campaign_extra(rig.pacing))
    ctrl = rig.open()
    rig.drive()
    assert len(ctrl.observations()) == 8
    assert len(ctrl.plans()) == 4
    for plan in ctrl.plans():
        generation = plan["generation"]
        inputs = json.loads(plan["inputs_json"])
        keys = {candidate["canonical_key"] for candidate in inputs["candidates"]}
        assert len(keys) == 6
        snapshot = json.loads((rig.root / "policy_harness/rounds" /
                               f"round_{generation:03d}/research_snapshot.json").read_text())
        assert {candidate["canonical_key"] for candidate in snapshot["candidate_catalog"]} == keys
        application = ctrl.policy.application(inputs["prior_application_id"])
        row = ctrl.store.row("SELECT query_keys_json FROM prior_applications WHERE application_id=?",
                             (application.application_id,))
        assert set(json.loads(row[0])) == keys
        assert inputs["head_ingest_seq"] == (generation - 1) * 2
    assert ctrl.policy.check() == []
    assert all(ctrl.audit(row["decision_id"]) == [] for row in ctrl.decisions())
