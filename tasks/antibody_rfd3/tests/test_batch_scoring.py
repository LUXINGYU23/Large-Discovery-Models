"""Checks of the batch audit's plan linkage and sparse-history policy handling."""
import hashlib
import json

from tasks.antibody_rfd3.repro.h20 import audit_batch_ldm as audit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Store:
    def __init__(self, plans, applications=()):
        self.plans, self.applications = plans, applications

    def rows(self, query, _args=()):
        if "FROM plans" in query:
            return self.plans
        if "FROM prior_applications" in query:
            return self.applications
        raise AssertionError(query)


def synthetic_audit(tmp_path):
    checker = audit.BatchAudit(tmp_path / "campaign", 1, 2, 2)
    chosen = [{"canonical_key": "A"}, {"canonical_key": "B"}]
    probabilities = [0.5, 1.0]
    steps = [{"candidate_ids": ["ca", "cb"], "canonical_keys": ["A", "B"],
              "selected_candidate_id": candidate, "selected_probability": probability,
              "metadata": {"gp_ready": False}}
             for candidate, probability in zip(("ca", "cb"), probabilities)]
    record = {"selector": {"objective": "energy_utility", "feature_version": "cdrh3-one-hot-v1"},
              "selection": {"steps": steps, "parameters": {"alpha": 1.0, "eta": 1.0},
                            "metadata": {"prior_applied": False}}}
    inputs = {"candidates": chosen, "head_ingest_seq": 0, "pending": [], "prior_application_id": "a1"}
    plan = {"plan_id": 1, "generation": 1, "count": 2, "consumed": 2, "closed_reason": "complete",
            "created_at": 10, "keys_json": json.dumps(chosen), "probabilities_json": json.dumps(probabilities),
            "inputs_json": json.dumps(inputs), "record_json": json.dumps(record)}
    checker.db = Store([plan])
    checker.decisions = [{"decision_id": i + 1, "snapshot_json": json.dumps(
        {"plan": {"id": 1, "step": i}, "probability": probabilities[i]})} for i in range(2)]
    checker.experiments = [{"decision_id": i + 1, "experiment_id": f"e{i}", "canonical_key": key}
                           for i, key in enumerate(("A", "B"))]
    checker.observation_rows = [{"experiment_id": f"e{i}", "ingest_seq": i + 1} for i in range(2)]
    checker.pools = {1: {"A", "B"}}
    return checker, record, inputs


def test_batch_audit_connects_every_step_to_its_own_experiment(tmp_path):
    checker, _record, _inputs = synthetic_audit(tmp_path)
    checker.plans()
    assert checker.problems == []
    checker.experiments[1]["canonical_key"] = "A"
    checker.plans()
    assert any("candidate, probability or observation differs" in problem for problem in checker.problems)


def test_one_batch_policy_does_not_require_a_fitted_gp(tmp_path, monkeypatch):
    from ldm_tts.async_runtime import policy, selectors

    checker, record, inputs = synthetic_audit(tmp_path)
    checker.db.applications = [{"application_id": "a1", "generation": 1, "degraded": 0, "prior_free": 1,
                                "created_at": 9, "superseded_at": None, "alpha": 1.0, "eta": 1.0}]
    checker.generations = [{"generation": 1, "outcome_json": json.dumps({"metadata": {"policy_round": {
        "status": "accepted", "degraded": False, "epoch_id": None, "artifact_sha256": None}}})}]
    root = checker.root / "policy_harness"
    dump(checker.root.parent / "launch.json", {"model": "test-model"})
    dump(root / "manifest.json", {"model": "test-model", "profiles": [
        {"profileId": "policy_architect", "session": "session.jsonl"}]})
    policy_round = root / "rounds/round_001"
    policy_round.mkdir(parents=True)
    compiled = b"compiled-test-data"
    (policy_round / "compiled.npz").write_bytes(compiled)
    dump(policy_round / "manifest.json", {"input_sha256": "frozen-input", "files": {}})
    dump(policy_round / "result.json", {"status": "accepted", "degraded": False, "epoch_id": None,
        "artifact_sha256": None, "action": "disable", "input_sha256": "frozen-input",
        "compiled_arrays_sha256": hashlib.sha256(compiled).hexdigest(), "submission_sha256": "submission",
        "harness_turn": {"turn_id": "t1", "usage": {"providerCalls": 1}}})
    dump(root / "turns/t1/turn_committed.json", {"profileId": "policy_architect", "submissionStatus": "accepted",
        "submissionDigest": "submission", "submission": {"action": "disable"}})
    monkeypatch.setattr(audit, "successful_reads", lambda _path: set())
    monkeypatch.setattr(audit, "read_skills", lambda _reads, names: names)
    monkeypatch.setattr(policy, "check_prior_invariants", lambda *_args: [])
    monkeypatch.setattr(selectors, "decision_choice", lambda _db, decision: (record, inputs, decision - 1))
    monkeypatch.setattr(checker, "check_application_context", lambda *_args: {"context_checked": True})
    replays = []
    monkeypatch.setattr(checker, "replay_decisions", lambda values, *_args: replays.extend(values))
    checker.policy()
    assert checker.problems == []
    assert checker.report["coverage"]["fitted_gp_with_custom_prior"] == "not_exercised"
    assert replays == [(1, "a1"), (2, "a1")]
