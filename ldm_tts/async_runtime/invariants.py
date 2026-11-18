"""Invariant checks over the controller store (design 5.7 I1-I6, I8), optionally against the ES.

Used by the property/crash tests after every step and usable by operators as a self-check.
Returns a list of human-readable violations (empty means consistent).
"""
from __future__ import annotations

import json

from .config import CampaignConfig
from .ledger import dedup_keys
from .store import Store
from .wire import TERMINAL, experiment_id_for


def expected_budget_state(row) -> str:
    if row["written_off_at"] is not None:
        return "CONSUMED"
    if row["es_state"] in TERMINAL:
        return "CONSUMED" if row["es_started_at"] else "REFUNDED"
    return "HOLD"


def check_invariants(store: Store, config: CampaignConfig, es_items=None, *, synced: bool = False) -> list[str]:
    problems: list[str] = []
    add = problems.append
    decisions = store.rows("""SELECT d.decision_id, d.experiment_id, d.snapshot_json, e.experiment_id AS eid
        FROM decisions d LEFT JOIN experiments e ON e.decision_id = d.decision_id ORDER BY d.decision_id""")
    for d in decisions:
        if d["eid"] != d["experiment_id"]:
            add(f"I1: decision {d['decision_id']} maps to {d['eid']!r}, recorded {d['experiment_id']!r}")
        if d["experiment_id"] != experiment_id_for(config.campaign_id, d["decision_id"]):
            add(f"I1: experiment id {d['experiment_id']} is not derived from decision {d['decision_id']}")
        probability = json.loads(d["snapshot_json"]).get("probability")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 < probability <= 1:
            add(f"I8: decision {d['decision_id']} has no recorded selection probability")
    experiments = store.rows("SELECT * FROM experiments ORDER BY decision_id")
    if len(experiments) != len(decisions):
        add("I1: experiments and decisions differ in number")
    held = sum(r["budget_state"] in ("HOLD", "CONSUMED") for r in experiments)
    if held > config.budget:
        add(f"I2: hold + consumed = {held} exceeds budget {config.budget}")
    live_keys: dict[str, str] = {}
    tried: dict[str, int] = {}
    for r in experiments:
        eid, terminal = r["experiment_id"], r["es_state"] in TERMINAL
        if r["pending"] != int(not terminal and r["written_off_at"] is None):
            add(f"I3: {eid} pending={r['pending']} but mirror={r['es_state']} written_off={r['written_off_at']}")
        if not terminal:
            if r["canonical_key"] in live_keys:
                add(f"I3: key {r['canonical_key']} has two non-terminal experiments ({live_keys[r['canonical_key']]}, {eid})")
            live_keys.setdefault(r["canonical_key"], eid)
        if r["es_state"] == "SUCCEEDED" or (r["es_state"] == "FAILED" and r["es_failure_class"] != "infrastructure"):
            tried[r["canonical_key"]] = tried.get(r["canonical_key"], 0) + 1
        if r["budget_state"] != expected_budget_state(r):
            add(f"budget: {eid} is {r['budget_state']}, expected {expected_budget_state(r)}")
        if (r["disposition"] is not None) != terminal:
            add(f"disposition: {eid} has {r['disposition']!r} with mirror {r['es_state']}")
        if (r["es_state"] is None) != (r["es_revision"] is None):
            add(f"I6: {eid} mirror state and revision disagree")
        if r["es_state"] is not None:
            if r["submit_state"] != "ACKED":
                add(f"I6: {eid} has an ES mirror but submit_state {r['submit_state']}")
            stored = json.loads(r["es_resource_json"])
            if int(stored["revision"]) != r["es_revision"] or stored["state"] != r["es_state"]:
                add(f"I6: {eid} mirror columns differ from the stored resource")
    for key, count in tried.items():
        if count > config.max_attempts_per_key:
            add(f"dedup: key {key} was measured {count} times (max {config.max_attempts_per_key})")
    observations = store.rows("""SELECT o.*, e.es_state, e.es_result_digest, e.disposition FROM observations o
        JOIN experiments e ON e.experiment_id = o.experiment_id ORDER BY o.ingest_seq""")
    for index, o in enumerate(observations, start=1):
        if o["ingest_seq"] != index:
            add(f"I4: ingest_seq {o['ingest_seq']} is not contiguous (expected {index})")
        if o["es_state"] not in ("SUCCEEDED", "FAILED") or o["disposition"] != "ingested":
            add(f"I4/I5: observation {o['experiment_id']} for mirror {o['es_state']} / {o['disposition']}")
        if o["result_digest"] != o["es_result_digest"]:
            add(f"I4: observation {o['experiment_id']} digest differs from the ES mirror")
    ingested = {r["experiment_id"] for r in experiments if r["disposition"] == "ingested"}
    if ingested != {o["experiment_id"] for o in observations}:
        add("I4: ingested dispositions and observations differ")
    if es_items is not None:
        problems.extend(_against_es(experiments, es_items, config, synced))
    return problems


def _against_es(experiments, es_items, config, synced) -> list[str]:
    problems = []
    local = {r["experiment_id"]: r for r in experiments}
    remote = {item["experiment_id"]: item for item in es_items if item.get("campaign_id") == config.campaign_id}
    for eid in sorted(set(remote) - set(local)):
        problems.append(f"I1: ES holds {eid}, unknown to the controller")
    for eid, r in local.items():
        item = remote.get(eid)
        if item is None:
            if r["submit_state"] == "ACKED":
                problems.append(f"I1: acknowledged {eid} is missing from the ES")
            continue
        if not item.get("tombstone") and item.get("spec_digest") != r["spec_digest"]:
            problems.append(f"I1: {eid} spec digest differs from the ES")
        if r["es_revision"] is not None and int(item["revision"]) < r["es_revision"]:
            problems.append(f"I6: {eid} mirror revision is ahead of the ES")
        if r["es_state"] in TERMINAL and (item["state"] != r["es_state"]
                                          or item.get("result_digest") != r["es_result_digest"]):
            problems.append(f"I4: {eid} terminal mirror differs from the ES")
        if synced and (item["state"] != r["es_state"] or int(item["revision"]) != r["es_revision"]):
            problems.append(f"sync: {eid} mirror {r['es_state']}@{r['es_revision']} vs ES {item['state']}@{item['revision']}")
    if synced:
        consumed = sum(1 for r in local.values() if r["written_off_at"] is None and r["budget_state"] == "CONSUMED")
        es_started = sum(1 for eid, i in remote.items() if i["state"] in TERMINAL and i.get("started_at")
                         and local.get(eid) is not None and local[eid]["written_off_at"] is None)
        refunded = sum(1 for r in local.values() if r["budget_state"] == "REFUNDED")
        es_unstarted = sum(1 for i in remote.values() if i["state"] in TERMINAL and not i.get("started_at"))
        if consumed != es_started or refunded != es_unstarted:
            problems.append(f"budget: consumed/refunded {consumed}/{refunded} vs ES terminal started/unstarted "
                            f"{es_started}/{es_unstarted}")
    return problems


def check_pacing_invariants(store: Store) -> list[str]:
    """controller-pacing additions: reservoir bookkeeping (I12), plans, the lane, decision records (I8, I12)."""
    problems: list[str] = []
    add = problems.append
    config = json.loads(store.value("SELECT config_json FROM campaign WHERE singleton=1"))
    pacing = config["campaign"]["extra"].get("pacing", {})
    live = store.rows("SELECT canonical_key, watermark FROM occurrences WHERE removed_at IS NULL")
    booked = store.value("SELECT COALESCE(MAX(ingest_seq), 0) FROM observations")
    # The controller's dedup set, not "has an observation": a key whose attempts failed below
    # ``max_attempts_per_key`` (or only with infrastructure faults) may be proposed again (review item 4).
    with store.lock:
        taken = dedup_keys(store.db, int(config["campaign"].get("max_attempts_per_key", 1)))
    keys = {r["canonical_key"] for r in live}
    for key in sorted(keys & taken):
        add(f"I12: {key} is pending or deduplicated but still has live occurrences")
    if any(r["watermark"] > booked for r in live):
        add("I12: an occurrence watermark is ahead of the booked results")
    if pacing and len(keys) > pacing["reservoir_size"]:
        add(f"reservoir: {len(keys)} live candidates exceed R = {pacing['reservoir_size']}")
    plans = {r["plan_id"]: r for r in store.rows("SELECT * FROM plans")}
    used: dict[int, int] = {}
    for row in store.rows("SELECT decision_id, snapshot_json FROM decisions ORDER BY decision_id"):
        snapshot = json.loads(row["snapshot_json"])
        if pacing and "pacing_state" not in snapshot:
            add(f"I12: decision {row['decision_id']} does not record the reservoir state it used")
        if "plan" in snapshot:
            used[snapshot["plan"]["id"]] = used.get(snapshot["plan"]["id"], 0) + 1
            plan = plans.get(snapshot["plan"]["id"])
            if plan is None or json.loads(plan["probabilities_json"])[snapshot["plan"]["step"]] != snapshot["probability"]:
                add(f"I8: decision {row['decision_id']} probability differs from its plan step")
        elif pacing and "record" not in snapshot:
            add(f"I8: decision {row['decision_id']} has no selection record")
    for plan_id, plan in plans.items():
        if used.get(plan_id, 0) != plan["consumed"] or plan["consumed"] > plan["count"]:
            add(f"plan {plan_id}: consumed {plan['consumed']} of {plan['count']}, decisions {used.get(plan_id, 0)}")
    if len([p for p in plans.values() if p["closed_reason"] is None]) > 1:
        add("plans: more than one open plan")
    active = store.rows("SELECT generation FROM generations WHERE state IN ('running','blocked')")
    latest = store.value("SELECT MAX(generation) FROM generations")
    if len(active) > 1 or (active and active[0]["generation"] != latest):
        add("lane: more than one generation in flight, or one behind a newer generation")
    return problems
