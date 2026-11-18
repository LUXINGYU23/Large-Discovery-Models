"""Read-only audit of a completed barrier-batched real HER2 campaign.

The run directory is the campaign directory, not its parent launch directory.
The audit reads receipts and replays each frozen selection plan once; it does
not run scientific models, contact a provider, or execute policy artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from tasks.antibody_rfd3.repro.h20.audit_full_ldm import (
    Audit, ROLES, file_sha256, read_json, read_json_value, read_jsonl, read_skills, successful_reads,
)


class BatchAudit(Audit):
    def __init__(self, directory, rounds, batch, pool):
        super().__init__(directory, rounds, rounds * batch)
        self.batch, self.pool = batch, pool
        self.report["expected_batch_size"] = batch
        self.report["expected_pool_size"] = pool
        self.report["coverage"].update({
            "controlled_cas_race": "not_tested_by_receipt_audit",
            "controlled_crash_recovery": "not_tested_by_receipt_audit",
            "controlled_cancellation": "not_tested_by_receipt_audit",
            "independent_native_recomputation": "not_performed",
        })

    def configuration(self):
        self.config = c = read_json(self.root / "config.json")
        self.require(c.get("method") == "ldm_harness" and c.get("antigen") == "1N8Z_C" and
                     c.get("oracle") == "absolut" and c.get("mock") is False and c.get("dry_run") is False,
                     "expected real HER2 generation and Absolut scoring")
        self.require(c.get("research_rounds") == self.expected_rounds and c.get("batch_size") == self.batch and
                     c.get("pool_size") == self.pool and c.get("iterations") == self.minimum_evaluations and
                     c.get("research_candidates_per_role") * 2 == self.pool,
                     "round, batch, pool or evaluation budgets differ from the requested values")
        summary = read_json(self.root / "summary.json")
        self.require(not summary.get("error") and not summary.get("invariant_problems") and
                     summary.get("successful_evaluation_count") == self.minimum_evaluations,
                     "campaign summary reports an error or incomplete evaluations")
        launch = read_json(self.root.parent / "launch.json")
        source = self.root.parent / "absolut"
        self.require(launch.get("status") == "completed" and launch.get("case", {}).get("status") == "completed" and
                     Path(c["absolut_path"]).resolve() == source.resolve(), "launch did not complete with this input library")
        manifest = read_json(source / "assets.json")
        self.require(manifest.get("schema_version") == 1 and manifest.get("antigen") == "1N8Z_C" and
                     manifest.get("assets") == launch.get("assets"), "launch and input asset manifests differ")
        self.asset_names = []
        for item in manifest.get("assets", []):
            name = item["name"]
            relative = Path(name)
            path = source / relative
            self.require(not relative.is_absolute() and ".." not in relative.parts and not path.is_symlink() and
                         path.is_file() and path.stat().st_size == item["bytes"] and not path.stat().st_mode & 0o222 and
                         file_sha256(path) == item["sha256"], f"invalid immutable input asset: {name}")
            self.asset_names.append(name)
        self.require("src/bin/Absolut" in self.asset_names and len(self.asset_names) == len(set(self.asset_names)),
                     "missing executable or duplicate input assets")
        self.report["configuration"] = {key: c.get(key) for key in
            ("method", "antigen", "oracle", "iterations", "research_rounds", "batch_size", "pool_size", "concurrency")}
        self.report["stop_reason"] = summary.get("stop_reason")

    def evaluations(self):
        super().evaluations()
        attempts = self.es.rows("SELECT * FROM attempts ORDER BY started_at")
        self.require(len(self.experiments) == self.minimum_evaluations, "unexpected number of experiments")
        for attempt in attempts:
            root = self.root / "absolut-attempts" / attempt["executor_ref"]
            workspace = root / "absolut"
            for name in self.asset_names:
                link = workspace / name
                self.require(link.is_symlink() and link.resolve() == (self.root.parent / "absolut" / name).resolve(),
                             f"{attempt['experiment_id']}: input link differs: {name}")
            self.require(not (workspace / f".antbo_llm_acq_{self.config['antigen']}.lock").is_symlink(),
                         f"{attempt['experiment_id']}: worker lock remains")
            self.require((root / "worker.log").is_file(), f"{attempt['experiment_id']}: worker log is missing")

    def designs(self):
        self.catalog, self.catalog_jobs = {}, {}
        used = defaultdict(lambda: {"backbones": 0, "sequences": 0})
        receipts = sorted((self.root / "design_jobs").glob("job-*/receipt.json"))
        self.require(bool(receipts), "no real design receipts")
        for path in receipts:
            receipt = read_json(path)
            generation, role = receipt["round"] + 1, receipt.get("profile_id")
            self.require(1 <= generation <= self.expected_rounds and role in ("explorer", "refiner") and
                         receipt.get("status") == "succeeded" and receipt.get("mock_generator") is False,
                         f"{path.parent.name}: invalid real-generation receipt")
            used[(generation, role)]["backbones"] += int(receipt["action"] == "design_h3")
            used[(generation, role)]["sequences"] += receipt["count"]
            design = read_json(path.parent / "output/designs.json")
            mpnn = read_json(path.parent / "output/proteinmpnn-0000.execution.json")
            self.require(design.get("status") == "success" and len(design["candidates"]) == receipt["count"] and
                         len(receipt["candidates"]) == receipt["count"] and mpnn.get("returncode") == 0 and
                         mpnn.get("elapsed_seconds", 0) > 0, f"{path.parent.name}: generation artifact incomplete")
            if receipt["action"] == "design_h3":
                rfd = read_json(path.parent / "output/rfd3.execution.json")
                self.require(rfd.get("returncode") == 0 and rfd.get("elapsed_seconds", 0) > 0,
                             f"{path.parent.name}: missing successful RFD3 execution")
            for candidate in receipt["candidates"]:
                design_id = candidate["design_id"]
                self.require(design_id not in self.catalog, f"duplicate design id: {design_id}")
                self.catalog[design_id], self.catalog_jobs[design_id] = candidate, receipt
        for generation in range(1, self.expected_rounds + 1):
            self.require(sum(used[(generation, role)]["sequences"] for role in ("explorer", "refiner")) <=
                         self.config["sequences_per_round"] and
                         sum(used[(generation, role)]["backbones"] for role in ("explorer", "refiner")) <=
                         self.config["backbones_per_round"], f"generation {generation}: generation budget exceeded")
            for role in ("explorer", "refiner"):
                self.require(used[(generation, role)]["sequences"] >= self.pool // 2,
                             f"generation {generation} {role}: insufficient real generated sequences")
        self.report["designs"] = {"jobs": len(receipts), "by_generation_and_role":
                                  {f"{g}:{role}": value for (g, role), value in sorted(used.items())}}

    def research(self):
        from ldm_tts.async_runtime.wire import digest

        visibility = read_jsonl(self.root / "history/visibility.jsonl")
        history = read_jsonl(self.root / "history/log.jsonl")
        self.require([event["seq"] for event in history] == list(range(1, len(history) + 1)),
                     "history event sequence is not contiguous")
        by_turn = defaultdict(list)
        for event in visibility:
            by_turn[(event["profile_id"], event["turn_id"])].append(event)
        reads = {}
        launch = read_json(self.root.parent / "launch.json")
        for role in ROLES:
            root = self.root / "harness" / role
            manifest = read_json(root / "manifest.json")
            profiles = manifest["profiles"]
            self.require(manifest.get("model") == launch.get("model") and len(profiles) == 1 and
                         profiles[0].get("profileId") == role, f"{role}: profile or model differs")
            session = (root / profiles[0]["session"]).resolve()
            self.require(session.is_relative_to(root.resolve()), f"{role}: invalid session path")
            reads[role] = successful_reads(session)
            expected = {"cdrh3-design", "cdrh3-evidence-review" if role == "evidence_reviewer"
                        else "cdrh3-experiment-design"}
            self.require(read_skills(reads[role], expected) == expected, f"{role}: task skill reads missing")
        stale = [e for e in visibility if e["kind"] == "cas_stale"]
        self.require(all(e["visible"] < e["head"] for e in stale), "invalid stale CAS receipt")
        self.report["coverage"]["cas_stale_in_campaign"] = "exercised" if stale else "not_exercised"
        pools = {}
        for row in self.generations:
            generation = row["generation"]
            file = self.root / "controller.sqlite3.research" / row["input_path"]
            inputs = read_json(file)
            saved = read_json(file.with_name(f"g{generation:06d}.outcome.json"))
            outcome = read_json_value(row["outcome_json"])
            self.require(digest(inputs) == row["input_digest"] == saved.get("input_digest") and
                         row["base_ingest_seq"] == (generation - 1) * self.batch and
                         inputs["watermarks"]["base_ingest_seq"] == row["base_ingest_seq"] and
                         not inputs["pending_keys"] and row["proposal_count"] == self.pool and row["admitted"] == self.pool,
                         f"generation {generation}: research input, barrier, or pool differs")
            metadata, proposals = outcome["metadata"], outcome["proposals"]
            self.require(metadata.get("bootstrap") is False and set(metadata.get("sessions", {})) == set(ROLES) and
                         metadata.get("candidate_counts") == {"explorer": self.pool // 2, "refiner": self.pool // 2},
                         f"generation {generation}: panel sessions or role counts differ")
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
                self.require(receipt["round"] + 1 == generation and not issued["violations"] and
                             issued["sequence"] == candidate["payload"]["sequence"] == key and key not in keys,
                             f"generation {generation}: invalid or repeated proposed design {design_id}")
                keys.add(key)
            self.require(len(keys) == self.pool and role_counts == {"explorer": self.pool // 2, "refiner": self.pool // 2},
                         f"generation {generation}: distinct proposal pool differs")
            pools[generation] = keys
            submissions = {}
            for role in ROLES:
                session = metadata["sessions"][role]
                events = by_turn[(role, session["turn_id"])]
                validations = [event for event in events if event["kind"] == "validate"]
                accepts = [event for event in events if event["kind"] == "decision" and event.get("decision") == "accept"]
                queries = [event for event in events if event["kind"] == "query" and not event.get("error")]
                commit = read_json(self.root / "harness" / role / "turns" / session["turn_id"] / "turn_committed.json")
                self.require(session.get("submission_status") == "accepted" and
                             session.get("usage", {}).get("providerCalls", 0) > 0 and validations and accepts and queries and
                             commit.get("submissionStatus") == "accepted" and
                             accepts[-1].get("submission_digest") == commit.get("submissionDigest"),
                             f"generation {generation} {role}: missing accepted real submission")
                if validations:
                    check = validations[-1]
                    self.require(check.get("history_seq") == check.get("head") == session.get("history_seq") and
                                 (check.get("visible") == check.get("head") or check.get("forced")),
                                 f"generation {generation} {role}: invalid CAS validation receipt")
                submissions[role] = commit["submission"]
            for role in ("explorer", "refiner"):
                ids = submissions[role].get("design_ids", [])
                self.require(len(ids) == self.pool // 2 and len(set(ids)) == len(ids) and
                             set(ids) == role_ids[role],
                             f"generation {generation} {role}: submitted designs differ from the pool")
            review = submissions["evidence_reviewer"]
            self.require(review == metadata.get("review") and bool(review.get("evidence", "").strip()) and
                         bool(review.get("hypotheses", "").strip()) and
                         set(review.get("design_ids", [])) <= role_ids["explorer"] | role_ids["refiner"],
                         f"generation {generation}: reviewer used a design outside the panel")
            peer_catalogs = metadata["peer_catalogs"]
            self.require(set(peer_catalogs) == {"refiner", "evidence_reviewer"},
                         f"generation {generation}: missing peer catalog")
            for role, info in peer_catalogs.items():
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
        self.pools = pools
        self.report["research"] = {"generations": len(pools), "cas_stale_events": len(stale)}

    def plans(self):
        plans = self.db.rows("SELECT * FROM plans ORDER BY plan_id")
        self.require(len(plans) == self.expected_rounds and
                     [p["generation"] for p in plans] == list(range(1, self.expected_rounds + 1)),
                     "expected exactly one frozen plan per generation")
        by_plan = defaultdict(list)
        for row in self.decisions:
            snapshot = read_json_value(row["snapshot_json"])
            plan = snapshot.get("plan") or {}
            by_plan[plan.get("id")].append((plan.get("step"), snapshot, row))
        experiment_by_decision = {r["decision_id"]: r for r in self.experiments}
        observation_by_id = {r["experiment_id"]: r for r in self.observation_rows}
        for plan in plans:
            generation, pid = plan["generation"], plan["plan_id"]
            chosen, probabilities = read_json_value(plan["keys_json"]), read_json_value(plan["probabilities_json"])
            inputs, record = read_json_value(plan["inputs_json"]), read_json_value(plan["record_json"])
            links = sorted(by_plan[pid], key=lambda row: row[0])
            self.require(plan["count"] == plan["consumed"] == self.batch and plan["closed_reason"] == "complete" and
                         len(chosen) == len(probabilities) == len(links) == self.batch and
                         [r[0] for r in links] == list(range(self.batch)), f"plan {pid}: incomplete frozen batch")
            all_candidates = {item["canonical_key"] for item in inputs["candidates"]}
            self.require(all_candidates == self.pools.get(generation) and len(all_candidates) == self.pool and
                         inputs["head_ingest_seq"] == (generation - 1) * self.batch and not inputs["pending"] and
                         len(record["selection"]["steps"]) == self.batch, f"plan {pid}: pool or frozen head differs")
            for step, snapshot, decision in links:
                if not isinstance(step, int) or step < 0 or step >= len(chosen) or step >= len(probabilities):
                    self.require(False, f"plan {pid}: invalid step")
                    continue
                experiment = experiment_by_decision[decision["decision_id"]]
                observation = observation_by_id[experiment["experiment_id"]]
                self.require(chosen[step]["canonical_key"] == experiment["canonical_key"] and
                             snapshot["probability"] == probabilities[step] and
                             (generation - 1) * self.batch < observation["ingest_seq"] <= generation * self.batch,
                             f"plan {pid} step {step}: candidate, probability or observation differs")
        self.require(set(by_plan) == {p["plan_id"] for p in plans}, "a decision points to an unknown plan")
        self.report["plans"] = {"count": len(plans), "selected": sum(p["consumed"] for p in plans)}

    def policy(self):
        from ldm_tts.async_runtime.policy import PriorApplication, check_prior_invariants
        from ldm_tts.async_runtime.selectors import decision_choice

        root = self.root / "policy_harness"
        manifest = read_json(root / "manifest.json")
        profiles = manifest["profiles"]
        launch = read_json(self.root.parent / "launch.json")
        self.require(manifest.get("model") == launch.get("model") and len(profiles) == 1 and
                     profiles[0].get("profileId") == "policy_architect", "policy profile or model differs")
        session = (root / profiles[0]["session"]).resolve()
        self.require(session.is_relative_to(root.resolve()) and
                     read_skills(successful_reads(session), {"compile-ldm-policy"}) == {"compile-ldm-policy"},
                     "policy architect did not read its policy skill")
        applications = self.db.rows("SELECT * FROM prior_applications ORDER BY seq")
        by_id = {row["application_id"]: row for row in applications}
        self.require(bool(applications) and not any(row["degraded"] for row in applications),
                     "missing or degraded frozen policy application")
        self.problems.extend(check_prior_invariants(self.db, "energy_utility"))
        contexts = []
        for application in applications:
            self.section(f"application {application['application_id']} context",
                         lambda app=application: contexts.append(self.check_application_context(app, decision_choice)))

        epochs, rounds = {}, []
        for generation in self.generations:
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
                self.require(file_sha256(path / name) == expected,
                             f"generation {number}: policy input digest differs: {name}")
            turn = result["harness_turn"]
            commit = read_json(root / "turns" / turn["turn_id"] / "turn_committed.json")
            self.require(turn.get("usage", {}).get("providerCalls", 0) > 0 and
                         commit.get("profileId") == "policy_architect" and
                         commit.get("submissionStatus") == "accepted" and
                         commit.get("submissionDigest") == result.get("submission_sha256") and
                         commit.get("submission", {}).get("action") == result.get("action"),
                         f"generation {number}: accepted policy turn is not backed by a provider receipt")
            epoch = result.get("epoch_id")
            if epoch:
                epoch_path = root / "epochs" / epoch
                epoch_manifest = read_json(epoch_path / "manifest.json")
                artifact = (root / epoch_manifest["artifact_path"]).resolve()
                self.require(artifact.is_relative_to(root.resolve()) and epoch_manifest["epoch_id"] == epoch and
                             epoch_manifest["artifact_sha256"] == result["artifact_sha256"] == file_sha256(artifact),
                             f"generation {number}: policy epoch artifact differs")
                for name, expected in epoch_manifest["files"].items():
                    self.require(file_sha256(epoch_path / name) == expected,
                                 f"generation {number}: policy epoch digest differs: {name}")
                epochs[epoch] = epoch_manifest
            rounds.append({"generation": number, "action": result.get("action"), "epoch": epoch})

        plans = {row["plan_id"]: row for row in self.db.rows("SELECT * FROM plans")}
        experiment_by_decision = {row["decision_id"]: row for row in self.experiments}
        used_generations, used_epochs, fitted = set(), set(), set()
        replay_inputs = []
        for decision in self.decisions:
            snapshot = read_json_value(decision["snapshot_json"])
            plan = plans[snapshot["plan"]["id"]]
            record, inputs, step = decision_choice(self.db, decision["decision_id"])
            application_id = inputs.get("prior_application_id")
            application = by_id.get(application_id)
            self.require(application is not None, f"decision {decision['decision_id']}: missing policy application")
            if application is None:
                continue
            used_generations.add(application["generation"])
            self.require(application["generation"] == plan["generation"] and
                         application["created_at"] <= plan["created_at"] and
                         (application["superseded_at"] is None or plan["created_at"] <= application["superseded_at"]),
                         f"decision {decision['decision_id']}: policy was not active when the plan was frozen")
            selection, chosen = record["selection"], record["selection"]["steps"][step]
            index = chosen["candidate_ids"].index(chosen["selected_candidate_id"])
            parameters = selection["parameters"]
            self.require(chosen["canonical_keys"][index] == experiment_by_decision[decision["decision_id"]]["canonical_key"] and
                         snapshot["probability"] == chosen["selected_probability"] and
                         math.isclose(float(parameters["alpha"]), application["alpha"], abs_tol=1e-12) and
                         math.isclose(float(parameters["eta"]), application["eta"], abs_tol=1e-12) and
                         record["selector"].get("objective") == "energy_utility" and
                         record["selector"].get("feature_version") == "cdrh3-one-hot-v1",
                         f"decision {decision['decision_id']}: frozen choice or policy weights differ")
            if not application["prior_free"]:
                epoch = application["epoch_id"]
                self.require(epoch in epochs and epochs[epoch]["artifact_sha256"] == application["artifact_sha256"] and
                             {item["canonical_key"] for item in inputs["candidates"]} <=
                             set(read_json_value(application["values_json"])),
                             f"decision {decision['decision_id']}: prior artifact or coverage differs")
                used_epochs.add(epoch)
                if chosen.get("metadata", {}).get("gp_ready") and selection.get("metadata", {}).get("prior_applied"):
                    fitted.add(plan["plan_id"])
            replay_inputs.append((decision["decision_id"], application_id))
        self.require(used_generations == set(range(1, self.expected_rounds + 1)),
                     "not every generation used its frozen policy application")
        self.report["coverage"]["fitted_gp_with_custom_prior"] = "exercised" if fitted else "not_exercised"
        self.report["policy"] = {"rounds": rounds, "applications": len(applications),
                                 "used_epochs": sorted(used_epochs), "fitted_prior_plans": sorted(fitted),
                                 "application_contexts": contexts}
        self.replay_decisions(replay_inputs, by_id, PriorApplication)

    def replay_decisions(self, replay_inputs, by_id, application_type):
        # audit_decision replays the entire selection, not just its requested step.
        # Every step and its decision/experiment linkage is checked in plans().
        unique, seen = [], set()
        for decision_id, application_id in replay_inputs:
            row = self.db.row("SELECT snapshot_json FROM decisions WHERE decision_id=?", (decision_id,))
            plan_id = read_json_value(row["snapshot_json"])["plan"]["id"]
            if plan_id not in seen:
                unique.append((decision_id, application_id))
                seen.add(plan_id)
        super().replay_decisions(unique, by_id, application_type)

    def run(self):
        self.section("configuration", self.configuration)
        if not hasattr(self, "asset_names"):
            return self.finish()
        self.section("controller and service", self.evaluations)
        if not hasattr(self, "db"):
            return self.finish()
        try:
            self.section("design receipts", self.designs)
            self.section("research and CAS", self.research)
            if hasattr(self, "pools"):
                self.section("frozen plans", self.plans)
            self.section("policy and selector replay", self.policy)
        finally:
            self.db.close()
            if hasattr(self, "es"):
                self.es.close()
        return self.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--pool", type=int, required=True)
    args = parser.parse_args()
    if min(args.rounds, args.batch, args.pool) < 1 or args.batch > args.pool or args.pool % 2:
        parser.error("rounds, batch and pool must be positive, batch <= pool, and pool even")
    report = BatchAudit(args.campaign_dir, args.rounds, args.batch, args.pool).run()
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
