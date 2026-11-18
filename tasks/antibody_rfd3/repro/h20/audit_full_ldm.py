"""Read-only acceptance audit for a real HER2 asynchronous LDM campaign.

Run on the host that ran the campaign, after it has stopped. This checks persisted
evidence and replays numerical selector decisions using frozen prior tables; it
never executes a submitted policy, calls a provider, or starts a scientific job.
CAS races, pending-aware conditioning, and crash recovery need separate controlled
tests when the corresponding events did not occur in the campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading

import numpy as np


ROLES = ("explorer", "refiner", "evidence_reviewer")
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
UTILITY = "energy_utility"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def successful_reads(path: Path) -> set[str]:
    """Find successful agent tool reads, not just resources listed in a manifest."""
    calls, found = {}, set()
    for entry in read_jsonl(path):
        message = entry.get("message") or {}
        if message.get("role") == "assistant":
            for content in message.get("content") or []:
                if content.get("type") == "toolCall" and content.get("name") == "read":
                    calls[content["id"]] = str(content.get("arguments", {}).get("path", ""))
        elif message.get("role") == "toolResult" and not message.get("isError", False):
            target = calls.get(message.get("toolCallId"))
            if target:
                found.add(target)
    return found


def read_skills(reads: set[str], names: set[str]) -> set[str]:
    return {name for name in names if any(target.endswith(f"/{name}/SKILL.md") for target in reads)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def onehot(sequence: str) -> np.ndarray:
    if len(sequence) != 11 or any(letter not in ALPHABET for letter in sequence):
        raise ValueError(f"invalid 11-residue sequence {sequence!r}")
    return np.asarray([float(letter == candidate) for letter in sequence for candidate in ALPHABET])


class ReadOnlyStore:
    """The minimal Store interface used by the framework's read-only audits."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()

    def rows(self, sql, args=()):
        return self.db.execute(sql, args).fetchall()

    def row(self, sql, args=()):
        return self.db.execute(sql, args).fetchone()

    def value(self, sql, args=()):
        row = self.row(sql, args)
        return None if row is None else row[0]

    def close(self):
        self.db.close()


class Audit:
    def __init__(self, directory: Path, rounds: int, evaluations: int):
        self.root = directory.resolve()
        self.expected_rounds = rounds
        self.minimum_evaluations = evaluations
        self.problems = []
        self.report = {"run_dir": str(self.root), "expected_rounds": rounds,
                       "minimum_evaluations": evaluations,
                       "coverage": {"crash_resume_in_campaign": "not_exercised"}}

    def require(self, condition, message):
        if not condition:
            self.problems.append(message)

    def section(self, name, fn):
        try:
            fn()
        except Exception as exc:  # A missing or malformed receipt is an audit failure, not a success.
            self.problems.append(f"{name}: {type(exc).__name__}: {exc}")

    def run(self):
        self.section("configuration", self.configuration)
        if not hasattr(self, "config"):
            return self.finish()
        self.section("controller and service", self.evaluations)
        if not hasattr(self, "db"):
            return self.finish()
        try:
            self.section("design receipts", self.designs)
            self.section("research and history", self.research)
            self.section("policy and selection", self.policy)
        finally:
            self.db.close()
            if hasattr(self, "es"):
                self.es.close()
        return self.finish()

    def finish(self):
        self.report["problems"] = self.problems
        self.report["passed"] = not self.problems
        return self.report

    def configuration(self):
        self.config = read_json(self.root / "config.json")
        c = self.config
        self.require(c.get("method") == "ldm_harness", "method must be ldm_harness")
        self.require(c.get("oracle") == "absolut" and c.get("antigen") == "1N8Z_C",
                     "expected the real HER2 (1N8Z_C) Absolut oracle")
        self.require(c.get("mock") is False and c.get("dry_run") is False,
                     "generation must be real and the campaign must not be a dry run")
        self.require(c.get("research_rounds") == self.expected_rounds,
                     "configured research_rounds differs from the expected count")
        self.require(isinstance(c.get("iterations"), int) and c["iterations"] >= self.minimum_evaluations,
                     "configured evaluation budget is below the requested minimum")
        summary = read_json(self.root / "summary.json")
        self.require(not summary.get("error") and not summary.get("invariant_problems"),
                     "campaign summary contains an error or an invariant violation")
        launch = read_json(self.root.parent / "launch.json")
        self.require(launch.get("status") == "completed" and launch.get("case", {}).get("status") == "completed" and
                     launch.get("model") == "deepseek-flash" and launch.get("thinking") == "low" and
                     Path(c["absolut_path"]).resolve() == (self.root.parent / "absolut").resolve(),
                     "launch metadata or isolated Absolut directory differs")
        assets = launch["assets"]
        self.require(bool(assets) and any(item["name"] == "src/bin/Absolut" for item in assets),
                     "launch manifest has no verified Absolut assets")
        for item in assets:
            path = (self.root.parent / "absolut" / item["name"]).resolve()
            self.require(path.is_relative_to((self.root.parent / "absolut").resolve()) and
                         file_sha256(path) == item["sha256"], f"Absolut asset digest differs: {item['name']}")
        self.report["configuration"] = {key: c.get(key) for key in
            ("method", "antigen", "oracle", "mock", "iterations", "research_rounds", "concurrency", "pool_size")}
        self.report["stop_reason"] = summary.get("stop_reason")

    def evaluations(self):
        self.db = ReadOnlyStore(self.root / "controller.sqlite3")
        self.es = ReadOnlyStore(self.root / "experiment-service.sqlite3")
        self.generations = self.db.rows("SELECT * FROM generations ORDER BY generation")
        self.experiments = self.db.rows("SELECT * FROM experiments ORDER BY decision_id")
        self.observation_rows = self.db.rows("SELECT * FROM observations ORDER BY ingest_seq")
        self.decisions = self.db.rows("SELECT * FROM decisions ORDER BY decision_id")
        remote = {r["experiment_id"]: r for r in self.es.rows("SELECT * FROM experiments ORDER BY seq")}
        attempts = self.es.rows("SELECT * FROM attempts ORDER BY started_at")
        self.require(len(self.generations) == self.expected_rounds and
                     [r["generation"] for r in self.generations] == list(range(1, self.expected_rounds + 1)) and
                     all(r["state"] == "ready" for r in self.generations),
                     "research generations are missing, noncontiguous, or not ready")
        self.require(len(self.experiments) >= self.minimum_evaluations and
                     len(self.experiments) == len(self.observation_rows) == len(self.decisions) == len(attempts),
                     "each evaluation must have exactly one decision, observation and service attempt")
        self.require(len(self.experiments) == self.config["iterations"], "the evaluation budget was not consumed")
        self.require({r["experiment_id"] for r in self.experiments} == set(remote),
                     "controller and service experiment IDs differ")
        self.require(len({r["canonical_key"] for r in self.experiments}) == len(self.experiments),
                     "a sequence was evaluated more than once")
        self.require(not self.db.value("SELECT COUNT(*) FROM anomalies WHERE resolved_at IS NULL"),
                     "controller has unresolved anomalies")
        self.observations = {r["experiment_id"]: read_json_value(r["observation_json"])
                             for r in self.observation_rows}
        self.by_candidate = {o["candidate"]["candidate_id"]: o for o in self.observations.values()}
        self.require([r["ingest_seq"] for r in self.observation_rows] == list(range(1, len(self.observation_rows) + 1)),
                     "observation ingest sequence is not contiguous")
        attempt_by_id = defaultdict(list)
        for attempt in attempts:
            attempt_by_id[attempt["experiment_id"]].append(attempt)
        for row in self.experiments:
            eid = row["experiment_id"]
            service, observation = remote.get(eid), self.observations.get(eid)
            self.require(row["submit_state"] == "ACKED" and row["es_state"] == "SUCCEEDED" and
                         row["budget_state"] == "CONSUMED" and row["pending"] == 0 and
                         row["disposition"] == "ingested", f"{eid}: unsettled controller record")
            if service is None or observation is None:
                continue
            candidate = observation["candidate"]
            sequence = candidate["payload"]["sequence"]
            onehot(sequence)
            metrics = observation["evaluation"]["metrics"]
            energy, utility = metrics.get("absolut_energy"), metrics.get(UTILITY)
            self.require(finite(energy) and finite(utility) and math.isclose(utility, -energy, abs_tol=1e-12),
                         f"{eid}: utility is not negative raw Absolut energy")
            self.require(sequence == row["canonical_key"] == candidate["canonical_key"],
                         f"{eid}: candidate and canonical sequence differ")
            spec, result = read_json_value(service["spec"]), read_json_value(service["result"])
            self.require(service["state"] == "SUCCEEDED" and spec ==
                         {"antigen": "1N8Z_C", "oracle": "absolut", "sequence": sequence} and
                         result.get("absolut_energy") == energy and
                         service["result_digest"] == row["es_result_digest"],
                         f"{eid}: experiment service result or request differs")
            linked = attempt_by_id[eid]
            self.require(len(linked) == 1, f"{eid}: expected one real scoring attempt")
            if len(linked) != 1:
                continue
            attempt = linked[0]
            reference = hashlib.sha256(json.dumps([eid, attempt["idx"]], separators=(",", ":")).encode()).hexdigest()
            self.require(attempt["state"] == "ended" and attempt["executor_ref"] == reference and
                         read_json_value(attempt["outcome"]).get("state") == "succeeded" and
                         finite(attempt["started_at"]) and finite(attempt["ended_at"]) and
                         attempt["ended_at"] > attempt["started_at"], f"{eid}: invalid attempt receipt")
            base = self.root / "absolut-attempts" / reference
            request, result_file, worker = (read_json(base / name) for name in
                                             ("request.json", "result.json", "worker.json"))
            self.require(request.get("sequence") == sequence and request.get("antigen") == "1N8Z_C" and
                         request.get("oracle") == "absolut" and
                         Path(request.get("absolut_path", "")).resolve() == Path(self.config["absolut_path"]).resolve() and
                         result_file.get("state") == "succeeded" and
                         result_file.get("metrics", {}).get("absolut_energy") == energy and
                         all(worker.get(k) is not None for k in ("pid", "start", "boot_id")),
                         f"{eid}: real Absolut process receipts do not match the observation")
        edges = [(r["started_at"], 1) for r in attempts] + [(r["ended_at"], -1) for r in attempts]
        active = peak = 0
        for _, delta in sorted(edges):
            active += delta
            peak = max(peak, active)
        self.require(peak <= self.config["concurrency"], "experiment service exceeded configured concurrency")
        self.report["evaluations"] = {"generations": len(self.generations), "decisions": len(self.decisions),
            "observations": len(self.observation_rows), "absolut_attempts": len(attempts), "peak_concurrency": peak,
            "energies": [self.observations[r["experiment_id"]]["evaluation"]["metrics"]["absolut_energy"]
                         for r in self.observation_rows]}

    def designs(self):
        receipts = [(p, read_json(p)) for p in sorted((self.root / "design_jobs").glob("job-*/receipt.json"))]
        self.require(bool(receipts), "no real generation receipts")
        budget = defaultdict(lambda: {"backbones": 0, "sequences": 0})
        self.catalog, self.catalog_jobs = {}, {}
        for path, receipt in receipts:
            generation = receipt["round"] + 1
            role = receipt.get("profile_id")
            self.require(1 <= generation <= self.expected_rounds and role in ROLES and
                         receipt.get("status") == "succeeded" and receipt.get("mock_generator") is False,
                         f"{path.parent.name}: invalid role, generation or real generation receipt")
            budget[(generation, role)]["backbones"] += int(receipt["action"] == "design_h3")
            budget[(generation, role)]["sequences"] += receipt["count"]
            self.require((path.parent / "output/designs.json").is_file(),
                         f"{path.parent.name}: missing model design artifact")
            design = read_json(path.parent / "output/designs.json")
            self.require(design.get("status") == "success" and len(design["candidates"]) == receipt["count"],
                         f"{path.parent.name}: model design artifact is incomplete")
            mpnn = read_json(path.parent / "output/proteinmpnn-0000.execution.json")
            self.require(mpnn.get("returncode") == 0 and mpnn.get("elapsed_seconds", 0) > 0,
                         f"{path.parent.name}: no successful ProteinMPNN execution")
            if receipt["action"] == "design_h3":
                rfd = read_json(path.parent / "output/rfd3.execution.json")
                self.require(rfd.get("returncode") == 0 and rfd.get("elapsed_seconds", 0) > 0,
                             f"{path.parent.name}: no successful RFD3 execution")
            self.require(len(receipt["candidates"]) == receipt["count"],
                         f"{path.parent.name}: generated candidate count differs")
            for item in receipt["candidates"]:
                self.require(item["design_id"] not in self.catalog, "duplicate issued design ID")
                self.catalog[item["design_id"]] = item
                self.catalog_jobs[item["design_id"]] = receipt
        for generation in range(1, self.expected_rounds + 1):
            total_b = total_s = 0
            for role in ROLES:
                used = budget[(generation, role)]
                if generation == 1:
                    cap = {"backbones": 1, "sequences": self.config["samples_per_call"]} if role == "explorer" else \
                          {"backbones": 0, "sequences": 0}
                elif role == "evidence_reviewer":
                    cap = {"backbones": 0, "sequences": 0}
                else:
                    extra = int(role == "explorer")
                    cap = {"backbones": (self.config["backbones_per_round"] + extra) // 2,
                           "sequences": (self.config["sequences_per_round"] + extra) // 2}
                self.require(all(0 <= used[k] <= cap[k] for k in cap),
                             f"generation {generation} role {role} exceeded its generation budget")
                total_b += used["backbones"]
                total_s += used["sequences"]
            self.require(total_b <= self.config["backbones_per_round"] and
                         0 < total_s <= self.config["sequences_per_round"],
                         f"generation {generation}: invalid shared generation budget")
        self.report["designs"] = {"jobs": len(receipts),
            "reused_initializations": sum(bool(r.get("initialization_source")) for _, r in receipts),
            "by_generation_and_role": {f"{g}:{role}": value for (g, role), value in sorted(budget.items())}}

    def research(self):
        from ldm_tts.async_runtime.wire import digest

        visibility = read_jsonl(self.root / "history/visibility.jsonl")
        history = read_jsonl(self.root / "history/log.jsonl")
        self.require([e["seq"] for e in history] == list(range(1, len(history) + 1)),
                     "history log sequence is not contiguous")
        by_turn = defaultdict(list)
        for event in visibility:
            by_turn[(event["profile_id"], event["turn_id"])].append(event)
        sessions_by_role = Counter()
        queries_by_role = Counter()
        skill_reads, role_reads = {}, {}
        for role in ROLES:
            role_root = self.root / "harness" / role
            required = {"cdrh3-design", "cdrh3-evidence-review" if role == "evidence_reviewer"
                        else "cdrh3-experiment-design"}
            manifest = read_json(role_root / "manifest.json")
            profiles = manifest["profiles"]
            self.require(manifest.get("model") == "deepseek-flash" and manifest.get("thinking") == "low" and
                         len(profiles) == 1 and profiles[0].get("profileId") == role and
                         len(profiles[0].get("skills", [])) == len(required),
                         f"{role}: profile, skills, model or reasoning configuration differs")
            session_path = (role_root / profiles[0]["session"]).resolve()
            self.require(session_path.is_relative_to(role_root.resolve()), f"{role}: invalid session path")
            role_reads[role] = successful_reads(session_path)
            skill_reads[role] = read_skills(role_reads[role], required)
            self.require(skill_reads[role] == required, f"{role}: no successful read of every registered task skill")
        generation_decisions = Counter()
        for row in self.decisions:
            snapshot = read_json_value(row["snapshot_json"])
            generation_decisions[snapshot["pacing_state"]["current_generation"]] += 1
        for row in self.generations:
            generation = row["generation"]
            round_file = self.root / "controller.sqlite3.research" / row["input_path"]
            round_input = read_json(round_file)
            saved_outcome = read_json(round_file.with_name(f"g{generation:06d}.outcome.json"))
            self.require(digest(round_input) == row["input_digest"] == saved_outcome.get("input_digest"),
                         f"generation {generation}: frozen research input or outcome digest differs")
            outcome = read_json_value(row["outcome_json"])
            metadata = outcome["metadata"]
            sessions = metadata["sessions"]
            self.require(set(sessions) == set(ROLES), f"generation {generation}: missing research role session")
            self.require(metadata.get("bootstrap") == (generation == 1),
                         f"generation {generation}: incorrect bootstrap marker")
            self.require(generation_decisions[generation] >= 1,
                         f"generation {generation}: no evaluation decision after its policy application")
            if self.config["iterations"] == self.expected_rounds:
                self.require(row["base_ingest_seq"] >= generation - 1,
                             f"generation {generation}: previous generation's result was not in its baseline")
            submissions = {}
            for role in ROLES:
                if role not in sessions:
                    continue
                session = sessions[role]
                events = by_turn[(role, session["turn_id"])]
                decisions = [e for e in events if e["kind"] == "decision" and e.get("decision") == "accept"]
                validations = [e for e in events if e["kind"] == "validate"]
                queries = [e for e in events if e["kind"] == "query" and not e.get("error")]
                commit = read_json(self.root / "harness" / role / "turns" / session["turn_id"] /
                                   "turn_committed.json")
                submissions[role] = commit["submission"]
                self.require(session.get("submission_status") == "accepted" and
                             session.get("usage", {}).get("providerCalls", 0) > 0 and bool(decisions) and
                             bool(validations) and bool(queries) and commit.get("profileId") == role and
                             commit.get("turnId") == session["turn_id"] and
                             commit.get("submissionStatus") == "accepted" and decisions[-1].get("submission_digest") ==
                             commit.get("submissionDigest"),
                             f"generation {generation} role {role}: no accepted real session")
                if validations:
                    validation = validations[-1]
                    self.require(validation.get("history_seq") == validation.get("head") and
                                 (validation.get("visible") == validation.get("head") or validation.get("forced")),
                                 f"generation {generation} role {role}: invalid CAS validation watermark")
                    self.require(session.get("history_seq") == validation.get("history_seq"),
                                 f"generation {generation} role {role}: accepted history watermark differs")
                sessions_by_role[role] += 1
                queries_by_role[role] += len(queries)
            if set(submissions) == set(ROLES):
                self.peer_catalogs(generation, metadata, submissions, by_turn, role_reads)
            for proposal in outcome["proposals"]:
                candidate = proposal["candidate"]
                design_id = candidate["payload"]["design_id"]
                issued = self.catalog.get(design_id)
                self.require(issued is not None and not issued["violations"] and
                             issued["sequence"] == candidate["payload"]["sequence"],
                             f"generation {generation}: unissued or invalid proposal {design_id}")
        self.require(all(queries_by_role[role] > 0 for role in ROLES),
                     "one or more research roles never queried experiment_history")
        stale_events = [e for e in visibility if e["kind"] == "cas_stale"]
        self.require(all(e["visible"] < e["head"] for e in stale_events), "invalid stale CAS event")
        stale = len(stale_events)
        self.report["research"] = {"sessions_by_role": dict(sessions_by_role),
            "history_queries_by_role": dict(queries_by_role), "accepted_decisions":
            sum(e["kind"] == "decision" and e.get("decision") == "accept" for e in visibility),
            "cas_stale": stale, "history_entries": len(history), "decisions_by_generation": dict(generation_decisions),
            "skills_read": {role: sorted(names) for role, names in skill_reads.items()}}
        self.report["coverage"]["cas_stale_in_campaign"] = "exercised" if stale else "not_exercised"

    def peer_catalogs(self, generation, metadata, submissions, by_turn, role_reads):
        peer = metadata["peer_catalogs"]
        self.require(set(peer) == {"refiner", "evidence_reviewer"},
                     f"generation {generation}: missing peer catalog")
        for role in ("refiner", "evidence_reviewer"):
            info = peer[role]
            record = (self.root / info["record"]).resolve()
            workspace = self.root / "harness" / role / "sessions" / role / "workspace"
            guest = workspace / Path(info["path"]).relative_to("/workspace")
            self.require(record.is_relative_to(self.root.resolve()) and file_sha256(record) == info["sha256"] ==
                         file_sha256(guest), f"generation {generation} role {role}: catalog digest differs")
            document = read_json(record)
            self.require(document.get("generation") == generation and document.get("recipient") == role and
                         document.get("base_history_seq") == metadata["base_history_seq"],
                         f"generation {generation} role {role}: catalog context differs")
            expected_ids = set(submissions["explorer"]["design_ids"])
            if role == "evidence_reviewer":
                expected_ids.update(submissions["refiner"]["design_ids"])
            rows = document["candidates"]
            self.require(bool(rows) and {r["design_id"] for r in rows} == expected_ids,
                         f"generation {generation} role {role}: catalog does not contain actual peer proposals")
            for item in rows:
                issued = self.catalog.get(item["design_id"])
                receipt = self.catalog_jobs.get(item["design_id"])
                self.require(issued is not None and receipt is not None and
                             all(item.get(k) == issued.get(k) for k in
                                 ("sequence", "backbone_id", "violations")) and
                             all(item["generation_job"].get(k) == receipt.get(k) for k in
                                 ("profile_id", "round", "action", "count", "temperature", "seed")) and
                             item.get("evidence_status") == "proposal, not an energy measurement",
                             f"generation {generation} role {role}: peer candidate provenance differs")
            session = metadata["sessions"][role]
            sent = next(e for e in by_turn[(role, session["turn_id"])] if e["kind"] == "turn_sent")
            settings = json.loads(sent["turn"]["message"].rsplit("\n\nTask settings: ", 1)[1])
            self.require(settings.get("peer_catalog") == info,
                         f"generation {generation} role {role}: turn did not receive the audited catalog")
            self.require(info["path"] in role_reads[role],
                         f"generation {generation} role {role}: no successful read of the peer catalog")
        review = metadata.get("review") or {}
        self.require(review == submissions["evidence_reviewer"] and
                     bool(review.get("evidence", "").strip()) and bool(review.get("hypotheses", "").strip()) and
                     set(review.get("design_ids", [])) <=
                     (set(submissions["explorer"]["design_ids"]) | set(submissions["refiner"]["design_ids"])),
                     f"generation {generation}: reviewer did not submit evidence on the peer catalog")
        if generation == 1:
            self.require(len({tuple(submissions[role]["design_ids"]) for role in ROLES}) == 1,
                         "bootstrap roles did not all inspect the same initial candidate")

    def policy(self):
        from ldm_tts.async_runtime.policy import PriorApplication, check_prior_invariants
        from ldm_tts.async_runtime.selectors import decision_choice

        root = self.root / "policy_harness"
        manifest = read_json(root / "manifest.json")
        profiles = manifest["profiles"]
        self.require(manifest.get("model") == "deepseek-flash" and manifest.get("thinking") == "low" and
                     len(profiles) == 1 and profiles[0].get("profileId") == "policy_architect" and
                     len(profiles[0].get("skills", [])) == 1,
                     "policy architect profile, skill, model or reasoning configuration differs")
        session_path = (root / profiles[0]["session"]).resolve()
        self.require(session_path.is_relative_to(root.resolve()) and
                     read_skills(successful_reads(session_path), {"compile-ldm-policy"}) == {"compile-ldm-policy"},
                     "policy architect did not successfully read the task policy skill")
        applications = self.db.rows("SELECT * FROM prior_applications ORDER BY seq")
        by_id = {r["application_id"]: r for r in applications}
        self.require(bool(applications), "no frozen policy applications")
        self.require(not any(a["degraded"] for a in applications), "one or more policy applications degraded")
        self.problems.extend(check_prior_invariants(self.db, UTILITY))
        epoch_manifests = {}
        rounds = []
        for row in self.generations:
            generation = row["generation"]
            policy_round = read_json_value(row["outcome_json"])["metadata"]["policy_round"]
            path = root / "rounds" / f"round_{generation:03d}"
            manifest = read_json(path / "manifest.json")
            result = read_json(path / "result.json")
            snapshot = read_json(path / "research_snapshot.json")
            input_document = read_json(path / "input.json")
            self.require(policy_round.get("status") == "accepted" and result.get("status") == "accepted" and
                         not policy_round.get("degraded") and not result.get("degraded"),
                         f"generation {generation}: no accepted nondegraded policy turn")
            self.require(result.get("harness_turn", {}).get("usage", {}).get("providerCalls", 0) > 0,
                         f"generation {generation}: policy turn has no provider calls")
            turn = result["harness_turn"]
            commit = read_json(root / "turns" / turn["turn_id"] / "turn_committed.json")
            self.require(commit.get("profileId") == "policy_architect" and
                         commit.get("submissionStatus") == "accepted" and
                         commit.get("submissionDigest") == result.get("submission_sha256") and
                         commit.get("submission", {}).get("action") == result.get("action"),
                         f"generation {generation}: policy turn receipt differs from the saved result")
            self.require(result.get("input_sha256") == manifest.get("input_sha256") and
                         result.get("compiled_arrays_sha256") == file_sha256(path / "compiled.npz"),
                         f"generation {generation}: policy result integrity mismatch")
            for name, expected in manifest["files"].items():
                self.require(file_sha256(path / name) == expected,
                             f"generation {generation}: policy input file digest mismatch: {name}")
            self.require(policy_round.get("epoch_id") == result.get("epoch_id") and
                         policy_round.get("artifact_sha256") == result.get("artifact_sha256"),
                         f"generation {generation}: policy round and saved result differ")
            feedback = input_document["execution_context"]["weight_context"]["prediction_feedback"]
            feedback_by_decision = {item["decision_id"]: item for item in feedback["measurements"]}
            for item in snapshot["measured_observations"]:
                decision_id = item["decision_id"]
                matched = feedback_by_decision.get(decision_id)
                decision_number = int(decision_id[1:])
                _, decision_inputs, _ = decision_choice(self.db, decision_number)
                self.require(matched is not None and matched.get("candidate_id") == item["candidate_id"] and
                             matched.get("application_id") == decision_inputs.get("prior_application_id") and
                             math.isclose(float(matched["measured_utility"]), float(item["utility"]), abs_tol=1e-12),
                             f"generation {generation}: no matching decision-keyed prediction feedback for {decision_id}")
            self.require(feedback["summary"]["count"] == len(feedback_by_decision) ==
                         len(snapshot["measured_observations"]),
                         f"generation {generation}: prediction feedback does not cover measured history")
            with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
                hf, hu, qf = arrays["history_features"], arrays["history_utilities"], arrays["query_features"]
                history_ids = snapshot["history_candidate_ids"]
                query = snapshot["candidate_catalog"]
                self.require(hf.shape == (len(history_ids), 220) and hu.shape == (len(history_ids),) and
                             qf.shape == (len(query), 220),
                             f"generation {generation}: policy feature dimensions differ from 11x20 one-hot")
                for index, candidate_id in enumerate(history_ids):
                    observation = self.by_candidate[candidate_id]
                    self.require(np.array_equal(hf[index], onehot(observation["candidate"]["payload"]["sequence"])) and
                                 math.isclose(float(hu[index]), observation["evaluation"]["metrics"][UTILITY], abs_tol=1e-12),
                                 f"generation {generation}: history feature or utility differs for {candidate_id}")
                for index, item in enumerate(query):
                    sequence = item["candidate"]["payload"]["sequence"]
                    self.require(np.array_equal(qf[index], onehot(sequence)),
                                 f"generation {generation}: query feature differs for {item['candidate_id']}")
                for item in snapshot["measured_observations"]:
                    observation = self.by_candidate[item["candidate_id"]]
                    self.require(math.isclose(float(item["utility"]),
                                              observation["evaluation"]["metrics"][UTILITY], abs_tol=1e-12),
                                 f"generation {generation}: policy measured utility differs")
            epoch = result.get("epoch_id")
            if epoch:
                ep = read_json(root / "epochs" / epoch / "manifest.json")
                artifact = (root / ep["artifact_path"]).resolve()
                self.require(artifact.is_relative_to(root.resolve()) and ep["epoch_id"] == epoch and
                             ep["artifact_sha256"] == result["artifact_sha256"] == file_sha256(artifact),
                             f"generation {generation}: policy epoch artifact digest mismatch")
                for name, expected in ep["files"].items():
                    self.require(file_sha256(root / "epochs" / epoch / name) == expected,
                                 f"generation {generation}: policy epoch file digest mismatch: {name}")
                epoch_manifests[epoch] = ep
            rounds.append({"generation": generation, "action": result.get("action"), "epoch": epoch,
                           "history": len(history_ids), "query": len(query),
                           "feedback_measurements": len(feedback_by_decision)})
        used_epochs = set()
        pending_decisions = 0
        pending_conditioned = 0
        prior_fitted_decisions = []
        replay_inputs = []
        used_generations = set()
        for row in self.decisions:
            record, inputs, step_index = decision_choice(self.db, row["decision_id"])
            selection = record["selection"]
            application_id = inputs.get("prior_application_id")
            application = by_id.get(application_id)
            self.require(application is not None, f"decision {row['decision_id']}: missing policy application")
            if application is None:
                continue
            used_generations.add(application["generation"])
            pending = {key for _, key in inputs["pending"]}
            candidate_keys = {c["canonical_key"] for c in inputs["candidates"]}
            self.require(not pending & candidate_keys, f"decision {row['decision_id']}: pending candidate selectable")
            pending_decisions += bool(pending)
            chosen = selection["steps"][step_index]
            self.require(pending <= set(chosen["pending_keys"]),
                         f"decision {row['decision_id']}: selection omitted a pending experiment")
            experiment = self.db.row("SELECT canonical_key FROM experiments WHERE decision_id=?", (row["decision_id"],))
            index = chosen["candidate_ids"].index(chosen["selected_candidate_id"])
            self.require(experiment is not None and chosen["canonical_keys"][index] == experiment["canonical_key"],
                         f"decision {row['decision_id']}: selection and submitted experiment differ")
            params = selection["parameters"]
            self.require(math.isclose(float(params["alpha"]), application["alpha"], abs_tol=1e-12) and
                         math.isclose(float(params["eta"]), application["eta"], abs_tol=1e-12),
                         f"decision {row['decision_id']}: policy weights differ from the application")
            self.require(record["selector"].get("objective") == UTILITY and
                         record["selector"].get("feature_version") == "cdrh3-one-hot-v1",
                         f"decision {row['decision_id']}: selector objective or features differ")
            self.require(record.get("pending_handling") == "kriging_believer",
                         f"decision {row['decision_id']}: selector is not configured for pending conditioning")
            pending_conditioned += bool(pending and chosen["metadata"].get("gp_ready") and
                                        chosen["metadata"].get("pending_handling") == "kriging_believer")
            self.require(application["created_at"] <= row["created_at"] and
                         (application["superseded_at"] is None or row["created_at"] <= application["superseded_at"]),
                         f"decision {row['decision_id']}: policy application was not active")
            if not application["prior_free"] and not application["degraded"]:
                epoch = application["epoch_id"]
                ep = epoch_manifests.get(epoch)
                self.require(ep is not None and ep["artifact_sha256"] == application["artifact_sha256"],
                             f"decision {row['decision_id']}: application is not linked to a verified epoch")
                used_epochs.add(epoch)
                if selection["metadata"].get("gp_ready") and selection["metadata"].get("prior_applied"):
                    prior_fitted_decisions.append(row["decision_id"])
            replay_inputs.append((row["decision_id"], application_id))
        self.require(bool(used_epochs), "no decision used a nondegraded custom policy application")
        self.require(bool(prior_fitted_decisions), "no fitted GP decision actually applied a custom policy prior")
        self.require(used_generations == set(range(1, self.expected_rounds + 1)),
                     "not every research generation's policy application was used by a decision")
        self.report["policy"] = {"rounds": rounds, "applications": len(applications),
            "verified_epochs": sorted(epoch_manifests), "used_epochs": sorted(used_epochs),
            "prior_fitted_decisions": prior_fitted_decisions, "pending_decisions": pending_decisions,
            "pending_conditioned_decisions": pending_conditioned}
        self.report["coverage"]["pending_at_decision_in_campaign"] = "exercised" if pending_decisions else "not_exercised"
        self.report["coverage"]["pending_conditioned_in_campaign"] = "exercised" if pending_conditioned else "not_exercised"
        self.replay_decisions(replay_inputs, by_id, PriorApplication)

    def replay_decisions(self, replay_inputs, by_id, application_type):
        # The framework replay consumes immutable rows and frozen tables, never policy code.
        from ldm_tts.async_runtime.config import CampaignConfig
        from ldm_tts.async_runtime.ledger import Ledger
        from ldm_tts.async_runtime.selectors import audit_decision
        from tasks.antibody_rfd3.core.policy import make_selector

        frozen = read_json_value(self.db.value("SELECT config_json FROM campaign WHERE singleton=1"))
        ledger = Ledger(self.db, CampaignConfig(**frozen["campaign"]), None, clock=None)
        selector = make_selector()
        priors = lambda application_id: application_type(by_id[application_id])
        for decision_id, _ in replay_inputs:
            self.problems.extend(f"decision {decision_id}: {problem}" for problem in
                                 audit_decision(self.db, ledger, decision_id, selector, priors))
        self.report["policy"]["decisions_replayed"] = len(replay_inputs)


def read_json_value(value):
    return json.loads(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--minimum-evaluations", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1 or args.minimum_evaluations < 1:
        parser.error("rounds and minimum evaluations must be positive")
    report = Audit(args.run_dir, args.rounds, args.minimum_evaluations).run()
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
