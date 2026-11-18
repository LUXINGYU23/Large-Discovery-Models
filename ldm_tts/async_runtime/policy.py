"""Policy turns and frozen prior applications in the asynchronous controller (design 6.6, 5.5 rule 5, 9.1).

A generation with a policy is ready only after three things: the panel committed, the policy turn finished and
the frozen policy was applied. ``PolicyPrior`` wires the existing compiled-policy machinery into the paced
controller:

* **Policy turn** (``PolicyRoundSource`` wraps the research source; runs in the research job after the panel):
  a ``PolicyRoundInput`` whose history is the generation's baseline (observations up to ``base_ingest_seq``),
  ``history_rounds`` = the arrival generation of each observation (the generation before the first round whose
  baseline contains it, so labels are chronological and below the round index), query rows = the reservoir and
  pending experiments frozen at generation start plus the panel's proposals. Prediction feedback is passed as
  records keyed by ``decision_id``. ``PolicyResearchController.resolve`` keeps, replaces or disables the epoch.
* **Application** (lane readiness step, committed inside ``generation_end``; re-applications in their own
  ``prior_application`` transaction): the epoch's artifact runs on ``H_A`` = every booked successful observation
  with the objective and ``Q_A`` = live reservoir (plus the generation's proposals) ∪ pending, with the
  authoritative GP's standardization constants ``(loc_A, scale_A)`` in its context. The result is stored as a
  table by canonical key in raw objective units, ``m_A(key) = loc_A + scale_A · prior_A(key)``. A default or
  disabled policy (no epoch) is a *prior-free* application: the plain GP, as the synchronous zero prior.
* **Use**: ``ReferenceSelector(prior=policy.current)`` refits on ``(head_ingest_seq, application_id)``; each
  decision records the application id and is replayed with that application (``PolicyPrior.application``).
  Observations without a value in the active table (a late result after a write-off) are booked but stay out of
  the residual GP until the next application. Reservoir candidates the table does not cover (merged after the
  application) are excluded from selection and trigger a supplementary application on the next step.

Re-application triggers: generation readiness, uncovered reservoir candidates (``supplement``) and, optionally,
``reapply_after_observations`` booked results since the last application (the application lifetime factor).
"""
from __future__ import annotations

import io
import json
import os
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ldm_tts.contracts import Candidate
from ldm_tts.harness.policy_contract import PolicyRoundInput
from ldm_tts.harness.policy_execution import PolicyExecutionError
from ldm_tts.harness.protocol import file_sha256
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.pending import PriorTable

from .lane import ResearchOutcome
from .ledger import candidate_from_json
from .proposals import results_booked
from .selectors import decision_choice
from .store import append_event
from .wire import canonical, decision_label, digest

PRIOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS prior_applications (
  application_id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, trigger TEXT NOT NULL, generation INTEGER NOT NULL,
  epoch_id TEXT, artifact_sha256 TEXT, source TEXT NOT NULL, degraded INTEGER NOT NULL, prior_free INTEGER NOT NULL,
  location REAL NOT NULL, scale REAL NOT NULL, alpha REAL NOT NULL, eta REAL NOT NULL, stage TEXT NOT NULL,
  head_ingest_seq INTEGER NOT NULL, occurrence_head INTEGER NOT NULL, history_keys_json TEXT NOT NULL,
  query_keys_json TEXT NOT NULL, values_json TEXT NOT NULL, input_digest TEXT NOT NULL, error TEXT,
  created_at REAL NOT NULL, superseded_at REAL, superseded_by TEXT, superseded_head INTEGER);
"""


class PriorApplication(PriorTable):
    """One frozen policy application A: raw-unit prior values by canonical key plus its identity and lifetime."""

    def __init__(self, row) -> None:
        super().__init__(row["application_id"], json.loads(row["values_json"]))
        self.seq, self.trigger, self.generation = row["seq"], row["trigger"], row["generation"]
        self.epoch_id, self.artifact_sha256, self.source = row["epoch_id"], row["artifact_sha256"], row["source"]
        self.degraded, self.prior_free = bool(row["degraded"]), bool(row["prior_free"])
        self.location, self.scale, self.stage = float(row["location"]), float(row["scale"]), row["stage"]
        self.weights = (float(row["alpha"]), float(row["eta"]))
        self.head_ingest_seq, self.occurrence_head = row["head_ingest_seq"], row["occurrence_head"]
        self.history_keys = tuple(json.loads(row["history_keys_json"]))
        self.query_keys = tuple(json.loads(row["query_keys_json"]))
        self.error = row["error"]

    def covers(self, key: str) -> bool:
        return self.prior_free or key in self


def arrival_generations(ingest_seqs: Sequence[int], bases: Sequence[int]) -> list[int]:
    """``history_rounds`` of a policy turn. ``bases[i]`` is the round baseline (``base_ingest_seq``) of generation
    ``i + 1``; an observation's label is ``h - 1`` for the first generation ``h`` whose baseline contains it, i.e.
    the last generation started before it was booked (0: before any generation). Monotone in ingest order and
    below the round index of every turn whose baseline contains the observation."""
    if any(a > b for a, b in zip(bases, bases[1:])):
        raise ValueError("round baselines must be non-decreasing")
    labels = []
    for seq in ingest_seqs:
        index = bisect_left(bases, seq)
        if index == len(bases):
            raise ValueError(f"observation {seq} lies outside every round baseline")
        labels.append(index)
    return labels


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _npz(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


class PolicyRoundSource:
    """Research source wrapper: the panel, then the policy turn on the generation's frozen inputs."""

    def __init__(self, inner, policy: "PolicyPrior"):
        self.inner, self.policy = inner, policy
        self.name = getattr(inner, "name", None) or type(inner).__name__
        self.version = getattr(inner, "version", "1")

    def round_input_extras(self, db, generation: int) -> dict:
        """The policy turn's frozen inputs, next to the wrapped source's own extras (the wrapper must not hide
        them; the lane only asks its direct source)."""
        inner = getattr(self.inner, "round_input_extras", None)
        extras = dict(inner(db, generation)) if inner is not None else {}
        if "policy" in extras:
            raise ValueError("the wrapped research source already uses the round input field 'policy'")
        extras["policy"] = self.policy.turn_extras(db, generation)
        return extras

    def research(self, round_input, history) -> ResearchOutcome:
        return self.policy.research(self.inner, round_input, history)


class PolicyPrior:
    """Policy turns and frozen prior applications for a ``PacedController`` (``policy=``).

    ``research`` is a ``PolicyResearchController`` (or anything with ``resolve(round_input, feedback=)``,
    ``store`` (``PolicyStore``), ``executor`` and ``contract``). Features, the objective and the GP settings come
    from the controller's ``ReferenceSelector``; ``prepare_turn(round_input, info)`` lets a task add diagnostics
    (e.g. ``ucb_diagnostics.prepare_ucb_policy_research``) to a policy turn's input."""

    name = "policy_prior"
    version = "1"

    def __init__(self, research, *, reapply_after_observations: int | None = None,
                 prepare_turn: Callable[[PolicyRoundInput, dict], PolicyRoundInput] | None = None):
        if reapply_after_observations is not None and (
                isinstance(reapply_after_observations, bool) or int(reapply_after_observations) < 1):
            raise ValueError("reapply_after_observations must be a positive integer or None")
        self.research_controller = research
        self.contract = research.contract
        self.reapply_after = None if reapply_after_observations is None else int(reapply_after_observations)
        self.prepare_turn = prepare_turn
        self.controller = None
        self._cache: dict[str, PriorApplication] = {}

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "contract_sha256": self.contract.sha256,
                "profile_id": getattr(self.research_controller, "profile_id", None),
                "reapply_after_observations": self.reapply_after}

    def wrap(self, source) -> PolicyRoundSource:
        return PolicyRoundSource(source, self)

    def bind(self, controller) -> None:
        selector = controller.reservoir_selector
        if not hasattr(selector, "training_rows"):
            raise ValueError("a policy prior needs the reference selector (ReferenceSelector)")
        self.controller, self.selector = controller, selector
        self.store, self.ledger, self.keeper, self.clock = controller.store, controller.ledger, controller.keeper, \
            controller.clock
        self.directory = controller.path.with_name(controller.path.name + ".policy")
        self._cache = {}
        if selector.prior is None or isinstance(getattr(selector.prior, "__self__", None), PolicyPrior):
            selector.prior = self.current  # (re)bind: a reopened controller gets a fresh PolicyPrior
        selector._fitted = selector._baseline = None
        with self.store.lock:
            self.store.db.executescript(PRIOR_SCHEMA)
        if controller.lane is not None:
            controller.lane.readiness = self

    # ---- reading applications -------------------------------------------------------------------------

    def application(self, application_id: str) -> PriorApplication:
        if application_id not in self._cache:
            row = self.store.row("SELECT * FROM prior_applications WHERE application_id=?", (application_id,))
            if row is None:
                raise KeyError(application_id)
            self._cache[application_id] = PriorApplication(row)
        return self._cache[application_id]

    def current(self) -> PriorApplication | None:
        """The active application (the selector's ``prior``); None before the first one."""
        found = self.store.value("SELECT application_id FROM prior_applications ORDER BY seq DESC LIMIT 1")
        return None if found is None else self.application(found)

    def applications(self) -> list[dict]:
        """Application records without their tables (identity, epoch, constants, lifetime) for reports."""
        return [{k: r[k] for k in r.keys() if k not in ("values_json", "history_keys_json", "query_keys_json")}
                for r in self.store.rows("SELECT * FROM prior_applications ORDER BY seq")]

    def covered(self, candidates) -> tuple[list, tuple[str, ...]]:
        """Split eligible candidates into those the active application covers and the keys it does not."""
        application = self.current()
        if application is None or application.prior_free:
            return list(candidates), ()
        kept = [c for c in candidates if c.canonical_key in application]
        return kept, tuple(c.canonical_key for c in candidates if c.canonical_key not in application)

    # ---- policy turn (research job) ---------------------------------------------------------------------

    def prediction_records(self, db, head: int) -> list[dict]:
        """Decision-keyed predictions for successful observations through a frozen ingest head."""
        feedback = []
        for row in db.execute("SELECT decision_id FROM observations WHERE status='succeeded' AND ingest_seq <= ? "
                              "ORDER BY ingest_seq", (head,)).fetchall():
            record, inputs, step = decision_choice(self.store, row[0])
            predictions = record.get("predictions") or []
            if step < len(predictions):
                feedback.append({"decision_id": decision_label(row[0]),
                                 "application_id": inputs.get("prior_application_id"),
                                 "predictions": [predictions[step]]})
        return feedback

    def turn_extras(self, db, generation: int) -> dict:
        """Frozen at generation start (inside the round input file): the reservoir and pending candidates the turn
        will act on, the round baselines of earlier generations and decision-keyed prediction records."""
        dedup = self.ledger.dedup_keys(db)
        reservoir, seen = [], set()
        retained = (self.keeper.live(db) if self.controller.pacing_config.reservoir != "generation_replacement"
                    else ())
        for _, occurrence in retained:
            key = occurrence.candidate.canonical_key
            if key not in dedup and key not in seen:
                seen.add(key)
                reservoir.append(occurrence.candidate.to_dict())
        pending = [json.loads(r[0]) for r in
                   db.execute("SELECT candidate_json FROM experiments WHERE pending=1 ORDER BY decision_id").fetchall()]
        bases = [[r[0], r[1]] for r in
                 db.execute("SELECT generation, base_ingest_seq FROM generations ORDER BY generation").fetchall()]
        feedback = self.prediction_records(db, results_booked(db))
        return {"reservoir": reservoir, "pending": pending, "generation_bases": bases, "feedback": feedback}

    def research(self, inner, round_input, history) -> ResearchOutcome:
        generation = round_input["generation"]
        path = self.directory / "panel" / f"g{generation:06d}.json"
        expected = digest(round_input)
        outcome = None
        if path.exists():  # the panel finished before a crash: the policy turn resumes without re-running it
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("input_digest") == expected:
                outcome = ResearchOutcome.from_json(saved["outcome"])
        if outcome is None:
            outcome = inner.research(round_input, history)
            if not isinstance(outcome, ResearchOutcome):
                raise TypeError(f"research source returned {type(outcome).__name__}, not ResearchOutcome")
            _write_atomic(path, canonical({"input_digest": expected, "outcome": outcome.to_json()}).encode())
        self.store.check("policy_turn.after_panel")
        if outcome.status != "committed":
            return outcome
        return replace(outcome, metadata={**outcome.metadata, "policy_round": self.turn(round_input, history, outcome)})

    def turn(self, round_input, history, outcome: ResearchOutcome) -> dict:
        policy_input, info = self.turn_input(round_input, history, outcome)
        if policy_input is None:
            return {"status": "skipped", "reason": "empty_query_set"}
        compiled = self.research_controller.resolve(policy_input, feedback=round_input["policy"]["feedback"])
        meta = compiled.metadata
        return {"status": meta.get("status", "accepted"), "action": meta.get("action"), "epoch_id": compiled.epoch_id,
                "artifact_sha256": compiled.artifact_digest, "source": compiled.source,
                "degraded": compiled.degraded, "stage": compiled.stage, "alpha": compiled.alpha, "eta": compiled.eta,
                **info}

    def turn_input(self, round_input, history, outcome: ResearchOutcome) -> tuple[PolicyRoundInput | None, dict]:
        extras, generation = round_input["policy"], round_input["generation"]
        base = round_input["watermarks"]["base_ingest_seq"]
        bases = [b for _, b in extras["generation_bases"]] + [base]
        entries = [o for o in history if o.evaluation.succeeded and self.selector.objective_name in o.metrics]
        labels = arrival_generations([o.metadata["ingest_seq"] for o in entries], bases)
        seen, query = {o.canonical_key for o in history}, []
        for candidate in [*(Candidate(**c) for c in extras["reservoir"]), *(p.candidate for p in outcome.proposals),
                          *(Candidate(**c) for c in extras["pending"])]:
            if candidate.canonical_key not in seen:
                seen.add(candidate.canonical_key)
                query.append(candidate)
        if not query:
            return None, {}
        width = len(self.contract.feature_names)
        rows, _ = self.selector.training_rows(entries)
        location, scale = self.location_scale(rows)
        objective = self.selector.objective_name
        measured = tuple({"candidate_id": o.candidate_id, "canonical_key": o.canonical_key,
                          "experiment_id": o.experiment_id, "decision_id": o.decision_id,
                          "ingest_seq": o.metadata["ingest_seq"], "arrival_generation": label,
                          "utility": float(o.metrics[objective])} for o, label in zip(entries, labels))
        policy_input = PolicyRoundInput(
            round_index=generation,
            history_features=np.asarray([r.feature_vector for r in rows], dtype=float).reshape(len(rows), width),
            history_utilities=np.asarray([r.scalar_score for r in rows], dtype=float),
            query_features=np.asarray([self.selector.vector(c).values for c in query], dtype=float).reshape(
                len(query), width),
            history_candidate_ids=tuple(o.candidate_id for o in entries),
            history_rounds=tuple(labels),
            measured_observations=measured,
            research_snapshot={"generation": generation, "base_ingest_seq": base,
                               "candidate_catalog": [{"candidate_id": c.candidate_id, "canonical_key": c.canonical_key,
                                                      "candidate": c.to_dict()} for c in query]},
            execution_context=self.context(generation, location, scale))
        info = {"history": len(entries), "query": len(query)}
        if self.prepare_turn is not None:
            policy_input = self.prepare_turn(policy_input, {"generation": generation, "query": query,
                                                            "history": entries, "round_input": round_input})
        return policy_input, info

    def context(self, generation: int, location: float, scale: float) -> dict:
        return {"mean_context": {"generation": generation, "round_index": generation,
                                 "target_location": location, "target_scale": scale},
                "weight_context": {"generation": generation, "round_index": generation,
                                   "default_alpha": float(self.contract.default_alpha),
                                   "default_eta": float(self.contract.default_eta)}}

    def application_context(self, generation: int, location: float, scale: float, *, history, head_ingest_seq: int,
                            generation_bases, feedback) -> dict:
        """Task hook for a context derived only from the application's frozen history and decisions."""
        return self.context(generation, location, scale)

    def location_scale(self, rows) -> tuple[float, float]:
        """The authoritative GP's target standardization on these rows (the conversion constants of m_A); a GP
        that is not ready yet uses (mean, 1), which converts exactly as well and is never applied before it is."""
        surrogate = RBFGPSurrogate(rows, feature_version=self.selector.feature_version, **self.selector.gp_settings)
        if surrogate.ready:
            return float(surrogate.y_mean), float(surrogate.y_std)
        return (float(np.mean([r.scalar_score for r in rows])) if rows else 0.0), 1.0

    # ---- application ------------------------------------------------------------------------------------

    def prepare(self, generation: int, outcome: ResearchOutcome) -> dict:
        """Lane readiness, outside the transaction: apply the turn's epoch to H_A and Q_A (with the proposals)."""
        return self._prepare("generation", generation, outcome.metadata.get("policy_round") or {}, outcome.proposals)

    def commit(self, db, generation: int, prepared: dict, now: float) -> str:
        """Inside ``generation_end`` / ``prior_application``: switch the whole table atomically."""
        seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM prior_applications").fetchone()[0]
        application_id = f"a{seq:06d}"
        p = prepared
        db.execute("""INSERT INTO prior_applications (application_id, seq, trigger, generation, epoch_id,
            artifact_sha256, source, degraded, prior_free, location, scale, alpha, eta, stage, head_ingest_seq,
            occurrence_head, history_keys_json, query_keys_json, values_json, input_digest, error, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (application_id, seq, p["trigger"], p["generation"], p["epoch_id"], p["artifact_sha256"],
                    p["source"], int(p["degraded"]), int(p["prior_free"]), p["location"], p["scale"], p["alpha"],
                    p["eta"], p["stage"], p["head_ingest_seq"], p["occurrence_head"], canonical(p["history_keys"]),
                    canonical(p["query_keys"]), canonical(p["values"]), p["input_digest"], p["error"], now))
        db.execute("""UPDATE prior_applications SET superseded_at=?, superseded_by=?, superseded_head=?
            WHERE superseded_by IS NULL AND application_id != ?""", (now, application_id, p["head_ingest_seq"],
                                                                     application_id))
        append_event(db, now, "prior.applied", application_id,
                     {"generation": p["generation"], "trigger": p["trigger"], "epoch_id": p["epoch_id"],
                      "prior_free": p["prior_free"], "degraded": p["degraded"], "history": len(p["history_keys"]),
                      "query": len(p["query_keys"]), "error": p["error"]})
        return application_id

    def step(self) -> bool:
        """Re-apply the current epoch when reservoir candidates are uncovered (``supplement``) or the lifetime
        threshold of booked results has passed (``observations``). Results and pending changes alone never do."""
        controller = self.controller
        if controller.collect_only or controller.ledger.campaign_row()["status"] != "active":
            return False
        application = self.current()
        if application is None:
            return False
        with self.store.lock:
            db = self.store.db
            head = results_booked(db)
            dedup = self.ledger.dedup_keys(db)
            live = {o.candidate.canonical_key for _, o in self.keeper.live(db)} - dedup
        uncovered = () if application.prior_free else sorted(k for k in live if k not in application)
        if uncovered:
            trigger = "supplement"
        elif self.reapply_after is not None and head - application.head_ingest_seq >= self.reapply_after:
            trigger = "observations"
        else:
            return False
        prepared = self._prepare(trigger, application.generation,
                                 {"epoch_id": application.epoch_id, "artifact_sha256": application.artifact_sha256,
                                  "degraded": application.degraded})
        with self.store.tx("prior_application") as db:
            if db.execute("SELECT MAX(seq) FROM prior_applications").fetchone()[0] != application.seq:
                return False
            self.commit(db, application.generation, prepared, self.clock.now())
        return True

    def _prepare(self, trigger: str, generation: int, turn: dict, proposals=()) -> dict:
        with self.store.lock:
            db = self.store.db
            head = results_booked(db)
            history = tuple(o for o in self.ledger.observations(0, head)
                            if o.evaluation.succeeded and self.selector.objective_name in o.metrics)
            bases = tuple(r[0] for r in db.execute(
                "SELECT base_ingest_seq FROM generations ORDER BY generation").fetchall())
            feedback = self.prediction_records(db, head)
            dedup = self.ledger.dedup_keys(db)
            live = (() if trigger == "generation" and
                    self.controller.pacing_config.reservoir == "generation_replacement" else self.keeper.live(db))
            occurrence_head = db.execute("SELECT COALESCE(MAX(id),0) FROM occurrences").fetchone()[0]
            pending = [candidate_from_json(r[0]) for r in db.execute(
                "SELECT candidate_json FROM experiments WHERE pending=1 ORDER BY decision_id").fetchall()]
            previous = db.execute("SELECT epoch_id, artifact_sha256, degraded FROM prior_applications "
                                  "ORDER BY seq DESC LIMIT 1").fetchone()
        rows, keys = self.selector.training_rows(history)
        query: dict[str, Candidate] = {}
        for candidate in [*(o.candidate for _, o in live), *(p.candidate for p in proposals)]:
            if candidate.canonical_key not in dedup:
                query.setdefault(candidate.canonical_key, candidate)
        for candidate in pending:
            query.setdefault(candidate.canonical_key, candidate)
        query_keys = sorted(query)
        location, scale = self.location_scale(rows)
        if "epoch_id" in turn:
            epoch, artifact_sha, degraded = turn["epoch_id"], turn["artifact_sha256"], bool(turn.get("degraded"))
        elif previous is not None:  # a skipped turn keeps the previous application's epoch
            epoch, artifact_sha, degraded = previous["epoch_id"], previous["artifact_sha256"], bool(previous["degraded"])
        else:
            epoch, artifact_sha, degraded = None, None, False
        utilities = [r.scalar_score for r in rows]
        context = self.application_context(generation, location, scale, history=history, head_ingest_seq=head,
                                           generation_bases=bases, feedback=feedback)
        prepared = {"trigger": trigger, "generation": generation, "epoch_id": epoch, "artifact_sha256": artifact_sha,
                    "location": location, "scale": scale, "head_ingest_seq": head, "occurrence_head": occurrence_head,
                    "history_keys": keys, "query_keys": query_keys, "degraded": degraded, "error": None,
                    "alpha": float(self.contract.default_alpha), "eta": float(self.contract.default_eta),
                    "stage": "default", "source": "default", "prior_free": True, "values": {}}
        prepared["input_digest"] = digest({**prepared, "utilities": utilities, "context": context,
                                           "contract": self.contract.sha256})
        if epoch is None:
            return prepared
        width = len(self.contract.feature_names)
        history_features = np.asarray([r.feature_vector for r in rows], dtype=float).reshape(len(rows), width)
        query_features = np.asarray([self.selector.vector(query[k]).values for k in query_keys],
                                    dtype=float).reshape(len(query_keys), width)
        prepared.update(source="artifact", prior_free=False)
        if not rows and not query_keys:
            return prepared
        try:
            artifact = self._artifact(epoch, artifact_sha)
            directory = self.directory / "applications" / prepared["input_digest"].split(":", 1)[1][:16]
            inputs = directory / "input"
            _write_atomic(inputs / "contract.json", canonical(self.contract.to_dict()).encode())
            _write_atomic(inputs / "input.json", canonical({"round_index": generation, "application": trigger,
                                                            "execution_context": context}).encode())
            _write_atomic(inputs / "arrays.npz", _npz(
                history_features=history_features, history_utilities=np.asarray(utilities, dtype=float),
                query_features=query_features if len(query_keys) else history_features))
            result = self.research_controller.executor.execute(artifact, inputs, directory / "output")
            history_prior = np.asarray(result.history_prior_mean, dtype=float)
            query_prior = np.asarray(result.query_prior_mean, dtype=float)
            if history_prior.shape != (len(rows),) or (query_keys and query_prior.shape != (len(query_keys),)) or \
                    not np.isfinite(history_prior).all() or not np.isfinite(query_prior).all():
                raise ValueError("policy application must return finite scalar priors aligned with its rows")
        except (PolicyExecutionError, OSError, ValueError, KeyError) as exc:
            # Like the synchronous fallback: a frozen artifact that fails on the application input degrades to the
            # default (prior-free) policy; the failure is recorded on the application.
            error = "; ".join(f"{e.code}: {e.message}" for e in exc.errors) if isinstance(exc, PolicyExecutionError) \
                else f"{type(exc).__name__}: {exc}"
            prepared.update(source="default", prior_free=True, degraded=True, error=error[:500])
            return prepared
        values = {k: location + scale * float(v) for k, v in zip(keys, history_prior)}
        values.update({k: location + scale * float(v) for k, v in zip(query_keys, query_prior)})
        prepared.update(values=values, alpha=float(result.alpha), eta=float(result.eta), stage=str(result.stage))
        return prepared

    def _artifact(self, epoch_id: str, artifact_sha: str) -> Path:
        store = self.research_controller.store
        manifest = json.loads((store.root / "epochs" / epoch_id / "manifest.json").read_text(encoding="utf-8"))
        path = store.active_artifact({"artifact_path": manifest["artifact_path"]})
        if manifest.get("artifact_sha256") != artifact_sha or file_sha256(path) != artifact_sha:
            raise ValueError(f"policy epoch {epoch_id} artifact digest mismatch")
        return path

    # ---- invariants ---------------------------------------------------------------------------------------

    def check(self, strict: bool = False) -> list[str]:
        return check_prior_invariants(self.store, self.selector.objective_name, strict=strict)


def check_prior_invariants(store, objective_name: str, *, strict: bool = False) -> list[str]:
    """I7 and the application chain. The active application covers the GP history it was applied on, every
    pending experiment and every live reservoir candidate merged before it (``strict``: every live candidate, as
    right after a switch); every decision's candidates are covered by the application it recorded."""
    if store.row("SELECT 1 FROM sqlite_master WHERE type='table' AND name='prior_applications'") is None:
        return []
    rows = store.rows("SELECT * FROM prior_applications ORDER BY seq")
    if not rows:
        return []
    problems: list[str] = []
    add = problems.append
    if [r["seq"] for r in rows] != list(range(1, len(rows) + 1)):
        add("prior: application sequence is not contiguous")
    for earlier, later in zip(rows, rows[1:]):
        if earlier["superseded_by"] != later["application_id"]:
            add(f"prior: {earlier['application_id']} is not superseded by {later['application_id']}")
    if rows[-1]["superseded_by"] is not None:
        add("prior: the newest application is marked superseded")
    tables = {r["application_id"]: (bool(r["prior_free"]), set(json.loads(r["values_json"]))) for r in rows}
    current = rows[-1]
    prior_free, keys = tables[current["application_id"]]
    if not prior_free:
        own = set(json.loads(current["history_keys_json"])) | set(json.loads(current["query_keys_json"]))
        if own - keys:
            add(f"I7: {current['application_id']} lacks values for its own keys {sorted(own - keys)[:5]}")
        for r in store.rows("SELECT canonical_key, observation_json FROM observations WHERE status='succeeded' "
                            "AND ingest_seq <= ?", (current["head_ingest_seq"],)):
            if objective_name in json.loads(r["observation_json"])["evaluation"]["metrics"] and \
                    r["canonical_key"] not in keys:
                add(f"I7: history key {r['canonical_key']} has no value in {current['application_id']}")
        for r in store.rows("SELECT canonical_key FROM experiments WHERE pending=1"):
            if r["canonical_key"] not in keys:
                add(f"I7: pending key {r['canonical_key']} has no value in {current['application_id']}")
        for r in store.rows("SELECT id, canonical_key FROM occurrences WHERE removed_at IS NULL"):
            if r["canonical_key"] not in keys and (strict or r["id"] <= current["occurrence_head"]):
                add(f"I7: reservoir key {r['canonical_key']} has no value in {current['application_id']}")
    for r in store.rows("SELECT decision_id FROM decisions ORDER BY decision_id"):
        record, inputs, _ = decision_choice(store, r["decision_id"])
        application_id = inputs.get("prior_application_id")
        if application_id is None:
            continue
        if application_id not in tables:
            add(f"prior: decision {r['decision_id']} records unknown application {application_id}")
            continue
        free, used = tables[application_id]
        missing = [] if free else [c["canonical_key"] for c in inputs.get("candidates", ())
                                   if c["canonical_key"] not in used]
        if missing:
            add(f"I7: decision {r['decision_id']} chose among keys {missing[:5]} not covered by {application_id}")
    return problems


__all__ = ["PRIOR_SCHEMA", "PolicyPrior", "PolicyRoundSource", "PriorApplication", "arrival_generations",
           "check_prior_invariants"]
