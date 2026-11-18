"""Selection adapters for the paced controller (design 6.4-6.7) and the decision audit.

``ReferenceSelector`` wraps ``PendingAwareUCBSelector``: it fits the authoritative GP on booked successful
observations (up to the decision's ``head_ingest_seq``), passes the pending experiments and the reservoir q0,
and draws ``count`` candidates by stepwise conditional sampling (Kriging Believer by default; ``dedup_only``,
``mc_fantasy`` and ``beta_inflation`` through ``SelectionRule``). Each choice returns the full selection record
and the inputs needed to replay it; ``audit_decision`` replays a stored decision and runs ``audit_selection``.
``LegacySelector`` wraps a controller-core ``Selector`` (no pending awareness: dedup only, labelled as such).

This module needs numpy (via ``ldm_tts.optimization``); the rest of ``async_runtime`` does not import it.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ldm_tts.contracts import Candidate
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.pending import PendingAwareUCBSelector, SelectionRule, audit_selection
from ldm_tts.optimization.records import (BOObservation, PendingAwareSelection, PendingExperiment, SelectionSeed,
                                          SelectionStep, SurrogateVector)

from .ledger import candidate_from_json
from .selection import Choice, SelectionContext, normalized as _normalized

CURRENT = object()  # "the selector's active prior" (as opposed to an explicit application or None)


class ReferenceSelector:
    """The framework reference selector: q0-tilted robust-z UCB with pending handling per ``SelectionRule``.

    ``prior`` returns the active policy application (a ``PriorTable``; ``prior_free`` applications fit the plain
    GP) or None. With ``weights_from_policy`` the application's ``weights`` (alpha, eta) replace the rule's."""

    name = "pending_ucb"
    version = "1"
    pending_aware = True

    def __init__(self, *, featurize: Callable[[Candidate], Sequence[float] | SurrogateVector], objective_name: str,
                 rule: SelectionRule | None = None, prior: Callable[[], Mapping[str, float] | None] | None = None,
                 feature_version: str = "", weights_from_policy: bool = False, **gp_settings):
        self.featurize, self.objective_name = featurize, str(objective_name)
        self.rule, self.prior, self.feature_version = rule or SelectionRule(), prior, str(feature_version)
        self.weights_from_policy = bool(weights_from_policy)
        self.gp_settings = dict(gp_settings)
        self._fitted: tuple | None = None
        self._model: PendingAwareUCBSelector | None = None
        self._baseline: tuple[int, Any] | None = None

    def describe(self) -> dict:
        described = {"name": self.name, "version": self.version, "rule": self.rule.to_dict(), "gp": self.gp_settings,
                     "objective": self.objective_name, "feature_version": self.feature_version}
        if self.weights_from_policy:
            described["weights_from_policy"] = True
        return described

    def vector(self, candidate: Candidate) -> SurrogateVector:
        value = self.featurize(candidate)
        if isinstance(value, SurrogateVector):
            return value
        return SurrogateVector(tuple(float(v) for v in value), self.feature_version, candidate.candidate_id)

    def training_rows(self, history) -> tuple[list[BOObservation], list[str]]:
        """GP rows: booked successful observations that carry the objective, in ingest order, with their keys."""
        rows, keys = [], []
        for observation in history:
            if not observation.evaluation.succeeded or self.objective_name not in observation.metrics:
                continue
            vector = self.vector(observation.candidate)
            rows.append(BOObservation.from_observation(
                observation, objective_names=[self.objective_name],
                feature=SurrogateVector(vector.values, vector.version, observation.candidate_id)))
            keys.append(observation.canonical_key)
        return rows, keys

    def resolve_prior(self, prior=CURRENT):
        if prior is not CURRENT:
            return prior
        return self.prior() if self.prior is not None else None

    def rule_for(self, prior) -> SelectionRule:
        weights = getattr(prior, "weights", None) if self.weights_from_policy else None
        return self.rule if weights is None else replace(self.rule, alpha=float(weights[0]), eta=float(weights[1]))

    def model(self, history, head: int, prior=CURRENT) -> PendingAwareUCBSelector:
        prior = self.resolve_prior(prior)
        table = None if prior is None or getattr(prior, "prior_free", False) else prior
        rule = self.rule_for(prior)
        key = (head, getattr(prior, "application_id", None) if prior is not None else None, rule)
        if self._fitted != key:
            rows, keys = self.training_rows(history)
            model = PendingAwareUCBSelector(objective_name=self.objective_name, rule=rule,
                                            feature_version=self.feature_version, **self.gp_settings)
            model.fit(rows, history_keys=keys, prior=table)
            self._model, self._fitted = model, key
        return self._model

    def baseline(self, history, head: int) -> RBFGPSurrogate:
        """The plain GP (no policy prior) on the same history: the baseline of prediction feedback."""
        if self._baseline is None or self._baseline[0] != head:
            rows, _ = self.training_rows(history)
            self._baseline = (head, RBFGPSurrogate(rows, feature_version=self.feature_version, **self.gp_settings))
        return self._baseline[1]

    def choose(self, context: SelectionContext, prior=CURRENT) -> Choice | None:
        prior = self.resolve_prior(prior)
        model = self.model(context.history, context.head_ingest_seq, prior)
        representations = {c.candidate_id: self.vector(c) for c in context.candidates}
        pending = [PendingExperiment(item.candidate, self.vector(item.candidate), item.state) for item in context.pending]
        selection = model.select_with_pending(context.candidates, representations, pending=pending, seed=context.seed,
                                              count=context.count, q0=context.q0)
        if not selection.steps:
            return None
        by_id = {c.candidate_id: c.canonical_key for c in context.candidates}
        inputs = {"head_ingest_seq": context.head_ingest_seq,
                  "candidates": [c.to_dict() for c in context.candidates],
                  "pending": [[item.experiment_id, item.canonical_key] for item in context.pending],
                  "q0": dict(context.q0), "count": context.count,
                  "seed": {"campaign_seed": context.seed.campaign_seed, "decision_id": context.seed.decision_id},
                  "prior_application_id": getattr(prior, "application_id", None) if prior is not None else None}
        record = {"selector": self.describe(), "pending_handling": model.rule.pending_handling,
                  "selection": selection.to_dict()}
        if self.prior is not None:  # policy campaigns: frozen predictions for decision-keyed feedback
            record["predictions"] = self._predictions(selection, context, representations,
                                                      None if model.prior is None else self.baseline(
                                                          context.history, context.head_ingest_seq))
        return Choice(tuple(by_id[cid] for cid in selection.selected_candidate_ids),
                      tuple(step.selected_probability for step in selection.steps), _normalized(record), inputs)

    @staticmethod
    def _predictions(selection, context, representations, baseline) -> list[dict]:
        keys = {c.candidate_id: c.canonical_key for c in context.candidates}
        rows = []
        for step in selection.steps:
            index = step.candidate_ids.index(step.selected_candidate_id)
            mean, std = step.mean[index], step.std[index]
            if baseline is not None:
                plain = baseline.predict_record(step.selected_candidate_id,
                                                representations[step.selected_candidate_id].values)
                base_mean, base_std = plain.scalar_mean, plain.scalar_std
            else:
                base_mean, base_std = mean, std
            rows.append({"step": step.index, "candidate_id": step.selected_candidate_id,
                         "canonical_key": keys[step.selected_candidate_id], "active_mean": mean, "active_std": std,
                         "baseline_mean": base_mean, "baseline_std": base_std, "q0": step.q0[index],
                         "selection_probability": step.selected_probability})
        return rows


# ---- audit ----------------------------------------------------------------------------------------------

def _selection_from_json(data) -> PendingAwareSelection:
    steps = []
    for step in data["steps"]:
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in step.items()}
        steps.append(SelectionStep(**values))
    return PendingAwareSelection(tuple(data["selected_candidate_ids"]), tuple(steps), data["parameters"],
                                 data["metadata"])


def decision_choice(store, decision_id: int) -> tuple[dict, dict, int]:
    """(record, inputs, step) of a stored decision; barrier decisions point into their plan."""
    snapshot = json.loads(store.value("SELECT snapshot_json FROM decisions WHERE decision_id=?", (decision_id,)))
    plan = snapshot.get("plan")
    if plan is None:
        return snapshot["record"], snapshot["inputs"], 0
    row = store.row("SELECT record_json, inputs_json FROM plans WHERE plan_id=?", (plan["id"],))
    return json.loads(row["record_json"]), json.loads(row["inputs_json"]), plan["step"]


def audit_decision(store, ledger, decision_id: int, selector: ReferenceSelector,
                   priors: Callable[[str], Any] | None = None) -> list[str]:
    """Design 6.5 audit of one decision: rebuild its inputs from the store (history up to the recorded head,
    candidates, pending experiments, q0, seed, and the policy application it used), redraw, compare with the
    record field by field, and check the record's internal consistency with ``audit_selection``. ``priors`` maps
    an application id to that application (``PolicyPrior.application``). Empty list: reproduced exactly."""
    record, inputs, step = decision_choice(store, decision_id)
    if "selection" not in record:
        return ["decision was not made by a pending-aware reference selector"]
    application_id = inputs.get("prior_application_id")
    prior = None
    if application_id is not None:
        try:
            if priors is not None:
                prior = priors(application_id)
            else:
                prior = selector.resolve_prior()
                if getattr(prior, "application_id", None) != application_id:
                    raise KeyError(application_id)
        except KeyError:
            return [f"prior application {application_id} is unavailable for the replay"]
    seed = SelectionSeed(inputs["seed"]["campaign_seed"], inputs["seed"]["decision_id"])
    problems = [f"audit: {p}" for p in audit_selection(_selection_from_json(record["selection"]),
                                                        selector.rule_for(prior), seed)]
    pending = []
    for experiment_id, key in inputs["pending"]:
        row = store.row("SELECT candidate_json, decision_id, submit_state, es_state FROM experiments WHERE experiment_id=?",
                        (experiment_id,))
        pending.append(_Pending(experiment_id, key, candidate_from_json(row["candidate_json"]), "pending"))
    history = tuple(ledger.observations(0, inputs["head_ingest_seq"]))
    context = SelectionContext(None, tuple(Candidate(**c) for c in inputs["candidates"]), inputs["q0"], tuple(pending),
                               seed, inputs["count"], history, inputs["head_ingest_seq"])
    selector._fitted = None  # replay fits from scratch
    choice = selector.choose(context, prior=prior)
    expected = None if choice is None else choice.record
    if expected is None:
        return problems + ["replay selected nothing"]
    left, right = expected["selection"], record["selection"]
    for name in ("selected_candidate_ids", "parameters", "metadata"):
        if left[name] != right[name]:
            problems.append(f"replay: {name} differs")
    for index, (a, b) in enumerate(zip(left["steps"], right["steps"])):
        problems.extend(f"replay: step {index}: {name} differs" for name in a if a[name] != b.get(name))
    if len(left["steps"]) != len(right["steps"]):
        problems.append("replay: step count differs")
    snapshot = json.loads(store.value("SELECT snapshot_json FROM decisions WHERE decision_id=?", (decision_id,)))
    if snapshot["probability"] != right["steps"][step]["selected_probability"]:
        problems.append("decision probability differs from the recorded step")
    return problems


@dataclass(frozen=True)
class _Pending:
    experiment_id: str
    canonical_key: str
    candidate: Candidate
    state: str


__all__ = ["CURRENT", "ReferenceSelector", "audit_decision", "decision_choice"]
