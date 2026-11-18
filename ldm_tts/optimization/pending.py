"""Pending-aware finite-reservoir selection.

Kriging Believer (strategy A) conditions the latent posterior on in-flight points
with fantasies at the current posterior mean: the mean is unchanged and the
covariance shrinks by a Schur complement. Monte Carlo fantasy (strategy B)
marginalizes the selection rule over sampled pending outcomes. Selection is
stepwise conditional sampling with the shared q0-tilted robust-z UCB rule from
``ldm_tts.harness.ucb_diagnostics``; every step returns its full probability table.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ldm_tts.contracts import AcquisitionSpec, Candidate
from ldm_tts.harness.ucb_diagnostics import robust_z_acquisition, selection_probability
from ldm_tts.optimization.acquisition import confidence_bound
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import (
    BOObservation,
    BOPrediction,
    BOSelectionResult,
    PendingAwareSelection,
    PendingExperiment,
    SelectionSeed,
    SelectionStep,
    SurrogateVector,
)

PENDING_HANDLINGS = ("dedup_only", "kriging_believer", "mc_fantasy")
BETA_INFLATIONS = ("none", "max_rho")
EPSILON = 1.0e-12
VARIANCE_FLOOR = 1.0e-9  # same floor as RBFGPSurrogate.predict_record


@dataclass(frozen=True)
class SelectionRule:
    """pi(x) ∝ q0(x)^alpha · exp(eta · clip(robust_z(mu + beta·sigma_j)))."""

    alpha: float = 1.0
    eta: float = 1.0
    beta: float = 1.0
    z_clip: float = 3.0
    pending_handling: str = "kriging_believer"
    beta_inflation: str = "none"
    fantasy_samples: int = 256

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.alpha, self.eta, self.beta, self.z_clip)):
            raise ValueError("selection rule parameters must be finite")
        if self.beta < 0 or self.z_clip <= 0:
            raise ValueError("selection rule requires beta >= 0 and z_clip > 0")
        if self.pending_handling not in PENDING_HANDLINGS:
            raise ValueError(f"pending_handling must be one of {PENDING_HANDLINGS}")
        if self.beta_inflation not in BETA_INFLATIONS:
            raise ValueError(f"beta_inflation must be one of {BETA_INFLATIONS}")
        if self.beta_inflation != "none" and self.pending_handling == "dedup_only":
            raise ValueError("beta inflation requires pending variance conditioning")
        if isinstance(self.fantasy_samples, bool) or self.fantasy_samples < 1:
            raise ValueError("fantasy_samples must be a positive integer")

    @property
    def normalization(self) -> dict[str, Any]:
        return {"name": "robust_z", "mad_scale": 1.4826, "epsilon": EPSILON, "z_clip": self.z_clip}

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "epsilon": EPSILON, "normalization": "robust_z"}


class PriorTable(Mapping):
    """One policy application's prior, raw objective units, indexed by canonical key."""

    def __init__(self, application_id: str, values: Mapping[str, float]) -> None:
        self.application_id = str(application_id)
        self._values = {str(k): float(v) for k, v in values.items()}
        if not all(math.isfinite(v) for v in self._values.values()):
            raise ValueError("prior table values must be finite")

    @classmethod
    def from_standardized(cls, application_id, keys, prior, *, location, scale) -> "PriorTable":
        """m_A(key) = loc_A + scale_A · prior_A(key), from the application-time constants."""
        values = np.asarray(prior, dtype=float)
        if values.shape != (len(keys),) or not float(scale) > 0:
            raise ValueError("standardized prior must align with keys and use a positive scale")
        return cls(application_id, {k: float(location) + float(scale) * v for k, v in zip(keys, values)})

    def standardized(self, keys: Sequence[str], location: float, scale: float) -> np.ndarray:
        """prior_z = (m_A − y_mean) / y_std under the current fit's constants."""
        return (np.asarray([self._values[k] for k in keys], dtype=float) - location) / scale

    def missing(self, keys) -> tuple[str, ...]:
        return tuple(k for k in keys if k not in self._values)

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def fit_residual_surrogate(
    history: Sequence[BOObservation],
    keys: Sequence[str],
    prior: Mapping[str, float] | None = None,
    **settings,
) -> tuple[RBFGPSurrogate, tuple[str, ...]]:
    """Fit the authoritative GP; with a prior table, rows lacking a value are left out."""
    if len(keys) != len(history):
        raise ValueError("history keys must align with history")
    if prior is None:
        return RBFGPSurrogate(history, **settings), ()
    rows = [item for item, key in zip(history, keys) if key in prior]
    excluded = tuple(item.candidate_id for item, key in zip(history, keys) if key not in prior)
    base = RBFGPSurrogate(rows, **settings)
    if not base.ready:
        return base, excluded
    # Standardization does not depend on the residual prior; reuse the plain fit's constants.
    values = np.asarray([prior[key] for key in keys if key in prior], dtype=float)
    residual = (values - base.y_mean) / base.y_std
    return RBFGPSurrogate(rows, residual_prior_mean=residual, **settings), excluded


def condition_on(covariance: np.ndarray, indices: Sequence[int], noise: float) -> np.ndarray:
    """Posterior covariance after noisy observations at ``indices`` (Schur complement).

    The values observed do not enter, so this is the Kriging Believer covariance;
    conditioning sequentially on one index at a time gives the same result.
    """
    index = list(indices)
    if not index:
        return covariance
    cross = covariance[:, index]
    gram = covariance[np.ix_(index, index)] + noise * np.eye(len(index))
    return covariance - cross @ np.linalg.solve(gram, cross.T)


def fantasy_mean_shifts(covariance, pending, targets, noise, rng, samples) -> np.ndarray:
    """MC fantasy mean shifts delta = C S⁻¹ (y_P − mu_P) with y_P ~ N(mu_P, S); latent units."""
    pending, targets = list(pending), list(targets)
    gram = covariance[np.ix_(pending, pending)] + noise * np.eye(len(pending))
    innovations = rng.standard_normal((samples, len(pending))) @ np.linalg.cholesky(gram).T
    return np.linalg.solve(gram, innovations.T).T @ covariance[np.ix_(targets, pending)].T


def selection_rng(seed: SelectionSeed, step: int, stream: str = "draw") -> np.random.Generator:
    digest = hashlib.sha256(f"{seed.campaign_seed}\x1f{seed.decision_id}\x1f{step}\x1f{stream}".encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([int(v) for v in np.frombuffer(digest, dtype="<u4")]))


def draw_index(probability, uniform: float) -> int:
    cumulative = np.cumsum(probability)
    return int(min(np.searchsorted(cumulative, uniform * cumulative[-1], side="right"), len(cumulative) - 1))


@dataclass(frozen=True, eq=False)
class PosteriorSnapshot:
    """Frozen inputs of one decision; ``covariance`` is the latent joint over pending then candidates."""

    candidate_ids: tuple[str, ...]
    canonical_keys: tuple[str, ...]
    pending_keys: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    covariance: np.ndarray | None
    y_std: float
    noise: float
    metadata: dict[str, Any] = field(default_factory=dict)


def posterior_snapshot(
    surrogate: RBFGPSurrogate,
    candidates: Sequence[Candidate],
    vectors,
    *,
    pending: Sequence[tuple[str, Sequence[float]]] = (),
    query_prior=None,
) -> PosteriorSnapshot:
    """Candidate means and the latent joint covariance from the authoritative model.

    ``query_prior`` is in the current fit's standardized units (see PriorTable). When
    the GP is not ready every candidate gets the same mean and std, so pi ∝ q0^alpha
    and pending experiments act only through deduplication.
    """
    ids = tuple(item.candidate_id for item in candidates)
    keys = tuple(item.canonical_key for item in candidates)
    pending_keys = tuple(key for key, _ in pending)
    if not surrogate.ready:
        records = [surrogate.predict_record(item, vector) for item, vector in zip(ids, vectors, strict=True)]
        return PosteriorSnapshot(
            ids, keys, pending_keys,
            np.asarray([r.scalar_mean for r in records], dtype=float),
            np.asarray([r.scalar_std for r in records], dtype=float),
            None, 1.0, surrogate.observation_noise,
            {"gp_ready": False, "fit_status": surrogate.fit_status, "prior_applied": False},
        )
    query = np.asarray([*[vector for _, vector in pending], *vectors], dtype=float)
    mean_z, covariance = surrogate.latent_posterior(query)
    prior = np.zeros(len(ids)) if query_prior is None else np.asarray(query_prior, dtype=float)
    if prior.shape != (len(ids),) or not np.isfinite(prior).all():
        raise ValueError("query prior must align with finite candidates")
    if prior.any() and surrogate.residual_prior_mean is None:
        raise ValueError("query prior requires a fitted residual GP")
    p = len(pending)
    return PosteriorSnapshot(
        ids, keys, pending_keys,
        surrogate.y_mean + surrogate.y_std * (prior + mean_z[p:]),
        surrogate.y_std * np.sqrt(np.maximum(np.diag(covariance)[p:], VARIANCE_FLOOR)),
        covariance, surrogate.y_std, surrogate.observation_noise,
        {"gp_ready": True, "fit_status": surrogate.fit_status, "prior_applied": query_prior is not None,
         "standardization": asdict(surrogate.standardization())},
    )


def stepwise_select(
    snapshot: PosteriorSnapshot,
    *,
    rule: SelectionRule,
    seed: SelectionSeed,
    count: int = 1,
    q0=None,
) -> PendingAwareSelection:
    """Stepwise conditional sampling; not Gumbel-top-k, because pi_j changes with every step.

    Step j samples x_j ~ pi_j on E_j (candidates not yet chosen), then x_j joins the
    conditioning set. q0 is renormalized on E_j, which is the q0 obtained after the
    chosen candidates' occurrences are invalidated.
    """
    n = len(snapshot.candidate_ids)
    if count < 1:
        raise ValueError("selection count must be positive")
    mass = np.full(n, 1.0 / max(n, 1)) if q0 is None else np.asarray(q0, dtype=float)
    if mass.shape != (n,) or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("q0 must be a finite non-negative vector aligned with candidates")
    p = len(snapshot.pending_keys)
    ready = snapshot.covariance is not None
    handling = rule.pending_handling if ready else "dedup_only"
    if ready:
        base_variance = np.maximum(np.diag(snapshot.covariance), VARIANCE_FLOOR)
        conditioned = condition_on(snapshot.covariance, range(p), snapshot.noise)
    remaining, chosen, steps = list(range(n)), [], []
    for j in range(min(count, n)):
        index = np.asarray(remaining)
        q = mass[index]
        if q.sum() <= 0:
            raise ValueError("q0 has no mass on the eligible set")
        q = q / q.sum()
        max_rho = None
        std = snapshot.std[index]
        if ready:
            variance = np.maximum(np.diag(conditioned)[p + index], VARIANCE_FLOOR)
            max_rho = float(np.sqrt(base_variance[p + index] / variance).max())
            if handling != "dedup_only":
                std = snapshot.y_std * np.sqrt(variance)
        beta = rule.beta * max_rho if rule.beta_inflation == "max_rho" and max_rho is not None else rule.beta
        mean = snapshot.mean[index]
        acquisition = np.asarray(confidence_bound(mean, std, kind="ucb", beta=beta, minimize=False), dtype=float)
        z = robust_z_acquisition(acquisition, rule.z_clip)
        conditioning = [*range(p), *(p + c for c in chosen)]
        kind = "closed_form"
        if handling == "mc_fantasy" and conditioning:
            shifts = snapshot.y_std * fantasy_mean_shifts(
                snapshot.covariance, conditioning, p + index, snapshot.noise,
                selection_rng(seed, j, "fantasy"), rule.fantasy_samples,
            )
            probability = np.mean([
                selection_probability(q, acquisition + shift, rule.alpha, rule.eta, rule.normalization)
                for shift in shifts
            ], axis=0)
            kind = "monte_carlo_estimate"
        else:
            probability = selection_probability(q, acquisition, rule.alpha, rule.eta, rule.normalization)
        uniform = float(selection_rng(seed, j).random())
        pick = draw_index(probability, uniform)
        selected = int(index[pick])
        steps.append(SelectionStep(
            index=j,
            candidate_ids=tuple(snapshot.candidate_ids[i] for i in index),
            canonical_keys=tuple(snapshot.canonical_keys[i] for i in index),
            pending_keys=(*snapshot.pending_keys, *(snapshot.canonical_keys[c] for c in chosen)),
            mean=tuple(mean.tolist()), std=tuple(std.tolist()),
            acquisition=tuple(acquisition.tolist()), normalized_acquisition=tuple(z.tolist()),
            q0=tuple(q.tolist()), probability=tuple(probability.tolist()),
            selected_candidate_id=snapshot.candidate_ids[selected],
            selected_probability=float(probability[pick]),
            uniform=uniform, beta=float(beta),
            effective_sample_size=float(1.0 / np.sum(probability ** 2)),
            max_rho=max_rho,
            metadata={
                "gp_ready": ready, "pending_handling": handling, "probability_kind": kind,
                "fantasy_samples": rule.fantasy_samples if kind != "closed_form" else 0,
                "logit_ranges": {
                    "proposal": float(np.ptp(rule.alpha * np.log(q + EPSILON))),
                    "acquisition": float(np.ptp(rule.eta * z)),
                },
            },
        ))
        chosen.append(selected)
        remaining.remove(selected)
        if ready:
            conditioned = condition_on(conditioned, [p + selected], snapshot.noise)
    return PendingAwareSelection(
        selected_candidate_ids=tuple(snapshot.candidate_ids[c] for c in chosen),
        steps=tuple(steps),
        parameters={**rule.to_dict(), "campaign_seed": seed.campaign_seed, "decision_id": seed.decision_id},
        metadata={**snapshot.metadata, "pending_effect": handling},
    )


def recompute_step_probability(step: SelectionStep, rule: SelectionRule) -> np.ndarray:
    """Recompute a closed-form pi_j from the recorded table (Monte Carlo steps need the snapshot)."""
    if step.metadata["probability_kind"] != "closed_form":
        raise ValueError("Monte Carlo fantasy probabilities are reproduced by replaying the snapshot")
    return selection_probability(np.asarray(step.q0), np.asarray(step.acquisition),
                                 rule.alpha, rule.eta, rule.normalization)


def audit_selection(selection: PendingAwareSelection, rule: SelectionRule, seed: SelectionSeed) -> list[str]:
    """Check recorded tables against the rule and the seeded draws; returns problems found.

    This verifies the record's internal consistency only. A consistently rewritten
    table (mean, std, acquisition and probability together) passes; use replay()
    with the decision's snapshot to detect that.
    """
    problems = []
    for step in selection.steps:
        probability = np.asarray(step.probability)
        handling = rule.pending_handling if step.metadata["gp_ready"] else "dedup_only"
        if step.metadata["pending_handling"] != handling:
            problems.append(f"step {step.index}: pending handling differs from the rule")
        conditioned = bool(step.pending_keys) and step.metadata["gp_ready"]
        kind = "monte_carlo_estimate" if handling == "mc_fantasy" and conditioned else "closed_form"
        if step.metadata["probability_kind"] != kind:
            problems.append(f"step {step.index}: probability kind differs from the rule")
        if step.metadata["gp_ready"] != (step.max_rho is not None):
            problems.append(f"step {step.index}: max_rho must be recorded exactly when the GP is ready")
        inflate = rule.beta_inflation == "max_rho" and step.max_rho is not None
        if step.beta != (rule.beta * step.max_rho if inflate else rule.beta):
            problems.append(f"step {step.index}: beta differs from the rule and recorded max_rho")
        if not math.isclose(sum(step.q0), 1.0, abs_tol=1e-9) or min(step.q0, default=0) < 0:
            problems.append(f"step {step.index}: q0 is not normalized on the eligible set")
        expected = np.asarray(confidence_bound(np.asarray(step.mean), np.asarray(step.std),
                                               kind="ucb", beta=step.beta, minimize=False))
        if not np.allclose(expected, step.acquisition, rtol=0, atol=1e-12):
            problems.append(f"step {step.index}: acquisition differs from mean + beta*std")
        if kind == "closed_form" and not np.allclose(
                recompute_step_probability(step, rule), probability, rtol=0, atol=1e-12):
            problems.append(f"step {step.index}: probability table does not recompute")
        if step.uniform != float(selection_rng(seed, step.index).random()):
            problems.append(f"step {step.index}: uniform does not match the seed stream")
        pick = draw_index(probability, step.uniform)
        if (step.candidate_ids[pick] != step.selected_candidate_id
                or step.selected_probability != probability[pick]):
            problems.append(f"step {step.index}: draw does not reproduce")
    return problems


def replay(
    selection: PendingAwareSelection,
    snapshot: PosteriorSnapshot,
    *,
    rule: SelectionRule,
    seed: SelectionSeed,
    q0=None,
) -> list[str]:
    """Design 6.5 audit: recompute the decision from its snapshot and compare with the record.

    ``q0`` must be the vector passed to the original decision (None for uniform).
    Returns the differing fields; empty means the record is reproduced exactly.
    """
    if not selection.steps:
        return ["record has no steps"]
    recomputed = stepwise_select(snapshot, rule=rule, seed=seed, count=len(selection.steps), q0=q0)
    expected, recorded = recomputed.to_dict(), selection.to_dict()
    if expected == recorded:
        return []
    problems = [f"{name} differs" for name in ("selected_candidate_ids", "parameters", "metadata")
                if expected[name] != recorded[name]]
    if len(expected["steps"]) != len(recorded["steps"]):
        return problems + ["step count differs"]
    for index, (left, right) in enumerate(zip(expected["steps"], recorded["steps"])):
        problems.extend(f"step {index}: {name} differs" for name in left if left[name] != right.get(name))
    return problems


class PendingAwareUCBSelector:
    """Reference pending-aware selector: q0-tilted robust-z UCB with Kriging Believer."""

    def __init__(
        self,
        *,
        objective_name: str,
        rule: SelectionRule | None = None,
        feature_version: str = "",
        **gp_settings,
    ) -> None:
        self.objective_name = str(objective_name)
        self.rule = rule or SelectionRule()
        self.feature_version = str(feature_version)
        self.gp_settings = {"feature_version": self.feature_version, **gp_settings}
        self.fit([])

    def describe(self) -> AcquisitionSpec:
        return AcquisitionSpec(
            name="ucb",
            objective_names=(self.objective_name,),
            score_direction="maximize",
            selection_rule="stepwise conditional sampling, q0-tilted robust-z UCB",
            parameters=self.rule.to_dict(),
            metadata={"pending_aware": True, "propensity": "per-step conditional probabilities"},
        )

    def fit(self, history, *, history_keys=None, prior=None) -> None:
        if any(len(item.objectives) != 1 for item in history):
            raise ValueError("PendingAwareUCBSelector requires one objective")
        keys = list(history_keys) if history_keys is not None else [
            str(item.metadata.get("canonical_key", item.candidate_id)) for item in history
        ]
        self.history, self.history_keys, self.prior = list(history), keys, prior
        self.surrogate, self.prior_excluded = fit_residual_surrogate(self.history, keys, prior, **self.gp_settings)

    def select_with_pending(
        self,
        candidates: Sequence[Candidate],
        representations: Mapping[str, SurrogateVector],
        *,
        pending: Sequence[PendingExperiment],
        seed: SelectionSeed,
        count: int = 1,
        q0: Mapping[str, float] | None = None,
    ) -> PendingAwareSelection:
        blocked = set(self.history_keys) | {item.candidate.canonical_key for item in pending}
        eligible, seen = [], set()
        for item in candidates:
            if item.canonical_key not in blocked and item.canonical_key not in seen:
                seen.add(item.canonical_key)
                eligible.append(item)
        missing = [item.candidate_id for item in eligible if item.candidate_id not in representations]
        if missing:
            raise ValueError("missing surrogate representation for candidate(s): " + ", ".join(missing))
        keys = [item.canonical_key for item in eligible]
        query_prior = None
        # Pending points need no prior value: the Kriging Believer innovation is zero and the
        # covariance ignores the mean, and MC fantasy shifts depend only on y_P − mu(P).
        if self.prior is not None and self.surrogate.ready:
            uncovered = [k for k in keys if k not in self.prior]
            if uncovered:
                raise ValueError("active prior application does not cover candidate(s): " + ", ".join(uncovered))
            query_prior = (np.asarray([self.prior[k] for k in keys], dtype=float)
                           - self.surrogate.y_mean) / self.surrogate.y_std
        if not eligible:
            return PendingAwareSelection((), (), self.rule.to_dict(), {"reason": "empty_eligible_set"})
        snapshot = posterior_snapshot(
            self.surrogate, eligible, [representations[item.candidate_id].values for item in eligible],
            pending=[(item.candidate.canonical_key, item.feature.values) for item in pending],
            query_prior=query_prior,
        )
        if q0 is not None and any(k not in q0 for k in keys):
            raise ValueError("q0 lacks eligible candidate key(s): " + ", ".join(k for k in keys if k not in q0))
        mass = None if q0 is None else [q0[k] for k in keys]
        selection = stepwise_select(snapshot, rule=self.rule, seed=seed, count=count, q0=mass)
        return PendingAwareSelection(selection.selected_candidate_ids, selection.steps, selection.parameters, {
            **selection.metadata,
            "surrogate": self.surrogate.summary(),
            "prior_application_id": getattr(self.prior, "application_id", None),
            "prior_excluded_history": list(self.prior_excluded),
            "excluded_candidates": len(candidates) - len(eligible),
        })

    def select(self, candidates, representations, *, count: int = 1, round_idx: int = 0) -> BOSelectionResult:
        """AcquisitionSelector compatibility: no pending, uniform q0, seed derived from the round."""
        selection = self.select_with_pending(
            candidates, representations, pending=(), seed=SelectionSeed(0, f"round:{round_idx}"), count=count
        )
        first = selection.steps[0] if selection.steps else None
        predictions = () if first is None else tuple(
            BOPrediction.scalar(cid, mean=m, std=s, acquisition_score=a,
                                metadata={"selection_probability": pi})
            for cid, m, s, a, pi in zip(first.candidate_ids, first.mean, first.std,
                                        first.acquisition, first.probability)
        )
        return BOSelectionResult(selection.selected_candidate_ids, predictions,
                                 metadata={"pending_aware": selection.to_dict()})


__all__ = [
    "BETA_INFLATIONS",
    "PENDING_HANDLINGS",
    "PendingAwareUCBSelector",
    "PosteriorSnapshot",
    "PriorTable",
    "SelectionRule",
    "audit_selection",
    "condition_on",
    "draw_index",
    "fantasy_mean_shifts",
    "fit_residual_surrogate",
    "posterior_snapshot",
    "recompute_step_probability",
    "replay",
    "selection_rng",
    "stepwise_select",
]
