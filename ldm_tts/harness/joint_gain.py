"""A one-step, joint expected-improvement policy over research actions.

The model predicts a branch's best *measured* score relative to its incumbent,
conditional on that branch receiving evaluations. It is a scheduling surrogate,
not the candidate GP or a counterfactual model of the global candidate selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from ldm_tts.harness.research_tree import ForkAssignment, ForkContext, ResearchNode


@dataclass(frozen=True)
class JointGainForks:
    samples: int = 512
    seed: int = 0
    noise_variance: float = 0.25

    def __post_init__(self) -> None:
        if type(self.samples) is not int or self.samples < 2 or self.samples % 2:
            raise ValueError("joint gain requires an even sample count of at least two")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("joint gain seed must be a nonnegative integer")
        if (type(self.noise_variance) not in (int, float)
                or not np.isfinite(self.noise_variance) or self.noise_variance <= 0):
            raise ValueError("joint gain noise variance must be finite and positive")

    def identity(self) -> dict:
        return {"name": "joint_gain", "version": 1, **asdict(self)}

    def plan(self, context: ForkContext) -> tuple[ForkAssignment, ...]:
        if len(context.profile_ids) != 4 or len(set(context.profile_ids)) != 4 or not context.main_nodes:
            raise ValueError("joint gain requires four distinct profiles and a main node")
        offset = (context.round_index - 1) % 4
        roles = context.profile_ids[offset:] + context.profile_ids[:offset]
        older = context.main_nodes[:-1] or context.main_nodes
        restarts = tuple(ForkAssignment(roles[0], n.node_id, "restart",
                                       f"explore_{context.round_index:04d}_{roles[0]}") for n in older)
        continuations = tuple(ForkAssignment(
            roles[1], n.node_id, "continue" if context.archive else "restart",
            n.lineage_id if context.archive else f"explore_{context.round_index:04d}_{roles[1]}",
        ) for n in (context.archive or older))
        fixed = tuple(ForkAssignment(p, context.main_nodes[-1].node_id, "main", "main") for p in roles[2:])
        actions = fixed + restarts + continuations
        pairs = [(i + 2, j + 2 + len(restarts))
                 for i, j in product(range(len(restarts)), range(len(continuations)))
                 if context.archive or len(older) == 1 or restarts[i].parent_id != continuations[j].parent_id]
        draws = self._sample_gaps(context, actions)
        fixed_best = np.maximum(0, draws[:, :2].max(axis=1))
        scores = np.asarray([np.maximum(fixed_best, draws[:, pair].max(axis=1)).mean() for pair in pairs])
        # All pairs share the same joint draws, including the two fixed branches.
        chosen = pairs[int(np.argmax(scores))]
        by_profile = {a.profile_id: a for a in (*fixed, actions[chosen[0]], actions[chosen[1]])}
        return tuple(by_profile[p] for p in context.profile_ids)

    def _sample_gaps(self, context: ForkContext, actions: tuple[ForkAssignment, ...]) -> np.ndarray:
        evidence = [o for o in context.outcomes if o.best_score is not None and o.incumbent is not None]
        if any(o.round_index >= context.round_index for o in evidence):
            raise ValueError("joint gain cannot train on current or future rounds")
        gaps = np.asarray([o.best_score - o.incumbent for o in evidence], dtype=float)
        if not np.isfinite(gaps).all():
            raise ValueError("joint gain requires finite measured outcomes")
        nonzero = np.abs(gaps[gaps != 0])
        scale = float(np.median(nonzero)) if len(nonzero) else 1.0
        nodes = {n.node_id: n for n in (*context.main_nodes, *context.archive)}

        def features(parent: ResearchNode, profile: str, purpose: str, round_index: int,
                     incumbent: float | None) -> tuple[float, ...]:
            known = parent.best_score is not None and incumbent is not None
            descriptor = np.asarray(parent.descriptor, dtype=float)
            if not len(descriptor) or not np.isfinite(descriptor).all():
                raise ValueError("joint gain requires finite parent descriptors")
            return (
                np.log1p(round_index), (round_index - parent.round_index) / round_index,
                np.log1p(parent.expansions), float(known),
                float(np.tanh((parent.best_score - incumbent) / scale)) if known else 0.0,
                *(float(profile == p) for p in context.profile_ids),
                *(float(purpose == p) for p in ("main", "restart", "continue")),
                *descriptor,
            )

        query = np.asarray([features(nodes[a.parent_id], a.profile_id, a.purpose,
                                     context.round_index, context.incumbent) for a in actions])
        covariance = _kernel(query, query)
        mean = np.zeros(len(actions))
        if evidence:
            train = np.asarray([features(o.parent, o.profile_id, o.purpose, o.round_index, o.incumbent)
                                for o in evidence])
            # A shared winning candidate is one measurement, even when several
            # profiles proposed it. Correlated noise prevents independent credit.
            shared = np.asarray([[float(i == j or (a.best_key is not None and a.best_key == b.best_key))
                                  for j, b in enumerate(evidence)] for i, a in enumerate(evidence)])
            noise = self.noise_variance * shared
            chol = np.linalg.cholesky(_kernel(train, train) + noise + 1e-8 * np.eye(len(train)))
            cross = _kernel(query, train)
            projected = np.linalg.solve(chol, cross.T)
            mean = projected.T @ np.linalg.solve(chol, gaps / scale)
            covariance -= projected.T @ projected
        # This small GP needs joint covariance and correlated observation noise;
        # the existing candidate surrogate exposes marginal predictions only.
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        if values.min() < -1e-7:
            raise ValueError("joint gain posterior covariance is not positive semidefinite")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, context.round_index]))
        normal = rng.standard_normal((self.samples // 2, len(actions)))
        normal = np.concatenate((normal, -normal))
        return (mean + normal @ (vectors * np.sqrt(np.maximum(values, 0))).T) * scale


def _kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distance = (left * left).sum(axis=1)[:, None] + (right * right).sum(axis=1)[None, :] - 2 * left @ right.T
    return np.exp(-0.5 * np.maximum(distance, 0))
