"""Fixed syntax features, shared GP posterior, and empirical-mass LDM selection."""

import ast
import math
import re
import numpy as np

from ldm_tts.contracts import AcquisitionSpec, SurrogateSpaceSpec
from ldm_tts.optimization.gp import RBFGPSurrogate
from ldm_tts.optimization.records import BOSelectionResult, SurrogateVector
from .grammar import FIELDS, GRAMMAR_VERSION, REGISTRY, parse_expression
from .protocol import digest

OPS = tuple(sorted(set(REGISTRY["qlib"]) | set(REGISTRY["assay"])))
TYPES = ("field", "literal", "operator", "binary", "unary", "group")
ROLES = ("value", "number", "window", "lag", "quantile")
LAYOUT = ("dialect:qlib", "dialect:assay", "depth", "nodes") + tuple("op:"+op for op in OPS) + tuple("field:"+f for f in FIELDS) + tuple(
    f"edge:{op}:{position}:{kind}" for op in OPS for position in range(3) for kind in TYPES) + tuple(
    f"constant:{role}:{stat}" for role in ROLES for stat in ("count", "mean", "min", "max")) + tuple(f"group:{index}" for index in range(32))
INDEX = {name: index for index, name in enumerate(LAYOUT)}
FEATURE_VERSION = "t3-ast-v1:" + digest([GRAMMAR_VERSION, LAYOUT])


class FactorEncoder:
    def describe(self):
        return SurrogateSpaceSpec("vector", "ordered AST operator/field/argument-role features", "fixed",
                                  dimension=len(LAYOUT), version=FEATURE_VERSION)

    def encode(self, candidate):
        expression = parse_expression(candidate.payload["expression"], backend="assay")
        # Parse canonical syntax so equivalent keyword/macro spellings have identical features.
        tree = ast.parse(re.sub(r"\$([A-Za-z]+)", r"__field_\1", expression.canonical), mode="eval")
        features = np.zeros(len(LAYOUT))
        features[INDEX["dialect:"+expression.dialect]] = 1
        features[INDEX["depth"]], features[INDEX["nodes"]] = expression.depth, expression.nodes
        numbers = {role: [] for role in ROLES}

        def kind(node):
            if isinstance(node, ast.Name): return "field"
            if isinstance(node, ast.Call): return "operator"
            if isinstance(node, ast.UnaryOp): return "unary"
            if isinstance(node, ast.Constant): return "group" if isinstance(node.value, str) else "literal"
            return "binary"

        def visit(node, role="value"):
            if isinstance(node, ast.Call):
                op = node.func.id
                features[INDEX["op:"+op]] += 1
                signature = REGISTRY[expression.dialect][op]
                for position, child in enumerate(node.args):
                    features[INDEX[f"edge:{op}:{position}:{kind(child)}"]] += 1
                    visit(child, signature[position].removesuffix("?").split(":")[-1])
            elif isinstance(node, ast.Name):
                features[INDEX["field:"+node.id.removeprefix("__field_")]] += 1
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    features[INDEX["group:"+str(int(digest(node.value)[:8], 16) % 32)]] += 1
                else:
                    numbers[role].append(math.copysign(math.log1p(abs(node.value)), node.value))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant) and isinstance(node.op, ast.USub):
                visit(ast.Constant(-node.operand.value), role)
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.expr): visit(child, role)
        visit(tree.body)
        for role, values in numbers.items():
            if values:
                for stat, value in zip(("count", "mean", "min", "max"), (len(values), np.mean(values), min(values), max(values))):
                    features[INDEX[f"constant:{role}:{stat}"]] = value
        return SurrogateVector(tuple(features), FEATURE_VERSION, source_id=candidate.candidate_id)


class FactorSelector:
    def __init__(self, objective, *, seed=42, alpha=2.0, eta=0.25, beta=1.0, pool_size=None):
        self.objective, self.seed, self.alpha, self.eta, self.beta = objective, seed, alpha, eta, beta
        self.pool_size = pool_size
        self.gp = None

    def describe(self):
        return AcquisitionSpec("ucb", (self.objective,), "maximize", "Gumbel top-k of alpha log(q0) + eta clipped robust-z UCB")

    def fit(self, history):
        self.gp = RBFGPSurrogate(history, lengthscale=1.5, noise=1e-4, prior_mean=0, prior_std=.25,
            min_training_observations=2, feature_scale_floor=1.0, target_scale_floor=.01, feature_version=FEATURE_VERSION)
        if len(history) >= 2 and self.gp.fit_status != "fitted":
            raise ValueError("GP numerical fit failed")

    def maintain_pool(self, candidates, round_idx):
        candidates = tuple(candidates)
        if self.pool_size is None:
            return candidates, {}
        ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        mass = np.array([item.metadata["q0"] for item in ordered], dtype=float)
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0) or not np.isclose(mass.sum(), 1):
            raise ValueError("empirical q0 must sum to one before BO pool maintenance")
        if len(ordered) <= self.pool_size:
            return candidates, {"configured_bo_pool_size": self.pool_size, "bo_pool_size": len(candidates),
                                "bo_pool_candidate_ids": [item.candidate_id for item in candidates],
                                "pool_maintenance": "all_unique_candidates"}
        pool_seed = int(digest([self.seed, round_idx, [item.candidate_id for item in ordered]])[:16], 16)
        scores = np.log(mass) + np.random.default_rng(pool_seed).gumbel(size=len(ordered))
        retained = tuple(ordered[index] for index in np.argsort(-scores, kind="stable")[:self.pool_size])
        retained = tuple(sorted(retained, key=lambda item: item.candidate_id))
        return retained, {"configured_bo_pool_size": self.pool_size, "bo_pool_size": len(retained),
                          "bo_pool_candidate_ids": [item.candidate_id for item in retained],
                          "pool_seed": pool_seed, "pool_maintenance": "q0_gumbel_top_k_without_replacement"}

    def select(self, candidates, representations, *, count=1, round_idx=0, predictions=None, alpha=None, eta=None,
               pool_prepared=False):
        candidates, pool_metadata = (tuple(candidates), {}) if pool_prepared else self.maintain_pool(candidates, round_idx)
        if predictions is None:
            predictions = tuple(self.gp.predict_record(candidate.candidate_id, representations[candidate.candidate_id].values,
                                                       beta=self.beta) for candidate in candidates)
        else:
            predictions = tuple(predictions)
            if tuple(item.candidate_id for item in predictions) != tuple(item.candidate_id for item in candidates):
                raise ValueError("predictions must align with the active candidate reservoir")
        ucb = np.array([item.acquisition_score for item in predictions])
        scale = float(np.median(np.abs(ucb - np.median(ucb)))) * 1.4826
        if scale <= 1e-12: scale = float(np.std(ucb))
        z = np.clip((ucb - np.median(ucb)) / scale, -5, 5) if scale > 1e-12 else np.zeros(len(ucb))
        mass = np.array([candidate.metadata.get("q0", 1 / len(candidates)) for candidate in candidates])
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0) or (not pool_prepared and not pool_metadata
                and not np.isclose(mass.sum(), 1)):
            raise ValueError("empirical q0 must sum to one over the unique reservoir")
        mass /= mass.sum()
        logits = (self.alpha if alpha is None else alpha) * np.log(mass) + (self.eta if eta is None else eta) * z
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, round_idx]))
        perturbed = logits + rng.gumbel(size=len(candidates))
        selected = np.argsort(-perturbed, kind="stable")[:count]
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        return BOSelectionResult(tuple(candidates[index].candidate_id for index in selected), predictions,
            metadata={**pool_metadata, "q0": mass.tolist(), "logits": logits.tolist(), "probabilities": probabilities.tolist(),
                      "gumbel_scores": perturbed.tolist(), "fit": self.gp.summary(), "round": round_idx})
