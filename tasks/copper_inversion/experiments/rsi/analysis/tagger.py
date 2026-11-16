"""Design tags, program families and the clustering fallback (spec §6.2, H7).

Rules are data: a JSON rule set is hashed and frozen at G3. The draft rules
below are placeholders until the rehearsal calibration (AT-35).
"""

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

DRAFT_RULES = {
    "version": "draft-v0",
    "rules": [
        {"name": "fast_marching", "top_level": True,
         "patterns": [r"fast_?march", r"\bfmm\b", r"skfmm", r"eikonal", r"travel_?time"]},
        {"name": "annealing", "top_level": True, "require_loop": True,
         "patterns": [r"anneal", r"temperature", r"\btemp\b", r"metropolis"]},
        {"name": "random_search", "top_level": True, "require_loop": True,
         "patterns": [r"^(random|rand|randn|uniform|normal|choice|integers|default_rng)$"]},
        {"name": "implicit_interpolation", "top_level": True,
         "patterns": [r"rbf", r"griddata", r"krig", r"\bidw\b", r"interpolat", r"implicit"]},
        {"name": "convex_optimization", "top_level": True,
         "patterns": [r"^(linprog|lsq_linear|nnls|lstsq|least_squares|minimize|cvxpy|quadprog)$"]},
        {"name": "level_set", "top_level": True,
         "patterns": [r"distance_transform", r"signed_distance", r"level_?set", r"\bsdf\b"]},
        {"name": "morphology", "top_level": False,
         "patterns": [r"binary_(closing|opening|dilation|erosion|fill_holes)", r"morpholog",
                      r"^label$"]},
    ],
}


def rules_sha256(rules):
    return hashlib.sha256(json.dumps(rules, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_rules(path=None):
    if path is None:
        return DRAFT_RULES
    rules = json.loads(Path(path).read_text())
    names = [r["name"] for r in rules["rules"]]
    if len(set(names)) != len(names) or any(not r.get("patterns") for r in rules["rules"]):
        raise ValueError("tag rules need unique names and at least one pattern")
    return rules


def _identifiers(tree):
    """(identifier, inside_loop) for names, attributes, calls, defs, imports and keywords."""
    found = []

    def visit(node, loop):
        inside = loop or isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.comprehension))
        if isinstance(node, ast.Name):
            found.append((node.id, inside))
        elif isinstance(node, ast.Attribute):
            found.append((node.attr, inside))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.append((node.name, inside))
        elif isinstance(node, ast.alias):
            for part in node.name.split("."):
                found.append((part, inside))
            if node.asname:
                found.append((node.asname, inside))
        elif isinstance(node, ast.ImportFrom) and node.module:
            for part in node.module.split("."):
                found.append((part, inside))
        elif isinstance(node, ast.keyword) and node.arg:
            found.append((node.arg, inside))
        elif isinstance(node, ast.arg):
            found.append((node.arg, inside))
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(tree, False)
    return found


class Tagger:
    def __init__(self, rules=None):
        self.rules = rules or DRAFT_RULES
        self.sha256 = rules_sha256(self.rules)
        self._compiled = [(r["name"], bool(r.get("top_level", True)), bool(r.get("require_loop")),
                           [re.compile(p, re.IGNORECASE) for p in r["patterns"]])
                          for r in self.rules["rules"]]
        self.top_level = [name for name, top, _, _ in self._compiled if top]

    def tags(self, source):
        tree = ast.parse(source.decode("utf-8") if isinstance(source, bytes) else source)
        names = _identifiers(tree)
        out = []
        for name, _, loop_only, patterns in self._compiled:
            if any(any(p.search(ident) for p in patterns) and (inside or not loop_only)
                   for ident, inside in names):
                out.append(name)
        return sorted(out)

    def family(self, tags):
        top = [t for t in tags if t in self.top_level]
        return "+".join(sorted(top)) if top else "untagged"


def free_text_chars(source, minimum=40):
    """Total length of string constants longer than `minimum` characters (contracts §4.2)."""
    tree = ast.parse(source.decode("utf-8") if isinstance(source, bytes) else source)
    return sum(len(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > minimum)


# -- clustering fallback (spec §6.2 step 3); needs numpy and scipy

def cluster_labels(vectors, threshold):
    """Average-linkage cosine clusters cut at `threshold`; labels c0, c1, ... by first appearance."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    matrix = np.asarray(vectors, dtype=float)
    if len(matrix) == 1:
        return ["c0"]
    raw = fcluster(linkage(matrix, method="average", metric="cosine"), t=threshold, criterion="distance")
    order = {}
    for value in raw:
        order.setdefault(value, f"c{len(order)}")
    return [order[value] for value in raw]


def assign_clusters(vectors, reference, labels, threshold):
    """Assign new vectors to the reference cluster with the smallest average cosine
    distance when that distance is within `threshold`, else to "unassigned"."""
    import numpy as np
    ref = np.asarray(reference, dtype=float)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    names = sorted(set(labels), key=lambda s: int(s[1:]))
    out = []
    for vector in np.asarray(vectors, dtype=float):
        vector = vector / np.linalg.norm(vector)
        distance = 1.0 - ref @ vector
        means = [(float(distance[[l == n for l in labels]].mean()), n) for n in names]
        best, name = min(means)
        out.append(name if best <= threshold else "unassigned")
    return out


def calibrate_threshold(round_vectors_by_run, grid=None, low=2, high=6):
    """Choose the cut for which the median round of the rehearsal runs has low..high clusters.

    round_vectors_by_run: [ [ (round, vector), ... ] per run ]. Among cuts that meet
    the condition, take the one whose median is closest to the middle; ties go to
    the smaller cut. Returns (threshold, {threshold: median}).
    """
    import numpy as np
    grid = grid or [round(x, 3) for x in np.arange(0.02, 1.0, 0.02)]
    medians = {}
    for threshold in grid:
        counts = []
        for rows in round_vectors_by_run:
            labels = cluster_labels([v for _, v in rows], threshold)
            by_round = {}
            for (r, _), label in zip(rows, labels):
                by_round.setdefault(r, set()).add(label)
            counts.extend(len(v) for v in by_round.values())
        medians[threshold] = float(np.median(counts))
    middle = (low + high) / 2
    eligible = [t for t in grid if low <= medians[t] <= high]
    if not eligible:
        raise ValueError("no clustering threshold gives the required number of clusters")
    return min(eligible, key=lambda t: (abs(medians[t] - middle), t)), medians


# -- calibration against human labels (AT-35, after the rehearsal)

def cohen_kappa(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal, nonempty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def decide_mode(kappa, largest_family_share, kappa_min=0.7, concentration_max=0.9):
    """spec §6.2 step 3: fall back to clustering on low agreement or one dominant family."""
    return "cluster" if kappa < kappa_min or largest_family_share > concentration_max else "tags"


def select_target_tag(tag_lists, best_tags, low=0.3, high=0.9):
    """E2 rule v1: among tags of the current best program whose share of the
    proposals (rounds 9 and 10) lies in [low, high], take the most frequent; ties by name."""
    n = len(tag_lists)
    share = {tag: sum(tag in tags for tags in tag_lists) / n for tag in best_tags}
    eligible = [t for t in sorted(best_tags) if low <= share[t] <= high]
    if not eligible:
        return None
    return min(eligible, key=lambda t: (-share[t], t))


def checkpoint_target_tag(run_dir, round_index=10, rules=None):
    """E2 target for a checkpoint (contracts §8 selection_rule "v1"): proposals of rounds
    round_index - 1 and round_index, and the best valid record up to round_index."""
    from .rundir import Run
    run = Run(run_dir)
    t = Tagger(rules)
    tag_lists = [t.tags(run.source(p["program_id"])) for r in (round_index - 1, round_index)
                 for p in run.proposals.get(r, [])]
    valid = [r for r in run.ledger if r["status"] == "valid" and r["round"] <= round_index]
    if not tag_lists or not valid:
        return None
    best = min(valid, key=lambda r: (r["risk"], r["program_id"]))
    return select_target_tag(tag_lists, t.tags(run.source(best["program_id"])))
