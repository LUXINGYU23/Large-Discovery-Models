"""Process measures of spec §6.2 for one run, from its public side plus the export."""

import math
from collections import Counter

import numpy as np

from .tagger import Tagger, assign_clusters, cluster_labels, free_text_chars

CLEAR = {"ldm4geo": "every_k", "accumulated": "never", "restart_best": "every_k",
         "independent_sampling": "every_round", "single_conversation": "never",
         "separate_runs": "every_k", "program_evolution": "every_round"}


def clear_policy(arm):
    try:
        from ....rsi.arms import policy
        return policy(arm).clear
    except (ImportError, ValueError):
        return CLEAR.get(arm, "every_k")


def line_edit_distance(a, b):
    """Levenshtein distance over lines, normalized by the longer program (0 = identical)."""
    a, b = a.splitlines(), b.splitlines()
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / max(len(a), len(b))


def _unit(matrix):
    matrix = np.asarray(matrix, float)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def run_vectors(run, ids):
    """Stored surrogate features when every program has them, else AST features for all."""
    stored = {pid: run.encoding(pid) for pid in ids}
    if all(v is not None for v in stored.values()):
        versions = {v["version"] for v in stored.values()}
        if len(versions) == 1:
            return {pid: np.asarray(v["values"], float) for pid, v in stored.items()}, versions.pop()
    from ....core.encoding import FEATURE_VERSION, ast_features, normalized_code
    return {pid: ast_features(normalized_code(run.source(pid))) for pid in ids}, FEATURE_VERSION + "+recomputed"


def reference_vectors(run, ids, version):
    """Use the frozen feature space, even when a subset has cached encodings."""
    from ....core.encoding import FEATURE_VERSION, ast_features, normalized_code
    if version == FEATURE_VERSION + "+recomputed":
        return {pid: ast_features(normalized_code(run.source(pid))) for pid in ids}, version
    values, actual = run_vectors(run, ids)
    if actual != version:
        raise ValueError("run feature version differs from frozen cluster reference")
    return values, actual


class RunMeasures:
    """All per-run process measures. `family` is {"mode": "tags"|"cluster", "threshold": t}."""

    def __init__(self, run, outcomes=None, tagger=None, family=None, options=None):
        self.run, self.outcomes = run, outcomes
        self.tagger = tagger or Tagger()
        self.family_config = family or {"mode": "tags"}
        self.options = {"top_quarter": 0.25, "lock_min_rounds": 1, "coverage_statuses": ["valid"],
                        "free_text_min_chars": 40, **(options or {})}
        self._compute()

    # -- helpers
    def _train_iou(self, pid):
        return None if self.outcomes is None else self.outcomes.train_iou.get((self.run.experiment, self.run.run_id, pid))

    def _validation(self, pid):
        if self.outcomes is None:
            return None
        row = self.outcomes.validation.get((self.run.experiment, self.run.run_id, pid))
        return None if row is None else row.get("risk")

    def _epochs(self, last_round):
        k = self.run.reset_period
        starts = sorted({((t - 1) // k) * k + 1 for t in self.proposal_rounds})
        return [(s, min(s + k - 1, last_round)) for s in starts]

    def _compute(self):
        run, opts = self.run, self.options
        ledger = run.ledger
        proposals = run.proposals
        self.proposal_rounds = sorted(proposals)
        self.last_round = max([r["round"] for r in ledger] + self.proposal_rounds + [0])
        records = {r["program_id"]: r for r in ledger}

        # programs: every proposal plus every charged record (the initial solver, imported prefixes)
        programs = {}
        for t in self.proposal_rounds:
            for p in proposals[t]:
                programs[p["program_id"]] = {"program_id": p["program_id"], "round": t, "branch": p["branch"],
                                             "parent": p.get("parent_sha256")}
        for r in ledger:
            programs.setdefault(r["program_id"], {"program_id": r["program_id"], "round": r["round"],
                                                  "branch": r["branch"], "parent": None})
        for pid, row in programs.items():
            source = run.source(pid)
            row["tags"] = self.tagger.tags(source)
            row["free_text_chars"] = free_text_chars(source, opts["free_text_min_chars"])
            record = records.get(pid)
            row.update(evaluated=record is not None, status=record["status"] if record else None,
                       risk=record["risk"] if record else None, train_iou=self._train_iou(pid))
        order = sorted(programs, key=lambda p: (programs[p]["round"], programs[p]["branch"], p))
        expected_version = self.family_config.get("feature_version")
        if expected_version is None:
            self.vectors, self.feature_version = run_vectors(run, order)
        else:
            self.vectors, self.feature_version = reference_vectors(run, order, expected_version)
        if self.family_config.get("mode") == "cluster":
            reference = self.family_config.get("reference")
            if reference is not None:
                labels = assign_clusters([self.vectors[p] for p in order], reference["vectors"],
                                         reference["labels"], self.family_config["threshold"])
            else:
                labels = cluster_labels([self.vectors[p] for p in order], self.family_config["threshold"])
            for pid, label in zip(order, labels):
                programs[pid]["family"] = label
        else:
            for row in programs.values():
                row["family"] = self.tagger.family(row["tags"])
        self.programs = programs
        self.order = order

        # per round
        self.trajectory, rounds = [], {}
        best = math.inf
        valid_by_round = {}
        for r in ledger:
            valid_by_round.setdefault(r["round"], []).append(r)
        incumbent = None
        for t in range(0, run.rounds + 1):
            for r in valid_by_round.get(t, []):
                if r["status"] == "valid" and (incumbent is None
                                               or (r["risk"], r["program_id"]) < (incumbent["risk"], incumbent["program_id"])):
                    incumbent = r
            best = incumbent["risk"] if incumbent else math.inf
            self.trajectory.append(None if math.isinf(best) else best)
            if t == 0:
                continue
            row = {"round": t, "best_risk": self.trajectory[-1],
                   "incumbent": incumbent["program_id"] if incumbent else None,
                   "holder_train_iou": self._train_iou(incumbent["program_id"]) if incumbent else None}
            val = self._validation(incumbent["program_id"]) if incumbent else None
            row["optimism"] = None if val is None else incumbent["risk"] - val
            pool = proposals.get(t, [])
            row["proposals"] = len(pool)
            if pool:
                v = _unit([self.vectors[p["program_id"]] for p in pool])
                d = 1.0 - v @ v.T
                iu = np.triu_indices(len(pool), 1)
                row["dispersion"] = float(d[iu].mean()) if len(pool) > 1 else 0.0
                row["active_families"] = len({programs[p["program_id"]]["family"] for p in pool})
                edits = [line_edit_distance(run.stripped(p["parent_sha256"]), run.stripped(p["program_id"]))
                         for p in pool if p.get("parent_sha256") and run.exists(f"candidates/{p['parent_sha256']}/solver.py")]
                row["edit_size"] = float(np.mean(edits)) if edits else None
                row["edit_n"] = len(edits)
                stated = {p["program_id"]: p for p in pool}
                covered = [stated[r["program_id"]]["interval80_low"] <= r["risk"] <= stated[r["program_id"]]["interval80_high"]
                           for r in valid_by_round.get(t, []) if r["program_id"] in stated
                           and r["status"] in opts["coverage_statuses"] and "interval80_low" in stated[r["program_id"]]]
                row["coverage"] = float(np.mean(covered)) if covered else None
                row["coverage_n"] = len(covered)
            else:
                row.update(dispersion=None, active_families=None, edit_size=None, edit_n=0,
                           coverage=None, coverage_n=0)
            usage = run.usage.get(t)
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "compactions", "requests"):
                row[key] = None if usage is None else sum(u.get(key, 0) for u in usage.values())
            rounds[t] = row
        self.rounds = rounds
        self.compactions_by_branch = Counter()
        for usage in run.usage.values():
            for branch, u in usage.items():
                self.compactions_by_branch[branch] += u.get("compactions", 0)

        # hidden training IoU against training risk
        pairs = [(row["risk"], row["train_iou"]) for row in programs.values()
                 if row["evaluated"] and row["status"] == "valid" and row["train_iou"] is not None]
        from .stats import spearman
        self.train_iou_spearman = spearman(*zip(*pairs)) if len(pairs) >= 3 else None
        self.train_iou_n = len(pairs)

        self._branch_agreement()
        self._reentry()
        self._lockin()

    def _branch_families(self, t, branch):
        return [self.programs[p["program_id"]]["family"] for p in self.run.proposals.get(t, [])
                if p["branch"] == branch]

    def _branches(self):
        return sorted({p["branch"] for t in self.proposal_rounds for p in self.run.proposals[t]})

    def _branch_agreement(self):
        self.branch_epochs = []
        k = self.run.reset_period
        for start, end in self._epochs(self.last_round):
            if end != start + k - 1 or end not in self.run.proposals:
                continue  # only completed epochs
            seen = [p for p in self.programs.values() if p["evaluated"] and p["round"] <= end and p["status"] == "valid"]
            validated = [(self._validation(p["program_id"]), p["program_id"]) for p in seen
                         if self._validation(p["program_id"]) is not None]
            if validated:
                reference, source = self.programs[min(validated)[1]]["family"], "validation"
            elif seen:
                holder = min(seen, key=lambda p: (p["risk"], p["program_id"]))
                reference, source = holder["family"], "training"
            else:
                continue
            for branch in self._branches():
                families = Counter(self._branch_families(end, branch))
                if not families:
                    continue
                majority = min(families, key=lambda f: (-families[f], f))
                self.branch_epochs.append({"epoch_end": end, "branch": branch, "family": majority,
                                           "reference_family": reference, "reference": source,
                                           "off_best": int(majority != reference)})

    def _reentry(self):
        self.reentry = []
        if clear_policy(self.run.arm) != "every_k":
            return
        quarter = self.options["top_quarter"]
        for start, end in self._epochs(self.last_round):
            if start == 1:
                continue
            for branch in self._branches():
                hit = None
                for t in range(start, end + 1):
                    predictions = self.run.selection.get(t)
                    if not predictions:
                        continue
                    ranked = sorted(predictions.values(), key=lambda p: (-p["acquisition"], p["program_id"]))
                    top = {p["program_id"] for p in ranked[:max(1, math.ceil(quarter * len(ranked)))]}
                    if any(p["branch"] == branch and p["program_id"] in top for p in self.run.proposals.get(t, [])):
                        hit = t
                        break
                self.reentry.append({"epoch_start": start, "branch": branch,
                                     "r": (hit - start) if hit else end - start + 1, "censored": hit is None})

    def _lockin(self):
        self.lockin = self._lock_spells(self.options["lock_min_rounds"])
        # Reported sensitivity: with 1, one all-one-family final round already counts as lock-in.
        self.lockin_sensitivity = self._lock_spells(self.options.get("lock_min_rounds_sensitivity", 2))

    def _lock_spells(self, minimum):
        spells = []
        for start, end in self._epochs(self.last_round):
            for branch in self._branches():
                per_round = [set(self._branch_families(t, branch)) for t in range(start, end + 1)]
                if any(not f for f in per_round):
                    continue
                lock = None
                for i in range(len(per_round)):
                    tail = per_round[i:]
                    if len(tail) >= minimum and all(len(f) == 1 for f in tail) and len(set().union(*tail)) == 1:
                        lock = i
                        break
                spells.append({"epoch_start": start, "branch": branch,
                               "time": lock if lock is not None else len(per_round), "event": lock is not None})
        return spells

    # -- E2 support
    def tag_share(self, target, t):
        pool = self.run.proposals.get(t, [])
        if not pool:
            return None
        if self.family_config.get("mode") == "cluster":
            return float(np.mean([self.programs[p["program_id"]]["family"] == target for p in pool]))
        return float(np.mean([target in self.programs[p["program_id"]]["tags"] for p in pool]))
