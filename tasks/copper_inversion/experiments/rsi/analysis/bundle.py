"""Build the results bundle of contracts §9.2 from an archive and the outcome export.

    python -m tasks.copper_inversion.experiments.rsi.analysis.bundle \
        --archive ARCHIVE --outcomes OUTCOMES.json --out RESULTS_DIR [--config CONFIG.json]

ARCHIVE holds <exp>/<run_id>/ public run directories. Nothing is written outside
--out, and the same inputs, configuration and code give byte-identical files.
"""

import argparse
import copy
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from . import stats
from .measures import RunMeasures
from .outcomes import Outcomes
from .rundir import discover
from .tagger import DRAFT_RULES, Tagger

PACKAGE = Path(__file__).resolve().parent
ARMS = {  # internal name: (results.json name, group, inherits)
    "ldm4geo": ("LDM4Geo", "core", "history"),
    "accumulated": ("Accumulated context", "core", "conversation + history"),
    "restart_best": ("Restart from best", "core", "best program"),
    "independent_sampling": ("Independent sampling", "core", "nothing"),
    "single_conversation": ("Single conversation", "extended", "one conversation"),
    "separate_runs": ("Separate runs", "extended", "nothing shared"),
    "program_evolution": ("Program evolution", "extended", "population"),
}
CORE = [a for a, v in ARMS.items() if v[1] == "core"]
DEFAULTS = {
    "seed": 20261115, "n_boot": 10000, "n_perm": 20000, "alpha": 0.05, "level": 0.95,
    "mde_z": 2.80, "mde_ns": [5, 8, 12, 16], "mde_bound_relative": 0.20, "pilot_seeds": 4,
    "e3_standard_seeds": 8, "budget_rounds": [5, 10], "checkpoint_round": 10, "checkpoint_experiment": "M1", "e2_late_rounds": [16, 20],
    "e1_curve_rounds": [12, 14, 15, 16, 18],
    # Prefix freezes each experiment's batches must request (frozen at G3 with this file; launch.py checks).
    "freeze_prefixes": {"M1": ["full", 5, 10], "E1": ["full", 12, 14, 15, 16, 18], "E2": ["full"],
                        "E3": ["full"], "X1": ["full"], "F1": ["full"]},
    "judges": {"quarter": "q25", "all": "standard", "oracle": "oracle"},
    "primary": ["ldm4geo", "accumulated"],
    "secondary": [["ldm4geo", "restart_best", "test_id"], ["ldm4geo", "independent_sampling", "test_id"],
                  ["ldm4geo", "accumulated", "test_ood"], ["ldm4geo", "restart_best", "test_ood"],
                  ["ldm4geo", "independent_sampling", "test_ood"]],
    "family": {"mode": "tags", "threshold": None},
    "tagger_rules": DRAFT_RULES,
    "measures": {"top_quarter": 0.25, "lock_min_rounds": 1, "lock_min_rounds_sensitivity": 2, "coverage_statuses": ["valid"],
                 "free_text_min_chars": 40},
    "round_digits": 6,
}


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        config.update(json.loads(Path(path).read_text()))
    return config


def package_sha256():
    h = hashlib.sha256()
    for path in sorted(PACKAGE.glob("*")):
        if path.suffix in (".py", ".sh") and path.is_file():
            h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _clean(value, digits):
    if isinstance(value, dict):
        return {str(k): _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    if isinstance(value, (bool, type(None), str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def _column_means(rows):
    if not rows:
        return []
    return [_mean(c) for c in itertools.zip_longest(*rows)]


class Analysis:
    def __init__(self, archive, outcomes, config):
        self.config = config
        self.archive = Path(archive)
        self.outcomes = outcomes
        self.tagger = Tagger(config["tagger_rules"])
        self.runs = discover(archive)
        self.measures = {}
        family = config["family"]
        primary = [r for r in self.runs if not r.continuation]
        for run in primary + [r for r in self.runs if r.continuation]:
            fam = dict(family)
            if family["mode"] == "cluster" and run.continuation:
                fam["reference"] = self._cluster_reference(run.continuation["source_run"])
            self.measures[run.run_id] = RunMeasures(run, outcomes, self.tagger, fam, config["measures"])
        self.by_exp = {}
        for run in self.runs:
            self.by_exp.setdefault(run.experiment, []).append(run)
        for runs in self.by_exp.values():
            runs.sort(key=lambda r: (r.arm or "", r.seed if r.seed is not None else -1, r.run_id))
        self.missing = []

    def _cluster_reference(self, source_id):
        source = self.measures.get(source_id)
        if source is None:
            return None
        cut = self.config["checkpoint_round"]
        ids = [p for p in source.order if source.programs[p]["round"] <= cut]
        return {"vectors": [source.vectors[p] for p in ids], "labels": [source.programs[p]["family"] for p in ids]}

    def rng(self, label):
        return stats.stream(self.config["seed"], label)

    # -- outcome access
    def final_round(self, run):
        return self.outcomes.final(run.experiment, run.run_id)

    def sealed(self, run, prefix, split, metric="iou"):
        return None if prefix is None else self.outcomes.sealed(run.experiment, run.run_id, prefix, split, metric,
                                                                 rounds=run.rounds)

    def validation(self, run, prefix, metric="risk"):
        return None if prefix is None else self.outcomes.validation_value(run.experiment, run.run_id, prefix,
                                                                           metric, rounds=run.rounds)

    def freeze(self, run, prefix):
        return None if prefix is None else self.outcomes.freeze(run.experiment, run.run_id, prefix, run.rounds)

    def sealed_matrix(self, runs, split, final=True, round_index=None):
        """(run-level values, runs x bodies matrix) for runs that have the outcome."""
        values, rows = [], []
        for run in runs:
            r = self.final_round(run) if final else round_index
            value = self.sealed(run, r, split)
            if value is None:
                self.missing.append({"run_id": run.run_id, "experiment": run.experiment,
                                     "outcome": f"{split}@{'final' if final else round_index}"})
                continue
            values.append(value)
            rows.append(self.outcomes.body_row(run.experiment, run.run_id, r, split, rounds=run.rounds))
        if rows and all(row is not None for row in rows) and len({len(row) for row in rows}) == 1:
            matrix = np.asarray(rows, float)
        else:
            matrix = np.asarray(values, float).reshape(-1, 1)
        return values, matrix

    def arm_runs(self, experiment, arm, judge=None):
        return [r for r in self.by_exp.get(experiment, []) if r.arm == arm and not r.intervention
                and (judge is None or r.judge == judge)]

    def contrast(self, label, groups, coefs):
        draws = stats.bootstrap_contrast(groups, coefs, self.config["n_boot"], self.rng(label))
        return stats.t_interval(stats.contrast_estimate(groups, coefs), draws, stats.contrast_df(groups),
                                self.config["level"])

    def body_method(self, matrices):
        bodies = {m.shape[1] for m in matrices}
        return "runs_and_paired_bodies" if len(bodies) == 1 and bodies.pop() > 1 else "runs_only"

    # -- blocks
    def e4(self):
        out = {}
        for arm, (name, group, inherits) in ARMS.items():
            runs = self.arm_runs("M1" if group == "core" else "X1", arm)
            same, ms = self.sealed_matrix(runs, "test_id")
            held, mh = self.sealed_matrix(runs, "test_ood")
            entry = {"group": group, "n": len(runs), "inherits": inherits, "same": same, "held_out": held,
                     "seeds": [r.seed for r in runs]}
            for key, values, matrix in (("same", same, ms), ("held", held, mh)):
                if values:
                    ci = self.contrast(f"E4/{arm}/{key}", [matrix], [1])
                    entry[f"{key}_mean"] = mean(values)
                    entry[f"{key}_hw"] = (ci[1] - ci[0]) / 2
                    entry[f"{key}_ci95"] = ci
                else:
                    entry.update({f"{key}_mean": None, f"{key}_hw": None, f"{key}_ci95": None})
            out[name] = entry
        return out

    def comparison(self, a, b, split, experiment="M1"):
        x, mx = self.sealed_matrix(self.arm_runs(experiment, a), split)
        y, my = self.sealed_matrix(self.arm_runs(experiment, b), split)
        if len(x) < 2 or len(y) < 2:
            return {"diff": None, "ci95": None, "perm_p": None, "n": [len(x), len(y)]}
        label = f"{experiment}/{a}-{b}/{split}"
        return {"diff": mean(x) - mean(y), "ci95": self.contrast(label, [mx, my], [1, -1]),
                "perm_p": stats.permutation_test(x, y, self.config["n_perm"], self.rng(label + "/perm")),
                "n": [len(x), len(y)], "interval_method": self.body_method([mx, my])}

    def e4_primary(self):
        a, b = self.config["primary"]
        return {"id": self.comparison(a, b, "test_id"), "shift": self.comparison(a, b, "test_ood")}

    def e4_secondary(self):
        rows = [{"comparison": f"{a} - {b}", "split": split, **self.comparison(a, b, split)}
                for a, b, split in self.config["secondary"]]
        tested = [i for i, r in enumerate(rows) if r["perm_p"] is not None]
        adjusted = stats.holm([rows[i]["perm_p"] for i in tested])
        for i, p in zip(tested, adjusted):
            rows[i]["p_holm"] = p
        return rows

    def pilot_values(self):
        values = []
        for arm in self.config["primary"]:
            runs = sorted(self.arm_runs("M1", arm), key=lambda r: r.seed)[:self.config["pilot_seeds"]]
            group = []
            for run in runs:
                r = self.final_round(run)
                v = self.validation(run, r, "iou")
                if v is not None:
                    group.append(v)
            values.append(group)
        return values

    def pilot(self):
        groups = self.pilot_values()
        s_v = stats.pooled_sd(groups)
        pooled = [v for g in groups for v in g]
        if s_v is None or not pooled:
            return {"s_v": None, "m_v": None, "bound": None, "mde_n12": None, "mde_n16": None, "runs_per_arm": None}
        m_v = mean(pooled)
        bound = self.config["mde_bound_relative"] * m_v
        z = self.config["mde_z"]
        decision = stats.replicate_decision(s_v, bound, z)
        return {"s_v": s_v, "m_v": m_v, "bound": bound, "mde_n12": stats.mde(s_v, 12, z),
                "mde_n16": stats.mde(s_v, 16, z), "runs_per_arm": decision["runs_per_arm"],
                "primary_mode": decision["primary"]}

    def f1(self):
        runs, matrices = {}, []
        for arm in self.config["primary"]:
            values, matrix = self.sealed_matrix(self.arm_runs("F1", arm), "test_id")
            runs[ARMS[arm][0]] = values
            matrices.append(matrix)
        names = [ARMS[a][0] for a in self.config["primary"]]
        out = {"runs": runs, "mean": {k: _mean(v) for k, v in runs.items()}}
        if all(len(runs[n]) >= 2 for n in names):
            out["diff"] = mean(runs[names[0]]) - mean(runs[names[1]])
            out["ci95"] = self.contrast("F1", matrices, [1, -1])
        else:
            out.update(diff=None, ci95=None)
        return out

    # -- E1 / E2
    def continuations(self, experiment, planted):
        rows = []
        for run in self.by_exp.get(experiment, []):
            if run.continuation and bool(run.intervention) == planted:
                rows.append(run)
        return sorted(rows, key=lambda r: (r.continuation["checkpoint"], r.continuation["mode"],
                                           r.continuation.get("replicate", 0), r.run_id))

    def e1(self):
        runs = self.continuations("E1", False)
        cut = self.config["checkpoint_round"]
        checkpoints = {}
        for run in runs:
            ck, source = run.continuation["checkpoint"], run.continuation["source_run"]
            checkpoints[ck] = self.outcomes.validation_value(self.config["checkpoint_experiment"], source, cut, "risk")
        modes = ("conversation", "history", "best_program")
        trajectories, finals, labels = {m: [] for m in modes}, {}, []
        rounds = max([r.rounds for r in runs] + [20])
        curve = [cut] + [t for t in self.config["e1_curve_rounds"] if cut < t < rounds] + [rounds]
        for mode in modes:
            for run in runs:
                if run.continuation["mode"] != mode:
                    continue
                ck = run.continuation["checkpoint"]
                if mode == "conversation":
                    labels.append(ck)
                path = [checkpoints[ck]] + [self.validation(run, t, "risk") for t in curve[1:]]
                trajectories[mode].append(path)
                final = self.final_round(run)
                value = self.validation(run, final, "risk")
                if value is None:
                    self.missing.append({"run_id": run.run_id, "experiment": "E1", "outcome": "validation@final"})
                    continue
                finals.setdefault((mode, ck), []).append(value)
        cks = sorted(checkpoints)
        out = {"checkpoints": checkpoints, "checkpoint": labels, "rounds": curve, "runs": trajectories,
               "final_mean": {m: _mean([v for (mm, _), vs in finals.items() if mm == m for v in vs]) for m in modes}}
        s0 = stats.pooled_sd(list(finals.values()))
        strata = [(finals[("history", c)], finals[("conversation", c)]) for c in cks
                  if len(finals.get(("history", c), [])) and len(finals.get(("conversation", c), []))]
        if strata and s0 is not None:
            diff = float(np.mean([np.mean(b) - np.mean(a) for b, a in strata]))
            upper = stats.shift_upper_bound(strata, alpha=self.config["alpha"])
            out.update(s0=s0, diff_history_minus_conversation=diff, upper95_one_sided=upper, margin=s0,
                       verdict=stats.e1_verdict(diff, upper, s0), perm_p_two_sided=stats.stratified_p(strata))
        else:
            out.update(s0=s0, diff_history_minus_conversation=None, upper95_one_sided=None, margin=s0,
                       verdict="no_data", perm_p_two_sided=None)
        out["diff_by_checkpoint"] = {c: float(np.mean(finals[("history", c)]) - np.mean(finals[("conversation", c)]))
                                     for c in cks if finals.get(("history", c)) and finals.get(("conversation", c))}
        best = [(finals[("best_program", c)], finals[("history", c)]) for c in cks
                if finals.get(("best_program", c)) and finals.get(("history", c))]
        out["diff_best_minus_history"] = float(np.mean([np.mean(b) - np.mean(a) for b, a in best])) if best else None
        out["active_families"] = {}
        for mode in modes:
            values = [row["active_families"] for run in runs if run.continuation["mode"] == mode
                      for t, row in self.measures[run.run_id].rounds.items() if t > cut and row["active_families"] is not None]
            out["active_families"][mode] = _mean(values)
        self.e1_rows = [{"run_id": r.run_id, "checkpoint": r.continuation["checkpoint"], "mode": r.continuation["mode"],
                         "replicate": r.continuation.get("replicate"),
                         "validation_risk": self.validation(r, self.final_round(r), "risk"),
                         "checkpoint_risk": checkpoints[r.continuation["checkpoint"]]} for r in runs]
        return out

    def e2(self):
        planted = self.continuations("E2", True)
        controls = self.continuations("E1", False)
        first = min([r.intervention["round"] for r in planted] or [self.config["checkpoint_round"] + 1])
        k = planted[0].reset_period if planted else 5
        rounds = list(range(first, first + 10))
        out_runs, strata, self.e2_rows = {}, {}, []
        for mode in ("conversation", "history"):
            out_runs[mode] = []
            for run in planted:
                if run.continuation["mode"] != mode:
                    continue
                ck, tag = run.continuation["checkpoint"], run.intervention["target_tag"]
                peers = [c for c in controls if c.continuation["checkpoint"] == ck and c.continuation["mode"] == mode]
                share = [self.measures[run.run_id].tag_share(tag, t) for t in rounds]
                control = [_mean([self.measures[c.run_id].tag_share(tag, t) for c in peers]) for t in rounds]
                # shift away from the target tag: control share minus planted share
                row = [None if s is None or c is None else c - s for s, c in zip(share, control)]
                out_runs[mode].append(row)
                control_rows, plant_rows = strata.setdefault((mode, ck), ([], []))
                plant_rows.append(share)
                if not control_rows:
                    control_rows.extend([self.measures[c.run_id].tag_share(tag, t) for t in rounds] for c in peers)
                self.e2_rows.extend({"run_id": run.run_id, "checkpoint": ck, "mode": mode, "target_tag": tag,
                                     "round": t, "planted_share": s, "control_share": c, "excess": d}
                                    for t, s, c, d in zip(rounds, share, control, row))
        means = {m: _column_means(rows) for m, rows in out_runs.items()}
        lo, hi = self.config["e2_late_rounds"]
        late_idx = [i for i, t in enumerate(rounds) if lo <= t <= hi]
        out = {"reset_period": k, "first_round": first, "runs": out_runs, "mean": means,
               "sign": "control share minus planted share of proposals carrying the target tag",
               "late_mean_rounds_16_20": {m: _mean([v[i] for i in late_idx]) if v else None for m, v in means.items()},
               "half_life_rounds": {m: stats.half_life(v) for m, v in means.items()}}
        out["late_ci95"], out["ci95_by_round"] = {}, {}
        for mode in out_runs:
            cells = [(c, p) for (m, _), (c, p) in sorted(strata.items()) if m == mode and p and c
                     and all(v is not None for row in p + c for v in row)]
            if not cells:
                out["late_ci95"][mode], out["ci95_by_round"][mode] = None, None
                continue
            draws = stats.stratified_bootstrap_rows(cells, self.config["n_boot"], self.rng(f"E2/{mode}"))
            est, df, level = stats.stratified_estimate(cells), stats.stratified_df(cells), self.config["level"]
            out["ci95_by_round"][mode] = [stats.t_interval(float(est[i]), draws[:, i], df, level) for i in range(len(rounds))]
            out["late_ci95"][mode] = stats.t_interval(float(est[late_idx].mean()), draws[:, late_idx].mean(axis=1), df, level)
        return out

    def coverage(self):
        out = {}
        for arm in CORE:
            runs = self.arm_runs("M1", arm)
            if not runs:
                continue
            rows = [[self.measures[r.run_id].rounds.get(t, {}).get("coverage") for t in range(1, r.rounds + 1)] for r in runs]
            out[arm] = _column_means(rows)
        return out

    def e3(self):
        block, gaps, matrices, self.e3_rows = {}, {}, {}, []
        for key, judge in sorted(self.config["judges"].items()):  # config key order is not stable
            block[key] = {}
            for arm in self.config["primary"]:
                if judge == "standard":
                    runs = sorted(self.arm_runs("M1", arm), key=lambda r: r.seed)[:self.config["e3_standard_seeds"]]
                else:
                    runs = self.arm_runs("E3", arm, judge)
                values, matrix = self.sealed_matrix(runs, "test_id")
                block[key][arm] = values
                matrices[(key, arm)] = matrix
                self.e3_rows.extend({"judge": key, "arm": arm, "run_id": r.run_id, "seed": r.seed,
                                     "sealed_iou": self.sealed(r, self.final_round(r), "test_id")}
                                    for r in runs)
            a, b = self.config["primary"]
            gaps[key] = (mean(block[key][a]) - mean(block[key][b])) if block[key][a] and block[key][b] else None
        a, b = self.config["primary"]
        weak, oracle = "quarter", "oracle"
        if gaps.get(weak) is not None and gaps.get(oracle) is not None and all(
                len(block[j][x]) >= 2 for j in (weak, oracle) for x in (a, b)):
            groups = [matrices[(weak, a)], matrices[(weak, b)], matrices[(oracle, a)], matrices[(oracle, b)]]
            ci = self.contrast("E3/gap", groups, [1, -1, -1, 1])
            diff = {"diff": gaps[weak] - gaps[oracle], "ci95": ci, "interval_method": self.body_method(groups)}
        else:
            diff = {"diff": None, "ci95": None}
        return block, gaps, diff

    # -- Proposition 3 and branch agreement
    def _prop3_from(self, measures):
        reentry = [x for m in measures for x in m.reentry]
        lock = [x for m in measures for x in m.lockin]
        r = _mean([x["r"] for x in reentry])
        h = stats.exponential_rate([x["time"] for x in lock], [x["event"] for x in lock])
        return r, h, stats.k_star(r, h), reentry, lock

    def prop3(self):
        arm = self.config["primary"][0]
        measures = [self.measures[r.run_id] for r in self.arm_runs("M1", arm)]
        r, h, k, reentry, lock = self._prop3_from(measures)
        out = {"r": r, "h": h, "K_star": k, "K_used": measures[0].run.reset_period if measures else None,
               "hr": None if r is None or h is None else r * h,
               "n_reentries": len(reentry), "n_reentries_censored": sum(x["censored"] for x in reentry),
               "n_lock_spells": len(lock), "n_lock_events": sum(x["event"] for x in lock)}
        spells = [x for m in measures for x in m.lockin_sensitivity]
        h2 = stats.exponential_rate([x["time"] for x in spells], [x["event"] for x in spells]) if spells else None
        out["sensitivity"] = {"lock_min_rounds": self.config["measures"]["lock_min_rounds_sensitivity"], "h": h2,
                              "K_star": None if r is None or h2 is None else stats.k_star(r, h2),
                              "n_lock_events": sum(x["event"] for x in spells)}
        if len(measures) >= 2:
            rng = self.rng("prop3")
            draws = []
            for _ in range(self.config["n_boot"] // 5):
                pick = [measures[i] for i in rng.integers(0, len(measures), len(measures))]
                draws.append(self._prop3_from(pick)[:3])
            for i, key in enumerate(("r", "h", "K_star")):
                finite = [d[i] for d in draws if d[i] is not None and math.isfinite(d[i])]
                out[f"{key}_ci95"] = stats.interval(finite, self.config["level"]) if finite else None
            out["K_star_undefined_share"] = sum(d[2] is None for d in draws) / len(draws)
        out["by_arm"] = {}
        for other in CORE:
            ms = [self.measures[x.run_id] for x in self.arm_runs("M1", other)]
            if ms:
                rr, hh, kk, _, _ = self._prop3_from(ms)
                out["by_arm"][other] = {"r": rr, "h": hh, "K_star": kk}
        return out

    @staticmethod
    def agreement(measures):
        xs, pairs = [], []
        for m in measures:
            groups = {}
            for row in m.branch_epochs:
                groups.setdefault(row["epoch_end"], []).append(row["off_best"])
            for values in groups.values():
                xs.extend(values)
                pairs.extend(a * b for a, b in itertools.combinations(values, 2))
        if not xs or not pairs:
            return {"q_bar": None, "icc": None, "var_q": None, "n_branch_epochs": len(xs)}
        q = mean(xs)
        var = mean(pairs) - q * q
        return {"q_bar": q, "icc": var / (q * (1 - q)) if 0 < q < 1 else None, "var_q": var,
                "n_branch_epochs": len(xs), "n_pairs": len(pairs)}

    def branch_agreement(self):
        arm = self.config["primary"][0]
        out = self.agreement([self.measures[r.run_id] for r in self.arm_runs("M1", arm)])
        out["by_arm"] = {a: self.agreement([self.measures[r.run_id] for r in self.arm_runs("M1", a)])
                         for a in CORE if self.arm_runs("M1", a)}
        for a in [arm] + [x for x in CORE if x != arm]:
            refs = [row["reference"] for r in self.arm_runs("M1", a) for row in self.measures[r.run_id].branch_epochs]
            source = {s: refs.count(s) for s in sorted(set(refs))}
            if a == arm:
                out["reference_family_source"] = source
            if a in out["by_arm"]:
                out["by_arm"][a]["reference_family_source"] = source
        return out

    # -- main
    def main_block(self):
        trajectories, families, tokens, totals, uncached, budget = {}, {}, {}, {}, {}, {}
        m1 = self.by_exp.get("M1", [])
        rounds = max([r.rounds for r in m1] or [20])
        k = m1[0].reset_period if m1 else 5
        budget_rounds = list(self.config["budget_rounds"]) + [rounds]
        for arm in CORE:
            runs = self.arm_runs("M1", arm)
            name = ARMS[arm][0]
            if not runs:
                continue
            ms = [self.measures[r.run_id] for r in runs]
            trajectories[name] = [m.trajectory for m in ms]
            families[name] = _column_means([[m.rounds.get(t, {}).get("active_families") for t in range(1, rounds + 1)] for m in ms])
            per_round = [[m.rounds.get(t, {}).get("input_tokens") for t in range(1, rounds + 1)] for m in ms]
            tokens[name] = [None if v is None else v / 1e6 for v in _column_means(per_round)]
            totals[name] = _mean([sum(v or 0 for v in row) / 1e6 for row in per_round])
            uncached[name] = _mean([sum((row.get("input_tokens") or 0) - (row.get("cached_input_tokens") or 0)
                                        for row in m.rounds.values()) / 1e6 for m in ms])
            budget[name] = [_mean(self.sealed_matrix(runs, "test_id", final=False, round_index=b)[0]) if b != rounds
                            else _mean(self.sealed_matrix(runs, "test_id")[0]) for b in budget_rounds]
        means = {n: _column_means(v) for n, v in trajectories.items()}
        a, b = (ARMS[x][0] for x in self.config["primary"])
        crossover = None
        if a in means and b in means:
            crossover = next((t for t in range(k + 1, len(means[a]))
                              if means[a][t] is not None and means[b][t] is not None and means[a][t] < means[b][t]), None)
        initial = [m.trajectory[0] for r in m1 for m in [self.measures[r.run_id]] if m.trajectory and m.trajectory[0] is not None]
        q1 = self.q1()
        return {"R0": _mean(initial), "rounds": rounds, "reset_period": k, "trajectories": trajectories,
                "trajectory_mean": means, "final_mean": {n: v[-1] if v else None for n, v in means.items()},
                "crossover_round": crossover, "families": families, "budget_iou": budget,
                "budget_rounds": budget_rounds, "budget_fractions": [x / rounds for x in budget_rounds],
                "tokens_per_round_M": tokens, "tokens_total_M": totals, "tokens_uncached_M": uncached,
                "token_saving": (1 - totals[a] / totals[b]) if totals.get(a) is not None and totals.get(b) else None,
                "extra_rows": self.references(), "tier_spearman": q1.get("tier_spearman"),
                "layout_spearman": q1.get("layout_spearman"), "proxy_spearman": q1.get("tier_spearman"),
                "proxy_programs": q1.get("programs")}

    def references(self):
        return {name: [v.get("test_id"), v.get("test_ood")] for name, v in sorted(self.outcomes.references.items())}

    def q1(self):
        data = self.outcomes.q1
        if not data:
            return {"programs": None, "tier_spearman": None, "layout_spearman": None, "repeat_sd": None}
        rows = data.get("programs", [])
        short = [r["short_12x1"] for r in rows]
        repeats = [r["risks"] for r in data.get("repeats", []) if len(r["risks"]) > 1]
        return {"programs": len(rows), "tier_spearman": stats.spearman(short, [r["full_24x1"] for r in rows]),
                "layout_spearman": stats.spearman(short, [r["short_24x2"] for r in rows]),
                "repeat_sd": stats.pooled_sd(repeats), "repeat_programs": len(repeats), "rows": rows}

    def process(self):
        out = {}
        for arm in CORE + [a for a in ARMS if a not in CORE]:
            runs = self.arm_runs("M1" if arm in CORE else "X1", arm)
            if not runs:
                continue
            ms = [self.measures[r.run_id] for r in runs]
            rounds = max(r.rounds for r in runs)
            series = lambda key: _column_means([[m.rounds.get(t, {}).get(key) for t in range(1, rounds + 1)] for m in ms])
            out[arm] = {key: series(key) for key in ("dispersion", "active_families", "edit_size", "coverage",
                                                       "optimism", "holder_train_iou", "input_tokens",
                                                       "cached_input_tokens", "output_tokens", "compactions")}
            out[arm]["train_iou_spearman"] = {"mean": _mean([m.train_iou_spearman for m in ms]),
                                              "per_run": [m.train_iou_spearman for m in ms]}
            out[arm]["compactions_per_run"] = _mean([sum(m.compactions_by_branch.values()) for m in ms])
            out[arm]["free_text_chars_mean"] = _mean([p["free_text_chars"] for m in ms for p in m.programs.values()])
            out[arm]["feature_spaces"] = sorted({m.feature_version for m in ms})
            # Rounds whose incumbent had a validation; the rest have no optimism value (B5).
            cells = [row.get("optimism") for m in ms for row in m.rounds.values()]
            out[arm]["optimism_coverage"] = sum(v is not None for v in cells) / len(cells) if cells else None
            # A capped run freezes early; the paper reports how many runs per arm hit the cap.
            out[arm]["token_capped_runs"] = sum(r.stop_reason == "token_cap" for r in runs)
        return out

    # -- assemble
    def results(self):
        e3, gaps, gap_diff = self.e3()
        results = {
            "MOCK": False,
            "warning": "Measured results. Every number is recomputed from the archive by recompute.sh.",
            "design": "experiment-design-20261112/spec.md (v16 of the draft)",
            "E1": self.e1(), "E2": self.e2(), "coverage": self.coverage(),
            "E3": e3, "E3_gap": gaps, "E3_gap_weak_minus_oracle": gap_diff,
            "E4": self.e4(), "E4_references": self.references(), "E4_primary": self.e4_primary(),
            "E4_secondary": self.e4_secondary(), "pilot": self.pilot(), "F1": self.f1(),
            "prop3": self.prop3(), "branch_agreement": self.branch_agreement(), "main": self.main_block(),
            "process": self.process(),
            "interval_method": "hierarchical bootstrap (n-1 of n at each level): runs within arm, bodies paired "
                               "across arms (runs only where body scores are absent); estimate +- t(df = sum of "
                               "runs - 1) x bootstrap SD. Proposition 3 intervals are percentile",
        }
        missing = sorted({json.dumps(m, sort_keys=True) for m in self.missing})
        results["missing_outcomes"] = [json.loads(m) for m in missing]
        return results

    # -- tables
    def runs_table(self):
        rows = []
        for run in sorted(self.runs, key=lambda r: (r.experiment, r.run_id)):
            m = self.measures[run.run_id]
            final = self.final_round(run)
            freeze = self.freeze(run, final)
            row = {"experiment": run.experiment, "run_id": run.run_id, "arm": run.arm, "tier": run.tier,
                   "judge": run.judge, "seed": run.seed, "status": run.status, "stop_reason": run.stop_reason,
                   "freeze_round": final, "records": len(run.ledger)}
            validation = (freeze or {}).get("validation") or {}
            row["validation_risk"], row["validation_iou"] = validation.get("risk"), validation.get("iou")
            sealed = (freeze or {}).get("sealed") or {}
            for split in ("test_id", "test_ood"):
                for metric in ("iou", "risk", "surface_distance", "volume_error", "seconds_per_solve"):
                    row[f"{split}_{metric}"] = (sealed.get(split) or {}).get(metric)
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "compactions", "requests"):
                row[key] = sum(r.get(key) or 0 for r in m.rounds.values())
            row["wall_seconds"] = sum(u.get("wall_seconds", 0) or 0 for t in run.usage.values() for u in t.values())
            rows.append(row)
        return rows

    def rounds_table(self):
        rows = []
        keys = ("best_risk", "dispersion", "active_families", "coverage", "coverage_n", "edit_size", "edit_n",
                "optimism", "holder_train_iou", "input_tokens", "cached_input_tokens", "output_tokens",
                "compactions", "requests", "proposals")
        for run in sorted(self.runs, key=lambda r: (r.experiment, r.run_id)):
            for t, r in sorted(self.measures[run.run_id].rounds.items()):
                rows.append({"experiment": run.experiment, "run_id": run.run_id, "round": t,
                             **{k: r.get(k) for k in keys}})
        return rows

    def programs_table(self):
        rows = []
        for run in sorted(self.runs, key=lambda r: (r.experiment, r.run_id)):
            m = self.measures[run.run_id]
            for pid in m.order:
                p = m.programs[pid]
                if p["evaluated"]:
                    rows.append({"experiment": run.experiment, "run_id": run.run_id, "program_id": pid,
                                 "round": p["round"], "branch": p["branch"], "status": p["status"], "risk": p["risk"],
                                 "tags": "+".join(p["tags"]), "family": p["family"], "train_iou": p["train_iou"],
                                 "free_text_chars": p["free_text_chars"]})
        return rows


def _csv(rows, digits):
    if not rows:
        return b""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        clean = _clean(row, digits)
        writer.writerow({k: "" if v is None else ("true" if v is True else "false" if v is False else v)
                         for k, v in clean.items()})
    return stream.getvalue().encode()


def _json(value, digits):
    return (json.dumps(_clean(value, digits), sort_keys=True, indent=1, allow_nan=False, ensure_ascii=False) + "\n").encode()


def build(archive, outcomes_path, config, code_commit="unknown"):
    paths = Outcomes.files(outcomes_path)
    blobs = {path.name: path.read_bytes() for path in paths}
    outcomes = Outcomes([json.loads(body) for body in blobs.values()])
    analysis = Analysis(archive, outcomes, config)
    digits = config["round_digits"]
    results = analysis.results()
    files = {
        "results.json": _json(results, digits),
        "runs.csv": _csv(analysis.runs_table(), digits),
        "rounds.csv": _csv(analysis.rounds_table(), digits),
        "programs.csv": _csv(analysis.programs_table(), digits),
        "e1.csv": _csv(analysis.e1_rows, digits),
        "e2.csv": _csv(analysis.e2_rows, digits),
        "e3.csv": _csv(analysis.e3_rows, digits),
        "prop3.json": _json(results["prop3"], digits),
        "q1.json": _json(analysis.q1(), digits),
        "analysis-config.json": _json(config, 12),
        "recompute.sh": (PACKAGE / "recompute.sh").read_bytes(),
    }
    protocol = outcomes.protocol_id or next(
        (r.manifest.get("protocol_id") for r in analysis.runs if r.manifest.get("protocol_id")), None)
    provenance = {
        "protocol_id": protocol, "code_commit": code_commit, "analysis_sha256": package_sha256(),
        "analysis_config_sha256": hashlib.sha256(files["analysis-config.json"]).hexdigest(),
        "tagger_sha256": analysis.tagger.sha256, "family_mode": config["family"]["mode"],
        "outcomes_sha256": {name: hashlib.sha256(body).hexdigest() for name, body in sorted(blobs.items())},
        "unseal_log_sha256": dict(sorted(outcomes.unseal_logs.items())),
        "inputs": {f"{r.experiment}/{r.run_id}": r.provenance() for r in analysis.runs},
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in sorted(files.items())},
        "field_sources": {
            "E4, E4_primary, E4_secondary, pilot, F1, E3": "outcomes freezes (sealed and validation) + body_scores",
            "E1": "outcomes freezes (validation risk) of E1 continuations; checkpoint = source run freeze at round 10",
            "E2, coverage, main.families, branch_agreement, prop3, process": "public run directories (proposals, "
                                                                            "selection, ledger, usage) via measures.py",
            "main.trajectories": "ledger.jsonl", "main.tokens_*": "usage/round_tt.json",
            "main.*_spearman": "outcomes q1", "E4_references": "outcomes references"},
    }
    files["provenance.json"] = _json(provenance, digits)
    return files


def write(out, files):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, body in files.items():
        (out / name).write_bytes(body)
    (out / "recompute.sh").chmod(0o755)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True, help="an outcome table or a directory of tables")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--code-commit", default="unknown")
    args = parser.parse_args(argv)
    write(args.out, build(args.archive, args.outcomes, load_config(args.config), args.code_commit))
    print(json.dumps({"out": str(args.out)}))


if __name__ == "__main__":
    main()
