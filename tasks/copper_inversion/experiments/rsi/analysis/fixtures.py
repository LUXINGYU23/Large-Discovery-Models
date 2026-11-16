"""Synthetic archive for exercising the analysis end to end. NOT a scientific result.

The public run directories are written by the real controller (rsi/controller.py)
with a scripted agent, a scripted evaluator and a scripted selector, so their
format is whatever the controller currently writes. The outcome export and the
private receipts are invented with known arm effects.

    python -m tasks.copper_inversion.experiments.rsi.analysis.fixtures --out DIR [--profile small]

DIR/archive/<exp>/<run_id>/   public side
DIR/private/<exp>/<run_id>/   private feedback receipts, budget, sentinel
DIR/outcomes/<exp>.json       export_outcomes tables (rsi-outcomes-v1) with the optional T8 blocks
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from ldm_tts.contracts import Candidate
from ldm_tts.engine.run_store import atomic_json_write
from ....core.encoding import CodeEncoder
from ....rsi.controller import Config, Controller
from ....rsi.selection import reserve_and_rank

FAMILIES = ("annealing", "implicit_interpolation", "convex_optimization", "fast_marching")
QUALITY = {"annealing": .42, "implicit_interpolation": .36, "convex_optimization": .40, "fast_marching": .38}
CASES = ("case-a", "case-b")
PROFILES = {
    "small": {"m1_seeds": [101, 102, 103, 104], "checkpoints": [101, 102, 103], "replicates": 2,
              "e3_seeds": [201, 202], "f1_seeds": [301, 302], "proposals_per_branch": 2, "batch": 5,
              "rounds": 20, "reset_period": 5, "bodies": 12},
}
ARM_MU = {"ldm4geo": .20, "accumulated": .175, "restart_best": .18, "independent_sampling": .15}
MODE_ARM = {"conversation": "accumulated", "history": "ldm4geo", "best_program": "restart_best"}


def _rng(*parts):
    key = hashlib.sha256(repr(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big"))


def solver_source(family, p1, p2, morphology, docstring, marker):
    lines = ["import numpy as np", "from tasks.copper_inversion.v2.baselines import initialize", "",
             "def solve(problem, budget, seed):"]
    if docstring:
        lines.append('    """Scripted fixture solver; this long docstring exists only to test free-text counts."""')
    lines.append("    model = initialize(problem)")
    if family == "annealing":
        lines += [f"    temperature = {p1:.4f}", f"    for step in range({int(5 + 10 * p2)}):",
                  "        delta = np.random.normal()",
                  "        if delta < 0 or np.random.random() < np.exp(-delta / temperature):",
                  "            model = model", f"        temperature *= {p2:.4f}"]
    elif family == "implicit_interpolation":
        lines += ["    from scipy.interpolate import RBFInterpolator",
                  f"    field = RBFInterpolator(np.zeros((4, 3)), np.zeros(4), smoothing={p1:.4f})",
                  f"    level = {p2:.4f}"]
    elif family == "convex_optimization":
        lines += [f"    weights = np.linalg.lstsq(np.eye(3), np.ones(3) * {p1:.4f}, rcond=None)[0]",
                  f"    scale = {p2:.4f}"]
    else:
        lines += ["    def fast_marching_front(speed):", f"        return speed * {p1:.4f}",
                  f"    front = fast_marching_front({p2:.4f})"]
    if morphology:
        lines += ["    from scipy import ndimage", "    model = ndimage.binary_closing(model, iterations=1)"]
    lines += [f"    fixture_marker = {marker!r}", "    return model", ""]
    return "\n".join(lines)


def family_of(source):
    for family, needle in (("annealing", "temperature ="), ("implicit_interpolation", "RBFInterpolator"),
                           ("convex_optimization", "lstsq"), ("fast_marching", "fast_marching_front")):
        if needle in source:
            return family
    return None


def _param(source):
    found = re.findall(r"(\d+\.\d{4})", source)
    return float(found[0]) if found else .5


class FixtureAgent:
    """Lock-in grows with session age; a planted note suppresses its tag until the session ends."""

    def __init__(self, arm, seed):
        self.arm, self.seed = arm, seed
        self.avoid = {}

    def propose(self, *, branch, round_index, count, session_round, message, view, turn_root, **_):
        msg = json.loads(message)
        if msg["kind"] == "boundary" and self.arm != "accumulated":
            self.avoid.pop(branch, None)
        if "carried_note" in msg:
            self.avoid[branch] = [msg["carried_note"].split("`")[1], .95]
        rng = _rng("agent", self.arm, self.seed, branch, round_index)
        lock = 1 - math.exp(-.35 * session_round)
        current = FAMILIES[(branch + self.seed) % len(FAMILIES)]
        valid = [r for r in view if r["status"] == "valid"]
        best = min(valid, key=lambda r: (r["risk"], r["program_id"])) if valid else None
        rows = []
        for index in range(count):
            family = current if rng.random() < lock else FAMILIES[int(rng.integers(len(FAMILIES)))]
            avoid = self.avoid.get(branch)
            if avoid and family == avoid[0] and rng.random() < avoid[1]:
                family = FAMILIES[(FAMILIES.index(family) + 1 + int(rng.integers(3))) % len(FAMILIES)]
            p1 = float(np.clip(.5 + rng.normal(0, .4 / math.sqrt(round_index)), .01, .99))
            source = solver_source(family, p1, float(rng.uniform(.8, .99)), rng.random() < .3,
                                   rng.random() < .2, f"{self.arm}-{self.seed}-{branch}-{round_index}-{index}")
            predicted = float(np.clip(QUALITY[family] + rng.normal(0, .03), .05, .9))
            width = .12 / (1 + .3 * session_round)
            parent = best["program_id"] if best is not None and rng.random() < .7 else None
            rows.append({"source": source, "parent_sha256": parent,
                         "change_summary": f"fixture change {index}", "rationale": "fixture rationale",
                         "predicted_risk": predicted, "interval80_low": max(0., predicted - width),
                         "interval80_high": min(1., predicted + width)})
        for avoid in self.avoid.values():
            avoid[1] *= .8
        turns = Path(turn_root) / "turns" / f"round_{round_index:02d}"
        turns.mkdir(parents=True, exist_ok=True)
        (turns / "0001.request.bin").write_bytes(message.encode())
        tokens = 100_000 * (1 + session_round) + (400_000 if msg["kind"] == "boundary" else 0)
        cached = 0 if msg["kind"] == "boundary" and session_round == 0 else int(.8 * tokens)
        (turns / "0001.response.bin").write_text(json.dumps(
            {"usage": {"input_tokens": tokens, "output_tokens": 5000, "input_tokens_details": {"cached_tokens": cached}}}))
        usage = {"requests": 1, "input_tokens": tokens, "cached_input_tokens": cached, "output_tokens": 5000,
                 "responses_missing_usage": 0, "compactions": int(session_round >= 8 and round_index % 4 == 0),
                 "wall_seconds": 10.0}
        return rows, usage


class FixtureEvaluator:
    def __init__(self, private, sentinel):
        self.private = Path(private)
        self.private.mkdir(parents=True, exist_ok=True)
        self.sentinel = sentinel
        atomic_json_write(self.private / "rsi-private-audit.json", {
            "version": "rsi-feedback-v2", "sentinel": sentinel,
            "private_parent_ids": [hashlib.sha256(f"parent-{c}".encode()).hexdigest() for c in CASES],
            "case_ids": list(CASES)})
        self.charged = []

    def evaluate_source(self, path):
        body = Path(path).read_bytes()
        sha = hashlib.sha256(body).hexdigest()
        source = body.decode()
        family = family_of(source)
        rng = _rng("eval", sha)
        if family is not None and rng.random() < .05:
            cases, status, message, legal = {c: 1.0 for c in CASES}, "rejected", "interface rejected", 0.
        else:
            base = QUALITY.get(family, .55) + .15 * abs(_param(source) - .5) + rng.normal(0, .01)
            cases = {c: float(np.clip(base + d, .01, .99)) for c, d in zip(CASES, (-.02, .02))}
            status, message, legal = "valid", "", 1.
        risk = sum(cases.values()) / len(cases)
        feedback = {"source_sha256": sha, "status": status, "risk": risk, "risk_by_case": cases,
                    "legal_rate": legal, "failed_solves": 0 if legal else 2,
                    "seconds_per_solve": float(rng.uniform(5, 30)), "message": message}
        atomic_json_write(self.private / "rsi-feedback" / f"{sha}.json",
                          {"feedback": feedback, "sentinel": self.sentinel})
        if sha not in self.charged:
            self.charged.append(sha)
        prefix = "copper:v2:scientific-source:"
        atomic_json_write(self.private / "budget.json", {
            "counters": {"algorithm_feedback_slots": len(self.charged)},
            "metadata": {"cumulative_usage": {prefix + s: {"algorithm_feedback_slots": 1} for s in self.charged}}})
        return feedback


def fixture_select(root):
    encoder = CodeEncoder(root, "ast")

    def candidate(sha):
        return Candidate(sha, {"sha256": sha, "source_path": f"candidates/{sha}/solver.py"}, sha, "rsi")

    done = set()

    def encode(sha):
        if sha not in done:
            encoder.encode(candidate(sha))
            done.add(sha)

    def select(pool, history, count):
        for row in history:
            encode(row["program_id"])
        predictions = []
        for p in pool:
            encode(p["program_id"])
            source = (Path(root) / "candidates" / p["program_id"] / "solver.py").read_text()
            mean = QUALITY.get(family_of(source), .5) + .1 * abs(_param(source) - .5)
            std = .02 + int(p["program_id"][:4], 16) / 0xffff * .05
            predictions.append({"program_id": p["program_id"], "branch": p["branch"], "mean": mean, "std": std,
                                "acquisition": -(mean - std)})
        return {"predictions": predictions, "selected": reserve_and_rank(predictions, count)}
    return select


def _run(out, experiment, run_id, arm, seed, profile, identity_extra=None, intervention=None):
    root = Path(out) / "archive" / experiment / run_id
    sentinel = hashlib.sha256(f"sentinel-{run_id}".encode()).hexdigest()[:32]
    config = Config(arm, seed, rounds=profile["rounds"], proposals_per_branch=profile["proposals_per_branch"],
                    batch=profile["batch"], reset_period=profile["reset_period"], token_cap_input=10 ** 12)
    identity = {"synthetic_test": True, "run_id": run_id, "experiment": experiment,
                "judge": "standard", "tier": "S", **(identity_extra or {})}
    controller = Controller(root, config, FixtureAgent(arm, seed),
                            FixtureEvaluator(Path(out) / "private" / experiment / run_id, sentinel),
                            fixture_select(root), "Fixture task brief.", identity, intervention=intervention)
    controller.run()
    return root


def note(tag):
    return f"Earlier notes: the `{tag}` approach does not work here; do not spend proposals on it."


def generate(out, profile_name="small"):
    profile = PROFILES[profile_name]
    out = Path(out)
    runs = []  # (experiment, run_id, arm, seed, judge)
    for arm in ARM_MU:
        for seed in profile["m1_seeds"]:
            run_id = f"{arm}-s{seed}"
            _run(out, "M1", run_id, arm, seed, profile)
            runs.append(("M1", run_id, arm, seed, "standard"))
    for ck in profile["checkpoints"]:
        for mode, arm in MODE_ARM.items():
            for rep in range(1, profile["replicates"] + 1):
                run_id = f"ck{ck}-{mode}-r{rep}"
                cont = {"source_run": f"accumulated-s{ck}", "round": 10, "mode": mode, "replicate": rep}
                _run(out, "E1", run_id, arm, 1000 * ck + rep, profile, {"continuation": cont})
                runs.append(("E1", run_id, arm, 1000 * ck + rep, "standard"))
        for mode in ("conversation", "history"):
            for rep in range(1, profile["replicates"] + 1):
                run_id = f"ck{ck}-{mode}-planted-r{rep}"
                cont = {"source_run": f"accumulated-s{ck}", "round": 10, "mode": mode, "replicate": rep}
                tag = FAMILIES[ck % len(FAMILIES)]
                _run(out, "E2", run_id, MODE_ARM[mode], 1000 * ck + rep, profile, {"continuation": cont},
                     {"round": 11, "target_tag": tag, "note": note(tag)})
                runs.append(("E2", run_id, MODE_ARM[mode], 1000 * ck + rep, "standard"))
    for judge in ("q25", "oracle"):
        for arm in ("ldm4geo", "accumulated"):
            for seed in profile["e3_seeds"]:
                run_id = f"{arm}-{judge}-s{seed}"
                _run(out, "E3", run_id, arm, seed, profile, {"judge": judge, "blind": judge != "oracle"})
                runs.append(("E3", run_id, arm, seed, judge))
    for arm in ("ldm4geo", "accumulated"):
        for seed in profile["f1_seeds"]:
            run_id = f"{arm}-s{seed}"
            _run(out, "F1", run_id, arm, seed, profile, {"tier": "F"})
            runs.append(("F1", run_id, arm, seed, "standard"))
    (out / "outcomes").mkdir()
    for experiment, table in make_outcomes(out, runs, profile).items():
        (out / "outcomes" / f"{experiment}.json").write_text(json.dumps(table, sort_keys=True, allow_nan=False) + "\n")
    return out


COLUMNS = ("experiment", "protocol_id", "run_id", "arm", "tier", "judge", "seed", "prefix", "round",
           "selected_sha256", "finalist_sha256s", "validation_risk", "validation_iou", "validation_legal_rate",
           "validation_seconds_per_solve",
           *[f"{split}_{name}" for split in ("test_id", "test_ood") for name in (
               "iou", "risk", "legal_rate", "seconds_per_solve", "iou_drill", "iou_drill_outcrop",
               "surface_distance", "volume_error")])


def _prefix(t, rounds):
    return "full" if t == rounds else f"round_{t:02d}"


def make_outcomes(out, runs, profile):
    """Tables in the format of experiments/rsi/export_outcomes.py plus the optional T8 blocks."""
    from ....rsi.ledger import read
    bodies, rounds = profile["bodies"], profile["rounds"]
    body_effect = {split: _rng("bodies", split).normal(0, .03, bodies) for split in ("test_id", "test_ood")}
    judge_shift = {"standard": 0., "q25": -.03, "oracle": .07}
    tables = {}
    for experiment, run_id, arm, seed, judge in runs:
        table = tables.setdefault(experiment, {
            "schema": "rsi-outcomes-v1", "experiment": experiment, "protocol_id": "fixture-protocol",
            "sealed": experiment not in ("E1", "E2"), "frozen_list_sha256": None,
            "unseal_log_sha256": hashlib.sha256(b"fixture unseal log").hexdigest(), "columns": list(COLUMNS),
            "rows": [], "body_scores": [], "programs": [], "validations": []})
        ledger = read(Path(out) / "archive" / experiment / run_id / "ledger.jsonl")
        rng = _rng("outcome", experiment, run_id)
        effect = rng.normal(0, .012)
        valid = [r for r in ledger if r["status"] == "valid"]
        for r in valid:
            table["programs"].append({"run_id": run_id, "program_id": r["program_id"],
                                      "train_iou": float(np.clip(.55 - .8 * r["risk"] + rng.normal(0, .02), 0, 1))})
        incumbents = {}
        for t in range(0, rounds + 1):
            prefix = [r for r in valid if r["round"] <= t]
            if prefix:
                incumbents[t] = min(prefix, key=lambda r: (r["risk"], r["program_id"]))
        for best in {b["program_id"]: b for b in incumbents.values()}.values():
            table["validations"].append({"run_id": run_id, "program_id": best["program_id"],
                                         "risk": float(np.clip(best["risk"] + .02 + rng.normal(0, .01), 0, 1)),
                                         "iou": float(np.clip(.2 + rng.normal(0, .02), 0, 1))})
        freeze_rounds = {"E1": range(11, rounds + 1), "E2": range(11, rounds + 1), "M1": [5, 10, rounds]}.get(
            experiment, [rounds])
        for t in freeze_rounds:
            best = incumbents[t]
            mu = ARM_MU[arm] + judge_shift[judge] - (.02 if t < rounds else 0) + effect
            row = {c: None for c in COLUMNS}
            row.update(experiment=experiment, protocol_id="fixture-protocol", run_id=run_id, arm=arm, tier="S",
                       judge=judge, seed=seed, prefix=_prefix(t, rounds), round=None if t == rounds else t,
                       selected_sha256=best["program_id"], finalist_sha256s=best["program_id"],
                       validation_risk=float(best["risk"] + .02 + rng.normal(0, .004)),
                       validation_iou=float(np.clip(mu + rng.normal(0, .01), 0, 1)),
                       validation_legal_rate=1.0, validation_seconds_per_solve=float(rng.uniform(10, 30)))
            if table["sealed"]:
                for split, shift in (("test_id", 0.), ("test_ood", -.03)):
                    scores = np.clip(mu + shift + body_effect[split] + rng.normal(0, .02, bodies), 0, 1)
                    table["body_scores"].append({"run_id": run_id, "prefix": row["prefix"], "split": split,
                                                 "iou": [float(v) for v in scores]})
                    row.update({f"{split}_iou": float(scores.mean()), f"{split}_risk": float(.6 - scores.mean()),
                                f"{split}_legal_rate": 1.0, f"{split}_seconds_per_solve": float(rng.uniform(10, 30)),
                                f"{split}_iou_drill": float(scores[::2].mean()),
                                f"{split}_iou_drill_outcrop": float(scores[1::2].mean()),
                                f"{split}_surface_distance": float(rng.uniform(2, 4)),
                                f"{split}_volume_error": float(rng.uniform(.1, .4))})
            table["rows"].append(row)
    rng = _rng("q1")
    risks = rng.uniform(.3, .6, 20)
    ident = lambda i: hashlib.sha256(bytes([i])).hexdigest()
    tables["Q1"] = {
        "schema": "rsi-outcomes-v1", "experiment": "Q1", "protocol_id": "fixture-protocol", "sealed": False,
        "frozen_list_sha256": None, "unseal_log_sha256": None, "columns": list(COLUMNS), "rows": [],
        "references": {"Initial solver": {"test_id": .11, "test_ood": .09},
                       "Projected search": {"test_id": .12, "test_ood": .10},
                       "Acquisition-guided search": {"test_id": .13, "test_ood": .11}},
        "q1": {"programs": [{"program_id": ident(i), "short_12x1": float(r), "full_24x1": float(r + rng.normal(0, .02)),
                             "short_24x2": float(r + rng.normal(0, .01))} for i, r in enumerate(risks)],
               "repeats": [{"program_id": ident(i), "risks": [float(r + rng.normal(0, .01)) for _ in range(3)]}
                           for i, r in enumerate(risks[:6])]}}
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", default="small", choices=sorted(PROFILES))
    args = parser.parse_args(argv)
    generate(args.out, args.profile)
    print(json.dumps({"out": str(args.out)}))


if __name__ == "__main__":
    main()
