"""Analysis package (WP8) on synthetic fixtures; AT-32, AT-33, AT-34, AT-36 and the read-only checks.

Nothing here is a scientific result: the fixture archive is written by the real
controller with scripted collaborators, and its outcome tables are invented.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tasks.copper_inversion.experiments.rsi.analysis import stats, tagger
from tasks.copper_inversion.experiments.rsi.analysis.bundle import build, load_config, write
from tasks.copper_inversion.experiments.rsi.analysis.checks import run_checks
from tasks.copper_inversion.experiments.rsi.analysis.fixtures import ARM_MU, generate, solver_source
from tasks.copper_inversion.experiments.rsi.analysis.measures import line_edit_distance, run_vectors
from tasks.copper_inversion.experiments.rsi.analysis.outcomes import Outcomes
from tasks.copper_inversion.experiments.rsi.analysis.pilot import SelectionReports, TableValues, pilot
from tasks.copper_inversion.experiments.rsi.analysis.rundir import discover

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = REPO / "tasks/copper_inversion/experiments/rsi/analysis"
FAST = {"n_boot": 2000, "n_perm": 2000}


# ---------------------------------------------------------------- unit checks

def test_statistics_reproduce_spec_and_manuscript_values():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.pooled_sd([[1, 3], [2, 4, 6]]) == pytest.approx(math.sqrt((2 + 8) / 3))
    # spec §7 table: 2.80 s sqrt(2/n)
    assert [round(r["mde_in_s"], 2) for r in stats.mde_table(1, [5, 8, 12, 16])] == [1.77, 1.40, 1.14, 0.99]
    assert round(stats.mde(0.041, 12), 3) == 0.047
    assert stats.replicate_decision(0.01, 0.02) == {"runs_per_arm": 12, "primary": "test"}
    assert stats.replicate_decision(0.05, 0.05)["runs_per_arm"] == 16
    assert stats.replicate_decision(0.2, 0.05)["primary"] == "estimate_only_author_confirms"
    # Proposition 3: r = 1, h = 0.05 gives K* = 7.1 (appendix); r = 1, h = 0.1 gives 5.32 (placeholder)
    assert round(stats.k_star(1, 0.05), 1) == 7.1
    assert round(stats.k_star(1, 0.1), 2) == 5.32
    assert stats.k_star(10, 0.1) is None and stats.k_star(1, 0) is None
    assert stats.exponential_rate([1, 2, 3], [True, False, True]) == pytest.approx(2 / 6)
    assert stats.half_life([0.4, 0.3, 0.19, 0.1]) == 2 and stats.half_life([0.4, 0.3]) is None
    assert stats.e1_verdict(-0.01, 0.002, 0.004) == "non-inferior"
    assert stats.e1_verdict(0.006, 0.01, 0.004) == "refuted"
    assert stats.e1_verdict(0.002, 0.006, 0.004) == "inconclusive"
    assert stats.spearman([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1)


def test_stratified_permutation_is_exact_and_bound_exceeds_estimate():
    strata = [([1.0, 2.0], [0.0, 0.5]), ([3.0, 3.5], [2.0, 4.0])]
    observed, null = stats.stratified_null(strata)
    brute = []
    from itertools import combinations
    per = []
    for b, a in strata:
        pool = b + a
        per.append([np.mean([pool[i] for i in c]) - np.mean([pool[i] for i in range(4) if i not in c])
                    for c in combinations(range(4), 2)])
    for x in per[0]:
        for y in per[1]:
            brute.append((x + y) / 2)
    assert observed == pytest.approx(np.mean([1.5 - 0.25, 3.25 - 3.0]))
    assert sorted(null) == pytest.approx(sorted(brute))
    rng = np.random.default_rng(1)
    strata = [(rng.normal(0, 1, 4), rng.normal(0, 1, 4)) for _ in range(3)]
    diff = np.mean([np.mean(b) - np.mean(a) for b, a in strata])
    assert stats.shift_upper_bound(strata) > diff


def test_tagger_families_and_calibration_helpers():
    t = tagger.Tagger()
    source = solver_source("annealing", .5, .9, True, True, "x")
    assert "annealing" in t.tags(source) and "morphology" in t.tags(source)
    assert "morphology" not in t.family(t.tags(source))
    assert t.family(t.tags(solver_source("convex_optimization", .5, .9, False, False, "y"))) == "convex_optimization"
    assert t.family(t.tags("def solve(problem, budget, seed):\n    return 1\n")) == "untagged"
    assert tagger.free_text_chars(source) > 40 and tagger.free_text_chars("x = 'short'\n") == 0
    custom = tagger.Tagger({"version": "test", "rules": [{"name": "lstsq", "patterns": ["lstsq"]}]})
    assert custom.sha256 != t.sha256 and custom.top_level == ["lstsq"]
    assert tagger.cohen_kappa(list("aabb"), list("aabb")) == 1
    assert tagger.cohen_kappa(list("abab"), list("aabb")) == 0
    assert tagger.decide_mode(0.8, 0.5) == "tags" and tagger.decide_mode(0.6, 0.5) == "cluster"
    assert tagger.decide_mode(0.9, 0.95) == "cluster"
    tags = [["a", "b"], ["a"], ["a", "c"], ["b"]]
    assert tagger.select_target_tag(tags, ["a", "b"]) == "a"
    assert tagger.select_target_tag(tags, ["c"]) is None
    rng = np.random.default_rng(0)
    centers = np.eye(8)[:3] * 5
    rows = [[(r, centers[k] + rng.normal(0, .1, 8)) for r in range(1, 6) for k in range(3) for _ in range(4)]]
    threshold, medians = tagger.calibrate_threshold(rows)
    assert 2 <= medians[threshold] <= 6
    labels = tagger.cluster_labels([v for _, v in rows[0]], threshold)
    assert labels[0] == "c0" and len(set(labels)) == 3
    assert tagger.assign_clusters([centers[1]], [v for _, v in rows[0]], labels, threshold) == [labels[4]]


def test_edit_distance_and_outcome_reader_refuse_body_identities():
    assert line_edit_distance("a\nb\nc", "a\nb\nc") == 0
    assert line_edit_distance("a\nb\nc", "a\nx\nc\nd") == pytest.approx(2 / 4)
    table = {"schema": "rsi-outcomes-v1", "experiment": "M1", "protocol_id": "p", "columns": [], "rows": [],
             "body_scores": [{"run_id": "r", "prefix": "full", "split": "test_id", "iou": [.1, .2]}]}
    assert Outcomes([table]).body_row("M1", "r", 20, "test_id", rounds=20) == [.1, .2]
    bad = json.loads(json.dumps(table))
    bad["body_scores"][0]["case_id"] = "x"
    with pytest.raises(ValueError, match="body-identifying"):
        Outcomes([bad])
    with pytest.raises(ValueError, match="schema"):
        Outcomes([{**table, "per_body": []}])


# ---------------------------------------------------------------- AT-34 simulations

def test_at34_permutation_test_null_rejection_rate():
    """400 null datasets of 12 vs 12 runs, 999 permutations: rejection rate at 0.05 within [0.02, 0.085]
    (about three binomial standard errors)."""
    rng = np.random.default_rng(34)
    sims = 400
    rejections = sum(stats.permutation_test(rng.normal(0, 1, 12), rng.normal(0, 1, 12), 999, rng) < 0.05
                     for _ in range(sims))
    assert 0.02 <= rejections / sims <= 0.085


@pytest.mark.parametrize("paired", [True, False])
def test_at34_hierarchical_bootstrap_interval_coverage(paired):
    """1,000 datasets with run effects, body effects shared by both arms and run-by-body noise; a known
    difference of 0.02. The t-calibrated 95% interval must cover it in [0.93, 0.97] of the datasets
    (three binomial standard errors) for the runs-only fallback (measured 0.954) and in [0.93, 0.99]
    body-paired: both levels see the run-by-body noise, so the paired interval is a little wide
    (measured 0.974, also with an arm-by-body interaction). The plain percentile interval covered 0.933
    paired and 0.903 runs only at 8 runs per arm."""
    rng = np.random.default_rng(35)
    sims, runs, bodies, truth, covered = 1000, 8, 16, 0.02, 0
    for _ in range(sims):
        body = rng.normal(0, .03, bodies)
        arms = [mu + rng.normal(0, .015, (runs, 1)) + body + rng.normal(0, .02, (runs, bodies))
                for mu in (.2 + truth, .2)]
        if not paired:
            arms = [a.mean(axis=1, keepdims=True) for a in arms]
        draws = stats.bootstrap_contrast(arms, [1, -1], 999, rng)
        lo, hi = stats.t_interval(stats.contrast_estimate(arms, [1, -1]), draws, stats.contrast_df(arms))
        covered += lo <= truth <= hi
    assert 0.93 <= covered / sims <= (0.99 if paired else 0.97)


def test_at34_e1_upper_bound_coverage_and_test_size():
    """E1 layout (3 checkpoints, 4 vs 4 continuations). With a true difference of 0.01 the one-sided 95%
    upper bound lies above it in [0.92, 0.98] of 600 datasets (three binomial standard errors; the
    bound inverts the permutation test under a common shift, so it needs no bootstrap correction);
    under no difference the exact two-sided stratified test rejects in [0.01, 0.10] of 200 datasets."""
    rng = np.random.default_rng(36)
    covered = 0
    for _ in range(600):
        strata = [(c + .01 + rng.normal(0, .005, 4), c + rng.normal(0, .005, 4)) for c in rng.normal(.4, .01, 3)]
        covered += stats.shift_upper_bound(strata, iterations=25) >= .01
    assert 0.92 <= covered / 600 <= 0.98
    rejections = 0
    for _ in range(200):
        strata = [(c + rng.normal(0, .005, 4), c + rng.normal(0, .005, 4)) for c in rng.normal(.4, .01, 3)]
        rejections += stats.stratified_p(strata) < 0.05
    assert 0.01 <= rejections / 200 <= 0.10


def test_at34_e2_stratified_bootstrap_coverage():
    """E2 layout (3 checkpoints, 4 planted vs 4 control continuations, 10 rounds); a known late-round
    difference of 0.15. Coverage of the t-calibrated 95% interval of the rounds 16-20 mean over 1,000
    datasets lies in [0.93, 0.97] (measured 0.947; the percentile interval covered 0.88)."""
    rng = np.random.default_rng(37)
    effect = np.r_[np.full(5, .3), np.full(5, .15)]
    covered = 0
    for _ in range(1000):
        strata = []
        for _ in range(3):
            base = rng.uniform(.3, .6)
            strata.append((base + effect + rng.normal(0, .05, (4, 10)), base + rng.normal(0, .05, (4, 10))))
        draws = stats.stratified_bootstrap_rows(strata, 999, rng)
        est = stats.stratified_estimate(strata)[5:].mean()
        lo, hi = stats.t_interval(est, draws[:, 5:].mean(axis=1), stats.stratified_df(strata))
        covered += lo <= .15 <= hi
    assert 0.93 <= covered / 1000 <= 0.97


# ---------------------------------------------------------------- fixture archive

@pytest.fixture(scope="session")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("rsi-analysis") / "fx"
    generate(root, "small")
    yield root
    shutil.rmtree(root, ignore_errors=True)  # about 300 MB; do not leave it in the retained basetemps


@pytest.fixture(scope="session")
def config(tmp_path_factory):
    config = load_config()
    config.update(FAST)
    path = tmp_path_factory.mktemp("cfg") / "config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture(scope="session")
def bundle(fixture, config, tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle") / "results"
    write(out, build(fixture / "archive", fixture / "outcomes", load_config(config), "fixture-commit"))
    return out


def _structure(skeleton, value, path, problems):
    if isinstance(skeleton, dict):
        if not isinstance(value, dict):
            problems.append(f"{path}: expected an object")
            return
        for key, sub in skeleton.items():
            if key not in value:
                problems.append(f"{path}.{key}: missing")
            else:
                _structure(sub, value[key], f"{path}.{key}", problems)
    elif skeleton == "list":
        if not isinstance(value, list):
            problems.append(f"{path}: expected a list")
    elif isinstance(value, (dict, list)):
        problems.append(f"{path}: expected a scalar")


def test_at32_results_have_the_placeholder_key_structure(bundle):
    skeleton = json.loads((ANALYSIS / "mock_skeleton.json").read_text())["skeleton"]
    results = json.loads((bundle / "results.json").read_text())
    problems = []
    _structure(skeleton, results, "results", problems)
    assert problems == []
    assert results["MOCK"] is False and skeleton["MOCK"] == "scalar"
    for name in ("results.json", "runs.csv", "rounds.csv", "programs.csv", "e1.csv", "e2.csv", "e3.csv",
                 "prop3.json", "q1.json", "provenance.json", "recompute.sh"):
        assert (bundle / name).is_file() and (bundle / name).stat().st_size > 0, name


def test_bundle_feeds_the_paper_plot_inputs(bundle):
    d = json.loads((bundle / "results.json").read_text())
    m = d["main"]
    T, K = m["rounds"], m["reset_period"]
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    for name in ("LDM4Geo", "Accumulated context", "Restart from best", "Independent sampling"):
        assert all(len(row) == T + 1 and all(map(number, row)) for row in m["trajectories"][name])
        assert len(m["families"][name]) == T and all(map(number, m["families"][name]))
        assert len(m["budget_iou"][name]) == 3 and all(map(number, m["budget_iou"][name]))
        assert all(a >= b for row in m["trajectories"][name] for a, b in zip(row, row[1:]))
    for name in ("LDM4Geo", "Accumulated context"):
        assert len(m["tokens_per_round_M"][name]) == T
    for mode in ("best_program", "conversation", "history"):
        assert all(len(r) == 7 and all(map(number, r)) for r in d["E1"]["runs"][mode])
    assert d["E1"]["rounds"] == [10, 12, 14, 15, 16, 18, 20]
    assert d["E1"]["verdict"] in {"non-inferior", "refuted", "inconclusive"}
    for mode in ("conversation", "history"):
        assert all(len(r) == 10 and all(map(number, r)) for r in d["E2"]["runs"][mode])
    # the scripted agent avoids the planted tag in round 11: a positive shift away from it
    assert d["E2"]["mean"]["conversation"][0] > 0
    assert d["E2"]["reset_period"] == K and d["E2"]["first_round"] == 11
    assert all(len(d["coverage"][k]) == 20 and all(0 <= v <= 1 for v in d["coverage"][k])
               for k in ("accumulated", "ldm4geo"))
    for j in ("quarter", "all", "oracle"):
        assert all(len(d["E3"][j][a]) >= 2 for a in ("ldm4geo", "accumulated")) and number(d["E3_gap"][j])
    primary = d["E4_primary"]["id"]
    assert primary["interval_method"] == "runs_and_paired_bodies"
    assert primary["ci95"][0] <= primary["diff"] <= primary["ci95"][1] and 0 < primary["perm_p"] <= 1
    assert all("p_holm" in row for row in d["E4_secondary"])
    assert d["E4"]["Single conversation"]["n"] == 0 and d["E4"]["LDM4Geo"]["n"] == 4
    assert number(d["prop3"]["r"]) and number(d["prop3"]["h"]) and d["prop3"]["K_used"] == K
    assert number(d["branch_agreement"]["q_bar"]) and d["missing_outcomes"] == []
    assert d["process"]["ldm4geo"]["train_iou_spearman"]["mean"] < 0
    assert all(0 <= v <= 2 for v in d["process"]["ldm4geo"]["dispersion"])
    programs = (bundle / "programs.csv").read_text().splitlines()
    rounds = (bundle / "rounds.csv").read_text().splitlines()
    assert len(programs) > 1 and len(rounds) > 1
    provenance = json.loads((bundle / "provenance.json").read_text())
    assert provenance["code_commit"] == "fixture-commit" and provenance["protocol_id"] == "fixture-protocol"
    assert all(len(v["sha256"]) == 64 and v["files"] > 0 for v in provenance["inputs"].values())


def _snapshot(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _chmod_tree(root, writable):
    for path in [Path(root), *Path(root).rglob("*")]:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        path.chmod(mode | 0o200 if writable else mode & ~0o222)


def test_at33_recompute_is_byte_identical_on_a_read_only_archive(fixture, bundle, tmp_path):
    inputs = [fixture / "archive", fixture / "outcomes"]
    before = {str(p): _snapshot(p) for p in inputs}
    for p in inputs:
        _chmod_tree(p, False)
    try:
        env = {**os.environ, "REPO": str(REPO), "PYTHON": sys.executable}
        done = subprocess.run(["sh", str(bundle / "recompute.sh"), str(fixture / "archive"), str(fixture / "outcomes"),
                               str(bundle), str(tmp_path / "again")], env=env, capture_output=True, text=True,
                              timeout=600)
    finally:
        for p in inputs:
            _chmod_tree(p, True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "identical" in done.stdout
    assert {str(p): _snapshot(p) for p in inputs} == before
    assert sorted(os.listdir(tmp_path / "again")) == sorted(os.listdir(bundle))


def test_at36_pilot_reports_only_pooled_values(fixture, config, tmp_path):
    runs = discover(fixture / "archive")
    outcomes = Outcomes.load(fixture / "outcomes")
    result = pilot(runs, TableValues(outcomes), load_config(config))
    assert set(result) == {"s_v", "m_v", "mde_table"}
    assert set(result["mde_table"]) == {"bound", "bound_relative", "z", "rows", "decision"}
    assert [row["n"] for row in result["mde_table"]["rows"]] == [5, 8, 12, 16]
    text = json.dumps(result)
    assert not any(arm in text for arm in ARM_MU)
    per_arm = []
    for arm in ("ldm4geo", "accumulated"):
        chosen = sorted((r for r in runs if r.experiment == "M1" and r.arm == arm), key=lambda r: r.seed)[:4]
        values = [outcomes.validation_value("M1", r.run_id, "full", "iou") for r in chosen]
        per_arm.append(np.mean(values))
    numbers = {round(v, 6) for v in (result["s_v"], result["m_v"], result["mde_table"]["bound"])}
    assert not {round(v, 6) for v in per_arm} & numbers
    assert result["m_v"] == pytest.approx(np.mean(per_arm), abs=1e-6)
    # the private-report route used before the unseal gives the same answer
    for run in runs:
        if run.experiment == "M1":
            value = outcomes.validation_value("M1", run.run_id, "full", "iou")
            path = tmp_path / "selection" / "M1" / run.run_id / "full" / "selection.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"selected_sha256": "a" * 64, "candidates": [
                {"source_sha256": "a" * 64, "metrics": {"mean_iou": value}},
                {"source_sha256": "b" * 64, "metrics": {"mean_iou": 0.0}}]}))
    assert pilot(runs, SelectionReports(tmp_path / "selection"), load_config(config)) == result
    done = subprocess.run([sys.executable, "-m", "tasks.copper_inversion.experiments.rsi.analysis.pilot",
                           "--archive", str(fixture / "archive"), "--outcomes", str(fixture / "outcomes")],
                          cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert set(json.loads(done.stdout)) == {"s_v", "m_v", "mde_table"}


def test_read_only_checks_pass_and_detect_tampering(fixture, tmp_path):
    report, failed = run_checks("all", fixture / "archive", fixture / "private")
    assert not failed, json.dumps(report)[:2000]
    entry = report["M1/ldm4geo-s101"]
    assert entry["briefs"]["briefs_checked"] == 80 and entry["sentinel"]["leaks"] == 0
    assert entry["sentinel"]["positive_control_receipts"] > 0
    assert entry["slots"]["feedback_receipts_matched"] == entry["slots"]["charged_sources"]
    archive, private = tmp_path / "archive" / "M1", tmp_path / "private" / "M1"
    shutil.copytree(fixture / "archive/M1/ldm4geo-s101", archive / "ldm4geo-s101")
    shutil.copytree(fixture / "private/M1/ldm4geo-s101", private / "ldm4geo-s101")
    run = archive / "ldm4geo-s101"
    sentinel = json.loads((private / "ldm4geo-s101/rsi-private-audit.json").read_text())["sentinel"]
    request = next(run.rglob("*.request.bin"))
    request.write_bytes(request.read_bytes() + sentinel.encode())
    brief = next(run.rglob("briefs/round_03.json"))
    brief.chmod(0o644)
    brief.write_text(brief.read_text().replace('"round":3', '"round":3 '))
    receipt = next((private / "ldm4geo-s101/rsi-feedback").glob("*.json"))
    data = json.loads(receipt.read_text())
    data["feedback"]["seconds_per_solve"] += 1
    receipt.write_text(json.dumps(data))
    report, failed = run_checks("all", tmp_path / "archive", tmp_path / "private")
    entry = report["M1/ldm4geo-s101"]
    assert failed and entry["sentinel"]["leaks"] == 1 and entry["sentinel"]["status"] == "failed"
    assert entry["briefs"]["status"] == "failed" and entry["slots"]["status"] == "failed"
    assert sentinel not in json.dumps(report)


def test_checkpoint_target_tag_follows_rule_v1(fixture):
    tag = tagger.checkpoint_target_tag(fixture / "archive/M1/accumulated-s101")
    assert tag is None or tag in {r["name"] for r in tagger.DRAFT_RULES["rules"]}


def test_cluster_fallback_builds_a_bundle(fixture, config, tmp_path):
    runs = [r for r in discover(fixture / "archive") if r.experiment == "M1"][:2]
    rows = []
    for run in runs:
        ids = [p["program_id"] for t in sorted(run.proposals) for p in run.proposals[t]]
        vectors, _ = run_vectors(run, ids)
        rows.append([(t, vectors[p["program_id"]]) for t in sorted(run.proposals) for p in run.proposals[t]])
    threshold, medians = tagger.calibrate_threshold(rows)
    cfg = load_config(config)
    cfg["family"] = {"mode": "cluster", "threshold": threshold}
    files = build(fixture / "archive", fixture / "outcomes", cfg, "fixture-commit")
    families = {line.split(",")[8] for line in files["programs.csv"].decode().splitlines()[1:]}
    assert families and all(f.startswith("c") or f == "unassigned" for f in families)
    results = json.loads(files["results.json"])
    assert results["E2"]["runs"]["conversation"] and json.loads(files["provenance.json"])["family_mode"] == "cluster"
