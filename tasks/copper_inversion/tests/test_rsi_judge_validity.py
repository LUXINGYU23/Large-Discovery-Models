"""Unit tests for the pre-registered judge validity battery (bench-design.md §3.3)."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from tasks.copper_inversion.experiments.rsi import judge_validity
from tasks.copper_inversion.v2.contract import LEGACY_OBJECTIVE, OBJECTIVE
from tasks.copper_inversion.v2.data import save_problem
from tasks.copper_inversion.v2.geological_policy import public_priors
from tasks.copper_inversion.v2.suite import file_hash


def _make_synthetic_case(shape=(16, 16, 16), box=(2, 9, 2, 9, 3, 8), seed=0):
    rng = np.random.default_rng(seed)
    origin = np.array([0., 0., 0.])
    spacing = np.array([2., 2., 2.])
    domain = np.ones(shape, dtype=bool)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[:, :, -1] = True
    feasible = domain & ~forbidden

    x0, x1, y0, y1, z0, z1 = box
    truth = np.zeros(shape, dtype=bool)
    truth[x0:x1, y0:y1, z0:z1] = True
    truth &= feasible

    def sample_obs(hole_coords):
        pts, labels, groups = [], [], []
        for g, (ix, iy) in enumerate(hole_coords):
            hx, hy = ix * 2. + 1., iy * 2. + 1.
            in_xy = x0 <= ix < x1 and y0 <= iy < y1
            for iz in range(shape[2] - 1):
                zc = iz * 2. + 1.
                in_ore = in_xy and (z0 <= iz < z1)
                pts.append([hx, hy, zc])
                labels.append(1. if in_ore else 0.)
                groups.append(g)
            if in_xy:
                pts.append([hx, hy, z0 * 2.])
                labels.append(.5)
                groups.append(g)
                pts.append([hx, hy, z1 * 2.])
                labels.append(.5)
                groups.append(g)
        disc = .5 * float(np.linalg.norm(spacing))
        return [dict(
            name="drill",
            points=np.asarray(pts, dtype=float),
            labels=np.asarray(labels, dtype=float),
            groups=np.asarray(groups, dtype=int),
            weight=1.,
            transition_scale=disc,
            contact_sigma=disc,
            label_error=0.,
        )]

    all_xy = [(ix, iy) for ix in range(1, shape[0] - 1) for iy in range(1, shape[1] - 1)]
    rng.shuffle(all_xy)
    pub_holes = all_xy[:20]
    conf_holes = all_xy[20:70]

    base = dict(
        schema_version="problem_schema_v2",
        geometry_version="occupancy_v2",
        objective_version=OBJECTIVE,
        shape=list(shape),
        origin=origin.tolist(),
        spacing=spacing.tolist(),
        domain_mask=domain,
        forbidden_mask=forbidden,
        open_boundary_faces=["x-", "x+", "y-", "y+", "z-"],
        priors=public_priors(feasible, spacing),
    )
    pub = {**base, "case_id": f"case-{seed}", "observations": sample_obs(pub_holes)}
    conf = {**base, "case_id": f"conf-{seed}", "observations": sample_obs(conf_holes)}
    return pub, conf, truth


def _write_release(root, split="calibration"):
    boxes = [
        ("folded", (1, 9, 1, 9, 2, 7)),
        ("folded", (2, 10, 1, 9, 3, 8)),
        ("faulted", (7, 15, 1, 9, 4, 9)),
        ("faulted", (6, 14, 2, 10, 5, 10)),
        ("lens", (1, 8, 8, 15, 6, 11)),
        ("lens", (2, 9, 7, 14, 7, 12)),
        ("multiple", (8, 15, 8, 15, 2, 7)),
        ("multiple", (7, 14, 7, 14, 8, 13)),
    ]
    split_dir = root / split
    cases, strata_rows = [], ["case_id,parent_id,split,family\n"]
    for idx, (fam, box) in enumerate(boxes):
        pub, conf, truth = _make_synthetic_case(box=box, seed=100 + idx)
        cid = pub["case_id"]
        pid = hashlib.sha256(f"parent-{split}-{idx}".encode()).hexdigest()
        pub_dir = split_dir / "public" / cid
        truth_dir = split_dir / "private" / cid
        save_problem(pub, pub_dir)
        save_problem(conf, truth_dir / "confirmation")
        np.save(truth_dir / "truth.npy", truth)
        files = ("problem.json", "problem.npz", "truth.npy",
                 "confirmation/problem.json", "confirmation/problem.npz")
        cases.append(dict(
            case_id=cid,
            parent_id=pid,
            modality="drill",
            public=str(pub_dir.relative_to(split_dir)),
            truth=str(truth_dir.relative_to(split_dir)),
            sha256={
                f: file_hash((pub_dir if f.startswith("problem.") else truth_dir) / f)
                for f in files
            },
        ))
        strata_rows.append(f"{cid},{pid},{split},{fam}\n")
    (split_dir / "manifest.json").write_text(json.dumps(dict(
        schema="copper-suite-v2", stage="development", split=split, seeds=[47], cases=cases,
    ), sort_keys=True))
    (root / "strata.csv").write_text("".join(strata_rows))
    for forbidden_split in ("validation", "test_id"):
        trap = root / forbidden_split
        trap.mkdir(parents=True)
        (trap / "manifest.json").write_text(json.dumps(dict(
            schema="copper-suite-v2", stage="development", split=forbidden_split, seeds=[47], cases=[],
        )))
    return root


def test_split_guard_refuses_validation_and_sealed_paths(tmp_path):
    release = _write_release(tmp_path / "release", split="calibration")
    for bad in ("validation", "test_id", "test_ood", "sealed", "calibration_ood"):
        with pytest.raises(ValueError, match="allows splits"):
            judge_validity.run_battery(release, split=bad, workers=1, auto_v6=False, run_replay=False)
        with pytest.raises(SystemExit):
            judge_validity.main(["--release", str(release), "--split", bad])

    cal_manifest = release / "calibration/manifest.json"
    data = json.loads(cal_manifest.read_text())
    data["cases"][0]["truth"] = "../validation/trap"
    cal_manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="forbidden split"):
        judge_validity.resolve_split_manifest(release, "calibration")


def test_battery_cli_passes_v3_and_exposes_v2_overlay_loophole(tmp_path, capsys):
    release = _write_release(tmp_path / "release", split="calibration")
    sponge_src = tmp_path / "sponge_solver.py"
    sponge_src.write_text(
        "import numpy as np\n"
        "def solve(problem, budget, seed):\n"
        "    feas = problem['domain_mask'] & ~problem['forbidden_mask']\n"
        "    i, j, k = np.indices(feas.shape)\n"
        "    m = (((k % 3) == 0) | ((i % 6 == 0) & (j % 6 == 0))) & feas\n"
        "    return np.where(m, 1., -1.)\n"
    )
    out_v3 = tmp_path / "validity-v3.json"
    rc = judge_validity.main([
        "--release", str(release),
        "--split", "calibration",
        "--objective", OBJECTIVE,
        "--workers", "1",
        "--adaptive-masks", "160",
        "--bootstrap", "30",
        "--sponge-solver", f"test_sponge={sponge_src}",
        "--no-auto-v6",
        "--no-replay",
        "--output", str(out_v3),
    ])
    assert rc == 0
    stdout_text = capsys.readouterr().out
    rep_v3 = json.loads(out_v3.read_text())
    assert json.loads(stdout_text) == rep_v3
    assert rep_v3["objective_version"] == OBJECTIVE
    assert rep_v3["body_count"] == 8
    assert rep_v3["archived_sponge_solvers"] == ["test_sponge"]
    for cname in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert rep_v3["criteria"][cname]["passed"] is True, (cname, rep_v3["criteria"][cname])
    assert rep_v3["status"] == "passed"
    assert rep_v3["k_tol_comparison"] is not None
    for k in range(8):
        assert f"case-{100 + k}" not in out_v3.read_text()

    rep_v2 = judge_validity.run_battery(
        release, split="calibration", objective=LEGACY_OBJECTIVE,
        workers=1, adaptive_masks=160, bootstrap=20,
        sponge_solvers=[f"test_sponge={sponge_src}"], auto_v6=False, run_replay=False,
    )
    assert rep_v2["objective_version"] == LEGACY_OBJECTIVE
    assert rep_v2["criteria"]["C4"]["max_overlay_win_share"] > rep_v3["criteria"]["C4"]["max_overlay_win_share"]


def test_c7_archived_replay_scoring(tmp_path):
    release = _write_release(tmp_path / "release", split="train")
    manifest = json.loads((release / "train/manifest.json").read_text())
    runs_root = tmp_path / "runs"
    services_root = tmp_path / "services"
    run_dir = runs_root / "ldm4geo-s101"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"status": "completed"}))

    ledger_lines = []
    for p_idx in range(6):
        pid = f"{p_idx + 1:064x}"
        ledger_lines.append(json.dumps({"program_id": pid, "status": "valid"}))
        eval_dir = services_root / "ldm4geo-s101" / "evaluations" / f"eval-{p_idx}"
        eval_dir.mkdir(parents=True)
        pc_rows, ious, v2_risks = [], [], []
        is_sponge = p_idx == 5
        for c in manifest["cases"]:
            cid = c["case_id"]
            truth = np.load(release / "train" / c["truth"] / "truth.npy")
            if is_sponge:
                ii, jj, kk = np.indices(truth.shape)
                mask = ((kk % 3 == 0) | ((ii % 6 == 0) & (jj % 6 == 0)))
                mask[:, :, -1] = False
                area = 12.
                v2_risk = .20  # sponge games v2!
            else:
                mask = np.roll(truth, p_idx, axis=0)
                mask[:, :, -1] = False
                area = 2.5
                v2_risk = .25 + .08 * p_idx
            m_dir = eval_dir / cid / "47" / "model"
            m_dir.mkdir(parents=True)
            np.save(m_dir / "model.npy", np.where(mask, 1., -1.))
            iou = float(np.sum(mask & truth) / max(1, np.sum(mask | truth)))
            ious.append(iou)
            v2_risks.append(v2_risk)
            pc_rows.append({"case_id": cid, "seed": 47, "status": "succeeded",
                            "metrics": {"iou": iou, "regularizer_area": area}})
        (eval_dir / "private-case-results.json").write_text(json.dumps(pc_rows))
        (eval_dir / "result.json").write_text(json.dumps({
            "status": "succeeded", "candidate_id": pid,
            "metrics": {"mean_iou": float(np.mean(ious)), "loss_total": float(np.mean(v2_risks))},
        }))
    (run_dir / "ledger.jsonl").write_text("\n".join(ledger_lines) + "\n")

    c7_v3 = judge_validity.evaluate_archived_replay(runs_root, services_root, release, OBJECTIVE, .5, workers=1)
    assert c7_v3["evaluated"] is True and c7_v3["passed"] is True
    assert c7_v3["sponge_rank_percentile"] >= .80
    assert c7_v3["pick_iou_gap"] <= .03

    c7_v2 = judge_validity.evaluate_archived_replay(runs_root, services_root, release, LEGACY_OBJECTIVE, .5, workers=1)
    assert c7_v2["evaluated"] is True and c7_v2["passed"] is False
    assert c7_v2["sponge_rank_percentile"] == 0.
