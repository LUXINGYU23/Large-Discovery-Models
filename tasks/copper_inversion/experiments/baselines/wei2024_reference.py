"""Wei et al. (2024, C&G 105779) faithful 3-D level-set MCMC Stage-A runner (wei2024).

Runs the pinned upstream repository (xiaolongw1223/Knowledge-driven-stochastic-modeling
at commit 94b8615bb2a9ca135583852a23ba114964a4b5af, packaged under /baseline in the
pinned evaluator image) with its published 3-D defaults and minimal I/O conversion:
  - Grid coordinates: unit-cell centers (0.5 .. n - 0.5) matching examples/01 & 02
    mesh.txt and shpmc/level_set_mc.py's dx = ones(3), max_step = 1.
  - Gaussian random field prior: mean=[0, 0], variance=[1, 1], range_x=[2, 20],
    range_y=[2, 20], range_z=[2, 10], anisotropy_xy=[0, 180], anisotropy_xz=[0, 180],
    via ReproducibleGaussianField tied to np.random.seed(seed).
  - Public observations voxelized onto 3-D NaN-padded grids (0 barren, 1 ore, 0.5 contact):
      * drill only (examples/01-drillholes-3d): contribution=[1], iter_num=1000,
        temperature=3, post-processing cut_off=500 (stride 2).
      * drill + outcrop (examples/02-drillholes-outcrops-3d): contribution=[1, 0.1],
        iter_num=1000, temperature=20, post-processing cut_off=800 (stride 2).
  - Initial model: [(initialize(problem) > 0).astype(float) - 0.5] (our public
    observation initializer shifted by -0.5, matching examples/01 & 02).
  - Minimal numerical/domain guards (from experiments/baselines/solver.py):
      * finite_binary_loss: stable logaddexp form of shpmc.loss_functions.loss_function_binary
        setting contact error to 0 when an observation grid has zero 0.5 contact cells.
      * safe_perturb: treats degenerate or single-sign perturbations ('no zero contour')
        as Metropolis self-loops instead of aborting the chain.
  - Output: posterior mean indicator model_mean = mean(model_cache[independence] >= 0)
    on feasible cells saved to field.npy (thresholded at >= 0.5 and legalised in Stage-B).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from tasks.copper_inversion.v2.baselines import initialize
from tasks.copper_inversion.v2.contract import validate
from tasks.copper_inversion.v2.data import read_problem


def finite_binary_loss(labels, distances, contacts, weight):
    """Algebraically identical logaddexp form of shpmc loss_function_binary; zero when contacts empty."""
    logistic = np.logaddexp(0.0, distances[labels == 0]).sum() + np.logaddexp(0.0, -distances[labels == 1]).sum()
    contact = float(contacts.mean() ** 2 + np.mean(contacts ** 2)) if contacts.size else 0.0
    return float(weight * (logistic + contact))


def obs_to_grid(problem, obs):
    """Voxelize one public observation modality onto a 3-D NaN-padded array in {0.0, 0.5, 1.0}."""
    shape = tuple(int(n) for n in problem["shape"])
    origin = np.asarray(problem["origin"], dtype=float)
    spacing = np.asarray(problem["spacing"], dtype=float)
    pts = np.asarray(obs["points"], dtype=float)
    labs = np.asarray(obs["labels"], dtype=float)
    grid = np.full(shape, np.nan, dtype=float)
    if not len(pts):
        return grid
    idx = np.clip(np.rint((pts - origin) / spacing - 0.5).astype(int), 0, np.asarray(shape) - 1)
    sums, counts, has_contact = {}, {}, set()
    for (ix, iy, iz), y in zip(idx, labs):
        key = (int(ix), int(iy), int(iz))
        if y == 0.5:
            has_contact.add(key)
        else:
            sums[key] = sums.get(key, 0.0) + float(y)
            counts[key] = counts.get(key, 0) + 1
    for key, s in sums.items():
        mean_y = s / counts[key]
        grid[key] = 1.0 if mean_y > 0.5 else (0.0 if mean_y < 0.5 else 0.5)
    for key in has_contact:
        grid[key] = 0.5
    return grid


def _load_upstream(vendor_dir=None):
    base = Path(vendor_dir or os.environ.get("COPPER_BASELINE_VENDOR_DIR", "/baseline"))
    repo = base / "Knowledge-driven-stochastic-modeling"
    for p in (str(base), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_3d_comparison as portable
    import shpmc.level_set_mc as upstream
    from shpmc.level_set_mc import StochasticLevelSet3D
    upstream.loss_function_binary = finite_binary_loss
    if not getattr(upstream, "_wei2024_patched", False):
        orig_perturb = upstream.level_set_perturbation

        def safe_perturb(phi, velocity, max_step):
            try:
                out = orig_perturb(phi, velocity, max_step)
                if not np.isfinite(out).all() or not ((out > 0).any() and (out < 0).any()):
                    return phi.copy()
                return out
            except (ValueError, ZeroDivisionError, FloatingPointError):
                return phi.copy()

        upstream.level_set_perturbation = safe_perturb
        upstream.tqdm = lambda it, **kw: it
        upstream._wei2024_patched = True
    return portable, StochasticLevelSet3D


def run_case(public_dir, out_dir, *, seed=47, iter_num=1000, vendor_dir=None):
    """Run upstream Wei et al. (2024) 3-D MCMC on one public problem and save field.npy + choice.json."""
    out_dir = Path(out_dir)
    choice_path = out_dir / "choice.json"
    field_path = out_dir / "field.npy"
    if choice_path.is_file() and (field_path.is_file() or json.loads(choice_path.read_text()).get("no_ore")):
        return json.loads(choice_path.read_text())

    t0 = time.monotonic()
    problem = read_problem(public_dir)
    validate(problem)
    out_dir.mkdir(parents=True, exist_ok=True)

    feasible = np.asarray(problem["domain_mask"], dtype=bool) & ~np.asarray(problem["forbidden_mask"], dtype=bool)
    any_ore = any(
        bool(((np.asarray(o["labels"], dtype=float) != 0.5) & (np.asarray(o["labels"], dtype=float) > 0.5)).any())
        for o in problem["observations"]
    )
    if not any_ore:
        choice = {
            "method": "wei2024",
            "case_id": problem["case_id"],
            "seed": int(seed),
            "no_ore": True,
            "fallback": False,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        choice_path.write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    portable, StochasticLevelSet3D = _load_upstream(vendor_dir=vendor_dir)
    random.seed(int(seed))
    np.random.seed(int(seed))

    shape = tuple(int(n) for n in problem["shape"])
    centers = tuple(np.arange(n, dtype=float) + 0.5 for n in shape)
    gf = portable.gaussian(centers, portable.default_config())
    gf.output = True

    init_bin = (initialize(problem) > 0).astype(float)
    initial_3d = [np.ascontiguousarray(init_bin - 0.5)]

    drill_obs = next((o for o in problem["observations"] if o["name"] == "drill"), problem["observations"][0])
    outcrop_obs = next((o for o in problem["observations"] if o["name"] == "outcrop" and len(o["points"]) > 0), None)
    if outcrop_obs is not None:
        data_3d = [obs_to_grid(problem, drill_obs), obs_to_grid(problem, outcrop_obs)]
        contribution = [1.0, 0.1]
        temperature = 20
        cut_off = min(800, int(iter_num))
    else:
        data_3d = [obs_to_grid(problem, drill_obs)]
        contribution = [1.0]
        temperature = 3
        cut_off = min(500, int(iter_num))

    sampler = StochasticLevelSet3D(
        data_3d, initial_3d, gaussian_field=gf, max_step=1, contribution=contribution,
    )
    loss_arr, model_cache, acceptance_count = sampler.mcmc_sampling_single_chain(
        iter_num=int(iter_num), temperature=temperature,
    )
    independence = -1 * np.arange(0, max(2, cut_off), 2)
    mean_occ = np.mean(model_cache[independence] >= 0.0, axis=0)
    field_feasible = np.asarray(mean_occ[feasible], dtype=np.float32)

    tmp_field = out_dir / ".field.tmp.npy"
    np.save(tmp_field, field_feasible, allow_pickle=False)
    os.replace(tmp_field, field_path)

    total_loss = np.sum(loss_arr, axis=-1)
    wall_s = round(float(time.monotonic() - t0), 4)
    choice = {
        "method": "wei2024",
        "upstream_commit": "94b8615bb2a9ca135583852a23ba114964a4b5af",
        "case_id": problem["case_id"],
        "seed": int(seed),
        "no_ore": False,
        "fallback": False,
        "iter_num": int(iter_num),
        "temperature": int(temperature),
        "contribution": contribution,
        "cut_off": int(cut_off),
        "retained_samples": int(len(independence)),
        "acceptance_count": int(acceptance_count),
        "initial_loss": round(float(total_loss[0]), 6),
        "final_loss": round(float(total_loss[-1]), 6),
        "min_loss": round(float(np.min(total_loss)), 6),
        "cv": {
            "selected_rule": "native",
            "selected_config": {
                "iter_num": int(iter_num),
                "temperature": int(temperature),
                "cut_off": int(cut_off),
                "max_step": 1,
            },
            "cv_score": None,
            "cv_evaluated": 0,
        },
        "wall_seconds": wall_s,
    }
    tmp_choice = out_dir / ".choice.tmp.json"
    tmp_choice.write_text(json.dumps(choice, indent=2, sort_keys=True))
    os.replace(tmp_choice, choice_path)
    return choice


def _worker(task):
    pub_dir, out_dir, seed, iter_num, vendor_dir = task
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    return run_case(pub_dir, out_dir, seed=seed, iter_num=iter_num, vendor_dir=vendor_dir)


def _run_manifest_in_process(manifest_file, workers=8):
    spec = json.loads(Path(manifest_file).read_text())
    tasks = spec["tasks"]
    results = []
    done = 0
    total = len(tasks)
    if workers <= 1:
        for t in tasks:
            r = _worker(t)
            done += 1
            results.append(r)
            print(json.dumps({"done": done, "total": total, "case_id": r["case_id"],
                              "wall_seconds": r["wall_seconds"], "acc": r.get("acceptance_count")}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), 16)) as pool:
            for r in pool.map(_worker, tasks):
                done += 1
                results.append(r)
                print(json.dumps({"done": done, "total": total, "case_id": r["case_id"],
                                  "wall_seconds": r["wall_seconds"], "acc": r.get("acceptance_count")}), flush=True)
    return {
        "runs": len(results),
        "median_seconds": float(np.median([r["wall_seconds"] for r in results])) if results else 0.0,
        "mean_seconds": float(np.mean([r["wall_seconds"] for r in results])) if results else 0.0,
    }


def run_splits_in_docker(release_dir, splits, output_root, *, seeds=(47,), workers=16,
                         iter_num=1000, image=None):
    """Stage public-only inputs (no truth mounted) and run Stage-A inside the pinned baseline image."""
    release_dir = Path(release_dir).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workers = min(max(1, int(workers)), 16)
    code_root = Path(__file__).resolve().parents[4]

    staging = Path(tempfile.mkdtemp(prefix="wei2024-pub-", dir=str(output_root)))
    staging.chmod(0o755)
    try:
        container_tasks = []
        for split in splits:
            manifest_path = release_dir / split / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for case in manifest["cases"]:
                cid = case["case_id"]
                src_pub = manifest_path.parent / case["public"]
                dst_pub = staging / "public" / split / cid
                dst_pub.mkdir(parents=True, exist_ok=True)
                dst_pub.chmod(0o755)
                for fn in ("problem.json", "problem.npz"):
                    shutil.copy2(src_pub / fn, dst_pub / fn)
                    (dst_pub / fn).chmod(0o644)
                for s in seeds:
                    host_out = output_root / split / cid / str(int(s))
                    if (host_out / "choice.json").is_file() and (
                        (host_out / "field.npy").is_file()
                        or json.loads((host_out / "choice.json").read_text()).get("no_ore")
                    ):
                        continue
                    host_out.mkdir(parents=True, exist_ok=True)
                    container_tasks.append((
                        f"/work/public/{split}/{cid}",
                        f"/out/{split}/{cid}/{int(s)}",
                        int(s),
                        int(iter_num),
                        "/baseline",
                    ))
        if not container_tasks:
            return {"status": "already_complete", "pending": 0}

        (staging / "public").chmod(0o755)
        for p in (staging / "public").rglob("*"):
            if p.is_dir():
                p.chmod(0o755)
        task_file = staging / "tasks.json"
        task_file.write_text(json.dumps({"tasks": container_tasks}))
        task_file.chmod(0o644)

        uid, gid = os.geteuid(), os.getegid()
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--cpus", str(workers),
            "--user", f"{uid}:{gid}",
            "-e", "OMP_NUM_THREADS=1",
            "-e", "OPENBLAS_NUM_THREADS=1",
            "-e", "MKL_NUM_THREADS=1",
            "-e", "NUMEXPR_NUM_THREADS=1",
            "-e", "MPLCONFIGDIR=/tmp/mpl",
            "-e", "PYTHONPATH=/code",
            "--mount", f"type=bind,src={code_root},dst=/code,readonly",
            "--mount", f"type=bind,src={staging},dst=/work,readonly",
            "--mount", f"type=bind,src={output_root},dst=/out",
            image,
            "python", "-u", "-m", "tasks.copper_inversion.experiments.baselines.wei2024_reference",
            "--in-container-manifest", "/work/tasks.json",
            "--workers", str(workers),
        ]
        subprocess.run(cmd, check=True)
        return {"status": "completed", "executed": len(container_tasks), "splits": list(splits)}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-container-manifest", type=Path, default=None,
                        help="Internal flag: execute a prepared task manifest inside the baseline container.")
    parser.add_argument("--release", type=Path, help="Release directory containing split manifests.")
    parser.add_argument("--splits", default="train_s,validation,test_id,test_ood",
                        help="Comma-separated split names.")
    parser.add_argument("--output", type=Path, help="Stage-A output root (e.g. .../stage_a/wei2024).")
    parser.add_argument("--seeds", default="47", help="Comma-separated execution seeds.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--iter-num", type=int, default=1000)
    parser.add_argument("--image", default="sha256:f6856385a0da9444b1e5edb5048b2f6081593fe568c3473c2fda524d0c4489f5")
    args = parser.parse_args(argv)

    if args.in_container_manifest is not None:
        summary = _run_manifest_in_process(args.in_container_manifest, workers=args.workers)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0

    if args.release is None or args.output is None:
        parser.error("--release and --output are required")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    summary = run_splits_in_docker(
        args.release, splits, args.output,
        seeds=seeds, workers=args.workers, iter_num=args.iter_num, image=args.image,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
