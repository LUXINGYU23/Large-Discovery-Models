"""M7 Stage-A Implicit Neural Representation runner (inr_mlp, re-implemented after Hillier et al. 2023).

Runs under $R/env/embedding/bin/python on CPU ONLY (never uses GPUs).
Architecture & optimiser from Hillier et al. (2023) GeoINR 1.0 Table 1 (Case Study 1: well data):
  - N_h = 3 hidden layers, d_rep = 256 hidden units per layer
  - Softplus activation with beta = 100
  - Adam optimiser with lr = 1e-4 and cosine annealing LR schedule
  - Last layer initialised so phi(x) ≈ -1 (barren) in absence of pre-training
  - Loss: mean_c |delta(x_c)| + weighted_mean_ore relu(-delta) + weighted_mean_barren relu(delta)
          + lam * mean_s | |grad phi(x_s)| - 1 |, delta = phi / |grad phi|
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from tasks.copper_inversion.v2 import baseline_harness as harness
from tasks.copper_inversion.v2.data import read_problem

INR_CONFIGS = (
    {"s_z": 4, "lam": 0.1},
    {"s_z": 4, "lam": 1.0},
    {"s_z": 2, "lam": 0.1},
    {"s_z": 2, "lam": 1.0},
    {"s_z": 4, "lam": 0.01},
    {"s_z": 2, "lam": 0.01},
)


def _get_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch, nn, F


def build_geoinr_net(hidden_layers=3, hidden_dim=256, beta=100.0):
    """3-hidden-layer 256-unit Softplus(beta=100) MLP (Hillier et al. 2023 Table 1)."""
    torch, nn, _ = _get_torch()

    class GeoINRNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_dim = 3
            for _ in range(hidden_layers):
                lin = nn.Linear(in_dim, hidden_dim)
                nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)
                layers.append(lin)
                in_dim = hidden_dim
            self.hidden = nn.ModuleList(layers)
            self.act = nn.Softplus(beta=float(beta))
            self.out = nn.Linear(hidden_dim, 1)
            nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
            nn.init.constant_(self.out.bias, -1.0)

        def forward(self, x):
            h = x
            for layer in self.hidden:
                h = self.act(layer(h))
            return self.out(h).squeeze(-1)

        def forward_with_grad(self, x):
            b_val = float(self.act.beta)
            s_list = []
            h = x
            for layer in self.hidden:
                z = layer(h)
                h = self.act(z)
                s_list.append(torch.sigmoid(b_val * z))
            phi = self.out(h).squeeze(-1)
            g = self.out.weight * s_list[-1]
            for i in range(len(self.hidden) - 1, 0, -1):
                g = (g @ self.hidden[i].weight) * s_list[i - 1]
            grad_x = g @ self.hidden[0].weight
            grad_norm = torch.sqrt(torch.sum(grad_x * grad_x, dim=-1) + 1e-8)
            return phi, grad_norm

    return GeoINRNet()


def _normalize_coords(points, center, extent, s_z):
    stretched = harness.stretch_points(points, center, s_z)
    half_box = 0.5 * np.array([extent[0], extent[1], float(s_z) * extent[2]], dtype=float)
    scale = float(np.max(half_box))
    return stretched / max(scale, 1e-12), half_box / max(scale, 1e-12)


def _phi_and_grad(net, x_tensor):
    torch, _, _ = _get_torch()
    x_req = x_tensor.detach().requires_grad_(True)
    phi = net(x_req)
    grad = torch.autograd.grad(
        outputs=phi.sum(), inputs=x_req, create_graph=True, retain_graph=True
    )[0]
    grad_norm = torch.sqrt(torch.sum(grad * grad, dim=-1) + 1e-8)
    return phi, grad_norm


def _validate_device(device):
    dev_str = str(device).strip()
    if "7" in dev_str:
        raise ValueError("GPU 7 is reserved for loop embeddings and must not be used")
    return dev_str


def train_geoinr(X_b, y_b, w_b, X_c, center, extent, *, s_z, lam, seed,
                 epochs=1000, domain_samples=2000, lr=1e-4, beta=100.0, device="cpu"):
    """Train GeoINR MLP on CPU or an explicitly free GPU and return predict(points_xyz)."""
    dev_str = _validate_device(device)
    torch, nn, F = _get_torch()
    rng = np.random.default_rng(int(seed))
    torch_seed = int(rng.integers(0, 2 ** 31 - 1))
    torch.manual_seed(torch_seed)

    dev = torch.device(dev_str)
    net = build_geoinr_net(hidden_layers=3, hidden_dim=256, beta=beta).to(dev)

    Xb_n, box_half = _normalize_coords(X_b, center, extent, s_z)
    ore_mask = y_b > 0.5
    bar_mask = ~ore_mask

    x_ore = torch.tensor(Xb_n[ore_mask], dtype=torch.float32, device=dev)
    w_ore = torch.tensor(w_b[ore_mask], dtype=torch.float32, device=dev)
    w_ore = w_ore / torch.clamp(w_ore.sum(), min=1e-12)

    x_bar = torch.tensor(Xb_n[bar_mask], dtype=torch.float32, device=dev) if bar_mask.any() else None
    w_bar = None
    if bar_mask.any():
        w_bar = torch.tensor(w_b[bar_mask], dtype=torch.float32, device=dev)
        w_bar = w_bar / torch.clamp(w_bar.sum(), min=1e-12)

    x_con = None
    if len(X_c):
        Xc_n, _ = _normalize_coords(X_c, center, extent, s_z)
        x_con = torch.tensor(Xc_n, dtype=torch.float32, device=dev)

    n_ore = len(x_ore)
    n_bar = len(x_bar) if x_bar is not None else 0
    n_con = len(x_con) if x_con is not None else 0

    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)))
    box_half_f32 = box_half.astype(np.float32)
    lam_f = float(lam)

    net.train()
    for _ in range(int(epochs)):
        u = rng.uniform(-1.0, 1.0, size=(int(domain_samples), 3)).astype(np.float32) * box_half_f32
        x_dom = torch.from_numpy(u).to(dev)

        parts = [x_ore]
        if n_bar:
            parts.append(x_bar)
        if n_con:
            parts.append(x_con)
        parts.append(x_dom)
        x_all = torch.cat(parts, dim=0)

        opt.zero_grad(set_to_none=True)
        phi_all, gnorm_all = net.forward_with_grad(x_all)
        delta_all = phi_all / gnorm_all

        idx = 0
        d_ore = delta_all[idx:idx + n_ore]
        idx += n_ore
        loss = torch.sum(w_ore * F.relu(-d_ore))

        if n_bar:
            d_bar = delta_all[idx:idx + n_bar]
            idx += n_bar
            loss = loss + torch.sum(w_bar * F.relu(d_bar))

        if n_con:
            d_con = delta_all[idx:idx + n_con]
            idx += n_con
            loss = loss + torch.mean(torch.abs(d_con))

        g_dom = gnorm_all[idx:]
        loss = loss + lam_f * torch.mean(torch.abs(g_dom - 1.0))

        if not torch.isfinite(loss):
            raise ValueError("non-finite GeoINR training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        opt.step()
        sched.step()

    net.eval()

    def predict(points_xyz):
        pts_n, _ = _normalize_coords(points_xyz, center, extent, s_z)
        with torch.no_grad():
            out_chunks = []
            for start in range(0, len(pts_n), 8192):
                xb = torch.tensor(pts_n[start:start + 8192], dtype=torch.float32, device=dev)
                out_chunks.append(net(xb).cpu().numpy().astype(np.float64))
        return np.concatenate(out_chunks, axis=0) if out_chunks else np.zeros(0, dtype=np.float64)

    return predict


def _limit_folds(folds, max_folds):
    if max_folds is None or len(folds) <= max_folds:
        return folds
    # Deterministic evenly spaced fold subset when fold count is capped for runtime (§3 M7).
    indices = [int(round(i * (len(folds) - 1) / max(1, max_folds - 1))) for i in range(max_folds)]
    seen = []
    for idx in indices:
        if idx not in seen:
            seen.append(idx)
    return [folds[i] for i in seen]


DEFAULT_INR_CONFIG = INR_CONFIGS[0]


def run_case(public_dir, out_dir, *, seed=0, cv_epochs=400, full_epochs=800,
             domain_samples=1000, max_cv_folds=3, lr=1e-3, wall_budget_seconds=600.0,
             device="cpu", mode="cv"):
    """Run M7 Stage-A on one (public_dir, seed) and write field.npy + choice.json."""
    if mode not in ("cv", "default"):
        raise ValueError(f"unknown GeoINR Stage-A mode: {mode!r}")
    dev_str = _validate_device(device)
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    if (out_dir / "choice.json").is_file() and (
        (out_dir / "field.npy").is_file()
        or json.loads((out_dir / "choice.json").read_text()).get("fallback")
        or json.loads((out_dir / "choice.json").read_text()).get("no_ore")
    ):
        return json.loads((out_dir / "choice.json").read_text())

    problem = read_problem(public_dir)
    data = harness.extract_public_data(problem)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (data["y_b"] > 0.5).any():
        choice = {
            "method": "inr_mlp",
            "mode": mode,
            "case_id": problem["case_id"],
            "seed": int(seed),
            "no_ore": True,
            "fallback": False,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    if mode == "default":
        cfg = DEFAULT_INR_CONFIG
        try:
            pred_fn = train_geoinr(
                data["X_b"], data["y_b"], data["w_b"], data["X_c"],
                data["center"], data["E"], s_z=cfg["s_z"], lam=cfg["lam"],
                seed=int(seed), epochs=full_epochs, domain_samples=domain_samples, lr=lr,
                device=dev_str,
            )
            field_def = harness.predict_chunked(pred_fn, data["G"]).astype(np.float32)
        except Exception:
            choice = {
                "method": "inr_mlp",
                "mode": "default",
                "case_id": problem["case_id"],
                "seed": int(seed),
                "no_ore": False,
                "fallback": True,
                "cv_evaluated": 0,
                "deadline_stopped": False,
                "wall_seconds": round(float(time.monotonic() - t0), 4),
            }
            (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
            return choice
        np.save(out_dir / "field.npy", field_def, allow_pickle=False)
        block = {
            "selected_rule": "native",
            "selected_config": cfg,
            "cv_score": None,
            "cv_evaluated": 0,
            "same_as_cv": True,
        }
        choice = {
            "method": "inr_mlp",
            "mode": "default",
            "case_id": problem["case_id"],
            "seed": int(seed),
            "no_ore": False,
            "fallback": False,
            "cv_evaluated": 0,
            "cv_folds_used": 0,
            "cv_epochs": 0,
            "full_epochs": int(full_epochs),
            "lr": float(lr),
            "deadline_stopped": False,
            "default": block,
            "native": block,
            "cv": block,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    all_folds = harness.make_hole_folds(data, problem)
    folds = _limit_folds(all_folds, max_cv_folds)
    deadline = t0 + 0.5 * float(wall_budget_seconds) if wall_budget_seconds > 0 else None

    best_cv_idx, best_cv_rule, best_cv_score = None, "share", -1.0
    best_nat_idx, best_nat_score = None, -1.0
    cv_evaluated = 0
    deadline_stopped = False

    if folds:
        for idx, cfg in enumerate(INR_CONFIGS):
            if deadline is not None and time.monotonic() >= deadline:
                deadline_stopped = True
                break
            counts = {"native": [0, 0, 0], "share": [0, 0, 0]}
            failed = False
            for f_i, fold in enumerate(folds):
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_stopped = True
                    failed = True
                    break
                fold_seed = int(seed) * 1009 + idx * 37 + f_i
                try:
                    pred_fn = train_geoinr(
                        fold["X_b_tr"], fold["y_b_tr"], fold["w_b_tr"], fold["X_c_tr"],
                        data["center"], data["E"], s_z=cfg["s_z"], lam=cfg["lam"],
                        seed=fold_seed, epochs=cv_epochs, domain_samples=domain_samples, lr=lr,
                        device=dev_str,
                    )
                    n_val = len(fold["X_val"])
                    f_both = harness.predict_chunked(pred_fn, np.vstack([fold["X_val"], data["S"]]))
                    f_val, f_S = f_both[:n_val], f_both[n_val:]
                except Exception:
                    failed = True
                    break
                y_true = fold["y_val"]
                for r in ("native", "share"):
                    if r == "native":
                        pred = f_val > 0.0
                    else:
                        q_cut = float(np.quantile(f_S, 1.0 - fold["p_train"]))
                        if np.any(f_S > q_cut) or float(f_S.max()) <= float(f_S.min()):
                            pred = f_val > q_cut
                        else:
                            pred = f_val >= q_cut
                    counts[r][0] += int(np.count_nonzero(pred & y_true))
                    counts[r][1] += int(np.count_nonzero(pred & ~y_true))
                    counts[r][2] += int(np.count_nonzero(~pred & y_true))
            if failed:
                if deadline_stopped:
                    break
                continue
            cv_evaluated += 1
            sc_nat = harness.pooled_jaccard(*counts["native"])
            sc_shr = harness.pooled_jaccard(*counts["share"])
            if sc_nat > best_nat_score + harness.TIE_TOL:
                best_nat_score = sc_nat
                best_nat_idx = idx
            for r, sc in (("native", sc_nat), ("share", sc_shr)):
                if sc > best_cv_score + harness.TIE_TOL:
                    best_cv_score = sc
                    best_cv_idx = idx
                    best_cv_rule = r

    def _fit_ordered(preferred_idx):
        order = (
            list(range(len(INR_CONFIGS)))
            if preferred_idx is None
            else [preferred_idx] + [i for i in range(len(INR_CONFIGS)) if i != preferred_idx]
        )
        for i in order:
            cfg = INR_CONFIGS[i]
            try:
                pred_fn = train_geoinr(
                    data["X_b"], data["y_b"], data["w_b"], data["X_c"],
                    data["center"], data["E"], s_z=cfg["s_z"], lam=cfg["lam"],
                    seed=int(seed) * 10007 + i, epochs=full_epochs, domain_samples=domain_samples, lr=lr,
                    device=dev_str,
                )
                f_grid = harness.predict_chunked(pred_fn, data["G"])
                return i, f_grid.astype(np.float32)
            except Exception:
                continue
        return None, None

    fit_cv_idx, field_cv = _fit_ordered(best_cv_idx)
    if field_cv is None:
        choice = {
            "method": "inr_mlp",
            "case_id": problem["case_id"],
            "seed": int(seed),
            "no_ore": False,
            "fallback": True,
            "cv_evaluated": cv_evaluated,
            "deadline_stopped": deadline_stopped,
            "wall_seconds": round(float(time.monotonic() - t0), 4),
        }
        (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
        return choice

    np.save(out_dir / "field.npy", field_cv, allow_pickle=False)

    target_nat_idx = best_nat_idx if best_nat_idx is not None else fit_cv_idx
    same_as_cv = target_nat_idx == fit_cv_idx
    fit_nat_idx = fit_cv_idx
    if not same_as_cv:
        cand_idx, field_nat = _fit_ordered(target_nat_idx)
        if field_nat is not None and cand_idx != fit_cv_idx:
            fit_nat_idx = cand_idx
            np.save(out_dir / "field_native.npy", field_nat, allow_pickle=False)
        else:
            same_as_cv = True

    choice = {
        "method": "inr_mlp",
        "case_id": problem["case_id"],
        "seed": int(seed),
        "no_ore": False,
        "fallback": False,
        "cv_evaluated": cv_evaluated,
        "cv_folds_used": len(folds),
        "cv_epochs": int(cv_epochs),
        "full_epochs": int(full_epochs),
        "lr": float(lr),
        "deadline_stopped": deadline_stopped,
        "cv": {
            "selected_rule": best_cv_rule if best_cv_idx is not None else "share",
            "selected_config": INR_CONFIGS[fit_cv_idx],
            "cv_score": best_cv_score if fit_cv_idx == best_cv_idx and best_cv_idx is not None else None,
            "cv_evaluated": cv_evaluated,
        },
        "native": {
            "selected_rule": "native",
            "selected_config": INR_CONFIGS[fit_nat_idx],
            "cv_score": best_nat_score if fit_nat_idx == best_nat_idx and best_nat_idx is not None else None,
            "cv_evaluated": cv_evaluated,
            "same_as_cv": same_as_cv,
        },
        "wall_seconds": round(float(time.monotonic() - t0), 4),
    }
    (out_dir / "choice.json").write_text(json.dumps(choice, indent=2, sort_keys=True))
    return choice


def _worker(task):
    pub_dir, out_dir, seed, cv_ep, full_ep, dom_s, max_f, lr, budget_s, dev_str, mode = task
    _validate_device(dev_str)
    if dev_str == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    _get_torch()
    return run_case(
        pub_dir, out_dir, seed=seed, cv_epochs=cv_ep, full_epochs=full_ep,
        domain_samples=dom_s, max_cv_folds=max_f, lr=lr, wall_budget_seconds=budget_s,
        device=dev_str, mode=mode,
    )


def run_split(release_dir, split, output_root, *, seeds=(0,), workers=8,
              cv_epochs=400, full_epochs=800, domain_samples=1000, max_cv_folds=3,
              lr=1e-3, wall_budget_seconds=900.0, device="cpu", mode="cv"):
    dev_str = _validate_device(device)
    release_dir = Path(release_dir).resolve()
    manifest_path = release_dir / split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = [
        (
            str(manifest_path.parent / c["public"]),
            str(Path(output_root) / split / c["case_id"] / str(int(s))),
            int(s), int(cv_epochs), int(full_epochs), int(domain_samples), int(max_cv_folds),
            float(lr), float(wall_budget_seconds), dev_str, mode,
        )
        for s in seeds
        for c in manifest["cases"]
    ]
    if workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), 16)) as pool:
            results = list(pool.map(_worker, tasks))
    return {
        "release": str(release_dir),
        "split": split,
        "mode": mode,
        "seeds": [int(s) for s in seeds],
        "runs": len(results),
        "fallbacks": sum(1 for r in results if r.get("fallback")),
        "median_seconds": float(np.median([r["wall_seconds"] for r in results])) if results else 0.0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run M7 GeoINR MLP Stage-A on public split inputs.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--split", required=True,
        choices=("calibration", "train", "train_s", "train_m", "validation", "test_id", "test_ood"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated integer seeds.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cv-epochs", type=int, default=400)
    parser.add_argument("--full-epochs", type=int, default=800)
    parser.add_argument("--domain-samples", type=int, default=1000)
    parser.add_argument("--max-cv-folds", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wall-budget-seconds", type=float, default=900.0)
    parser.add_argument("--device", default="cpu", help="Torch device (cpu or free cuda:<id>, never GPU 7).")
    parser.add_argument("--mode", default="cv", choices=("cv", "default"))
    args = parser.parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    summary = run_split(
        args.release, args.split, args.output, seeds=seeds, workers=args.workers,
        cv_epochs=args.cv_epochs, full_epochs=args.full_epochs,
        domain_samples=args.domain_samples, max_cv_folds=args.max_cv_folds,
        lr=args.lr, wall_budget_seconds=args.wall_budget_seconds, device=args.device,
        mode=args.mode,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
