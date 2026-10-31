"""core/ldm/acquisition/parallel_search.py — parallel local search + sampling.

Executes LocalSearch atoms via batch hill-climbing and Sampling atoms via
batch random generation.  All GP evaluations are batched for efficiency.

Returns ALL evaluated points (not just per-worker optima) so the session
can pool, rank, and present top-k to the LLM for review.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

from tasks.antibody.core.cdr_constraints import check_cdr_constraints
from tasks.antibody.core.ldm.dsl.alphabet import SEQ_LEN, hamming
from tasks.antibody.core.ldm.dsl.search_space import (
    LatinHyperCubeSampling,
    LocalSearch,
    NeighborSampling,
    Or,
    SearchSpaceAtom,
)


def _eval_batch(
    seqs: list[list[int]],
    gp,
    f_acq,
    bias_dsl,
    bias_weight: float,
    device: torch.device,
    acq_name: str,
) -> list[dict]:
    """Batch-evaluate sequences and return scored dicts."""
    if not seqs:
        return []
    arr = torch.tensor(np.array(seqs), dtype=torch.float32, device=device)
    with torch.no_grad():
        posterior = gp.likelihood(gp(arr))
        acq_vals = f_acq(arr).cpu().numpy().flatten()
        mu = posterior.mean.cpu().numpy().flatten()
        sigma = posterior.stddev.cpu().numpy().flatten()

    if bias_dsl is not None:
        bias_vals = np.array([bias_dsl(s) for s in seqs], dtype=float)
    else:
        bias_vals = np.zeros(len(seqs))

    combined = acq_vals + bias_weight * bias_vals

    return [
        {
            "seq": seqs[j],
            acq_name: float(acq_vals[j]),
            "mu": float(mu[j]),
            "sigma": float(sigma[j]),
            "bias": float(bias_vals[j]),
            f"bias+{acq_name}": float(combined[j]),
        }
        for j in range(len(seqs))
    ]


def _sample_neighbour(x: list[int], n_categories: list[int]) -> list[int]:
    """One-position random mutation."""
    x_pert = list(x)
    choice = np.random.randint(0, len(n_categories))
    curr = x_pert[choice]
    options = [i for i in range(n_categories[choice]) if i != curr]
    x_pert[choice] = int(np.random.choice(options))
    return x_pert


def parallel_local_search(
    atoms: list[LocalSearch],
    gp,
    f_acq: Callable,
    bias_dsl,
    bias_weight: float,
    config: np.ndarray,
    cdr_constraints: bool,
    device: torch.device,
    acq_name: str = "ei",
    timeout_s: float = 30.0,
) -> list[dict]:
    """Execute LocalSearch atoms via batched parallel hill-climbing.

    Returns ALL evaluated points with acquisition, posterior, and bias scores.
    """
    n_categories = list(config)
    deadline = time.time() + timeout_s

    # Build workers
    workers: list[dict] = []
    for atom_idx, atom in enumerate(atoms):
        center = atom.center_idx
        for r in range(atom.restart):
            workers.append({
                "atom_idx": atom_idx,
                "restart": r,
                "center": center,
                "radius": atom.radius,
                "fixed_positions": atom.fixed_positions,
                "max_steps": atom.steps,
                "step": 0,
                "x": list(center),
                "best_combined": None,
                "tol": 100,
                "active": True,
                "trajectory": set([tuple(center)]),
            })

    if not workers:
        return []

    all_evaluated: list[dict] = []

    # Evaluate all centers (batch)
    center_seqs = [w["x"] for w in workers]
    center_results = _eval_batch(
        center_seqs, gp, f_acq, bias_dsl, bias_weight, device, acq_name,
    )
    for i, w in enumerate(workers):
        r = center_results[i]
        r["source"] = f"LocalSearch(center={atoms[w['atom_idx']].center}, restart={w['restart']}, step=0)"
        all_evaluated.append(r)
        w["best_combined"] = r[f"bias+{acq_name}"]

    max_steps = max(w["max_steps"] for w in workers)

    for step in range(1, max_steps + 1):
        if time.time() > deadline:
            break

        # Deactivate workers past their step limit
        for w in workers:
            if w["active"] and w["step"] >= w["max_steps"]:
                w["active"] = False

        # Generate neighbours for active workers
        neighbours: list[list[int]] = []
        worker_ids: list[int] = []
        for i, w in enumerate(workers):
            if not w["active"]:
                continue
            found = False
            attempts = 0
            while attempts < w["tol"]:
                attempts += 1
                nb = _sample_neighbour(w["x"], n_categories)
                key = tuple(nb)

                # Check fixed positions
                if any(nb[p] != w["center"][p] for p in w["fixed_positions"]):
                    continue
                # Check radius
                if w["radius"] is not None:
                    if hamming(nb, w["center"]) > w["radius"]:
                        continue
                # Check not visited
                if key in w["trajectory"]:
                    continue
                # CDR constraints
                if cdr_constraints:
                    if not check_cdr_constraints(nb):
                        continue

                neighbours.append(nb)
                worker_ids.append(i)
                found = True
                break

            if not found:
                w["active"] = False

        if not neighbours:
            break

        # Batch GP evaluation
        results = _eval_batch(
            neighbours, gp, f_acq, bias_dsl, bias_weight, device, acq_name,
        )
        for j, res in enumerate(results):
            wid = worker_ids[j]
            res["source"] = (
                f"LocalSearch(center={atoms[workers[wid]['atom_idx']].center}, "
                f"restart={workers[wid]['restart']}, step={step})"
            )
            all_evaluated.append(res)

        # Greedy ascent
        for j, wid in enumerate(worker_ids):
            w = workers[wid]
            w["trajectory"].add(tuple(neighbours[j]))
            w["step"] += 1
            combined = results[j][f"bias+{acq_name}"]
            if combined > w["best_combined"]:
                w["x"] = list(neighbours[j])
                w["best_combined"] = combined

    return all_evaluated


def execute_sampling_atoms(
    atoms: list[NeighborSampling | LatinHyperCubeSampling],
    gp,
    f_acq: Callable,
    bias_dsl,
    bias_weight: float,
    rng: np.random.Generator,
    timeout_s: float,
    device: torch.device,
    acq_name: str = "ei",
) -> list[dict]:
    """Execute Sampling atoms and evaluate all candidates."""
    all_candidates: list[list[int]] = []

    for atom in atoms:
        n = atom.budget
        samples = atom.sample(n=n, rng=rng, timeout_s=timeout_s)
        all_candidates.extend(samples)

    return _eval_batch(
        all_candidates, gp, f_acq, bias_dsl, bias_weight, device, acq_name,
    )


def execute_atoms(
    search_dsl: SearchSpaceAtom,
    gp,
    f_acq: Callable,
    bias_dsl,
    bias_weight: float,
    config: np.ndarray,
    cdr_constraints: bool,
    rng: np.random.Generator,
    timeout_s: float,
    device: torch.device,
    acq_name: str = "ei",
) -> list[dict]:
    """Dispatch search_dsl atoms to parallel_local_search or sampling."""
    # Flatten Or into individual atoms
    atoms: list[SearchSpaceAtom] = []
    if isinstance(search_dsl, Or):
        atoms = search_dsl.children
    else:
        atoms = [search_dsl]

    ls_atoms = [a for a in atoms if isinstance(a, LocalSearch)]
    samp_atoms = [a for a in atoms if isinstance(a, (NeighborSampling, LatinHyperCubeSampling))]

    results: list[dict] = []

    if ls_atoms:
        results.extend(parallel_local_search(
            ls_atoms, gp, f_acq, bias_dsl, bias_weight,
            config, cdr_constraints, device, acq_name, timeout_s,
        ))

    if samp_atoms:
        results.extend(execute_sampling_atoms(
            samp_atoms, gp, f_acq, bias_dsl, bias_weight,
            rng, timeout_s, device, acq_name,
        ))

    return results
