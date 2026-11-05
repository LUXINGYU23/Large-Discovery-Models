"""Adapter around original published/package algorithms; no LDM calls or tuning."""
import sys
sys.path[:0] = ['/baseline', '/baseline/Knowledge-driven-stochastic-modeling']
import json
import time
from pathlib import Path
import numpy as np
import skfmm
import run_3d_comparison as portable
from shpmc.level_set_mc import StochasticLevelSet3D
import shpmc.level_set_mc as upstream


def finite_binary_loss(labels, distances, contacts, weight):
    # Original equation, stable log(1+exp(x)); absent contact constraints add zero.
    logistic = np.logaddexp(0, distances[labels == 0]).sum() + np.logaddexp(0, -distances[labels == 1]).sum()
    contact = contacts.mean()**2 + np.mean(contacts**2) if contacts.size else 0.
    return float(weight * (logistic + contact))


upstream.loss_function_binary = finite_binary_loss

# Filled by the experiment controller before sealing this source.
METHOD = 'paper_mcmc'
MAX_CALLS = 1000
OBJECTIVE = 'aligned'
ACQ_WARMUP = 30


class BudgetStop(Exception):
    pass


def solve(problem, budget, seed):
    started = time.monotonic()
    deadline = started + max(1., budget['wall_time_seconds'] - 10.)
    np.random.seed(seed)
    # ReproducibleGaussianField changes only the unseeded GSTools realization RNG;
    # ranges, kernel, proposal equation and native acceptance equation stay intact.
    centers = tuple(problem['origin'][i] + (np.arange(n)+.5)*problem['spacing'][i]
                    for i,n in enumerate(problem['shape']))
    initial = np.ascontiguousarray(problem['initial_level_set'])
    gaussian = portable.gaussian(centers, portable.default_config())
    objective = StochasticLevelSet3D([o['values'] for o in problem['observations']],
        [initial], gaussian, 1., [o['weight'] for o in problem['observations']])
    invalid_proposals = 0
    original_perturbation = portable.level_set_perturbation
    def bounded_perturbation(phi, velocity, max_step):
        nonlocal invalid_proposals
        try:
            return original_perturbation(phi, velocity, max_step)
        except ValueError as exc:
            if 'no zero contour' not in str(exc):
                raise
            # A proposal outside the valid two-phase domain is a self-loop.
            # Its subsequent objective call still consumes the query budget.
            invalid_proposals += 1
            return phi.copy()
    upstream.level_set_perturbation = bounded_perturbation
    acquisition_decisions = 0
    original_predict = portable.OnlineBayesianRidge.predict
    def counted_predict(surrogate, features):
        nonlocal acquisition_decisions
        acquisition_decisions += 1
        return original_predict(surrogate, features)
    portable.OnlineBayesianRidge.predict = counted_predict
    original = objective.loss_computation
    best, best_phi = float('inf'), initial.copy()
    rows = []
    def measured(phi):
        nonlocal best, best_phi
        if len(rows) >= MAX_CALLS or (rows and time.monotonic() >= deadline):
            raise BudgetStop()
        if OBJECTIVE == 'aligned':
            from tasks.copper_inversion.core.objective import score
            metrics = score(problem, phi)
            value = metrics['loss_total']
            parts = np.array([metrics[f"loss_{o['name']}"] for o in problem['observations']])
        else:
            value, parts = original(phi)
        if not np.isfinite(value):
            raise ValueError('original objective nonfinite')
        if value < best:
            best, best_phi = float(value), phi.copy()
        rows.append(dict(call=len(rows)+1, loss=float(value), best_loss=best,
                         elapsed_seconds=time.monotonic()-started))
        return value, parts
    objective.loss_computation = measured
    # Give aligned baselines the identical initial field as an explicitly counted query.
    if OBJECTIVE == 'aligned':
        measured(initial)
    stopped = 'call_budget'
    try:
        if METHOD == 'paper_mcmc':
            objective.mcmc_sampling_single_chain(MAX_CALLS,
                temperature=20 if 'outcrops' in problem['case_id'] else 3)
        else:
            portable.load_case = lambda: (centers, initial, objective)
            portable.run_method(METHOD, MAX_CALLS-1, seed, batch=4, warmup=ACQ_WARMUP, beta=.75)
    except BudgetStop:
        stopped = 'time_budget' if time.monotonic() >= deadline else 'call_budget'
    Path('/output/search.json').write_text(json.dumps(dict(method=METHOD, objective=OBJECTIVE, seed=seed,
        calls=len(rows), max_calls=MAX_CALLS, acq_warmup=ACQ_WARMUP, acquisition_decisions=acquisition_decisions, invalid_proposals=invalid_proposals, stop=stopped, elapsed_seconds=time.monotonic()-started,
        best_raw_loss=best, curve=rows)))
    return np.ascontiguousarray(best_phi)
