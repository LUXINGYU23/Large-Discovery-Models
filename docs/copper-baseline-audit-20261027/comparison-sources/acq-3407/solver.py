"""Projected stochastic / acquisition-guided optimization, adapted from MCMC.

Reuse package Gaussian fields, feature map and online Bayesian ridge. Preserve
the paper's squared-loss acceptance. Adapt distances, step and covariance ranges
to physical units; enforce public masks, score with objective_v2, retain only the
best legal state instead of allocating the original complete chain history.
The hard-prior projection makes proposals asymmetric: no posterior sampling
correctness is claimed. Historical r3 sources remain frozen separately.
"""
import time
import signal
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import skfmm
from scipy import ndimage

sys.path[:0]=["/baseline","/baseline/Knowledge-driven-stochastic-modeling"]
from run_3d_comparison import gaussian, default_config, cheap_features, OnlineBayesianRidge
from tasks.copper_inversion.v2.baselines import initialize, constrained_initial
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.objective import score

# Controller seals these parameters into each candidate source before evaluation.
CONFIG={'batch': 4, 'beta': 0.9091132879257202, 'initializer': 'random_field', 'method': 'acq_v2', 'range_scale': 0.13884450815016813, 'step_cells': 0.5111154545585631, 'temperature': 0.9886021103844687, 'warmup': 4}


def project_proposal(problem, field):
    """Project a stochastic optimization proposal onto public hard priors.

    Keep the largest permitted components; use the existing connected ranked
    construction only when a hard volume bound requires adding/removing mass.
    Legal proposals retain their continuous field exactly. This projection is
    not symmetric, so this optimizer must not be advertised as posterior MCMC.
    """
    mask=field>0
    maximum=problem['priors']['max_components']
    if maximum is not None:
        labels,count=ndimage.label(mask)
        if count>maximum:
            sizes=np.bincount(labels.ravel())[1:]
            keep=np.argsort(-sizes,kind='stable')[:maximum]+1
            mask=np.isin(labels,keep)
    bounds=problem['priors']['hard_volume_bounds']
    volume=int(mask.sum())*float(np.prod(problem['spacing']))
    if bounds is not None and not bounds[0]<=volume<=bounds[1]:
        mask=constrained_initial(problem,mask,field)>0
    changed=mask!=(field>0)
    if not changed.any():
        return field,0
    result=field.copy()
    magnitude=np.maximum(np.abs(field[changed]),float(min(problem['spacing'])))
    result[changed]=np.where(mask[changed],magnitude,-magnitude)
    return result,int(changed.sum())


def search_distance(problem, field):
    """Validate occupancy while retaining the proposal's subvoxel interface.

    The scorer canonicalizes occupancy independently. Applying that operation
    to the chain state erases accumulated motion smaller than one voxel.
    """
    mask, canonical = canonicalize(problem, field)
    if np.any(field > 0) and np.any(field < 0):
        # First-order marching preserves the signed interface on anisotropic
        # grids where the default second-order stencil can change voxel signs.
        # Keep subvoxel geometry; never quantize the chain to binary occupancy.
        distance = np.asarray(skfmm.distance(np.ascontiguousarray(field),
                                            dx=tuple(problem["spacing"]), order=1))
        # Retain the invariant even with first order: no silent sign repair.
        if not np.isfinite(distance).all() or not np.array_equal(distance > 0, mask):
            raise ValueError("distance reinitialization changed validated occupancy")
        return distance
    return canonical


class SearchDeadline(Exception):
    """Stop an unfinished proposal without discarding the best legal state."""


def stop_search(signum, frame):
    raise SearchDeadline


def solve(problem,budget,seed):
    # The runner has already deducted imports from the remaining allowance.
    # Start at solve entry so that setup time is not charged a second time.
    start=time.monotonic();deadline=start+.75*float(budget["wall_time_seconds"])
    np.random.seed(seed)
    spacing=np.asarray(problem["spacing"]);unit=float(spacing.min())
    feasible=problem["domain_mask"]&~problem["forbidden_mask"]
    centers=tuple(problem["origin"][i]+(np.arange(n)+.5)*spacing[i] for i,n in enumerate(problem["shape"]))
    cfg=default_config()
    for key in ("range_x","range_y","range_z"):
        cfg[key]=[v*unit*CONFIG["range_scale"] for v in cfg[key]]
    initial=initialize(problem)
    best_phi=initial.copy();best=None;curve=[]
    accepted=invalid=proposals=decisions=0
    invalid_reasons=Counter()
    acquisition_batch_sizes=Counter()
    first_acquisition_seconds=None
    surrogate_samples=0
    projected_proposals=projected_voxels=unchanged_occupancy=0
    previous_handler=signal.signal(signal.SIGALRM,stop_search)
    try:
        try:
            # Checking only between proposals cannot bound GSTools sampling or
            # surrogate fitting. Interrupt unfinished work at the soft deadline;
            # the independent container hard deadline remains unchanged.
            if time.monotonic()>=deadline:raise SearchDeadline
            signal.setitimer(signal.ITIMER_REAL,deadline-time.monotonic())
            if CONFIG["initializer"]=="random_field":
                random_field=gaussian(centers,cfg).field_3d[0]
                initial=constrained_initial(problem,(random_field>0)&feasible,random_field)
            phi=canonicalize(problem,initial)[1]
            current=score(problem,phi)["loss_total"];best=current;best_phi=phi.copy()
            surrogate=OnlineBayesianRidge(len(cheap_features(phi/unit,np.zeros(7))))
            curve.append(dict(call=1,loss=current,best_loss=best,seconds=time.monotonic()-start))
            while time.monotonic()<deadline:
                candidates=[];features=[]
                acquisition=CONFIG["method"]=="acq_v2" and len(surrogate.y)>=CONFIG["warmup"]
                for _ in range(CONFIG["batch"] if acquisition else 1):
                    if time.monotonic()>=deadline:break
                    velocity,theta=gaussian(centers,cfg).field_3d
                    proposals+=1
                    try:
                        if np.any(phi>0) and np.any(phi<0):
                            _,extended=skfmm.extension_velocities(np.ascontiguousarray(phi),velocity,dx=tuple(spacing),order=1)
                            speed=float(np.max(np.abs(extended)))
                            if not np.isfinite(speed) or speed==0:raise ValueError("degenerate velocity")
                            proposal=phi-np.random.uniform(0,CONFIG["step_cells"])*unit*extended/speed
                        else:
                            # Explicit birth move for empty/full states: no observed or
                            # hidden geometry centre is inserted into the initial model.
                            proposal=velocity-2.
                        proposal=np.where(feasible,proposal,-unit)
                        proposal,changed=project_proposal(problem,proposal)
                        projected_proposals+=int(changed>0)
                        projected_voxels+=changed
                        proposal=search_distance(problem,np.asarray(proposal,dtype=float))
                    except ValueError as exc:
                        invalid_reasons[str(exc)]+=1
                        invalid+=1
                        continue
                    candidates.append(proposal)
                    scaled_theta=theta.copy();scaled_theta[2:5]/=unit
                    features.append(cheap_features(proposal/unit,scaled_theta))
                if not candidates:continue
                if time.monotonic()>=deadline:break
                index=0
                if acquisition:
                    acquisition_batch_sizes[len(candidates)]+=1
                    if first_acquisition_seconds is None:
                        first_acquisition_seconds=time.monotonic()-start
                    surrogate.fit();mu,sigma=surrogate.predict(features)
                    index=int(np.argmin(mu-CONFIG["beta"]*sigma));decisions+=1
                candidate=candidates[index]
                unchanged_occupancy+=int(np.array_equal(candidate>0,phi>0))
                value=score(problem,candidate)["loss_total"]
                surrogate.add(features[index],value)
                surrogate_samples+=1
                if np.log(max(np.random.uniform(),np.finfo(float).tiny)) <= (current**2-value**2)/CONFIG["temperature"]:
                    phi,current=candidate,value;accepted+=1
                if value<best:best,best_phi=value,candidate.copy()
                curve.append(dict(call=len(curve)+1,loss=value,best_loss=best,seconds=time.monotonic()-start))
        finally:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
    except SearchDeadline:
        # The one-shot signal has been consumed, including if it interrupted
        # the inner finally before SIG_IGN was installed. No new alarm is armed.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    Path("/output/search.json").write_text(json.dumps(dict(config=CONFIG,seed=seed,calls=len(curve),
        proposals=proposals,accepted=accepted,invalid=invalid,invalid_reasons=dict(invalid_reasons),acquisition_decisions=decisions,
        acquisition_batch_sizes=dict(acquisition_batch_sizes),first_acquisition_seconds=first_acquisition_seconds,surrogate_samples=surrogate_samples,
        projected_proposals=projected_proposals,projected_voxels=projected_voxels,unchanged_occupancy=unchanged_occupancy,
        seconds=time.monotonic()-start,best_public_loss=best,curve=curve)))
    return best_phi

_native_solve = solve
def solve(problem, budget, seed):
    import os
    target=float(os.environ["COPPER_WORKLOAD_DEADLINE_MONOTONIC"])-5.0-1.0
    remaining=max(.01,target-time.monotonic())
    adjusted=dict(budget);adjusted["wall_time_seconds"]=remaining/0.75
    return _native_solve(problem,adjusted,seed)
