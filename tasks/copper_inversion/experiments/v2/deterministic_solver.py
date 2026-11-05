"""Public observations -> distance/RBF field -> shared-objective local search."""
import time
MODULE_STARTED=time.monotonic()
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,binary_dilation,binary_erosion
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from tasks.copper_inversion.v2.baselines import initialize
from tasks.copper_inversion.v2.geometry import canonicalize
from tasks.copper_inversion.v2.objective import score

METHOD="distance_local"


def solve(problem,budget,seed):
    start=MODULE_STARTED;deadline=start+.8*float(budget["wall_time_seconds"])
    spacing=np.asarray(problem["spacing"]);unit=float(spacing.min())
    feasible=problem["domain_mask"]&~problem["forbidden_mask"]
    best=initialize(problem);best_loss=score(problem,best)["loss_total"]
    calls=1;fallback=False
    points=np.concatenate([o["points"] for o in problem["observations"]])
    labels=np.concatenate([o["labels"] for o in problem["observations"]])
    xyz=np.stack(np.meshgrid(*[problem["origin"][i]+(np.arange(n)+.5)*spacing[i]
        for i,n in enumerate(problem["shape"])],indexing="ij"),axis=-1)
    field=canonicalize(problem,best)[1]/unit
    if METHOD=="contact_rbf":
        # Label constraints + zero-valued contacts, physical isotropic scaling.
        # Rank-deficient or insufficient measurements have a declared fallback.
        unique,inverse=np.unique(points,axis=0,return_inverse=True)
        values=np.bincount(inverse,weights=2*labels-1)/np.bincount(inverse)
        if len(unique)>=4 and np.linalg.matrix_rank(unique-unique.mean(0))==3:
            try:
                scale=float(np.linalg.norm(np.asarray(problem["shape"])*spacing))
                origin=np.asarray(problem["origin"])
                rbf=RBFInterpolator((unique-origin)/scale,values,smoothing=.05,kernel="thin_plate_spline",neighbors=min(64,len(unique)))
                field=rbf((xyz.reshape(-1,3)-origin)/scale).reshape(problem["shape"])
            except np.linalg.LinAlgError:
                fallback=True
        else:fallback=True
    curve=[dict(calls=calls,seconds=time.monotonic()-start,best_public_loss=best_loss)]

    def evaluate(candidate):
        nonlocal best,best_loss,calls
        candidate=np.where(feasible,candidate,-1.).astype(float)
        try:value=score(problem,candidate)["loss_total"]
        except ValueError:return
        calls+=1
        if value<best_loss:
            best,best_loss=candidate.copy(),value
            curve.append(dict(calls=calls,seconds=time.monotonic()-start,best_public_loss=best_loss))

    # Public target selects smoothness/threshold; no hidden score is queried.
    thresholds=(-.25,0.,.25) if METHOD=="contact_rbf" and not fallback else (-1.,0.,1.)
    for width in (0.,1.,2.):
        smoothed=gaussian_filter(field,width*unit/spacing) if width else field
        for threshold in thresholds:
            if time.monotonic()>=deadline:break
            evaluate(smoothed-threshold)
    # Deterministic boundary patches; physical radii, legal mask, strict descent.
    coords=np.indices(problem["shape"])
    sweeps=0
    while time.monotonic()<deadline:
        before=best_loss;mask=best>0
        boundary=(binary_dilation(mask)^binary_erosion(mask))&feasible
        locations=np.argwhere(boundary)
        if not len(locations):break
        # Prioritize measured regions without assuming observations determine
        # the missing-region sign. Fixed tie order makes this deterministic.
        distances=cKDTree(points).query(xyz[boundary])[0]
        order=np.argsort(distances,kind="stable")
        for j in order[::max(1,len(order)//128)]:
            if time.monotonic()>=deadline:break
            center=locations[j]
            for radius in (unit,2*unit):
                if time.monotonic()>=deadline:break
                patch=sum(((coords[k]-center[k])*spacing[k])**2 for k in range(3))<=radius**2
                patch &= feasible
                proposal=np.where(best>0,1.,-1.);proposal[patch]*=-1
                evaluate(proposal)
        sweeps+=1
        if best_loss>=before:break
    Path("/output/search.json").write_text(json.dumps(dict(method=METHOD,calls=calls,sweeps=sweeps,
        rbf_fallback=fallback,best_public_loss=best_loss,seconds=time.monotonic()-start,curve=curve)))
    return best
