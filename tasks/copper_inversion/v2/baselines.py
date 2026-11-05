"""Public-data-only development baselines for the shared v2 objective.

These implement deterministic initialization and a Metropolis sampler. They are
not yet a tuned or qualified reproduction of the original paper's sampler.
"""

import time
import heapq
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from .contract import validate
from .objective import score
from .geometry import canonicalize


def constrained_initial(problem, mask, rank):
    """Keep an already legal estimate; otherwise build a ranked connected body.

    This uses only observations and the public domain/prior, never hidden truth.
    It also makes a nonempty population prior explicit when no positive sample
    is observed, rather than silently returning an illegal empty initialization.
    """
    field=np.where(mask,1.,-1.)
    try:
        canonicalize(problem,field)
        return field
    except ValueError:
        pass
    feasible=problem['domain_mask'] & ~problem['forbidden_mask']
    labels,count=ndimage.label(feasible)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    cell=float(np.prod(problem['spacing']));bounds=problem['priors']['hard_volume_bounds']
    lo=0 if bounds is None else int(np.ceil(bounds[0]/cell-1e-9))
    hi=int(feasible.sum()) if bounds is None else int(np.floor(bounds[1]/cell+1e-9))
    maximum=problem['priors']['max_components']
    regions=sorted(range(1,count+1),key=lambda i:(-sizes[i],i))
    if maximum is not None:regions=regions[:maximum]
    capacity=int(sum(sizes[i] for i in regions))
    target=min(max(int(mask.sum()),lo),hi,capacity)
    if target<lo:raise ValueError('public domain cannot satisfy initial volume/connectivity bounds')
    result=np.zeros(feasible.shape,bool);visited=np.zeros(feasible.shape,bool);filled=0
    for region in regions:
        if filled>=target:break
        indices=np.argwhere(labels==region)
        start=tuple(indices[np.argmax(rank[labels==region])]);heap=[(-float(rank[start]),start)];visited[start]=True
        while heap and filled<target:
            _,position=heapq.heappop(heap);result[position]=True;filled+=1
            for axis in range(3):
                for direction in (-1,1):
                    q=list(position);q[axis]+=direction;q=tuple(q)
                    if not 0<=q[axis]<feasible.shape[axis] or visited[q] or labels[q]!=region:continue
                    visited[q]=True;heapq.heappush(heap,(-float(rank[q]),q))
    field=np.where(result,1.,-1.)
    canonicalize(problem,field)
    return field


def initialize(problem):
    """Nearest positive/negative observations, with no truth-derived seed body."""
    validate(problem)
    xyz = np.stack(np.meshgrid(*[
        problem["origin"][i] + (np.arange(n) + .5) * problem["spacing"][i]
        for i, n in enumerate(problem["shape"])], indexing="ij"), axis=-1)
    points = np.concatenate([o["points"] for o in problem["observations"]])
    labels = np.concatenate([o["labels"] for o in problem["observations"]])
    positive, negative = points[labels == 1], points[labels == 0]
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    if not len(positive):
        mask = np.zeros(tuple(problem["shape"]), bool)
        center=np.asarray(problem['origin'])+.5*np.asarray(problem['shape'])*problem['spacing']
        rank=-np.linalg.norm(xyz-center,axis=-1)
    else:
        dp = cKDTree(positive).query(xyz.reshape(-1, 3))[0].reshape(problem["shape"])
        if len(negative):
            dn = cKDTree(negative).query(xyz.reshape(-1, 3))[0].reshape(problem["shape"])
            mask = dp < dn
            rank=dn-dp
        else:
            # A public resolution-based radius, not a hidden volume estimate.
            mask = dp <= 2 * np.linalg.norm(problem["spacing"])
            rank=-dp
        mask &= feasible
    return constrained_initial(problem,mask,rank)


def deterministic(problem, budget, seed):
    return initialize(problem)


def metropolis(problem, budget, seed, *, temperature=.02, radius_fraction=.1):
    """Symmetric local occupancy flips, minimizing the same public objective."""
    started = time.monotonic()
    seconds = float(budget["wall_time_seconds"])
    if not np.isfinite(seconds) or seconds <= 0 or temperature <= 0 or radius_fraction <= 0:
        raise ValueError("invalid sampler budget or parameters")
    # Reserve time for process teardown/serialization; same convention for LDM.
    deadline = started + .9 * seconds
    rng = np.random.default_rng(seed)
    field = initialize(problem)
    current = score(problem, field)["loss_total"]
    best, best_loss = field.copy(), current
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    centers = np.argwhere(feasible)
    coords = np.indices(problem["shape"], dtype=float)
    spacing = np.asarray(problem["spacing"])
    length = float(np.linalg.norm(np.asarray(problem["shape"]) * spacing))
    proposals = accepted = 0
    while time.monotonic() < deadline:
        center = centers[rng.integers(len(centers))]
        radius = max(float(spacing.min()), length * radius_fraction * rng.uniform(.25, 1.))
        patch = sum(((coords[i] - center[i]) * spacing[i])**2 for i in range(3)) <= radius**2
        patch &= feasible
        proposal = field.copy(); proposal[patch] *= -1
        proposals += 1
        try:
            loss = score(problem, proposal)["loss_total"]
        except ValueError:
            continue
        if np.log(max(rng.random(), np.finfo(float).tiny)) < min(0., (current - loss) / temperature):
            field, current = proposal, loss
            accepted += 1
            if loss < best_loss:
                best, best_loss = proposal.copy(), loss
    return best, dict(proposals=proposals, accepted=accepted, loss_total=best_loss,
                     wall_seconds=time.monotonic() - started)
