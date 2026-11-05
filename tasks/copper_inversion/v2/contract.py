"""Strict public input contract, with no private geology or legacy fields."""

import numpy as np

SCHEMA = "problem_schema_v2"
GEOMETRY = "occupancy_v2"
OBJECTIVE = "objective_v2"
PUBLIC_KEYS = {
    "schema_version", "geometry_version", "objective_version", "case_id",
    "shape", "origin", "spacing", "domain_mask", "forbidden_mask",
    "open_boundary_faces", "observations", "priors",
}
PRIOR_KEYS = {"area_weight", "volume_weight", "volume_bounds", "fragment_weight",
              "fragment_volume", "hard_volume_bounds", "max_components"}
OBS_KEYS = {"name", "points", "labels", "groups", "weight", "transition_scale",
            "contact_sigma", "label_error"}


def validate(problem):
    if set(problem) != PUBLIC_KEYS:
        raise ValueError("public problem fields differ from v2 contract")
    for key, value in (("schema_version", SCHEMA), ("geometry_version", GEOMETRY),
                       ("objective_version", OBJECTIVE)):
        if problem[key] != value:
            raise ValueError(f"unsupported {key}")
    shape = problem["shape"]
    if len(shape) != 3 or any(type(n) is not int or n < 2 for n in shape):
        raise ValueError("v2 requires a 3D grid with at least two cells per axis")
    if np.prod(shape, dtype=np.int64) > 200_000:
        raise ValueError("grid exceeds bounded model decoding budget")
    spacing = np.asarray(problem["spacing"], dtype=float)
    origin = np.asarray(problem["origin"], dtype=float)
    if spacing.shape != (3,) or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("invalid physical coordinates")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing must be finite and positive")
    for key in ("domain_mask", "forbidden_mask"):
        a = problem[key]
        if not isinstance(a, np.ndarray) or a.dtype != bool or a.shape != tuple(shape):
            raise ValueError(f"invalid {key}")
    domain, forbidden = problem["domain_mask"], problem["forbidden_mask"]
    if not domain.any() or not (domain & ~forbidden).any() or np.any(forbidden & ~domain):
        raise ValueError("invalid feasible domain")
    faces = problem["open_boundary_faces"]
    if len(set(faces)) != len(faces) or not set(faces) <= {"x-", "x+", "y-", "y+", "z-", "z+"}:
        raise ValueError("invalid open boundary faces")
    # A closed face is encoded in the public forbidden mask, not scorer-only logic.
    for axis, name in enumerate("xyz"):
        for index, suffix in ((0, "-"), (-1, "+")):
            face = [slice(None)] * 3
            face[axis] = index
            face = tuple(face)
            if name + suffix not in faces and np.any(domain[face] & ~forbidden[face]):
                raise ValueError("closed boundary requires a forbidden shell")
    priors = problem["priors"]
    if set(priors) != PRIOR_KEYS:
        raise ValueError("invalid prior fields")
    for key in ("area_weight", "volume_weight", "fragment_weight", "fragment_volume"):
        if not np.isfinite(priors[key]) or priors[key] < 0:
            raise ValueError("prior scales must be finite and nonnegative")
    for key in ("volume_bounds", "hard_volume_bounds"):
        bounds = priors[key]
        if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all()
                                   or not 0 <= bounds[0] <= bounds[1]):
            raise ValueError("invalid physical volume bounds")
    if priors["volume_weight"] and priors["volume_bounds"] is None:
        raise ValueError("volume regularization requires public bounds")
    count = priors["max_components"]
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("invalid component constraint")
    names = set()
    if not problem["observations"]:
        raise ValueError("at least one observation modality required")
    for obs in problem["observations"]:
        if set(obs) != OBS_KEYS or obs["name"] in names:
            raise ValueError("invalid or duplicate observation modality")
        names.add(obs["name"])
        points, labels, groups = (np.asarray(obs[k]) for k in ("points", "labels", "groups"))
        if labels.ndim != 1 or not len(labels) or points.shape != (len(labels), 3) or groups.shape != labels.shape:
            raise ValueError("observation array shapes differ")
        if not np.isfinite(points).all() or not np.isin(labels, [0, 0.5, 1]).all():
            raise ValueError("invalid observation coordinates or labels")
        if groups.dtype.kind not in "iu" or np.any(groups < 0):
            raise ValueError("groups must be nonnegative integers")
        # Interpolation never extrapolates outside the cell-center box.
        index = (points - origin) / spacing - 0.5
        if np.any(index < -1e-9) or np.any(index > np.asarray(shape) - 1 + 1e-9):
            raise ValueError("observation outside interpolation domain")
        nearest = tuple(np.rint(index).astype(int).T)
        if not domain[nearest].all():
            raise ValueError("observation outside physical domain")
        for key in ("weight", "transition_scale", "contact_sigma"):
            if not np.isfinite(obs[key]) or obs[key] <= 0:
                raise ValueError("observation scales must be positive")
        if not np.isfinite(obs["label_error"]) or not 0 <= obs["label_error"] < 0.5:
            raise ValueError("invalid label noise rate")
    return problem
