"""Public-only serialization. Hidden truth is never a problem field."""

import json
from pathlib import Path
import numpy as np
from .contract import validate


def save_problem(problem, directory):
    validate(problem)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in problem.items() if k not in {"domain_mask", "forbidden_mask", "observations"}}
    arrays = {k: problem[k] for k in ("domain_mask", "forbidden_mask")}
    meta["observations"] = []
    for i, obs in enumerate(problem["observations"]):
        meta["observations"].append({k: v for k, v in obs.items() if k not in {"points", "labels", "groups"}})
        for key in ("points", "labels", "groups"):
            arrays[f"{key}_{i}"] = obs[key]
    (directory / "problem.json").write_text(json.dumps(meta, sort_keys=True, allow_nan=False))
    np.savez(directory / "problem.npz", **arrays)


def read_problem(directory):
    directory = Path(directory)
    problem = json.loads((directory / "problem.json").read_text())
    with np.load(directory / "problem.npz", allow_pickle=False) as arrays:
        for key in ("domain_mask", "forbidden_mask"):
            problem[key] = arrays[key]
        for i, obs in enumerate(problem["observations"]):
            for key in ("points", "labels", "groups"):
                obs[key] = arrays[f"{key}_{i}"]
    validate(problem)
    for arr in [problem["domain_mask"], problem["forbidden_mask"],
                *[o[k] for o in problem["observations"] for k in ("points", "labels", "groups")]]:
        arr.flags.writeable = False
    return problem


ORIENTATION_COUNT = 8
_ROT90_FACE = {"x-": "y-", "y-": "x+", "x+": "y+", "y+": "x-", "z-": "z-", "z+": "z+"}
_FLIPX_FACE = {"x-": "x+", "x+": "x-", "y-": "y-", "y+": "y+", "z-": "z-", "z+": "z+"}


def _check_orientation(orientation):
    if type(orientation) is not int or not 0 <= orientation < ORIENTATION_COUNT:
        raise ValueError("orientation must be an integer in 0..7")
    return orientation % 4, bool(orientation // 4)


def apply_orientation(array, orientation):
    """Apply horizontal D4 symmetry (optional x-flip then k*90 deg xy-rotation) to a 3D array."""
    k, flip = _check_orientation(orientation)
    out = np.asarray(array)
    if flip:
        out = np.flip(out, axis=0)
    if k:
        out = np.rot90(out, k=k, axes=(0, 1))
    return np.ascontiguousarray(out)


def invert_orientation(array, orientation):
    """Exact inverse of apply_orientation on a 3D grid."""
    k, flip = _check_orientation(orientation)
    out = np.asarray(array)
    if k:
        out = np.rot90(out, k=(4 - k) % 4, axes=(0, 1))
    if flip:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def _transform_faces(faces, k, flip):
    out = []
    for face in faces:
        f = _FLIPX_FACE[face] if flip else face
        for _ in range(k):
            f = _ROT90_FACE[f]
        out.append(f)
    return out


def canonical_problem(problem, *, case_id=None, canonical_origin=True, orientation=0):
    """Return a validated copy with optional opaque case_id, zero-shifted origin, and horizontal D4 orientation."""
    validate(problem)
    k, flip = _check_orientation(orientation)
    origin = np.asarray(problem["origin"], dtype=float)
    shape = list(problem["shape"])
    spacing = list(problem["spacing"])
    new_origin = [0.0, 0.0, 0.0] if canonical_origin else list(problem["origin"])
    if k in (1, 3):
        new_shape = [shape[1], shape[0], shape[2]]
        new_spacing = [spacing[1], spacing[0], spacing[2]]
    else:
        new_shape = shape
        new_spacing = spacing
    extent = np.asarray(shape, dtype=float) * np.asarray(spacing, dtype=float)
    observations = []
    for obs in problem["observations"]:
        points = np.asarray(obs["points"], dtype=float)
        if orientation == 0:
            shifted = points - origin if canonical_origin else points.copy()
        else:
            rel = points - origin
            rx, ry, rz = rel[:, 0].copy(), rel[:, 1].copy(), rel[:, 2].copy()
            lx, ly = extent[0], extent[1]
            if flip:
                rx = lx - rx
            if k == 1:
                rx, ry = ly - ry, rx
            elif k == 2:
                rx, ry = lx - rx, ly - ry
            elif k == 3:
                rx, ry = ry, lx - rx
            shifted = np.column_stack((rx, ry, rz))
            if not canonical_origin:
                shifted = shifted + origin
        observations.append({
            **obs,
            "points": shifted,
            "labels": np.asarray(obs["labels"]).copy(),
            "groups": np.asarray(obs["groups"]).copy(),
        })
    domain = apply_orientation(np.asarray(problem["domain_mask"], dtype=bool), orientation)
    forbidden = apply_orientation(np.asarray(problem["forbidden_mask"], dtype=bool), orientation)
    faces = _transform_faces(problem["open_boundary_faces"], k, flip) if orientation else list(problem["open_boundary_faces"])
    result = {
        **problem,
        "case_id": problem["case_id"] if case_id is None else case_id,
        "shape": new_shape,
        "origin": new_origin,
        "spacing": new_spacing,
        "open_boundary_faces": faces,
        "domain_mask": domain,
        "forbidden_mask": forbidden,
        "observations": observations,
    }
    validate(result)
    for arr in [result["domain_mask"], result["forbidden_mask"],
                *[o[k] for o in result["observations"] for k in ("points", "labels", "groups")]]:
        arr.flags.writeable = False
    return result
