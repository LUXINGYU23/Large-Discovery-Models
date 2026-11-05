"""Canonical occupancy geometry in physical units, shared by all solvers."""

import numpy as np
import skfmm
from scipy import ndimage
from .contract import validate


def components(mask, spacing):
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    volumes = np.bincount(labels.ravel())[1:] * float(np.prod(spacing))
    return count, volumes


def canonicalize(problem, field):
    validate(problem)
    if (not isinstance(field, np.ndarray) or np.ma.isMaskedArray(field)
            or field.shape != tuple(problem["shape"]) or field.dtype.kind != "f"
            or field.dtype.itemsize not in (4, 8) or not np.isfinite(field).all()):
        raise ValueError("model requires matching shape and finite float32/64 values")
    mask = field > 0
    if np.any(mask & (~problem["domain_mask"] | problem["forbidden_mask"])):
        raise ValueError("invalid_geometry: positive voxel outside feasible domain")
    spacing = problem["spacing"]
    volume = int(mask.sum()) * float(np.prod(spacing))
    bounds = problem["priors"]["hard_volume_bounds"]
    if bounds is not None and not bounds[0] <= volume <= bounds[1]:
        raise ValueError("invalid_geometry: public hard volume bounds violated")
    maximum = problem["priors"]["max_components"]
    if maximum is not None and components(mask, spacing)[0] > maximum:
        raise ValueError("invalid_geometry: public component bound violated")
    # No exterior padding: open grid faces are not artificial contact surfaces.
    # Closed faces already have an explicit public forbidden shell.
    cap = float(np.linalg.norm(np.asarray(problem["shape"]) * spacing))
    if not mask.any():
        phi = np.full(mask.shape, -cap)
    elif mask.all():
        phi = np.full(mask.shape, cap)
    else:
        phi = np.asarray(skfmm.distance(np.ascontiguousarray(np.where(mask, 1., -1.)), dx=tuple(spacing)))
        if not np.isfinite(phi).all():
            raise ValueError("invalid canonical distance")
        phi = np.clip(phi, -cap, cap)
    return mask, phi


def diagnostics(problem, mask):
    spacing = np.asarray(problem["spacing"])
    cell = float(np.prod(spacing))
    feasible = problem["domain_mask"] & ~problem["forbidden_mask"]
    area = 0.0
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        faces = (mask[a] != mask[b]) & feasible[a] & feasible[b]
        area += int(faces.sum()) * cell / spacing[axis]
    count, volumes = components(mask, spacing)
    volume = int(mask.sum()) * cell
    domain_volume = int(problem["domain_mask"].sum()) * cell
    bounds = problem["priors"]["volume_bounds"]
    volume_penalty = 0.0 if bounds is None else (
        max(0, bounds[0] - volume)**2 + max(0, volume - bounds[1])**2) / domain_volume**2
    fragment = float(volumes[volumes < problem["priors"]["fragment_volume"]].sum())
    return dict(positive_volume=volume, component_count=count,
                largest_component_fraction=float(volumes.max() / volume) if volume else 0.,
                fragment_volume=fragment, internal_area=area,
                regularizer_area=area / domain_volume**(2/3),
                regularizer_volume=volume_penalty, regularizer_fragment=fragment / domain_volume)
