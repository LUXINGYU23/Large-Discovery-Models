"""Compact geometry feedback derived only from a formally scored frozen model."""

import hashlib
import numpy as np
from .geometry import read_model


def model_slices(path, problem, *, seed):
    field = read_model(path, problem["shape"])
    axes = problem["axis_names"]
    slices = []
    for normal in ([None] if field.ndim == 2 else range(3)):
        index = None if normal is None else field.shape[normal] // 2
        plane = field if normal is None else np.take(field, index, axis=normal)
        remaining = [i for i in range(field.ndim) if i != normal]
        indices = [np.arange(0, n, max(1, (n + 31) // 32)) for n in plane.shape]
        sampled = plane[np.ix_(*indices)]
        slices.append({
            "normal_axis": None if normal is None else axes[normal],
            "normal_index": index,
            "normal_coordinate": None if normal is None else
                problem["origin"][normal] + (index + 0.5) * problem["spacing"][normal],
            "row_axis": axes[remaining[0]], "column_axis": axes[remaining[1]],
            "row_indices": indices[0].tolist(), "column_indices": indices[1].tolist(),
            "rows": ["".join("+" if v > 0 else "-" if v < 0 else "0" for v in row)
                     for row in sampled],
        })
    return {"schema_version": 1, "seed": seed,
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(field.shape), "axis_names": axes,
            "origin": problem["origin"], "spacing": problem["spacing"],
            "legend": {"+": "interior", "-": "exterior", "0": "interface"},
            "sampling": "central index planes; strided samples, at most 32 per axis; no interpolation",
            "slices": slices}
