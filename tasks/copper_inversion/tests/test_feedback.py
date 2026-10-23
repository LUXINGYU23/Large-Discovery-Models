"""Measured geometry feedback preserves physical axes and bounded output."""

import hashlib
import numpy as np
import pytest
from tasks.copper_inversion.core.feedback import model_slices


@pytest.mark.parametrize("shape,axes", [((7, 5), ["x", "z"]), ((65, 33, 17), ["x", "y", "z"])])
def test_slice_axes_and_sampling(tmp_path, shape, axes):
    field = np.indices(shape).sum(axis=0).astype(float) - 10
    path = tmp_path / "model.npy"
    np.save(path, np.asfortranarray(field))
    problem = dict(shape=shape, axis_names=axes, origin=[-7.0] * len(shape), spacing=[2.0] * len(shape))
    result = model_slices(path, problem, seed=47)
    assert result["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for s in result["slices"]:
        normal = None if s["normal_axis"] is None else axes.index(s["normal_axis"])
        plane = field if normal is None else np.take(field, shape[normal] // 2, axis=normal)
        remaining = [i for i in range(len(shape)) if i != normal]
        assert s["row_axis"] == axes[remaining[0]]
        assert s["column_axis"] == axes[remaining[1]]
        assert len(s["rows"]) <= 32 and all(len(row) <= 32 for row in s["rows"])
        if normal is not None:
            assert s["normal_coordinate"] == -7 + (shape[normal] // 2 + 0.5) * 2
        for row, i in zip(s["rows"], s["row_indices"]):
            for char, j in zip(row, s["column_indices"]):
                assert char == ("+" if plane[i, j] > 0 else "-" if plane[i, j] < 0 else "0")
