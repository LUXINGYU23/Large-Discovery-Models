"""Trusted bounded array decoding and canonical signed-distance geometry."""

from pathlib import Path
import stat
import numpy as np
import skfmm

MAX_MODEL_BYTES = 2 * 1024 * 1024


def read_model(path, shape):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MODEL_BYTES:
        raise ValueError("model must be a bounded regular file")
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError("unsupported npy format")
        actual, order, dtype = header
        if (
            tuple(actual) != tuple(shape)
            or dtype.kind != "f"
            or dtype.itemsize not in (4, 8)
        ):
            raise ValueError("model shape/dtype differs from contract")
        if info.st_size != f.tell() + int(np.prod(shape)) * dtype.itemsize:
            raise ValueError("model byte count differs")
    return np.load(path, allow_pickle=False)


def signed_distance(field, shape, spacing):
    if np.ma.isMaskedArray(field) or not isinstance(field, np.ndarray):
        raise ValueError("model must be an unmasked ndarray")
    if (
        field.shape != tuple(shape)
        or field.dtype.kind != "f"
        or not np.isfinite(field).all()
    ):
        raise ValueError(
            "model requires matching shape and finite real floating values"
        )
    if not (np.any(field > 0) and np.any(field < 0)):
        raise ValueError("model must contain interior and exterior")
    # scikit-fmm 2025.6.23 misreads Fortran-strided 3D inputs.
    # Canonicalize storage without changing physical axes or geometry.
    phi = skfmm.distance(np.ascontiguousarray(field, dtype=np.float64), dx=tuple(spacing))
    if np.ma.isMaskedArray(phi) or not np.isfinite(phi).all():
        raise ValueError("invalid reconstructed distance")
    return phi
