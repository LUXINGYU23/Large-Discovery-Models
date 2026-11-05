"""Development observation scales; physical noise and grid error stay distinct."""
import math
import numpy as np


def scales(spacing, noisy):
    h=np.asarray(spacing,dtype=float)
    if h.shape!=(3,) or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError("positive finite 3D spacing required")
    measurement_sigma=float(.25*h.min()) if noisy else 0.
    # A conservative grid-resolution scale, not measured sensor uncertainty or
    # a claimed mathematical error bound for fast-marching reconstruction.
    discretization_scale=float(.5*np.linalg.norm(h))
    effective=math.hypot(measurement_sigma,discretization_scale)
    return dict(measurement_sigma=measurement_sigma,
                discretization_scale=discretization_scale,effective_scale=effective)
