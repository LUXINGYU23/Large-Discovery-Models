"""Development observation scales; physical noise and grid error stay distinct."""
import math
import numpy as np

DENSITIES={"lo":(3,12),"mid":(12,24),"hi":(24,48)}


def parse_holes(holes):
    if holes is None:return (3,12)
    if isinstance(holes,str):
        if holes not in DENSITIES:raise ValueError(f"unknown hole density {holes!r}")
        return DENSITIES[holes]
    if isinstance(holes,int) and not isinstance(holes,bool):lo=hi=holes
    elif isinstance(holes,(list,tuple)) and len(holes) in (1,2):
        lo=holes[0];hi=holes[0] if len(holes)==1 else holes[1]
    else:raise ValueError("holes must be an integer, (lo, hi) pair, or density name")
    if type(lo) is not int or type(hi) is not int or lo<1 or lo>hi:
        raise ValueError("invalid hole count bounds")
    return (lo,hi)


def parse_cli_holes(values):
    if not values:return (3,12)
    if len(values)==1 and values[0] in DENSITIES:return DENSITIES[values[0]]
    try:parsed=[int(v) for v in values]
    except ValueError as exc:raise ValueError(f"invalid --holes {values!r}") from exc
    return parse_holes(parsed)


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
