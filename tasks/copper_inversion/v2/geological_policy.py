"""Public population assumptions, fixed before generating evaluation geology.

This is a constrained synthetic benchmark, not a universal mine prior. The
volume range is the existing generator's population range; four components
defines the revised population for all families (including faulted bodies).
A deposit model is built only after discovery: in-distribution cases have ore
intersected by at least two public boreholes.
No per-case truth, family label, measured label or true volume enters this map.
"""
import numpy as np

POLICY_VERSION = 'synthetic-volume-connectivity-1'
MAX_COMPONENTS = 4
VOLUME_FRACTIONS = (.02, .65)
# Registered fallback only if calibration validity fails C2: area_weight=0.05
# with hinge at 1.1 * max calibration regularizer_area. All weights stay 0.
FALLBACK_AREA_WEIGHT = .05
FALLBACK_AREA_HINGE_SCALE = 1.1


def public_priors(feasible, spacing):
    cell=float(np.prod(spacing))
    available=int(np.asarray(feasible,dtype=bool).sum())
    lo=int(np.ceil(VOLUME_FRACTIONS[0]*available))
    hi=int(np.floor(VOLUME_FRACTIONS[1]*available))
    if not 0 < lo <= hi:raise ValueError('domain too small for geological policy')
    return dict(area_weight=0.,volume_weight=0.,volume_bounds=None,
                fragment_weight=0.,fragment_volume=0.,
                hard_volume_bounds=[lo*cell,hi*cell],max_components=MAX_COMPONENTS)
