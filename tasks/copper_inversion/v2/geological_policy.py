"""Public population assumptions, fixed before generating evaluation geology.

This is a constrained synthetic benchmark, not a universal mine prior. The
volume range is the existing generator's population range; four components
defines the revised population for all families (including faulted bodies).
No per-case truth, family label, measured label or true volume enters this map.
"""
import numpy as np

POLICY_VERSION = 'synthetic-volume-connectivity-1'
MAX_COMPONENTS = 4
VOLUME_FRACTIONS = (.02, .65)


def public_priors(feasible, spacing):
    cell=float(np.prod(spacing))
    available=int(np.asarray(feasible,dtype=bool).sum())
    lo=int(np.ceil(VOLUME_FRACTIONS[0]*available))
    hi=int(np.floor(VOLUME_FRACTIONS[1]*available))
    if not 0 < lo <= hi:raise ValueError('domain too small for geological policy')
    return dict(area_weight=0.,volume_weight=0.,volume_bounds=None,
                fragment_weight=0.,fragment_volume=0.,
                hard_volume_bounds=[lo*cell,hi*cell],max_components=MAX_COMPONENTS)
