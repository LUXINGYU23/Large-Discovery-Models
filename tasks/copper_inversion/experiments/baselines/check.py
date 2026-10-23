"""Remote preflight: objective agreement and immutable adapter source checks."""
import json
import numpy as np
import skfmm
from tasks.copper_inversion.core.data import load_problem
from tasks.copper_inversion.core.objective import score
from shpmc.level_set_mc import StochasticLevelSet3D
for case in ('drillholes_3d','drillholes_outcrops_3d'):
    p=load_problem(case)
    phi=skfmm.distance(np.ascontiguousarray(p['initial_level_set']))
    original=StochasticLevelSet3D([o['values'] for o in p['observations']],[phi],None,1,[o['weight'] for o in p['observations']])
    a=original.loss_computation(phi)[0]
    b=score(p,np.ascontiguousarray(p['initial_level_set']))['loss_total']
    assert np.isclose(a,b,rtol=1e-12), (case,a,b)
    print(json.dumps(dict(case=case,original=a,independent=b,status='passed')))
