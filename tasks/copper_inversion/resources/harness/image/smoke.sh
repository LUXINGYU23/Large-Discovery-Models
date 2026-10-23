#!/bin/sh
set -eu
python - <<'PYCODE'
import numpy as np
import scipy.optimize
import skfmm
from discretize import TensorMesh
x, y = np.indices((12, 9))
phi = 9.0 - (x - 5.5)**2 - (y - 4)**2
distance = skfmm.distance(phi)
assert np.isfinite(distance).all()
assert TensorMesh([np.ones(12), np.ones(9)]).n_cells == 108
assert scipy.optimize.minimize(lambda x: (x[0]-1)**2, [0.0]).success
PYCODE
mkdir -p /workspace/research
printf 'numpy_scipy_skfmm_discretize_ok\n' > /workspace/research/proof.txt
