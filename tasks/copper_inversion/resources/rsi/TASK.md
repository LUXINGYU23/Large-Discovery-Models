# Geological inversion task

Implement `solve(problem, budget, seed)` and return a finite float32 or float64
three-dimensional NumPy array matching `problem['shape']`. Positive values are
the predicted unit. Input observations are physical-coordinate borehole and
outcrop points with uncertainty, masks, spacing, and public geological priors.
Missing observations are unknown. Solvers must honor forbidden regions, boundary
conditions, and all hard priors in the supplied problem. Public case names are
opaque per-evaluation tokens with no geological meaning, and grid origins are
canonicalized. Design a general inversion algorithm for unseen geometries rather
than case-specific dispatch.

Read `cases.json` and load a case with
`tasks.copper_inversion.v2.data.read_problem`. The public contract, geometry,
objective, and fixtures are provided in `tasks/copper_inversion/v2/`.
NumPy, SciPy, scikit-fmm, and discretize are available to solvers. The per-solve
wall time, CPU count, and memory limit are given by `budget`.

An independent evaluator scores a frozen result against withheld confirmation
measurements using the same public priors and `objective.score`: noise-corrected
line-Jaccard loss `1 - (TP + 1) / (TP + FP + FN + 1)` on binary samples outside
the observed-contact tolerance band, weighted by modality, plus any active prior
penalties. Risk is that loss averaged within each geology body and then equally
across bodies; a failed or illegal solve counts as risk 1. Only the aggregate
risk across the evaluation suite is reported for measured programs. Neither the
withheld data nor reference geometry is available to the researcher or the
solver.

Write the requested complete programs to `candidates/<index>/solver.py` and
submit them with the prescribed metadata and risk intervals; the i-th submitted
entry (counting from 0) must point to `candidates/<i>/solver.py`. Each solver
must stay within 32,000 bytes once comments are removed and the code is
re-formatted. Public interface
checks can be run with `python selfcheck.py /workspace/candidates/0/solver.py`
from the exported task directory; repeat for every candidate. These checks do
not evaluate withheld measurements. Do not launch external evaluations.
