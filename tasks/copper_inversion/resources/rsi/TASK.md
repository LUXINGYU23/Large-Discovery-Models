# Geological inversion task

Implement `solve(problem, budget, seed)` and return a finite float32 or float64
three-dimensional NumPy array matching `problem['shape']`. Positive values are
the predicted unit. Input observations are physical-coordinate borehole and
outcrop points with uncertainty, masks, spacing, and public geological priors.
Missing observations are unknown. Solvers must honor forbidden regions, boundary
conditions, and all hard priors in the supplied problem. Public case names do
not encode geological meaning.

Read `cases.json` and load a case with
`tasks.copper_inversion.v2.data.read_problem`. The public contract, geometry,
objective, and fixtures are provided in `tasks/copper_inversion/v2/`.
NumPy, SciPy, scikit-fmm, and discretize are available to solvers. The per-solve
wall time, CPU count, and memory limit are given by `budget`.

An independent evaluator scores a frozen result against withheld confirmation
measurements using the same public priors. If its loss is L, risk is L/(1+L),
averaged within each geology body and then equally across bodies. A failed solve
counts as risk 1. The controller may provide the per-case risk for measured
programs. Neither the withheld data nor reference geometry is available to the
researcher or the solver.

Write the requested complete programs to `candidates/<index>/solver.py` and
submit them with the prescribed metadata and risk intervals. Public interface
checks can be run with `python selfcheck.py /workspace/candidates/0/solver.py`
from the exported task directory; repeat for every candidate. These checks do
not evaluate withheld measurements. Do not launch external evaluations.
