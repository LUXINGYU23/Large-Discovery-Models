# Versioned public resources

`cases/` contains supplied synthetic geological tutorial inputs, with SHA-256
identities in `provenance.json`. `LICENSE.upstream` preserves the scientific code
license. `seeds/solver.py` returns the original initial geometry; it contains no
optimization algorithm. `selfcheck.py` is the public convenience interface check;
the controller repeats authoritative checks on sealed source. `TASK.md` defines
exact input semantics. Public geological loss/data are not a hidden benchmark.

The task's guest recipe provides a reproducible Python research environment.
No model keys, endpoint bindings or raw run directories belong here.
`evidence/` contains compact verification records and hashes of retained remote
artifacts; it is excluded from runtime identity along with qualification metadata.
