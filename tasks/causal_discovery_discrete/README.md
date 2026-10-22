# Discrete Causal Discovery

This adapter searches bounded causal-discovery configurations against the pinned
[MLS-Bench task](https://github.com/Imbernoulli/MLS-Bench/tree/cfd57a7e0139c72753e32e31bca593719b098717/tasks/causal-discovery-discrete).
The benchmark samples five discrete bnlearn Bayesian networks at seed 42 and
compares estimated graphs with the true CPDAG using SHD, adjacency precision and
recall, and arrow precision and recall.

## Candidate Domain

Candidates contain a normalized-mutual-information cutoff and a hard per-node
degree cap. The evaluator computes a sparse undirected skeleton, which is a valid
`causallearn` graph accepted by the upstream interface. This intentionally bounded
representation prevents arbitrary generated source from entering the evaluator.
The deterministic catalog is collectable only in mock mode; real catalog actions
are not written as model-training examples.

## Qualification State

The checked-in contract starts at `draft`. A successful real seed evaluation and
tiny campaign must be promoted into compact evidence under `resources/` before
changing it to `qualified`. The 20-iteration profile is explicitly extended-budget
and is not the upstream task's official comparison budget.

## Run

Mock verification requires no external assets:

```bash
python3 scripts/run_ldm_tts.py config/causal_discovery_discrete/mock.yaml
```

Real runs require a clean MLS-Bench checkout at the pinned commit plus `numpy`,
`pandas`, `pgmpy`, and `causal-learn`. Set `upstream-root` in a local copy of the
real config. The official profile runs one selected candidate; the extended
profile runs twenty. Completed runs include `result.json`, `trajectory.csv`,
budget/status snapshots, search/selection records, and per-candidate evaluation
manifests below the run directory.

`--evaluation-timeout` bounds each network's graph construction and scoring in a
separate process. Expired workers are terminated and joined; the evaluation and
its manifest report `timed_out`, including the networks attempted and elapsed
time. The final MLS-Bench scoring subprocess also has a bounded timeout.
