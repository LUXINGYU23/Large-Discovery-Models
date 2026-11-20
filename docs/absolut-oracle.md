# Absolut scoring in the streaming PoC

The `ldm_harness` method of `antibody_rfd3` submits scoring to the HTTP experiment service. Each
attempt runs in its own directory and uses read-only inputs verified by
`assets.json`. The controller ingests the score and records the experiment
and attempt identity in SQLite. A failed run does not substitute mock scores.

The real configurations use HER2 (`1N8Z_C`) and the custom PD-1 lattice
(`5GGR_Y_FuC_5.25`). The [Absolut build instructions](../tools/absolut/README.md)
describe the native binary and lattice provenance. These are computational scores.

On H20 the HER2 configurations reuse the verified input bundle from the
completed streaming run. The PD-1 configurations use the prepared bundle at
`/Aidata/shared/ldm/antibody-poc/cache/absolut-async/5GGR_Y_FuC_5.25`.
For a different workspace, set `args.absolut-path` to a bundle below the
active data root. To stage a fresh bundle, run
`python -m tasks.antibody_rfd3.repro.h20.prepare_absolut_assets` with
`--antigen`, `--binary`, `--output-dir`, and an explicit `--asset` for each
matching precomputed structure and the detailed and compact interaction files,
plus the self-folding inputs from the same cache. The tool verifies the read-only
bundle before publishing it.

The fixed, Pi and AntBO baselines use the original synchronous Absolut
adapter with the published initialization receipt for each target. They
write native evaluation results and logs under their own run directories. See the
[task instructions](../tasks/antibody_rfd3/README.md) for the run command.
The historical [research](../tasks/antibody_rfd3/resources/h20-real-sessions.json)
and [baseline](../tasks/antibody_rfd3/resources/h20-real-baselines.json)
records were produced on `antibody-poc-rsd3` before this merge.
