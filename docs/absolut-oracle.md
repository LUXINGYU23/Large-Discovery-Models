# Absolut Oracle

The module `tasks.antibody.core.absolut_oracle` runs real Absolut repertoire evaluations in a unique directory per invocation. It does not change the process working directory. Targets: `HER2` → `1N8Z_C`; `PD-1` → custom `5GGR_Y_FuC_5.25`.

## H20 example

```bash
source /Aidata/shared/ldm/antibody-poc/env/activate_absolut.sh
cd /Aidata/shared/ldm/antibody-poc/code
/Aidata/shared/ldm/antibody-poc/env/antibody_campaign/bin/python -m tasks.antibody.core.absolut_oracle \
  --resources "$ABSOLUT_PATH" \
  --resources /Aidata/shared/ldm/antibody-poc/runs/absolut-her2-check-fUwLve \
  --run-root /Aidata/shared/ldm/antibody-poc/runs/oracle-evaluations \
  --target HER2 --sequence CAGPSTTVPYYFDYW
```

Repeat `--sequence` for batches. Each invocation preserves `input.tsv`, `evaluation.log`, raw FinalBindings output and `result.json`. The score is the minimum finite Energy per input sequence. Results include antigen ID, executable SHA256 and run directory. Missing resources, invalid sequences, failed commands, timeouts and incomplete output raise errors; no mock fallback is used. Timeout defaults to 600 seconds.

HER2 validation returned -99.72 for the sequence above in `runs/oracle-validation/1N8Z_C-ztg3aqcr/result.json`.

PD-1 resources are available under `runs/absolut-pd1-5GGR-Y`. Add that directory as `--resources` and use `--target PD-1`. Its structure is a custom lattice of PDB 5GGR author chain Y, resolution 5.25, FuC, with jumps allowed across missing residues and two preserved subchains. Scores measure this computational proxy.

This module is independently callable. The `antibody_rfd3` real HER2 and PD-1 configurations also call it through a Campaign adapter, enforcing 11 aa H3 admission and preserving Oracle results and their provenance in each observation. Other antigen configurations retain their existing evaluator. Cached resource files are shared through symlinks and evaluation outputs have isolated directories.

## Additional acceptance checks

Recorded in `/Aidata/shared/ldm/antibody-poc/runs/oracle-validation/acceptance.json`:

- Four invalid input cases rejected before evaluation (plain string instead of batch, invalid amino acids, empty batch, unknown target).
- Missing resources rejected before evaluation.
- Actual Absolut process interrupted by a 0.01 second timeout; failure details preserved in the invocation directory as `error.json`.
- Two real HER2 invocations ran concurrently in distinct directories without output collisions.
- A three-item batch preserved input order and duplicate sequence results: `CAGPSTTVPYYFDYW` = -99.72, `CARAYYSNDYW` = -94.77, repeated first sequence = -99.72. The concurrent single-item invocation also returned -99.72.

Result rows must contain a finite energy and the original CDR3 corresponding to their input index. Failures after creation of the run directory preserve an `error.json`; failures during preflight are raised directly. Structure caches are reused. The Campaign adapter records failed attempts and exposes successful energies and artifact references to subsequent research turns.

## PD-1 validation completed

The precomputation completed with 4,902,973 receptor structures and 2,409,798 non-redundant interaction profiles. Oracle batch evaluation succeeded for custom antigen `5GGR_Y_FuC_5.25`: `CAGPSTTVPYYFDYW` → -85.41; `CARAYYSNDYW` → -72.85. Record: `/Aidata/shared/ldm/antibody-poc/runs/oracle-validation/5GGR_Y_FuC_5.25-o0ye09cu/result.json`. Earlier statements that PD-1 precomputation and validation were pending are superseded by this result. Use `--resources /Aidata/shared/ldm/antibody-poc/runs/absolut-pd1-5GGR-Y --target PD-1`. These scores describe the custom lattice specified above, including missing-residue handling; they do not establish experimental affinity.
