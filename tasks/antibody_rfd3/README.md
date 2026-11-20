# Structure-assisted CDRH3 design

`antibody_rfd3` registers a parallel LDM task for fixed-framework H3 design. One
campaign owns proposal admission, sequence deduplication, observations, budgets
and selection. Its structural template is 1N8Z chain B residues 1–120. The 11
Kabat H3 positions B99–B109 are followed through the generated sequence-chain
mapping. The default scoring antigen is 1ADQ_A.

## Running the case

Run from the repository root. The offline smoke exercises the actual Campaign
with deterministic generator fixtures and a sequence-hash oracle.

```sh
python scripts/validate_tasks.py --task antibody_rfd3
python scripts/run_ldm_tts.py config/antibody_rfd3/mock.yaml
```

The four other configs run real proposal methods, with `--oracle mock` as the
default for integration. `mock_energy` is a deterministic hash of the antigen
and H3 mapped to [-1, 0]. It has no biological interpretation. To measure
`absolut_energy`, select `--set args.oracle=absolut` and export `ABSOLUT_PATH`
to an [immutable Absolut input bundle](../../docs/absolut-oracle.md) containing
`assets.json` and `src/bin/Absolut`. The shared HTTP evaluator scores only the
11-residue sequence.

| Config | Proposal and selection after the shared initialization |
|---|---|
| `fixed.yaml` | RFD3 and ProteinMPNN at a fixed temperature, then the first eligible new H3 in sampling order |
| `harness.yaml` | One persistent Pi session uses the tools and submits its preferred eligible design ID |
| `ldm_harness.yaml` | The research panel generates and reviews distinct design IDs. Pending-aware BO selects candidates using the fitted GP and dynamic policy |
| `antbo.yaml` | Upstream AntBO receives measured H3s through `observe` and proposes through `suggest`, subject to the common campaign admission rules |

Use the deployed independent structure design environments and archived weights
described in [the tool README](../antibody/tools/structure_design/README.md).
The controller uses Python 3.10+ with the task's project dependencies. Install
the `gp` extra for GP selection. H20's controller is
`/Aidata/shared/ldm/antibody-poc/env/antibody_campaign/bin/python`.
Generation uses one Linux worker slot across campaigns sharing the project
root. The worker launches the full model process tree under `taskset` with two
allowed logical CPUs by default and caps BLAS/OpenMP threads. GPU 0 is the
default. Override with `--set args.gpu=N`. For intentional GPU sharing add
`--set args.share-gpu=true` (24 GiB PyTorch allocator cap).

For Pi, provide `LLM_BASE_URL`, `LLM_MODEL_NAME` and `LLM_API_KEY` via the
environment. The Responses endpoint must accept tool calls and be reachable
from both the controller and the Docker sidecar. Use a sidecar compatible with
this checkout. Set `LDM_PI_IMAGE` for the Docker image and `LDM_GONDOLIN_CACHE`
for the guest cache. On Linux prepare and smoke the task guest with the
[repository Pi build instructions](../../harnesses/pi/README.md) using
`--task antibody_rfd3 --cache-dir "$LDM_GONDOLIN_CACHE"`. The sidecar receives
the task resources and a private Unix socket. The model weights stay on the host.

For paired comparisons, run the fixed case once per seed, then give every
comparison method its output directory as `args.initialization-run`. The worker
replays the original initialization batch, including the mapping and backbone
artifacts, so every method evaluates the same first eligible H3 even if GPU
kernels yield slightly different sequences on independent launches. It records
the source in the receipt. Match seed, temperature and samples per call.

```sh
ROOT=/Aidata/shared/ldm/antibody-poc
PY="$ROOT/env/antibody_campaign/bin/python"
"$PY" scripts/run_ldm_tts.py config/antibody_rfd3/fixed.yaml \
  --set args.out-dir="$ROOT/runs/antibody_rfd3/anchor"
"$PY" scripts/run_ldm_tts.py config/antibody_rfd3/ldm_harness.yaml \
  --set args.initialization-run="$ROOT/runs/antibody_rfd3/anchor" \
  --set args.out-dir="$ROOT/runs/antibody_rfd3/ldm"
```

Native AntBO uses a separate Python environment. Set `ANTBO_HOME` to the
`AntBO/` subdirectory of HEBO at commit
`77636146d2a2208d971e589062d4ec3d050029f0`. Install the task dependencies plus
PyTorch 2.6.0, BoTorch 0.9.5, GPyTorch 1.11, pymoo 0.6.1.3, scikit-learn,
matplotlib and tqdm there. Run the same command with that interpreter and
`antbo.yaml`. Its native trust region starts from the shared measured H3.

## Evidence and budgets

The smoke configs allocate two evaluations, one per round including the
initialization. Each structural method has the same per-round caps on new
backbones and sampled sequences. Failed jobs consume their reserved generation
budget. A `sequence_h3` call uses only ProteinMPNN on the requested earlier
backbone. All accepted H3s satisfy the original antibody sequence constraints.

Each run retains the standard `events.jsonl`, `budget.json`, `checkpoint.json`,
`status.json` and `summary.json`. Under `design_jobs/`, each receipt maps design
IDs and backbone IDs to full candidate records, coordinates, residue mappings,
tool logs and rejection reasons. Pi writes its session artifacts under
`harness/`, including the input history, skill snapshot, tool calls and
submissions. The fine-tuning collection boundary for a future export is the
validated design-ID submission. Native Pi traces remain raw research records.

The [H20 smoke record](resources/h20-smoke.json) contains the two observations
for each of the four methods, the Pi trace location and the second-round GP
scores. The runs used seed 42 and shared the first generated backbone and its
measured H3. Both Pi sessions used Qwen3-8B through a Responses adapter for
vLLM 0.11.2. In the LDM run, Pi read the design skill, received the first mock
score and reused the first backbone with ProteinMPNN. The GP compared two
eligible submissions. The original run directories, including CIFs, full
sequences and event logs, are under the H20 path recorded in the JSON.

The full asynchronous panel is exercised by `repro/h20/run_full_ldm.py` and
independently checked by `repro/h20/audit_full_ldm.py`. This uses the configured
LLM endpoint, real generation, and HER2 Absolut scoring. See
[`poc-full-ldm-harness.md`](../../docs/poc-full-ldm-harness.md) for the configuration
and H20 preparation. Role budgets, history replay, pending conditioning and
policy recovery also have provider-free tests under `tests/`; these tests are
run on H20. The obsolete synchronous scripted-provider smoke has been removed.

## Paired real baselines on H20

For each target (`her2` or `pd1`), the configs
`<target>_fixed_real.yaml`, `<target>_antbo_real.yaml` and
`<target>_harness_real.yaml` use the original synchronous campaign. The next
proposal step starts after the previous score is recorded. All three reuse the
same published initialization job for the target and retain their original
Absolut evaluator and per-run artifacts. Native AntBO records the measurements
handed to its optimizer. The single Pi baseline receives measured history in
its turn input and submits its preferred candidate.

```sh
ROOT=/Aidata/shared/ldm/antibody-poc
export ANTBO_HOME="$ROOT/env/sources/HEBO-77636146/AntBO"
"$ROOT/env/antbo_native/bin/python" scripts/run_ldm_tts.py \
  config/antibody_rfd3/her2_antbo_real.yaml \
  --set args.out-dir="$ROOT/runs/antibody_rfd3/her2-antbo-async"
```

Use the campaign interpreter for fixed and Pi. For Pi, load the configured
Responses endpoint and a compatible sidecar before running. The
`<target>_real.yaml` configurations start a fresh three-evaluation LDM reference
run. For a fresh paired comparison, pass that completed run directory as
`args.initialization-run` to each baseline. The existing full streaming and
barrier configs retain their own budgets and scheduling settings.

The [historical baseline report](resources/h20-real-baselines.json) and
`repro/h20/verify_baselines.py` document and verify the original branch's
completed synchronous runs. The LDM reference uses controller and
experiment-service ledgers, and its full streaming configuration permits
research and evaluation to overlap.
