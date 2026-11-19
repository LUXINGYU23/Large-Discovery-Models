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
to the Absolut installation containing the antigen lattice files and
`src/bin/Absolut`. The shared evaluator scores only the 11-residue sequence.

| Config | Proposal and selection after the shared initialization |
|---|---|
| `fixed.yaml` | RFD3 and ProteinMPNN at a fixed temperature, then the first eligible new H3 in sampling order |
| `harness.yaml` | One persistent Pi session uses the tools and submits its preferred eligible design ID |
| `ldm_harness.yaml` | The same Pi skill and tools produce distinct design IDs. The categorical antibody GP selects one using expected improvement for minimization |
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
other method its output directory as `args.initialization-run`. The worker
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

The real HER2 and PD-1 configurations are
`config/antibody_rfd3/her2_real.yaml` and `config/antibody_rfd3/pd1_real.yaml`.
Each runs three evaluations with one persistent Pi session. They use the
isolated Absolut Oracle and the existing H20 precomputed resources. Additional
resource directories can be supplied with repeated `--absolut-resources`.
`--absolut-timeout` defaults to 600 seconds. Preflight requires structures and
interaction caches. The campaign admits exactly 11 residues, records raw
Oracle outputs under `oracle/`, and sends successful energies, design IDs,
backbone IDs and result references to the next research turn.

On H20, start both targets sequentially with the deployed Qwen3-8B model.
Choose a fresh output directory. This launcher starts and cleans up its own
model service and the Responses transport adapter, and preserves per-target
exit codes and logs.

```sh
/Aidata/shared/ldm/antibody-poc/env/antibody_campaign/bin/python \
  tasks/antibody_rfd3/repro/h20/run_real_sessions.py \
  --out-dir /Aidata/shared/ldm/antibody-poc/runs/antibody_rfd3/my-real-sessions \
  --gpu 0
```

The [real-session acceptance record](resources/h20-real-sessions.json) contains
all six energies, Oracle artifact paths, GP candidate scores, tool usage and
trace hashes from the 2026-11-19 H20 run. Recheck its original run artifacts with
the controller Python from the repository root using
`python -m tasks.antibody_rfd3.repro.h20.verify_real_sessions RUN_ROOT`.

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

`tests/pi_transport_smoke.py <fresh-output-dir>` is a Linux integration check
with a **scripted Responses provider**. It uses the real sidecar, real GPU tools
and the mock oracle. It verifies that the skill was read and the second turn
received the first measurement. Run a configured LLM endpoint for agent
performance experiments.
