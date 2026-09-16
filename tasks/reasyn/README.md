# ReaSyn Science Benchmark for LDM

This task implements **the first/main reconstruction evaluation** and the
released TDC goal-directed optimization evaluation from the source-pinned
ReaSyn v2 code and paper. Both execute through `ldm_tts.campaign.run_campaign`,
including shared candidate admission, budgets, acquisition, checkpoints and
resume. No model training is needed. Registration is valid; scientific
qualification remains **draft** because official assets and a real GPU seed run
have not been verified. Synthetic mock values are not ReaSyn scores.

## What LDM controls

| Track | Candidate and reservoir expansion | Expensive evaluation | Reported metric |
| --- | --- | --- | --- |
| Reconstruction (default) | LDM proposes molecular SMILES to use as projection queries, sees original target and measured history; Empirical-q0 weighted Tanimoto GP/UCB sampling selects independent query/seed trials. Restart seeds are assigned by the task. | One frozen ReaSyn projection of the selected query; preserve every returned, replay-verified pathway. | Exact reconstruction, best target similarity, product and building-block diversity, averaged over **all requested targets**. |
| TDC | LDM replaces Graph GA offspring generation with SMILES targets; frozen ReaSyn projects each target, retaining first/highest-similarity product as in the released script; Empirical-q0 weighted Tanimoto GP/UCB sampling selects products. | One previously unseen canonical product passed to the released TDC oracle. | Top-10 score and released coarse-trapezoid AUC top-10 versus unique oracle calls. |

The molecular encoder is Morgan radius 2 / 2048 bits; an exact Tanimoto GP fits
at most the last 256 observations. Acquisition uses the shared posterior UCB implementation. Independent
`--proposal-batch-size` requests collect `--proposal-samples` valid occurrences
(`--reservoir-size` is its alias) before canonical deduplication. A separately
configured `--bo-pool-size` maintains the selection pool by q0-weighted sampling.
LDM then samples without replacement using
`alpha * log(q0 + epsilon) + eta * robust_z(UCB)`, controlled by
`--acquisition-alpha`, `--acquisition-eta`, and `--acquisition-z-clip`. It does not standardize sparse bits into an inappropriate
Euclidean RBF space. Mock features are explicitly synthetic hash features.

Reconstruction queries may repeat with different task-assigned sampling seeds:
these are different stochastic projection trials. Their evaluation identities
include canonical query and sampling seed; q0 groups occurrences by canonical
query alone. In TDC, one independent projection
occurrence contributes its first verified product, and products converging from
different queries accumulate frequency under the same canonical product
identity. Within-round occurrences remain legal. Products in the TDC track
are deduplicated by canonical **isomeric** SMILES before the oracle. The public
reconstruction target is legitimately visible to the proposal model. Passing
the target verbatim gives credit only when the frozen stock/reaction replay
actually supports a pathway. Zero-reaction stock entries are allowed, matching
the released sampler.

Both reconstruction and TDC expose `prior_mean@1` and `ldm_weights@1`.
Reconstruction allocates a retained query group's empirical mass uniformly to
its retained independent trials, then uses per-trial logits
`alpha*log(group_q0+epsilon) - log(group_trial_count) + eta*robust_z(UCB)`.
This avoids counting multiplicity twice and makes alpha effective: with three
trials of query A and one of B, equal acquisitions give A total first-draw mass
0.5, 0.75 and 0.9 for alpha 0, 1 and 2 respectively. BO-pool maintenance preserves
the original frequency of each surviving query group. Trial seeds, independent
projection receipts and diversity multiplicity remain unchanged. This grouping
version is frozen in scientific identity; pre-change reconstruction runs cannot
silently resume under the new sampling rule. TDC's one-group-per-product rule
and oracle deduplication are unchanged.

Provider settings are explicit: `--llm-wire-api` defaults to `responses` and
can be `chat_completions` for direct calls. `--llm-reasoning` defaults to `off`,
`--llm-temperature` to 0.7, and `--llm-extra-body-json` to an empty object.
Harness requires Responses and uses the same generation settings; its legacy
`--harness-thinking` alias must agree with `--llm-reasoning`. These settings are
frozen in scientific identity and recorded in the native Harness manifest.
Rebuild the shared sidecar for the new provider-request options.

When a proposal batch is smaller than the session pool, durable allocation
receipts rotate through every configured session, including across refills and
resumes. The history tool defaults to a compact index; use `detail=detailed`,
`candidate_ids`, `round`, `sort_by` and `order` for selected evidence.
Pilot diagnostics record baseline/compiled predictions, prior impact, q0 and
tilted entropy, effective sample size, KL divergence and top-k overlap.

## Scientific contract and changes

Primary local evidence is in `resources/source_provenance.json` and the
upstream `scripts/eval_recon.py`, `scripts/optimize_tdc.py`,
`reasyn/sampler/sampler.py`, plus paper sections 5 and appendix C.

* Reconstruction follows upstream nonstereochemical exact equality and Morgan
  radius 2 / **4096** bit similarity. Diversity uses `tdc.Evaluator('diversity')`
  on analog pathways with similarity **>= 0.8**. Missing or failed requested
  targets contribute zero. A subset reports its actual requested denominator;
  a paper-size dataset must contain all 1000 requested targets.
* Published reconstruction projector settings are width 8, exhaustiveness 4,
  100 EB samples/cycle; Enamine/ChEMBL/ZINC250k use 12/24/16 cycles respectively.
  ZINC250k additionally needs the released extra building-block index.
* TDC has **13 released oracles**, listed by both the script and Table pmo.
  Paper prose says 15; a 23-oracle PMO suite is not this released ReaSyn table.
  The contract explicitly reports the discrepancy. Use three seeds (0, 1, 2).
* TDC max budget is **10,000 unique canonical valid oracle queries**. Invalid
  SMILES and repeats cost no oracle query. The adapter fixes upstream's `>` vs
  `>=` off-by-one without changing the metric for legal trajectories.
* Released AUC uses trapezoids at frequency 100, starting from value zero;
  `core/metrics.py` matches the source formula. A partial campaign reports
  `auc_top10_observed`; `auc_top10` stays null until its declared query budget
  is exhausted. The original GA convergence criterion is algorithm-specific
  and is not silently applied to LDM. The metric helper supports explicit
  terminal padding for evaluating a documented upstream early stop.
* LDM changes the search algorithm, not the frozen oracle or synthesis model.
  A single original-input baseline projection is available for reconstruction.
  Multi-round LDM feedback/search uses extra projection trials and must be
  compared to a baseline with the **same projection trial/parameter budget**.
  The full LDM configs use four selected projection trials per target.
  Their cycle caps are 4 x 3 / 4 x 6 / 4 x 4 for Enamine / ChEMBL / ZINC.
  Matched original-query restart baselines use the same four trials. Resetting
  sampler state between trials differs from the original single 12 / 24 / 16
  evolving-cycle baseline, which is separately configured. Equal summed cycle
  allowances do not imply identical wall time or effective sampling work.
* `projection_cycle_allowance` is the sum of configured cycle caps reserved for
  started targets; it is not measured cycles completed. Upstream exact-match
  stopping or time limits can reduce actual work. Projection trials, configured
  cycles, width, exhaustiveness, EB samples, model parameters and worker logs
  are retained separately from TDC oracle calls.

The subprocess bridge loads the released AR/EB pair once per target batch and
constructs the **upstream** `Sampler` directly. It overrides environment-specific
index paths without changing checkpoint parameters. Every returned pathway is
replayed against the supplied stock/reaction matrix; unknown building blocks,
invalid reaction IDs, reaction failure and terminal-product mismatch are
rejected. For Torch >=2.6, the bridge explicitly uses `weights_only=False` for
user-supplied trusted Lightning checkpoints. GPU model reload overhead is
currently incurred per selected reconstruction query/per TDC expansion batch;
full production throughput has not been qualified.

## Setup

Obtain the official [ReaSyn v2 source](https://github.com/NVIDIA-BioNeMo/ReaSyn/tree/reasyn_v2)
matching `resources/source_provenance.json`, and set `REASYN_ROOT` to its absolute
checkout or extracted-source path. A conventional `../ReaSyn-reasyn_v2` path is
the fallback; a sibling checkout is not required. Use upstream `env.yml` for
the dedicated GPU environment. LDM's lightweight
runner can use its own environment; set `REASYN_PYTHON` to the upstream Python.
Install the task's `chemistry` extra in the runner environment for real
canonicalization, similarity, TDC oracle and diversity. `projection` lists
bridge dependencies; upstream dependency/data conversion instructions remain
authoritative for pickle compatibility.

Required assets (not bundled or downloaded by the adapter):

1. AR/EB checkpoints: `data/trained_model/nv-reasyn-ar-166m-v2.ckpt` and
   `nv-reasyn-eb-174m-v2.ckpt`. Override both with comma-separated
   `REASYN_MODEL_PATHS` or `--model-paths`, AR first.
2. Converted `data/processed/comp_2048/fpindex.pkl` and `matrix.pkl`; upstream
   explains SynFormer pickle conversion and the scikit-learn version change.
3. ZINC reconstruction: additional `data/processed/zinc250k_2048/fpindex.pkl`,
   generated from the supplied building-block file using upstream preprocessing.
4. The requested reconstruction target set. ZINC250k's supplied TXT has a
   `SMILES` header which the loader removes. Enamine and ChEMBL target files
   must be obtained as described upstream. Do not regenerate/re-filter the
   supplied ZINC building-block list from contradictory prose size thresholds.
5. For LDM proposals: `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY` in the
   environment. `LDM_LLM_*` / `OPENAI_*` aliases are accepted where applicable.
   The direct backend uses Chat Completions and checks that wire API before
   proposals. Persistent Harness methods use the same configured provider
   through the shared sidecar and require Docker/KVM and the task guest image
   environment described below.

```bash
# Run from the repository root with Python 3.10 or 3.11.
uv sync --project tasks/reasyn --group dev
# Add --extra chemistry to the sync command for real chemistry/oracle runs.
# Use tasks/reasyn/.venv/bin/python as python below.
python scripts/validate_tasks.py --task reasyn
python scripts/check_task_dependencies.py config/reasyn/mock.yaml --no-optional
python scripts/run_ldm_tts.py config/reasyn/mock.yaml --dry-run
python scripts/run_ldm_tts.py config/reasyn/mock.yaml
python scripts/run_ldm_tts.py config/reasyn/mock_tdc.yaml
python -m pytest tasks/reasyn/tests -q

# Inventory local real prerequisites; never downloads or modifies upstream.
python tasks/reasyn/scripts/prepare_official_data.py --upstream-root "$REASYN_ROOT"
python scripts/check_task_dependencies.py config/reasyn/reconstruction_tiny.yaml
python scripts/run_ldm_tts.py config/reasyn/reconstruction_tiny.yaml --dry-run
# Execute after assets/environment are available:
python scripts/run_ldm_tts.py config/reasyn/reconstruction_tiny.yaml
python scripts/run_ldm_tts.py config/reasyn/tdc_tiny.yaml
```

Named runner-enforced full profiles are
`reconstruction_{enamine,chembl,zinc250k}_{original,matched_baseline,ldm}`
and `tdc_10000`. Each of the nine reconstruction configs is directly runnable:

```bash
python scripts/run_ldm_tts.py config/reasyn/reconstruction_enamine_ldm.yaml
python scripts/run_ldm_tts.py config/reasyn/reconstruction_chembl_ldm.yaml
python scripts/run_ldm_tts.py config/reasyn/reconstruction_zinc250k_ldm.yaml
# Replace _ldm with _matched_baseline or _original for corresponding baselines.
```

Each reconstruction config accepts an explicit `seed` argument; use seeds 0, 1,
2 in separate named run directories for replicated comparisons. The helper
`scripts/write_suite_configs.py` generates all dataset/method/seed configs and
the 13-oracle x 3-seed TDC grid without launching network requests. These are draft reproducible configurations, not a qualification
claim. For a targeted debug run, call the task module with `--target-smiles`,
`--dataset custom` and `--out-dir`. A suite runner can call
`tdc_10000` separately for each released oracle and seed; do not average partial
or missing oracle runs. `scripts/aggregate_tdc_results.py` enforces the full
13-by-3 inventory, 10k real-call condition, and a predeclared shared
`scientific_identity.json` configuration with `oracle`/`seed` omitted. If a
field is intentionally oracle-specific, declare it under `oracle_overrides`
before aggregation.

For paper-scale reconstruction tables, `scripts/aggregate_reconstruction_results.py`
requires a predeclared JSON suite with `benchmark: reconstruction`, `seeds: [0,1,2]`,
the complete ordered `targets` and frozen `assets` dictionaries for all three
datasets, `methods`, and `cases`. Each method declares `id`, `label`,
`search_method`, `proposal_mode`, `model`, and `trials_per_target` (1 or 4);
search methods also freeze `settings_by_dataset` before evaluation. Each case
declares `method`, `dataset`, `seed`, and its run `path`. Paths may be relative to
the manifest. Run with the chemistry task environment:

```bash
python tasks/reasyn/scripts/aggregate_reconstruction_results.py /path/to/suite.json \
  --output /path/to/report
```

The checker requires every 1,000-target, three-seed case, verifies individual
completion and selected projection receipts, checks frozen assets/model IDs and
paper projector settings, and recomputes all four metrics. Completed empty
projections contribute zero; missing or unfinished targets are not silently
dropped. It emits a 12-metric horizontal Markdown table and full-precision JSON,
using mean and population standard deviation (`ddof=0`) across seeds, consistent
with the existing TDC aggregator's convention. Passing this result audit does
not independently attest source cleanliness, identical upstream trajectories,
or global SOTA.

An explicitly requested full-dataset single-seed run can instead declare
`replication_protocol: single_seed` and `seeds: [42]`. It still requires every
1,000-target dataset for every method, checks the exact seed in every receipt,
and emits all twelve metric point estimates without CI or a fabricated
cross-seed standard deviation. This is full dataset scale, not the paper's
three-replicate statistical protocol. The original three-seed contract remains
the default; dropping seeds from it without declaring the alternate protocol
is rejected.

## Persistent research and pilot comparison

`--search-method` selects `ldm`, `bo`, `llm`, `harness`, `ldm_harness`, or
`ldm_harness_compiled`. Direct LDM and LLM use independently sampled provider
requests. Pure BO draws from a supplied score-blind SMILES pool
(`--bo-targets-file` or `REASYN_BO_TARGETS`) and selects by Tanimoto GP/UCB;
it makes no proposal-provider requests. Real BO never substitutes the mock
molecular fixture for an external candidate space.

Harness methods create parallel independent persistent research sessions.
`harness` uses accepted candidates in submission order; its shipped configs
request exactly the evaluation batch from one persistent session.
`ldm_harness` samples the empirical proposal distribution tilted by GP/UCB. `ldm_harness_compiled` additionally runs a separate persistent
policy researcher through the shared policy controller and isolated runner.
The task packages role instructions, molecular research and policy skills,
research tools, and a pinned guest-image recipe in `resources/harness`.
Sessions receive incremental completed history and can query full measured
history or product membership on demand. They use the guest sandbox to research
molecules, submit an exact-count candidate file, and repair task validation
errors in the same session. Provider messages and native sessions are captured
by the shared Harness alongside candidate artifact digests and lineage.

```bash
# Build on a Linux host with Docker and /dev/kvm available.
docker build -t ldm-pi-harness:latest harnesses/pi
python scripts/check_task_dependencies.py config/reasyn/tdc_ldm_harness_tiny.yaml
python scripts/run_ldm_tts.py config/reasyn/tdc_ldm_harness_tiny.yaml --dry-run
python scripts/run_ldm_tts.py config/reasyn/tdc_ldm_harness_tiny.yaml
python scripts/run_ldm_tts.py config/reasyn/tdc_ldm_harness_compiled_tiny.yaml
```

Corresponding `reconstruction_{harness,ldm_harness,ldm_harness_compiled}_tiny.yaml`
and `tdc_harness_tiny.yaml` configs are provided. Configure a remote Docker host
with `--harness-docker-host` only when source/artifact/cache bind mounts resolve
on that host. `--harness-mcp-config` uses the shared MCP configuration;
repeat `--harness-tool-budget NAME=LIMIT` to bound declared tools. The default
session deadline is 1800 seconds, with a 2100-second response timeout. These
are research budgets, separately accounted from projection and oracle work.
Direct `llm`/`harness` configurations use the first accepted reservoir entries;
if an explicit configuration requests more candidates than evaluations, the
remaining entries are not automatically evaluated.

The pilot matrix uses one identical `shared_start` initialization round per
seed, then five optimization rounds across all six methods, with seeds 0, 1,
and 2. It currently compares the released `jnk3` case and two evaluations per
round. Set `REASYN_RUNS_ROOT` to a new output root and `REASYN_BO_TARGETS` to an
absolute, score-blind, one-SMILES-per-line pool file before generating plans:

```bash
python scripts/run_pilot_evaluation.py config/pilot_evaluation/reasyn.yaml --dry-run
python scripts/run_pilot_evaluation.py config/pilot_evaluation/reasyn.yaml
# Continue an interrupted matrix with the same configuration and provenance:
python scripts/run_pilot_evaluation.py config/pilot_evaluation/reasyn.yaml --resume
```

The shared pilot runner requires a clean committed checkout for real execution.
Tiny/pilot runs validate integration and measure a declared partial trajectory;
they do not establish full-budget TDC AUC or scientific qualification.

## Resume and artifacts

`python -m tasks.reasyn.ldm_task.procedure ... --resume-from <run-directory>`
restores the shared checkpoint. Target manifest, scientific settings and source
archive identity must match. Iterations can extend. Model name/token cap,
projector settings, file paths and actual checkpoint/index/upstream-source
SHA-256 digests are locked across resume. Source files are checked against the
supplied archive provenance before a real run. All real asset digests are
computed once at campaign startup and included in cached projection requests. Completed independent proposal responses are cached before projection. Each
projection target writes an atomic checkpoint, including empty searches; a
retry reuses completed occurrences with their original seeds and reruns only
pending targets. A changed request is rejected even if an earlier worker wrote
no result. `--projection-timeout` is the single-target batch allowance; each
additional pending serial target adds `--projection-time-limit` seconds. With
defaults, a 32-target batch has 34,600 seconds instead of 3,600. Worker timeout
or failure preserves progress and pauses the campaign for explicit resume. Completed TDC
scores come from an ordered cache; an interrupted, charged oracle request is
explicitly marked and refused on retry instead of silently evaluated again.

`events.jsonl`, `checkpoint.json`, `summary.json`, `budget.json`, `status.json`,
`result.json`, `trajectory.csv`, `search_manifest.json`, `selection_record.json`
and an experiment contract snapshot are written per campaign. Multiple
reconstruction targets have individual subdirectories and an aggregate
`benchmark_result.json`. Resume may retry interrupted proposal rounds; all
attempted endpoint preflights and proposal calls remain separately counted.
`--recovery-attempts` is a fixed additional attempt allowance (default 3);
retries do not increase the requested scientific round/evaluation count.
`--projection-retry-targets` separately limits reattempted projection work.
TDC canonical historical products are rejected in the task before reservoir
admission and trigger feedback plus bounded refill.
`--max-replenishment-batches` bounds these extra minibatches. If valid-occurrence
or unique-product quotas cannot be reached, the run stops with the recorded
`proposal_replenishment_budget_exhausted` reason and does not report a complete
fixed-budget run.

Fine-tuning collection is opt-in with `LDM_DATA_COLLECTION_ENABLED=1`. The
accepted action boundary is parsed, chemically validated projection-target
JSON, before projecting or measuring outcomes. Original target/observed scores
are legitimate prompt state; generated products, source artifacts, hidden
outcomes and run metadata are excluded. Synthetic mock records are tagged in
collection provenance and must not be mixed into scientific training data.

## Current verification limit

Both mock tracks and task contract tests run locally. Tests execute a bridge
with fake model modules to verify exact sampler settings, checkpoint loading,
stock/reaction replay, invalid routes and output capture; metric tests compare
AUC directly with the released source function and exercise real RDKit
stereochemistry and Tanimoto acquisition. They do not establish neural-model
correctness or a measured benchmark score. See `resources/verification_record.json`
and the repository `docs/science-benchmarks.md` for the qualification state.
Source and asset file SHA-256 identities are recorded. A real provider/session,
agent research, candidate submission, frozen-model projection, and oracle
measurement must be validated together before upgrading qualification.
Mocked session/model tests are infrastructure evidence only.
