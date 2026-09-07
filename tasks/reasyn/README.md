# ReaSyn Science Benchmark for LDM

This task implements **the first/main reconstruction evaluation** and the
released TDC goal-directed optimization evaluation from the locally supplied
ReaSyn v2 code and paper. Both execute through `ldm_tts.campaign.run_campaign`,
including shared candidate admission, budgets, acquisition, checkpoints and
resume. No model training is needed. Registration is valid; scientific
qualification remains **draft** because official assets and a real GPU seed run
have not been verified. Synthetic mock values are not ReaSyn scores.

## What LDM controls

| Track | Candidate and reservoir expansion | Expensive evaluation | Reported metric |
| --- | --- | --- | --- |
| Reconstruction (default) | LDM proposes molecular SMILES to use as projection queries, sees original target and measured history; Tanimoto GP/UCB selects queries. Restart seeds are assigned by the task. | One frozen ReaSyn projection of the selected query; preserve every returned, replay-verified pathway. | Exact reconstruction, best target similarity, product and building-block diversity, averaged over **all requested targets**. |
| TDC | LDM replaces Graph GA offspring generation with SMILES targets; frozen ReaSyn projects each target, retaining first/highest-similarity product as in the released script; Tanimoto GP/UCB selects products. | One previously unseen canonical product passed to the released TDC oracle. | Top-10 score and released coarse-trapezoid AUC top-10 versus unique oracle calls. |

The molecular encoder is Morgan radius 2 / 2048 bits; an exact Tanimoto GP fits
at most the last 256 observations. Acquisition uses the shared LDM posterior
UCB implementation. It does not standardize sparse bits into an inappropriate
Euclidean RBF space. Mock features are explicitly synthetic hash features.

Reconstruction queries may repeat with different task-assigned sampling seeds:
these are different stochastic projection trials. Products in the TDC track
are deduplicated by canonical **isomeric** SMILES before the oracle. The public
reconstruction target is legitimately visible to the proposal model. Passing
the target verbatim gives credit only when the frozen stock/reaction replay
actually supports a pathway. Zero-reaction stock entries are allowed, matching
the released sampler.

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

The source checkout defaults to sibling `ReaSyn-reasyn_v2`, or set `REASYN_ROOT`.
Use the upstream `env.yml` for its dedicated GPU environment. LDM's lightweight
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
   proposals. No live sidecar/harness backend is claimed in this adapter.

```bash
# Run from LDM repository root with Python >=3.10.
python scripts/validate_tasks.py --task reasyn
python scripts/check_task_dependencies.py config/reasyn/mock.yaml --no-optional
python scripts/run_ldm_tts.py config/reasyn/mock.yaml --dry-run
python scripts/run_ldm_tts.py config/reasyn/mock.yaml
python scripts/run_ldm_tts.py config/reasyn/mock_tdc.yaml
python -m pytest tasks/reasyn/tests -q

# Inventory local real prerequisites; never downloads or modifies upstream.
python tasks/reasyn/scripts/prepare_official_data.py --upstream-root ../ReaSyn-reasyn_v2
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
13-by-3 inventory and 10k real-call condition.

## Resume and artifacts

`python -m tasks.reasyn.ldm_task.procedure ... --resume-from <run-directory>`
restores the shared checkpoint. Target manifest, scientific settings and source
archive identity must match. Iterations can extend. Model name/token cap,
projector settings, file paths and actual checkpoint/index/upstream-source
SHA-256 digests are locked across resume. Source files are checked against the
supplied archive provenance before a real run. All real asset digests are
computed once at campaign startup and included in cached projection requests. Repeated successful
projection requests reuse their durable request/result pair. Completed TDC
scores come from an ordered cache; an interrupted, charged oracle request is
explicitly marked and refused on retry instead of silently evaluated again.

`events.jsonl`, `checkpoint.json`, `summary.json`, `budget.json`, `status.json`,
`result.json`, `trajectory.csv`, `search_manifest.json`, `selection_record.json`
and an experiment contract snapshot are written per campaign. Multiple
reconstruction targets have individual subdirectories and an aggregate
`benchmark_result.json`. Resume may retry interrupted proposal rounds; all
attempted endpoint preflights and proposal calls remain separately counted.

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
and the repository `docs/science-benchmarks.md` for actual local/SSH diagnostics.
Local archives have no Git metadata; file SHA-256 identities are recorded and
formal clean-checkout qualification gates remain pending.
