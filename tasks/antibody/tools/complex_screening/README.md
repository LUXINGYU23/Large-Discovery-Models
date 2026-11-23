# Validated optional protein tools

## SolubleMPNN

Structure-design config accepts `"proteinmpnn_variant": "soluble"`; vanilla remains default. Bootstrap assets fetch the soluble checkpoint at the pinned ProteinMPNN commit to a separate `soluble/` directory, with the existing source/hash sidecar handling. Fixed framework and H3 mapping remain unchanged.

Pi now exposes `sequence_h3_soluble(backbone_id, count, temperature)`. It requires a backbone already generated in the campaign, consumes the shared sequence budget, and does not alter the shared initial vanilla design. Model weights do not establish soluble expression.

H20 standalone CPU smoke passed in 6.99 seconds. A guided real Qwen3-8B HER2 session completed three evaluations in ~214 seconds: vanilla initialization -81.87, then two genuine soluble calls -92.22/-79.06. Raw Oracle output, mapped H3, actual soluble checkpoint and next-turn measured history were checked. This validates tool execution and feedback, not unbiased autonomous model choice. Evidence: `validation/h20-soluble-llm.json`. No PD-1 soluble LLM session was tested.

Matched pilot: identical generated backbone, temperature 0.5, seeds 101/202/303, 16 attempted sequences per method per seed. No replacement for invalid candidates. Vanilla admitted 24/48; soluble 14/48. Best HER2 energies: vanilla -97.47, soluble -96.64. Best PD-1 energies: vanilla -75.11, soluble -75.41. Keep vanilla default and soluble optional. A single backbone is not a universal comparison; PD-1 scores are controlled sequence-oracle comparisons on a HER2-derived backbone, not PD-1-conditioned structural designs. Plan/summary are in `validation/h20-mpnn-paired-*.json`; H20-only reproduction script: `tasks/antibody_rfd3/repro/h20/compare_mpnn.py` (fixed output directory, refuses overwrite).

## ipSAE

A confidence-analysis tool, not a structure predictor or affinity oracle. Supply genuine structure and matching PAE. No automatic candidate rejection or change to Absolut objectives is introduced.

```bash
python -m tasks.antibody.tools.complex_screening --out /absolute/new/score ipsae \
  --python /isolated/env/bin/python --script /pinned/IPSAE/ipsae.py \
  --pae /absolute/pae.json --structure /absolute/model.cif
```

Each job copies inputs into a unique/empty output directory, preserves their hashes, command, logs, chain-pair directional scores, raw output and status. Child failure/timeout raises and preserves failed status. ipSAE and NumPy must be installed externally; upstream code is not vendored. Verified upstream commit `6174cf9e71cb1bd660cc805856a18c4871a6dec3`, script SHA256 `10cf9b08c68c91e06cb28526cf2026f47a3980c9048fd3226d13e3304eaf1c27`. Upstream: https://github.com/DunbrackLab/IPSAE (MIT).

Pi exposes `score_interface(artifact_id)`. Host sets `IPSAE_SCRIPT` and optionally `IPSAE_PYTHON`. Before use, prepare `RUN/interface_inputs/ID.json`:

```json
{"pae": "prediction/pae.json", "structure": "prediction/model.cif"}
```

Both paths resolve inside `interface_inputs/`. Arbitrary host paths and escaping paths are rejected. IDs use letters, digits, underscore and hyphen. Output directories are under `RUN/interface_scores/`. Missing data fails explicitly: the current RFD3 pipeline does not produce PAE, so this tool is optional and requires prepared prediction artifacts. It is not automatically applied to every candidate. Consider CPU time per invocation (120 second host timeout); this version has no separate global scoring budget.

Official AURKA–TPX2 example: A→B ipSAE 0.448952, B→A 0.866498. Both direct invocation and actual agent Unix-socket route succeeded on H20. Two fixture runner tests cover output isolation and failure status; seven optional-tool/shared-procedure tests passed in the experimental checkout. No fresh real-LLM session autonomously invoking ipSAE was tested.

## Scope

New tools are optional. Protenix v1 is an experimental auxiliary predictor only; no rejection threshold or campaign objective is changed. Boltz-2, BoltzGen, ESMFold2 and PXDesign now have real GPU smoke evidence, with different scopes described below. None is a validated hard screening rule. Scientific efficacy and large-scale load performance are not established by smoke tests. Large weights, structures and runtime outputs remain on H20; the small validation records here retain paths and hashes for audit.


## Protenix v1 auxiliary prediction

Uses the public `protenix_base_default_v1.0.0` model through an external isolated CLI. The installed package version (`protenix==2.0.0` on H20) is distinct from the model version. V2 is not enabled by this adapter.

```bash
python -m tasks.antibody.tools.complex_screening --out /absolute/new/prediction \
  protenix --input /absolute/native-input.json \
  --executable /isolated/protenix/bin/protenix --gpu 0 \
  --seeds 19 43 101 --samples 1
python -m tasks.antibody.tools.complex_screening.protenix_pae \
  /absolute/native_full_data_sample_0.json /absolute/native_sample_0.cif \
  /absolute/new/ipsae-confidence.json
```

Use complete molecular sequences. `pdb_sequences` builds native input from PDB SEQRES, refusing missing or noncanonical chains; it does not silently delete unresolved residues. MSA is enabled by default. Precomputed MSA paths must be absolute and their hashes are recorded. Without precomputed MSA, the upstream CLI may contact its configured remote service. `--no-use-msa` requests single-sequence inference. Full confidence output is required; every complex/seed must produce the requested CIF samples and native confidence files.

The converter accepts canonical protein ATOM records with one ordered CA per residue. It checks atom/token ordering, chain partitions, finite PAE, and native pLDDT scale before converting units. PAE is never synthesized. Feed the converted JSON and matching CIF into the existing `ipsae` command.

H20 calibration used PD-1/nivolumab and PD-1/trastuzumab mismatch, complete sequences, real unpaired MSA, query-only paired MSA, and fixed seeds 19/43/101. Heavy-chain/antigen minimum directional ipSAE was 0/0/0.115505 for the positive and 0/0/0 for mismatch. The mismatch is not an experimentally proven nonbinder. Earlier exploratory seed 7 had a much higher positive interface score; it does not establish a calibrated screening threshold. Global ipTM can be dominated by heavy/light pairing.

Consequently `run.json` marks `hard_filter_eligible: false`. A low score means uncertain prediction and must not reject an antibody. Calibration records are in `validation/h20-protenix-control-calibration.json`, `validation/protenix-calibration-status.json`, and `validation/protenix-positive-seed-sensitivity.json`.

New wrapper validation on H20: real complete-sequence/MSA positive seed 19 prediction succeeded, followed by the strict PAE converter and isolated ipSAE runner. Four Protenix/ipSAE adapter tests passed with predictor dependencies present. Fresh wrapper evidence: `validation/h20-protenix-wrapper-v1.json` and `validation/h20-protenix-wrapper-v1-ipsae.json`. This validates the independent command line route; the predictor is not registered as a new autonomous Pi tool.

Verified H20 dependencies and model SHA256 are recorded in `validation/h20-protenix-environment.json`. Set `PROTENIX_ROOT_DIR` to the model cache root before running; on H20 this is `/Aidata/shared/ldm/antibody-poc/cache/protenix`. Put the isolated Protenix environment `bin/` on PATH so upstream kernel builds can locate ninja.

## Boltz-2 auxiliary complex prediction

`boltz.py` calls the isolated `boltz==2.2.1` native CLI with model `boltz2`, provided native YAML/MSAs, explicit seeds, full PAE, and native PyTorch kernels. It requires every requested CIF plus matching PAE, pLDDT and confidence files; a successful process exit with missing predictions is a failure. It does not invoke the small-molecule affinity head as antibody affinity.

```bash
python -m tasks.antibody.tools.complex_screening.boltz \
  --input /absolute/input.yaml --out /absolute/new/boltz \
  --executable /isolated/boltz2/bin/boltz --cache /absolute/boltz-cache \
  --gpu 0 --seed 19
```

Pass native `pae_*.npz` and matching CIF to the existing `ipsae` command. The runner requires and copies the matching `plddt_*.npz`, plus confidence JSON when present. It uses relative isolated aliases to avoid the upstream script accidentally replacing `pae` within an absolute directory name. Symlink companions are rejected.

Real complete-Fab PD-1 controls with provided unpaired MSAs, seeds 19/43/101, one sample each: all six predictions and ipSAE runs succeeded. The positive heavy/antigen minimum directional ipSAE is 0.011141/0.010767/0; mismatch is 0.140091/0.045320/0.013382. This control **fails to establish useful discrimination**. Keep it auxiliary; do not reject on low scores or treat a higher mismatch score as experimental affinity. Evidence: `validation/h20-boltz2-calibration.json`.

## BoltzGen optional H3 proposal tool

Pinned `boltzgen==0.3.2`, with official adherence and inverse-fold checkpoints. `boltzgen_design.py` snapshots the nested native YAML/CIF graph, generates **one backbone**, then **N native inverse-folded sequences** with the upstream antibody C-avoidance restriction. It verifies the complete reference heavy/light framework and exactly 11 mapped H3 residues, rather than trimming arbitrary generated binders.

Native proposal coordinates can have small fixed-framework numerical jitter. We retain raw output, require pre-projection framework CA RMSD ≤0.25 Å, rigidly align to the scaffold, restore matching fixed-framework atoms, and require projected CA RMSD ≤0.05 Å and both H3 boundary C–N distances in [1.0, 1.8] Å. H3 sequence and aligned H3 coordinates remain native. **This projection is exclusively for generated proposals and must never modify predictor CIF/PAE pairs.** Geometry checks do not establish binding.

The host must explicitly enable `--enable-boltzgen` for `harness`/`ldm_harness`. Pi then exposes `design_h3_boltzgen(count)`; otherwise the tool is absent. One call consumes one shared backbone slot and `count` shared sequence slots, including invalid proposals. Existing 11-residue sequence constraints remain unchanged. BoltzGen backbones cannot enter the RFD3 ProteinMPNN resequencing route. Vanilla initialization and defaults remain unchanged.

Guided real Qwen3-8B LDM sessions completed three real Absolut evaluations for each antigen. PD-1 BoltzGen H3 `SSSSYGGAMDY` scored -69.01; HER2 `SGSSGYSAMDY` scored -84.54. Actual acquisition-selection events and measured feedback were audited. Admission varied: PD-1 13/16 then 0/16; HER2 3/16 then 0/16, with duplicate sequences among admitted rows. Later rounds used existing alternatives/fallbacks; these are not all-BoltzGen campaigns or efficacy comparisons. Full ranking/refolding pipeline is not validated. Evidence: `validation/h20-boltzgen-llm.json`.

## ESMFold2 auxiliary complex prediction

Pinned `esm==3.5.0`, official `biohub/ESMFold2` snapshot `69869f737beffec5294845ede23db5fc0b4f509e`. All six model shards and CCD were checked against official LFS SHA256 metadata (`validation/esmfold2-official-model-manifest.json`). Uses local files only, no MSA submission, native model precision, explicit pure-PyTorch backend. No CUDA driver upgrade is required for the tested fallback. Forcing the entire model to BF16 caused an output-head dtype mismatch; the adapter preserves native defaults.

```bash
python -m tasks.antibody.tools.complex_screening.esmfold2 \
  --input /absolute/one-complex.json --out /absolute/new/esmfold2 \
  --model-dir /absolute/esmfold2-snapshot --gpu 0 --seed 19
python -m tasks.antibody.tools.complex_screening.esmfold2_pae \
  /absolute/new/esmfold2/native-confidence.npz /absolute/new/esmfold2/model.cif \
  /absolute/new/esmfold2/ipsae-confidence.json
```

Input is a list containing one native protein complex, with explicit unique one-character IDs and canonical complete sequences, as in the Protenix controls. Defaults are 20 loops, 200 sampling steps, one diffusion sample. The strict converter validates native token atom ranges, CIF residue/CA order, chain partitions, dimensions, finite PAE, and normalized pLDDT. It broadcasts each native token pLDDT only to that token's native atoms and converts to the 0–100 scale; PAE is unchanged. Run the converted JSON and original CIF through `ipsae`.

Default-parameter complete-Fab PD-1 positive and mismatch predictions and ipSAE completed for seeds 19/43/101 (six runs). Positive heavy/antigen minimum directional ipSAE was 0.446642/0/0.421704; mismatch was 0/0/0. Seed 43 misses the known positive interface, so this remains auxiliary: one control pair and three seeds do not establish a rejection threshold or binding measurement. Seed 19 runtime was ~109 seconds per prediction including model load. Evidence: `validation/h20-esmfold2.json`.

## PXDesign standalone protein binder generation

Pinned official source `f788441313c84c3074fe9596ac2433f96b15c763`, package 0.1.0, PXDesignBench 0.1.2, Protenix `v0.5.0+pxd`, Python 3.11/Torch 2.3.1. The wheel omits the namespace `configs` package; `pxdesign.py` explicitly loads the pinned source checkout rather than inventing configs. The CCD must be the official `components.v20240608` assets, not another Protenix release's CCD.

```bash
python -m tasks.antibody.tools.complex_screening.pxdesign \
  --spec /absolute/target.yaml --out /absolute/new/pxdesign \
  --source-root /absolute/PXDesign --checkpoint-dir /absolute/pxdesign/checkpoint \
  --ccd-dir /absolute/pxdesign/ccd_cache --gpu 0 --seed 19
```

Native YAML must specify one target chain and `binder_length`. The wrapper snapshots the target/spec, hashes resources, executes native raw inference, and requires a real predicted CIF with target A0 and requested-length binder B0. The raw API avoids the upstream full-pipeline downloader's three unused evaluation checkpoints; no missing checkpoints are faked. Full AF2/Protenix selection is not claimed.

Real 80-residue binder generation succeeded for PD-1 and HER2, 200 steps, seed 19, ~46/67 seconds including initialization. **These are whole protein binders, not 11-residue CDRH3 proposals.** This tool stays independent of the current antibody campaign, with no Absolut H3 conversion or autonomous Pi registration. Generation does not establish binding or design quality. Evidence: `validation/h20-pxdesign-raw.json`.

## H20 operation and resource isolation

See `docs/protein-tools-expanded.md` for server paths, configuration, reproduction and deployment scope. Large checkpoints, chemical dictionaries and raw results remain outside Git. The primary `/Aidata/shared/ldm/antibody-poc/code` worktree is not changed by these validations. Protenix v2 is excluded: official weights retrieval and authorized sourcing remain unresolved; public repository code alone does not grant access to third-party mirrored weights.
