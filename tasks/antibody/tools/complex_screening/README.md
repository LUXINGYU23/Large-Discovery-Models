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

New tools are optional. Protenix v1 is an experimental auxiliary predictor only; no rejection threshold or campaign objective is changed. Boltz-2 and BoltzGen work in this branch remains pending real GPU validation; installation and configuration checks alone do not establish readiness. Scientific efficacy and large-scale load performance are not established by smoke tests. Large weights, structures and runtime outputs remain on H20; the small validation records here retain paths and hashes for audit.


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
