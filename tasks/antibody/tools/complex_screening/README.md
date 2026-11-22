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

This delivery contains no Protenix predictor, weight mirrors, score thresholds, experimental predictor calibration, or BoltzGen adapter. New tools are optional. Scientific efficacy and large-scale load performance are not established by smoke tests. Large weights, structures and runtime outputs remain on H20; the small validation records here retain paths and hashes for audit.
