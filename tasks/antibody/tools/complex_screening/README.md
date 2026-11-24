# Optional protein tools

Run each predictor with its isolated environment. Outputs must use a new or empty directory. Structure confidence is auxiliary; the adapters do not change Absolut objectives or reject candidates automatically.

## SolubleMPNN

Set `proteinmpnn_variant` to `soluble`, or enable `--enable-protein-tools` to use Pi's `sequence_h3_soluble(backbone_id, count, temperature)`. It shares the existing sequence budget and requires a generated backbone. Vanilla remains the default.

## Complex prediction

```bash
python -m tasks.antibody.tools.complex_screening --out /absolute/output protenix \
  --input /absolute/input.json --executable /isolated/bin/protenix \
  --gpu 0 --seeds 19 43 101 --samples 1

python -m tasks.antibody.tools.complex_screening.boltz \
  --input /absolute/input.yaml --out /absolute/output \
  --executable /isolated/bin/boltz --cache /absolute/cache --gpu 0

python -m tasks.antibody.tools.complex_screening.esmfold2 \
  --input /absolute/input.json --out /absolute/output \
  --model-dir /absolute/model-snapshot --gpu 0
```

Protenix supports `protenix_base_default_v1.0.0`. Precomputed MSA paths must be absolute; `--no-use-msa` disables upstream MSA queries. Set `PROTENIX_ROOT_DIR` to the cache root and put the isolated environment's `bin` on PATH.

Boltz uses native YAML and requires matching CIF, PAE, pLDDT and confidence artifacts. ESMFold2 requires `esm==3.5.0` and local model/CCD files; input is a JSON list containing one canonical protein complex with explicit unique one-character chain IDs. Its defaults are 20 loops and 200 sampling steps, without MSA.

## ipSAE

```bash
python -m tasks.antibody.tools.complex_screening --out /absolute/output ipsae \
  --python /isolated/bin/python --script /absolute/IPSAE/ipsae.py \
  --pae /absolute/confidence.json --structure /absolute/model.cif
```

Use genuine matching structure/confidence artifacts. Native Boltz `pae_*.npz` requires a matching regular `plddt_*.npz` companion in the same directory. The runner copies companions into isolated aliases. Protenix and ESMFold2 native confidence can be converted with `protenix_pae` and `esmfold2_pae`; both check atom/token order, chain partition and confidence scale.

With `--enable-protein-tools`, Pi's `score_interface(artifact_id)` requires host `IPSAE_SCRIPT`, optional `IPSAE_PYTHON`, and `RUN/interface_inputs/ID.json` containing relative `pae` and `structure` paths inside `interface_inputs/`.

## BoltzGen

Requires `boltzgen==0.3.2`, adherence and inverse-fold checkpoints, chemical data and native YAML/CIF inputs. Native examples are in `examples/boltzgen/`; place the corresponding original RCSB mmCIF files alongside them.

Enable Pi's `design_h3_boltzgen(count)` with `--enable-boltzgen` and host settings `BOLTZGEN_PYTHON`, `BOLTZGEN_SCAFFOLD`, `BOLTZGEN_CHECKPOINT`, `BOLTZGEN_INVERSE_FOLD_CHECKPOINT`, `BOLTZGEN_MOLDIR`, and `BOLTZGEN_SPEC_PD1` / `BOLTZGEN_SPEC_HER2`. One call consumes one backbone and `count` sequence slots, including invalid proposals. Existing H3 constraints remain enforced. The standalone module is `boltzgen_design`.

Fixed-framework projection applies only to generated proposals. Preserve raw outputs. Predictor structure/PAE pairs must retain their original coordinates. BoltzGen backbones do not enter the RFD3 ProteinMPNN resequencing route. The tool is disabled by default.

## PXDesign

```bash
python -m tasks.antibody.tools.complex_screening.pxdesign \
  --spec /absolute/target.yaml --out /absolute/output \
  --source-root /absolute/PXDesign --checkpoint-dir /absolute/checkpoints \
  --ccd-dir /absolute/ccd --gpu 0
```

Requires PXDesign 0.1.0 source, PXDesignBench 0.1.2 and compatible Protenix `0.5.0+pxd` dependencies. Native YAML specifies one target chain and `binder_length`. This entry generates whole protein binders independently of the 11-residue H3 campaign. Optional downstream selection is outside this adapter.

All modules expose `--help` for their complete arguments. Runtime logs and status are written to the selected output directory.
