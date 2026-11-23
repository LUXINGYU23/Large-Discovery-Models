# H20 calibration record

These are the actual research drivers used for the reported experiments. Paths are pinned to `/Aidata/shared/ldm/antibody-poc`, with isolated source under `tmp/expanded-protein-tools`. They rely on the already verified local model environments/caches and public native structures described in `docs/protein-tools-calibration.md`. They do not install software or download weights.

They preserve the historical experiment layout; choose new experiment directories before reproducing to keep existing evidence. Native wrappers reject nonempty output directories. These scripts are research reproduction helpers, not new production commands.

Execution order:

1. Prepare `runs/complex-construct-calibration-001/inputs` using `python -m tasks.antibody.tools.complex_screening.construct_controls` and the existing `runs/protenix-validation/calibration-001/prepared-controls.json`. Use the BoltzGen environment for Biopython/NumPy/YAML utilities.
2. Run `pose_audit` on 5GGR against itself with reference/prediction author chains A/B/Y to create `native-self-check.json` in that experiment root. `audit_baseline_poses.py` checks the previously completed full-Fab positives.
3. `run_construct_calibration.py` predeclares and runs 24 additional PD-1 jobs on GPUs 2/6/7, requires every native prediction and ipSAE result, and audits positives against the real native interface. Its completion marker is written only after all job futures succeed.
4. `prepare_her2_calibration.py` creates the full 607-aa HER2 extracellular sequence/Fv inputs and the 12-job plan. Run `pose_audit` on 1N8Z against itself with both chain lists B/A/C to create its `native-self-check.json`.
5. `run_her2_calibration.py` waits for the PD-1 experiment to finish before using those GPUs. `audit_boltz_confidence.py` checks native PAE/pLDDT and CIF ordering and compares PAE at native contact pairs for the declared diagnostic examples.
6. `run_pae_sensitivity.py` declares the exploratory 15/20-A grid and waits for HER2 completion. It scores every Fv/no-MSA positive/mismatch seed in both antigens and models using four CPU workers. It never selects a threshold or changes model predictions.
7. `collect_construct_calibration.py` and `collect_her2_calibration.py` assemble results, chain-pair scores, hashes and native audits. The first includes 12 existing Fab runs plus 24 new runs. Missing work remains explicitly listed; only complete reports are published.

GPU availability must match the plan when launching. The scripts use fixed seeds and one sample per complex/seed, with no outcome-based replacement or best-seed selection. A zero ipSAE is a valid numerical result, not an execution failure. Mismatch controls are computational controls, not experimentally certified nonbinders. Known positive structures may overlap training data.
