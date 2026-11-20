# Independent structural design tool

RFDiffusion 3 → original ProteinMPNN → optional CDRH3 extraction. This CLI does
not run Pi, LDM, a GP, Absolut!, structural relaxation, or binding-energy scoring.
Its controller uses Python's standard library. Heavy dependencies run in isolated
Linux Python 3.12 subprocesses, independently of the old AntBO task environment.

## H20 deployment

Workspace root: `/Aidata/shared/ldm/antibody-poc`. Run from its `code/` checkout:

```bash
python3 -u -m tasks.antibody.tools.structure_design.bootstrap \
  --root /Aidata/shared/ldm/antibody-poc
```

The installer creates `env/rfd3/`, `env/proteinmpnn/`, and a pinned original
ProteinMPNN source under `env/sources/ProteinMPNN/`. It fixes Python 3.12,
PyTorch 2.6.0 (CUDA 12.4), NumPy 1.26.4, rc-foundry 0.2.0, and AtomWorks 2.1.1.
The bundled `requirements/` locks replay the validated transitive versions.
Each environment also saves its resolved `requirements.lock.txt` after `pip check`;
the export uses package versions rather than machine-local wheel URLs. If bundled
locks are absent, the installer resolves dependencies before exporting them.
No system driver or AntBO dependency is changed.
`--stage assets` and `--stage env` can also be run separately.

Weights live outside Git on Ceph through the workspace's existing `archive/`
symlink:

- `archive/models/rfd3/2025-12-01/rfd3.ckpt` (about 2.69 GB).
- `archive/models/proteinmpnn/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/v_48_020.pt`.
- Original PDB input: `archive/data/structure_design/1N8Z.pdb`.

Downloads use a temporary filename and are promoted only when complete. Each
asset has a `.source.json` sidecar with URL, size, UTC timestamp, and local
SHA256. That digest detects subsequent modification; it is not an independently
published upstream checksum. The CLI verifies weights before model execution.
ProteinMPNN's inference scripts and license are downloaded from the pinned commit;
their individual source URLs and installed file hashes are recorded. The full
upstream repository, training data, and unrelated model weights are not needed.

Environments, pip/torch/Inductor caches, temporary files, and active runs use
local NVMe (`env/`, `cache/`, `tmp/`, `runs/`), not the repository or user home.
Runtime checkpoint paths are explicit: no implicit Foundry checkpoint selection.

## Commands

```bash
CFG=tasks/antibody/tools/structure_design/examples/1n8z_vh.json
RUNS=/Aidata/shared/ldm/antibody-poc/runs/structure_design

python3 -m tasks.antibody.tools.structure_design doctor \
  --config "$CFG" --out-dir "$RUNS/doctor-001"

python3 -m tasks.antibody.tools.structure_design pipeline \
  --config "$CFG" --out-dir "$RUNS/pipeline-001" --extract-h3

python3 -m tasks.antibody.tools.structure_design generate \
  --config "$CFG" --out-dir "$RUNS/generate-001"

python3 -m tasks.antibody.tools.structure_design sequence \
  --config "$CFG" --generation "$RUNS/generate-001/generation.json" \
  --out-dir "$RUNS/sequence-001" --extract-h3

python3 -m tasks.antibody.tools.structure_design extract-h3 \
  --config "$CFG" --designs "$RUNS/pipeline-001/designs.json" \
  --out-dir "$RUNS/extract-001"
```

Every command requires a new or empty output directory. `--root` overrides the
example's workspace root. Relative deployment/input paths are rooted there.
GPU commands try idle physical GPU 6, then 7, with at least 95% free memory;
`--gpu N` selects one physical GPU explicitly. They wait up to `gpu_wait_seconds`
(default 600). A project-local advisory lock prevents this tool's concurrent
workers from selecting the same GPU; it cannot reserve the GPU against unrelated
programs. No existing workloads are killed or modified. Batch size is always one.

Sharing a busy device requires explicit `--gpu N --allow-gpu-sharing`. The
default sharing limit is a 24 GiB **PyTorch allocator** cap, configurable with
`--gpu-memory-limit-gib`; startup requires that amount plus 2 GiB free headroom.
CUDA context/library allocations are outside this allocator cap, and compute
utilization is not capped. Sharing can slow other workloads and is never enabled
automatically. Original upstream model code is unchanged; a launcher sets the
allocator limit before invoking each model.

`--device cpu` explicitly disables all CUDA devices and runs the same toolchain
on CPU (default 8 CPU threads, configurable with `cpu_threads`). It is useful for
debugging when GPUs are occupied. CPU is never selected as an automatic fallback;
device choice is recorded in `device.json` and `run.json`. CPU runs do not
establish GPU speed or GPU-memory measurements.

`sequence` without `--generation` designs the configured input backbone directly.
It does not require first running RFD3. `extract-h3` needs no model environment or
GPU; it reads the full sequence and mapping from an existing successful manifest.

## Design configuration and mapping

The documented example selects only 1N8Z heavy chain B residues 1–120 (VH),
without light chain or HER2. Kabat HCDR3 corresponds to source author residues
B99–B109: **WGGDGFYAMDY**, 11 residues. This is a pipeline smoke example, not an
antigen-binding experiment. Sources: [PDB 1N8Z](https://www.rcsb.org/structure/1N8Z)
and [Kabat HCDR3 definition](https://pmc.ncbi.nlm.nih.gov/articles/PMC4794604/).

`h3_residues` is an ordered list of `{chain, number, insertion_code}` source
identities. It must be complete and contiguous. Insertion codes are part of
identity. Input residues are normalized to consecutive model-facing numbering,
with a complete source-to-model mapping retained. RFD3 contig slots explicitly
assign each output position to its source role; output chain lengths, all fixed
sequences, and aligned fixed-backbone coordinates are verified before reuse.
The tool never searches for an arbitrary 11-aa window or truncates/pads H3.

Fixed framework backbones are checked after rigid alignment with both RMSD ≤
0.05 Å and maximum atom error ≤ 0.1 Å. BF16 inference can quantize otherwise
fixed coordinates: the initial real CPU sample had RMSD 0.022 Å and maximum
error 0.058 Å. These numerical tolerances are explicit and recorded in every
candidate; they do not permit general framework remodeling. Upstream RFD3's
fixed-residue index map must also agree with the explicit contig slots.

`design_residues` defaults to H3. To design other positions, supply that explicit
list. All other selected protein residues keep fixed coordinates and sequence.
`fixed_residues` may additionally assert positions that must not be designed;
overlap with design positions is an error. `regions` selects source chain ranges;
omitting it selects all non-hetero protein atoms. Inputs must have standard
residues and complete N/CA/C/O backbones; unsupported or missing residues fail
explicitly. The v1 tool preserves selected lengths; variable-length remodeling,
automatic antibody numbering, ligands, and structure repair are outside this CLI.

`required_h3_length` is optional. The Absolut-oriented example sets 11; other
lengths are accepted when this constraint is omitted. `extract_h3` defaults to
false. `--extract-h3` enables it; full sequences and structures are always kept.
`num_designs`, `sequences_per_design`, `num_timesteps`, `sampling_temperature`,
and a positive `seed` configure generation. Zero seeds are rejected because
original ProteinMPNN interprets zero as random.

## Outputs and interpretation

- `config.resolved.json`, `provenance.json`, `gpu.json`, and `run.json` record the
  resolved setup, weights, packages, device, final status, and wall time.
- `input/` stores the normalized PDB, original H3, mapping, and actual RFD3 spec.
- `rfd3/` retains the original all-atom CIFs and upstream JSON metadata.
- `generated-*/` holds validated candidates and backbone-only PDBs for MPNN.
- `mpnn-*/` retains upstream parser inputs, fixed-position constraints and FASTA.
- `generation.json` and `designs.json` contain full candidates, parent IDs,
  sequences, original structures, mappings, tool parameters, seeds and timings.
- Each subprocess has a log and `.execution.json`. GPU memory is sampled with
  nvidia-smi at one-second intervals for that process; this is an observed peak,
  not an exact allocator maximum or total GPU memory budget.

After ProteinMPNN, the new sequence is associated with the generated backbone.
The old all-atom sidechains are **not** a repacked or relaxed structure for that
new sequence. No affinity or binding-energy claim is made.

Failures exit nonzero and write `run.json` with `status: failed` plus `error.log`.
Intermediate outputs remain available for diagnosis. Only a successful final
run status establishes end-to-end completion.

## Validation

```bash
python -m pytest tasks/antibody/tools/structure_design/tests/test_common.py -q
/Aidata/shared/ldm/antibody-poc/env/rfd3/bin/python -m pytest \
  tasks/antibody/tools/structure_design/tests -q
```

Tests cover explicit H3 identities, chain renaming, insertion codes, exact-length
admission, malformed mappings and structure constraints. The real H20 smoke run
separately establishes model execution; passing unit tests alone does not.

### Validated H20 smoke run

[Validation record](validation/h20-smoke.json): 1N8Z VH, seed 42, 200 diffusion
steps, one structure and one MPNN sequence, on shared physical GPU 1 with the
24 GiB allocator cap. The complete pipeline passed in 53.4 seconds; sampled
process GPU memory peaks were 2,688 MiB (RFD3) and 414 MiB (ProteinMPNN).
These are one-case measurements under shared load, not general performance or
memory guarantees. The final H3 was `SNGDWGSPGTV`; all non-design sequence
positions remained unchanged. No binding-energy evaluation was performed.

Full artifacts are under the deployment workspace's
`runs/structure_design/pipeline-gpu-002/`, including a checksum-verified
`tool-source.tar.gz` snapshot of the runtime sources used for that run. The
validation record also covers standalone sequence design with extraction off,
explicit H3 extraction, source/weight hashes and dependency checks. The installer
lock export was subsequently normalized to portable exact versions; it does not
change the model runtime or this measured generation result.
