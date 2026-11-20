# Reproducible Absolut configuration

Upstream: https://github.com/csi-greifflab/Absolut at `beb1821568f3ba5b7861255ab19172bbc65a7480`.

`bash tools/absolut/install.sh /absolute/new/Absolut` clones the fixed baseline, applies the headless build and custom PD-1 antigen patches, and compiles evaluation and discretization binaries. Requires git, make, GCC/G++, patch support through git. It creates `activate.sh` in the installation. The installer itself has not been executed end-to-end; the equivalent build and scoring operations were validated on H20. `activate_absolut.sh` is the existing H20-specific activation script.

The patch defines `5GGR_Y_FuC_5.25` with two separate subchain origins. `5GGR_Yinlattice.txt` records the generated lattice and blocked positions. The original PDB is https://files.rcsb.org/download/5GGR.pdb. Parameters: author chain Y, FuC, 5.25, allow jumps across missing residues. See `../../docs/absolut-oracle.md` (from repository root: `docs/absolut-oracle.md`) for limitations and real validation records.

HER2 resources: run `src/bin/Absolut info_filenames 1N8Z_C` and download/unzip the reported official structure resource. H20 already has this resource in the installation root.

For custom PD-1, run `singleBinding 5GGR_Y_FuC_5.25 CAGPSTTVPYYFDYW` in a persistent resource directory. This generates structures and interaction caches and can take over 13 hours on one CPU. H20 already has completed resources in `/Aidata/shared/ldm/antibody-poc/runs/absolut-pd1-5GGR-Y`; reuse them rather than repeating enumeration.

Binaries, downloaded PDBs, large structures, caches and evaluation outputs are not committed. H20 input bundles and the streaming evaluation path are documented in [Absolut scoring](../../docs/absolut-oracle.md).
