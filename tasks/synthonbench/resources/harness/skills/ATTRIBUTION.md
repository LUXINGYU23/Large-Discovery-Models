# Research Skills

The rdkit, experimental-design, scientific-critical-thinking, and statsmodels Skills
are task-local chemistry adaptations of
[K-Dense Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/1e5eeffbdad3749125afe7ab48a39694e27f181c/skills),
pinned to commit `1e5eeffbdad3749125afe7ab48a39694e27f181c`.
Each retains its upstream MIT notice. Sequence, clinical, manuscript, and
unrelated service workflows are omitted. The runtime installs the analysis
packages used by these instructions and checks them with the guest smoke.

The rdkit adaptation focuses on public source-synthon analysis and preserves
the repository's MIT notice for Skill text; the RDKit software has its separate
BSD license. It uses APIs available in the pinned task guest, not the upstream
Skill's newer environment assumptions. No datamol wrapper, additional service,
or unrelated bundled workflow is required.

Candidate sessions load these Skills on demand from digest-verified read-only
snapshots. Only names and descriptions are initially advertised. The separate
policy session loads only this task's compile-ldm-policy Skill. Research Skills
do not change candidate identity, the framework GP, or the official evaluator.
