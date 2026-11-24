---
name: cdrh3-protein-tools
description: Optional host-enabled protein generation and interface-confidence tools.
---

# Optional protein tools
When the host enables optional protein tools, `sequence_h3_soluble` resequences an issued RFD3 backbone with soluble ProteinMPNN weights using the same role/round sequence allowance. `design_h3_boltzgen`, when available, generates one backbone followed by the requested number of native inverse-folded H3 sequences; it consumes one role/round backbone slot and all requested sequence slots. It accepts no ProteinMPNN temperature. Its backbones cannot enter either ProteinMPNN resequencing route. All ordinary admission constraints still apply.

`score_interface` accepts only a host-prepared run-local artifact ID for a genuine matching prediction/PAE pair. It returns auxiliary interface confidence, independently of the measured Absolut history. Such confidence is not an affinity measurement and must not be submitted as an Oracle result. Optional tools can be absent; use only the registered tool set and retain the existing role submission contract.
