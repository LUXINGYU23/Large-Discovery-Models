---
name: cdrh3-design
description: Generate mapped 11-residue CDRH3 proposals with RFD3 and ProteinMPNN, reuse a measured design's backbone, and submit valid design IDs.
---

Read the campaign's scoring mode, measured sequences, energies, backbone IDs and remaining generation budgets at the start of each turn. Lower energy is preferred. A mock score tests feedback plumbing only.

Use design_h3 to explore a new loop geometry with a fixed 1N8Z VH framework. RFD3 builds the backbone and ProteinMPNN samples H3 on it. Use sequence_h3 with an existing backbone_id when a measured sequence suggests a promising scaffold, or to explore sequence diversity without spending another backbone. Set count and temperature to fit the remaining sequence budget. Repeated sequences on different backbones have one oracle identity.

For example, in the H20 validation the 120-residue VH originally had WGGDGFYAMDY at author chain B99–B109. The designed candidate had SNGDWGSPGTV at sequence chain A positions 99–109. Its residue map relates each output position to the source chain, number and insertion code. The worker retains the full VH, mapping and original CIF. Use the exact returned design_id for submission and backbone_id for resampling. The example sequence is not an evaluated lead.

Absolut evaluates the extracted 11 residues against the configured lattice antigen. The generated coordinates and ProteinMPNN scores are not inputs to this energy. Use only returned candidates with empty violations and distinct unmeasured H3 sequences. Submit the exact number requested for the method.

In a batch, each generating role must supply the exact requested number of eligible unique new H3 sequences. Raw generation count is not eligible yield. The generation tool returns a candidate_catalog path and role_remaining: read the catalog, then generate additional sequences if necessary and budget permits. Try a different backbone or permitted temperature if yield is poor. If the budget is exhausted before the required count, submit the available IDs so the host records the shortfall; never substitute an older or peer candidate.
