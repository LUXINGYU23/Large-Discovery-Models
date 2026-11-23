---
name: cdrh3-design
description: Use measured HER2 sequence history to design local variants, register exact H3 proposals, or sample RFD3 and ProteinMPNN structures within durable budgets.
---

Begin each research turn by reading the measured_history supplied in the task message. Lower measured energy is preferred. Use measured observations to guide candidate design and record your rationale. Pending candidates have no measured outcome yet.

Choose the tool that expresses the design you intend.

| Tool | Design control | Budget |
|---|---|---|
| mutate_h3 | Exact seed_sequence, count, mutable_positions, min_mutations, max_mutations, allowed_amino_acids, rationale | Requested sequence count |
| register_h3 | Exact sequences array, rationale, optional seed_sequence for parent attribution | Number of submitted sequences |
| design_h3 | New RFD3 backbone, count and ProteinMPNN temperature | One backbone and requested sequence count |
| sequence_h3 | Existing backbone_id, count and ProteinMPNN temperature | Requested sequence count |
| candidate_catalog | Refresh eligible candidates for your current generation | No generation budget |

For mutate_h3, mutable_positions contains 1-based H3 positions from 1 through 11. Omitted positions remain equal to the seed. If mutable_positions is omitted, all positions are available. min_mutations and max_mutations bound Hamming distance from the seed, with defaults 1 and 2. allowed_amino_acids maps position strings such as "3" or "8" to a string of permitted canonical amino acids. Use these controls to test an explicit position or residue hypothesis. Choose a search space large enough for the requested unique count, and inspect the returned catalog for eligibility. You can distribute batches across several measured seeds, vary the radius, or constrain selected positions according to the evidence.

Use register_h3 when you already know the exact complete sequences you want to test. Each sequence must contain 11 canonical amino acids. The host applies campaign admission rules, deduplication, budgets and durable issuance. Supply the relevant measured parent when one exists and explain your design in rationale. Submit the returned eligible design IDs through submit_designs.

The sequence tools produce sequence proposals with null backbone_id and record parent sequence, constraints and rationale. Their experimental evidence comes from subsequent Absolut measurement. The structure tools retain full VH structures and residue mapping. sequence_h3 conditions ProteinMPNN on the selected backbone, so use mutate_h3 when the intended condition is an exact measured sequence neighborhood.

Read remaining before calls. For an exact candidate_count, count eligible unique new sequences in the current-generation catalog. Refresh candidate_catalog before submission. Measured, pending, peer and excluded live-pool sequences can become ineligible. If yield is poor, revise the seed, permitted positions, mutation radius, residue options or generation tool within the available budget. Submit exactly the requested eligible IDs. If the budget is exhausted, submit the available IDs so the host records the shortfall.

Absolut scores the extracted H3 against the declared lattice antigen. Treat generated geometry and unmeasured changes as hypotheses. Each H3 has one oracle identity across all design tools and backbones.
