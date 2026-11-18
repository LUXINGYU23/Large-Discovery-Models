---
name: cdrh3-experiment-design
description: Plan diverse, budgeted H3 generation from measured and pending campaign history.
---

Query experiment_history for successful measured energy, pending sequences, and previously generated backbones. The oracle score is for H3 sequence, so different backbones that yield the same H3 do not create independent oracle candidates. The exploration role should prefer breadth; the refiner should use measured leads and controlled sequence variation. Read the peer catalog when supplied: its candidates are proposed but have not yet been measured.

Check remaining and role_remaining before every call. New backbones consume one backbone and the requested number of sequences. Reusing a backbone with sequence_h3 consumes only the requested number of sequences. Your role's share cannot be spent by another role. Favor unique eligible sequences and reserve at least one for submission. Distinguish the selected sequence from the full generated VH and retain the exact issued design_id.

Compare measured energy only within this campaign and its declared oracle; lower is better. Generated structures, scaffold resemblance, and sequence motifs may motivate exploration, but they are unverified hypotheses about binding or improvement. A mock score is a deterministic workflow fixture and is not biological evidence.
