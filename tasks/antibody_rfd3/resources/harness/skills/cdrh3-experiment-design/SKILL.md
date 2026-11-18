---
name: cdrh3-experiment-design
description: Plan diverse, budgeted H3 generation from measured and pending campaign history.
---

Query experiment_history for successful measured energy, pending sequences, and previously generated backbones. The oracle score is for H3 sequence, so different backbones that yield the same H3 do not create independent oracle candidates. The exploration role should prefer breadth; the refiner should use measured leads and controlled sequence variation. Read the peer catalog when supplied: its candidates are proposed but have not yet been measured.

Check remaining and role_remaining before every call. New backbones consume one backbone and the requested number of sequences. Reusing a backbone with sequence_h3 consumes only the requested number of sequences. Your role's share cannot be spent by another role. Favor unique eligible sequences and reserve at least one for submission. Distinguish the selected sequence from the full generated VH and retain the exact issued design_id.

For an exact candidate_count, count only distinct eligible new H3 sequences, not the number of raw samples. The candidate_catalog is rebuilt from this role's current generation and excludes measured and pending sequences, the live pool listed at the start of this generation, and already submitted peer sequences. Read it after each call and use candidate_catalog to refresh it before submission without spending generation budget. Query experiment_history as results arrive and generate more when necessary. If duplicates or constraint violations reduce the yield, consider another backbone or adjust temperature within the permitted range; do not repeat calls without regard to the finite sequence and backbone budgets. If the budget is exhausted below the requirement, submit the available IDs; if validation requests a retry, refresh the catalog and resubmit available IDs until the host closes the turn as a rejected shortfall.

Compare measured energy only within this campaign and its declared oracle; lower is better. Generated structures, scaffold resemblance, and sequence motifs may motivate exploration, but they are unverified hypotheses about binding or improvement. A mock score is a deterministic workflow fixture and is not biological evidence.
