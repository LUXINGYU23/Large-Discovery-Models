---
name: cdrh3-experiment-design
description: Turn measured and pending HER2 history into sequence-conditioned research batches with traceable hypotheses.
---

At the start of each turn, inspect successful leads, recently completed results and pending candidates through experiment_history. Read the peer catalog when supplied. Record which measured sequence and result motivates each local batch. The objective is to learn from feedback while proposing the required number of distinct candidates.

The explorer can compare multiple measured seeds, vary several positions, register a new sequence hypothesis, or explore a new scaffold. The refiner can use mutate_h3 to test neighborhoods around promising measured sequences. Small mutation radii and targeted positions help attribute changes to a specific hypothesis. Widen the radius or change seeds when local evidence suggests stagnation. These are research choices that should follow the observed history.

Use register_h3 for exact designs whose residues you choose yourself. Use mutable_positions and allowed_amino_acids with mutate_h3 when you want a controlled local batch. Explain the motivation in rationale and preserve the measured parent through seed_sequence. A structural resampling call uses a backbone as its condition. Select that operation when geometry is the hypothesis being explored.

Check role_remaining and remaining before every call. All proposed sequences consume generation budget, including candidates later rejected by constraints or deduplication. The reviewer has zero generation budget. Refresh candidate_catalog after generation and before final submission to obtain currently eligible IDs. It contains only your role's current generation and excludes measured, pending, live-pool and submitted peer sequences.

Compare measured energy within this campaign's declared oracle. Use successful scores to revise the next batch and describe untested explanations as hypotheses. If no evidence supports a positional preference, say so and use the batch to test one. Keep the exact requested count and reserve time for submission validation.
