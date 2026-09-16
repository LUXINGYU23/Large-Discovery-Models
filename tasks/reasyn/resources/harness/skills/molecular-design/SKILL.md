---
name: molecular-design
description: Research valid ReaSyn projection targets using public chemistry and measured campaign evidence.
---

Inspect the task context and current history delta, then query full measurements as needed. Use Python and RDKit in the sandbox to canonicalize SMILES, compare Morgan fingerprints, and inspect scaffold diversity. Form competing chemical hypotheses and record counterexamples. Distinguish reconstruction similarity to the original target from TDC utility. A valid query need not have a valid synthesis; the frozen projector verifies routes later. For TDC, query whether products were measured and follow host rejection feedback when different queries converge on an old product. Legal repeated occurrences, including within one session, contribute q0 before product deduplication. Reconstruction groups proposal frequency by canonical query but assigns each occurrence an independent trial seed; a previously seen query is not a previously evaluated trial. Submit exactly the requested target count with concise rationales.
