# NucleoBench Resources

This directory contains small, versioned protocol inputs only:

- `upstream_contract.json` pins the official source and verified public API.
- `cases/catalog.json` declares the 17 source-pinned benchmark cases.
- `qualification_evidence.json` records the highest completed release gate.
- `verification_record.json` records Malinois qualification outcomes.
- `enformer_qualification.json` records Enformer oracle and Harness-Compiled qualification outcomes.

Official source checkouts, model weights, start-sequence sets, generated
manifests, caches, traces, and campaign outputs must be stored outside the Git
repository. Runtime preparation must record their digests before a case can
move beyond `planned`. `malinois_k562` and `enformer_muscle_not_liver` are
qualified; 14 additional published cases are prepared.

The contract records the shared Malinois artifact, all 12 BPNet artifacts, the
Enformer checkpoint, the RiNALMo weights, the published CSV and Enformer
Parquet start tables, exact case row blocks, canonical start-set hashes, and
Enformer's per-start editable-mask hash. Mirrors are acceptable only when their
bytes match those digests.

`python -m tasks.nucleobench.scripts.prepare_official_data` writes an external
`prepared_manifest.json` for any declared case. The RiNALMo case remains
planned because the official paired-start input is not present in the published
benchmark artifacts. Generated manifest instances are not tracked here.
