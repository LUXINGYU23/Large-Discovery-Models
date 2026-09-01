# NucleoBench Resources

This directory contains small, versioned protocol inputs only:

- `upstream_contract.json` pins the official source and verified public API.
- `cases/catalog.json` declares the 17 source-pinned benchmark cases.
- `qualification_evidence.json` records the highest completed release gate.
- `verification_record.json` records compact, reproducible gate outcomes.

Official source checkouts, model weights, start-sequence sets, generated
manifests, caches, traces, and campaign outputs must be stored outside the Git
repository. Runtime preparation must record their digests before a case can
move beyond `planned`. `malinois_k562` is currently the only qualified case.

For Malinois, the contract records the official model artifact and the
`start_sequences_df.csv` table from Zenodo record `17079936`, including raw
file hashes, the 100-sequence extraction rule, and the canonical start-set
hash. Mirrors are acceptable only when their bytes match those digests.

`python -m tasks.nucleobench.scripts.prepare_official_data` writes the first
case's external `prepared_manifest.json`; generated manifest instances are not
tracked here.
