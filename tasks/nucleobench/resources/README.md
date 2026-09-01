# NucleoBench Resources

This directory contains small, versioned protocol inputs only:

- `upstream_contract.json` pins the official source and verified public API.
- `cases/catalog.json` declares the 17 source-pinned benchmark cases.
- `qualification_evidence.json` records the highest completed release gate.

Official source checkouts, model weights, start-sequence sets, generated
manifests, caches, traces, and campaign outputs must be stored outside the Git
repository. Runtime preparation must record their digests before a case can
move beyond `planned`.
