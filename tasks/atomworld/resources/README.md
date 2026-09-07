# Versioned resources

- `source_manifest.json`: SHA-256 identity of the local source archive and paper; no invented Git revision.
- `mock_fixture.json`: synthetic crystal, private target and deterministic transport responses used only in mock tests.
- `qualification_evidence.json`: formal gate state; remains scaffolded because this archive has no tracked Git evidence and no qualified real endpoint campaign.
- `local_verification.json`: actual local check results, counters and artifact hashes, without treating mock scores as model performance.

Released datasets and run outputs belong under ignored `runs/`; do not copy targets into model-visible state.
