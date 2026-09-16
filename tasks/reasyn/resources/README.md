# Versioned evidence and research resources

- `source_provenance.json` records SHA-256 digests of the supplied official source
  archive, paper and local release data. Archive identities are not Git commits.
- `qualification_evidence.json` retains the draft/scaffolded status until official
  assets and a real provider/projector/oracle seed run are verified.
- `verification_record.json` is a dated historical test snapshot. It does not
  describe current environment availability or certify live Harness integration.
- `harness/` packages roles, research Skills, structured measured-history tools,
  compiled-policy resources and a pinned guest dependency recipe.

Large checkpoints, stock/reaction indices, runs and collection records stay
outside versioned resources. Configure their locations explicitly and preserve
run-local artifact references and immutable source/asset digests.
