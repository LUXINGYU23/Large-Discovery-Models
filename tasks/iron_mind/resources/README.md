# Versioned Task Resources

This directory contains small, redistributable inputs and provenance required
to validate the Iron Mind task:

- `upstream_contract.json`: pinned upstream revisions, suite definitions,
  data hashes, and schema metadata.
- `reaction_schemas.json` and `mock_oracle.csv`: self-contained mock
  fixtures.
- `harness/`: digest-pinned multi-Agent LDM and single-Agent direct-research
  profiles plus the structured source-pinned reaction-space tool.
- `qualification_evidence.json` and `verification_record.json`: release
  validation summaries.

Official upstream checkouts, prepared data, provider credentials, and campaign
outputs are intentionally kept outside the repository. The data preparation
script validates the frozen source snapshot against `upstream_contract.json`
before a real campaign starts.
