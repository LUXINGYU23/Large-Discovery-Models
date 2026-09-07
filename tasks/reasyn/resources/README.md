# Versioned evidence

* `source_provenance.json`: SHA-256 digests of the supplied ReaSyn code, paper
  and released local data. No Git commit is invented for archive directories.
* `qualification_evidence.json`: formal gates remain pending because this
  workspace has no Git tracking metadata and no real checkpoint seed run.
* `verification_record.json`: compact actual local test/smoke observations;
  synthetic values are explicitly distinguished from scientific scores.

Large pretrained checkpoints, stock/reaction indices, task runs and collection
records stay outside versioned resources. Their locations are configured via
paths/environment and all run artifacts use relative references.
