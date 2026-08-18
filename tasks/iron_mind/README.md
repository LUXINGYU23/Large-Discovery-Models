# Iron Mind

Iron Mind is a source-pinned closed-loop optimization task over categorical
reaction-condition tables. Both mock and endpoint proposals use the same strict
four-candidate JSON parser, candidate admission, one-hot encoder, shared RBF
GP-UCB selector, frozen-table evaluator, and `LDMEngine` campaign lifecycle.

## Status

The task remains `draft` until the later seed and real tiny qualification gates
write their tracked evidence. The current workflow is suitable for deterministic
mock execution and endpoint-free contract testing; it does not claim an
all-dataset Iron Mind reproduction.

## Run locations

Run commands on the remote development server from the repository root. The
tracked mock config has no external data or endpoint requirement:

```bash
uv run --locked --project tasks/iron_mind python scripts/run_ldm_tts.py config/iron_mind/mock.yaml
```

`config/iron_mind/real_tiny.yaml` locks one Buchwald-Hartwig endpoint proposal,
four admitted candidates, one GP-UCB selection, and one frozen-table evaluation.
It reads pinned data from `/mnt/data1/ldm-for-sci/data/iron_mind/` and declares
the public endpoint/model only. The credential is read exclusively from
`LDM_LLM_API_KEY`; never add it to YAML, JSON, source code, or run artifacts.

Use `--dry-run` before a real invocation to inspect the task spec and active
experiment contract without contacting the endpoint.

## Artifacts

Completed campaigns write the shared `campaign.json`, `budget.json`,
`checkpoint.json`, `status.json`, `result.json`, and `trajectory.csv` artifacts,
plus Iron Mind `search_manifest.json`, `selection_record.json`, and
`evaluation_manifest.json`. Artifact references in task manifests are relative
to the campaign directory.
