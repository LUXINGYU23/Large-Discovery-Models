# NucleoBench Task Quickstart

The current release stage provides task registration, source provenance, a
strict 17-case catalog, mutation-patch validation, and external preparation for
the first case. It does not yet execute an official oracle campaign.

Run these commands from the repository root.

## Install the Task Environment

```bash
uv sync --locked --project tasks/nucleobench
```

## Validate Registration

```bash
uv run --locked --project tasks/nucleobench \
  python scripts/validate_tasks.py --task nucleobench

uv run --locked --project tasks/nucleobench \
  python -m pytest \
  tasks/nucleobench/tests/test_candidate.py \
  tasks/nucleobench/tests/test_official_adapter.py \
  tasks/nucleobench/tests/test_procedure.py \
  tasks/nucleobench/tests/test_task_isolation.py
```

## Inspect a Case

```bash
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.ldm_task.procedure \
  --case-id malinois_k562 --dry-run
```

This prints the source-pinned case metadata and declarative LDM task contract.
Commands without `--dry-run` are rejected until an executable campaign profile
has passed its qualification gates.

## Run the Mock Campaign

```bash
uv run --locked --project tasks/nucleobench \
  python scripts/run_ldm_tts.py config/nucleobench/mock.yaml
```

This deterministic two-round run exercises the shared engine and writes the
standard campaign, budget, status, event, checkpoint, summary, result,
trajectory, and data-collection artifacts under `tasks/nucleobench/runs/`.

## Prepare External Inputs

Supply the official 100-start set as a JSON array, a local official model
artifact, and the verified protocol values:

```bash
uv run --locked --project tasks/nucleobench \
  python -m tasks.nucleobench.scripts.prepare_official_data \
  --case malinois_k562 \
  --source-dir /external/nucleobench/source \
  --output-dir /external/nucleobench/data/malinois_k562 \
  --starts-file /external/nucleobench/inputs/malinois_k562_starts.json \
  --model-artifact /external/nucleobench/models/malinois_artifacts.tar.gz \
  --model-sha256 <verified-sha256> \
  --start-set-sha256 <verified-canonical-start-set-sha256> \
  --bending-factor <verified-official-value>
```

Source checkouts, model weights, starts, prepared manifests, caches, and run
outputs must remain outside the Git repository.
