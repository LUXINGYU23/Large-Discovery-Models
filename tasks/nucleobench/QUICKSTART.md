# NucleoBench Task Quickstart

The current release stage provides task registration, source provenance, and a
strict 17-case catalog. It does not yet execute an oracle campaign.

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

Official source checkouts, model weights, start sequences, caches, and run
outputs must remain outside the Git repository. Their preparation and runtime
commands will be documented when the first case reaches the corresponding
qualification stage.
