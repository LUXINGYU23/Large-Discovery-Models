# Quick start

See [README.md](README.md) for the scientific adaptation and assets.

```bash
uv sync --project tasks/reasyn --group dev
tasks/reasyn/.venv/bin/python scripts/run_ldm_tts.py config/reasyn/mock.yaml
tasks/reasyn/.venv/bin/python scripts/run_ldm_tts.py config/reasyn/mock_tdc.yaml
tasks/reasyn/.venv/bin/python -m pytest tasks/reasyn/tests -q
```

Mock outputs are synthetic. Real runs require frozen AR/EB checkpoints,
converted chemistry indices, the requested target data, and the task chemistry
extra (`uv sync --project tasks/reasyn --group dev --extra chemistry`). Set
`REASYN_ROOT` to the official source directory, `REASYN_PYTHON` to its projection
environment, and `LLM_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY` for proposals.

```bash
tasks/reasyn/.venv/bin/python tasks/reasyn/scripts/prepare_official_data.py --upstream-root "$REASYN_ROOT"
tasks/reasyn/.venv/bin/python scripts/check_task_dependencies.py config/reasyn/reconstruction_tiny.yaml
tasks/reasyn/.venv/bin/python scripts/run_ldm_tts.py config/reasyn/reconstruction_tiny.yaml
```

Persistent research additionally requires Linux Docker/KVM and the shared
sidecar. All task roles, skills, research tools and image recipes are bundled.

```bash
docker build -t ldm-pi-harness:latest harnesses/pi
tasks/reasyn/.venv/bin/python scripts/run_ldm_tts.py config/reasyn/tdc_ldm_harness_tiny.yaml
tasks/reasyn/.venv/bin/python scripts/run_ldm_tts.py config/reasyn/tdc_ldm_harness_compiled_tiny.yaml
# Set REASYN_RUNS_ROOT and REASYN_BO_TARGETS first; see README for pool semantics.
tasks/reasyn/.venv/bin/python scripts/run_pilot_evaluation.py config/pilot_evaluation/reasyn.yaml --dry-run
```

Resume with the original arguments plus `--resume-from RUN_DIRECTORY`. Proposal
responses and per-target projection progress are reused; recovery and refill
have their own finite limits. A real end-to-end seed/campaign remains required
before scientific qualification can advance beyond draft.
