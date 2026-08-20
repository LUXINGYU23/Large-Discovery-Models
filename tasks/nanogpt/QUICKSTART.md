# nanoGPT Clean-Room Quick Start

This guide reproduces the validated CPU-only nanoGPT mock path from a new
checkout. It builds only the lightweight dependencies, validates the task
contract, runs a deterministic LDM search, checks its artifacts, and runs the
relevant tests. It does not install Torch, prepare training data, contact a
model API, or run a real nanoGPT evaluation.

Run every command from the repository root:

```bash
cd /path/to/LDM
```

## Prerequisites

- `uv` is installed.
- Python 3.10 or newer is available to `uv`.
- The repository contains `tasks/nanogpt/uv.lock`.
- No GPU or model credentials are required for the mock path.

## 1. Verify The Checkout

```bash
test -f tasks/nanogpt/uv.lock
test -f tasks/nanogpt/resources/train/mock_train.py
test -f tasks/nanogpt/resources/schemas/mock_operations.json
python scripts/validate_tasks.py --task nanogpt
```

Keep credentials outside this checkout. Do not copy `api_credential.json`
into the repository.

## 2. Build The Lightweight Locked Environment

```bash
export CUDA_VISIBLE_DEVICES=''
uv sync --locked --project tasks/nanogpt
```

The default environment contains the runner, mock, and test dependencies. It
intentionally excludes the much larger CUDA-enabled Torch training stack.
Verify that split:

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/nanogpt python - <<'PY'
import importlib.util
import sys

print("python=", sys.version.split()[0])
print("torch_installed=", importlib.util.find_spec("torch") is not None)
PY
```

For this path, `torch_installed` must be `False`.

## 3. Check The Mock Dependencies

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/nanogpt \
  python scripts/check_task_dependencies.py \
  config/nanogpt/mock_best_of_n.yaml --no-optional
```

The model URL, model name, API key, training data, tokenizer, Torch, and CUDA
are not part of this mock plan.

## 4. Inspect The Resolved Plan

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/nanogpt \
  python scripts/run_ldm_tts.py \
  config/nanogpt/mock_best_of_n.yaml --dry-run
```

Confirm that the resolved plan uses:

- `tasks/nanogpt/resources/train/mock_train.py`
- `tasks/nanogpt/resources/schemas/mock_operations.json`
- `generator=operation_mock`
- `eval-command="python {train_path}"`

It must not resolve an LLM endpoint or a real training command.

## 5. Run The CPU-Only Mock Search

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/nanogpt \
  python scripts/run_ldm_tts.py \
  config/nanogpt/mock_best_of_n.yaml \
  --set args.run-name=nanogpt_clean_room_mock
```

This evaluates a small deterministic Python scoring function. References to
training time, VRAM, or validation BPB in its output are mock metrics and are
not measurements from nanoGPT training.

## 6. Verify The Mock Result

From a new checkout, the run directory is
`tasks/nanogpt/runs/nanogpt_clean_room_mock`. If that name already exists,
the runner appends a numeric suffix; use the newest matching directory below.

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/nanogpt python - <<'PY'
import json
from pathlib import Path

run_dir = sorted(
    Path("tasks/nanogpt/runs").glob("nanogpt_clean_room_mock*"),
    key=lambda path: path.stat().st_mtime,
)[-1]
summary = json.loads((run_dir / "summary.json").read_text())
failures = [
    state for state in summary["states"]
    if state["status"] == "generation_error"
]
assert not failures, failures
assert summary["best_state_id"] is not None
assert isinstance(summary["best_score"], (int, float))
print("run_dir=", run_dir)
print("best_state_id=", summary["best_state_id"])
print("best_score=", summary["best_score"])
print("evaluation_count=", summary["evaluation_count"])
PY
```

The assertions are the contract: candidate generation completed, at least one
candidate was evaluated, and the run selected a finite best result.

## 7. Run The Relevant Tests

```bash
env -u LLM_BASE_URL -u LLM_API_KEY -u LLM_MODEL_NAME -u LLM_MODEL \
  -u TTS_LLM_URL -u TTS_LLM_API_KEY -u TTS_LLM_MODEL \
  CUDA_VISIBLE_DEVICES='' uv run --locked --project tasks/nanogpt \
  pytest -q tasks/nanogpt/tests/test_search.py \
    tests/test_openai_api_configuration.py \
    tests/test_ldm_tts_core.py \
    tests/test_shared_coverage.py \
    tests/test_task_registration.py
```

These tests remain on the lightweight dependency set and do not run real
training.

## 8. Optional Real-Run Preparation

Real nanoGPT evaluation is a separate, GPU-dependent path. Install it only on
a suitable CUDA host:

```bash
uv sync --locked --group train --project tasks/nanogpt
```

The `train` group includes the CUDA 12.8 Torch wheel, kernels, tokenizer, data,
and plotting dependencies and can download several gigabytes. The committed
real configs call evaluation with the same explicit group:

```text
uv run --locked --group train --project {repo_root}/tasks/nanogpt python {train_path}
```

Configure the model through environment variables, never committed YAML:

```bash
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_API_KEY=your-api-key
export LLM_MODEL_NAME=your-served-model
```

When a protected credential JSON file contains `url`, `key`, and `model`, load
it without copying it into the checkout:

```bash
export LDM_CREDENTIAL_FILE=/secure/path/api_credential.json
export LLM_BASE_URL="$(jq -r .url "$LDM_CREDENTIAL_FILE")"
export LLM_API_KEY="$(jq -r .key "$LDM_CREDENTIAL_FILE")"
export LLM_MODEL_NAME="$(jq -r .model "$LDM_CREDENTIAL_FILE")"
```

Use an environment-only Python probe before a real search so the key is not
placed in shell arguments:

```bash
uv run --locked --project tasks/nanogpt python - <<'PY'
import os
import httpx

base_url = os.environ["LLM_BASE_URL"].rstrip("/")
headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
with httpx.Client(timeout=60, headers=headers) as client:
    models = client.get(f"{base_url}/models")
    models.raise_for_status()
    listed = any(
        item.get("id") == os.environ["LLM_MODEL_NAME"]
        for item in models.json().get("data", [])
    )
    print("configured_model_listed=", listed)
    reply = client.post(
        f"{base_url}/chat/completions",
        json={
            "model": os.environ["LLM_MODEL_NAME"],
            "messages": [{"role": "user", "content": "Reply with exactly OK"}],
            "max_tokens": 8,
        },
    )
    reply.raise_for_status()
    print("chat_reply=", reply.json()["choices"][0]["message"]["content"])
PY
```

Prepare the dataset and tokenizer as described in the task README, then run
the full real-config dependency check. This clean-room trial intentionally
stopped before these steps because actual evaluation requires a GPU.

## 9. Clear Runtime State

```bash
unset LLM_BASE_URL LLM_API_KEY LLM_MODEL_NAME LLM_MODEL
unset TTS_LLM_URL TTS_LLM_API_KEY TTS_LLM_MODEL
unset LDM_CREDENTIAL_FILE CUDA_VISIBLE_DEVICES
test ! -e api_credential.json
```
