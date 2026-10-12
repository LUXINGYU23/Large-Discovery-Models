# ResearchGym Quick Start

Run from the repository root. Steps 1-3 need no GPU, dataset, endpoint, or
container; later steps are the staged first real run.

## 1. Registration and tests

```bash
python scripts/validate_tasks.py --task researchgym
uv run --locked --project tasks/researchgym python -m pytest tasks/researchgym/tests
```

## 2. Mock campaigns through the shared runner

```bash
python scripts/check_task_dependencies.py config/researchgym/mock.yaml --no-optional
python scripts/run_ldm_tts.py config/researchgym/mock.yaml --dry-run
python scripts/run_ldm_tts.py config/researchgym/mock.yaml
python scripts/run_ldm_tts.py config/researchgym/mock_ldm_harness.yaml
```

`mock.yaml` runs direct-request `ldm` with a synthetic client that goes through
the real parser, one repair, empirical q0, and the LDM selector. The Harness
mock drives the real shared `HarnessClient` against a protocol-faithful fake
sidecar (`tests/fixtures/harness_sidecar.py`). Scores are analytic and labeled
`synthetic_fixture`. Outputs go to `tasks/researchgym/runs/` (Git-ignored).

## 3. Inspect a run

```bash
cat tasks/researchgym/runs/researchgym/result.json
head tasks/researchgym/runs/researchgym/evaluations.csv
```

## 4. Real prerequisites

1. `git clone https://github.com/Anikethh/ResearchGym && git -C ResearchGym checkout 0adc08e93754b606d8a76055ed2f1c9574504312`
2. Create the case environment and prepare case data (see README "Setup").
3. Export `RESEARCHGYM_ROOT`, `RESEARCHGYM_CASE_DATA_ROOT`,
   `RESEARCHGYM_CASE_PYTHON`, `RESEARCHGYM_DEVICES`, `RESEARCHGYM_RUNS_ROOT`,
   `LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY`.
4. For Harness methods, build and smoke the `researchgym` guest image with the
   current Pi sidecar (protocol 0.2). If you set `--api-budget-usd`, also set a
   conservative `--harness-provider-reserve-usd` upper bound for one provider
   call.

## 5. Staged first real run

```bash
python scripts/check_task_dependencies.py config/researchgym/continual_learning_tiny_ldm.yaml
python scripts/run_ldm_tts.py config/researchgym/continual_learning_tiny_ldm.yaml --dry-run
python scripts/run_ldm_tts.py config/researchgym/continual_learning_tiny_ldm.yaml
```

The run preflights the endpoint, evaluates the released seed in round 0, then
one LDM-selected candidate. Check `status.json`, `budget.json`, and
`evaluations/<candidate>/jobs.json` before larger profiles. Remove exported
credentials when finished.
