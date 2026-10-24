# Testing And Coverage

Validate task registration and conventional adapter layouts before running the
separate test lanes:

```bash
uv sync --locked --group dev
uv run --locked python scripts/validate_tasks.py
```

The repository has separate test lanes because the task packages use different
Python and scientific dependency stacks.

Coverage commands use `pytest-cov`. It is declared by the root and each
task-specific development group.

## Shared and NanoGPT Core

From the repository root:

```bash
uv run --locked python -m pytest
uv run --locked python -m pytest tests tasks/nanogpt/tests \
  --cov --cov-config=.coveragerc --cov-report=term-missing
```

The coverage command measures branch coverage for the shared `ldm_tts`
package and the dependency-free NanoGPT search engine, search methods, mock
trainer, and single-search wrapper. `.coveragerc` enforces a minimum total of
80%. GPU training code, remote model clients, and domain adapters are verified
in their package environments instead of being counted as uncovered core code.

## Pi tasks and shared recovery

Use the task environments for their scientific dependencies. In an environment
containing the six tasks' test dependencies, run from the repository root:

```bash
python -m pytest tests tasks/nucleobench/tests tasks/iron_mind/tests \
  tasks/synthonbench/tests tasks/reasyn/tests tasks/atomworld/tests \
  tasks/alphabench/tests
npm --prefix harnesses/pi ci
npm --prefix harnesses/pi test
npm --prefix harnesses/pi run typecheck -- --noUnusedLocals --noUnusedParameters
```

`tests/test_research_task_adapters.py` exercises task factories and default main-history 1+3
trees with synthetic Harness responses. Shared engine tests cover single and
batch result persistence, unknown-result pauses, failure replenishment and
completed replay. Task recovery tests verify receipt and checkpoint fault
windows, scientific budgets, private feedback and attribution. These tests do
not call a paid provider or qualify official benchmark scores.

Pi tests exercise the actual provider proxy against a local fixture server,
including research parameter overrides and native compaction's low reasoning
and output budget. Submission repair, partial commits, identity tampering,
timeouts and length-stop are tested in their owning modules.

For AlphaBench integration, set `ALPHABENCH_SOURCE_ROOT` to the pinned source
checkout to enable native algorithm tests. Set `ALPHABENCH_T3_SIDECAR_IMAGE` to
an immutable built image only after building the matching task guest in
`$ALPHABENCH_RUNTIME_ROOT/cache`; guest tests also require KVM. Then run
`test_harness_mcp.py`, `test_harness_guest.py` and `test_policy.py` under
`tasks/alphabench/tests`. A sidecar image alone supports MCP and policy artifact
execution but does not satisfy tests that initialize Pi sessions. Missing guest,
upstream or real-model assets must be reported separately from mock test passes.

Run builds and tests on the designated Linux server under `/mnt/data1/`.
Use an isolated copy, a short `TMPDIR` for Unix sockets, distinct image tags and
source/build hashes. An archive without `.git` must supply its actual archive
identity via `LDM_PILOT_EVALUATION_COMMIT` for pilot-run provenance. Never change
an old experiment's identity or checkpoint to test a new implementation.

## Antibody

Use the locked antibody environment documented in
`tasks/antibody/pyproject.toml`, then run:

```bash
uv sync --locked --project tasks/antibody --group dev
cd tasks/antibody
uv run --locked --group dev python -m pytest
uv run --locked --group dev python -m pytest \
  --cov --cov-config=.coveragerc --cov-report=term-missing
```

The acquisition search tests import PyTorch, which is a declared runtime
dependency of the antibody package. The coverage configuration measures all
of `core/ldm` with branch coverage and enforces a minimum total of 80%.

## Small Molecule

Install the package's scientific dependencies and run its suite from the
repository root. Prefer a standalone CPython rather than an Anaconda base
interpreter; some Anaconda macOS installations crash in the native `readline`
module before pytest collection begins.

```bash
uv sync --locked --project tasks/small_molecule --group dev
uv run --locked --project tasks/small_molecule \
  python -m pytest tasks/small_molecule/tests
cd tasks/small_molecule
uv run --locked --group dev python -m pytest \
  --cov --cov-config=.coveragerc --cov-report=term-missing
```

The small-molecule coverage configuration measures the unit-testable algorithm
core and service adapters with branch coverage and enforces a minimum total of
80%. It excludes the activity-model training script, docking implementation,
ReaSyn analogue client, dependency preflight, and top-level workflow. Those
operational boundaries are exercised by the mock workflow and opt-in real
integration lanes instead of being counted as uncovered unit-testable code.

The real ReaSyn, Vina, model-service, and GPU integration tests remain skipped
unless their documented environment variables and external assets are
available. Unit tests mock those process and service boundaries.
