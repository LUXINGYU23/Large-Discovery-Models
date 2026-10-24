# Default Pi research validation

Validated on 2026-10-24 in the isolated H20 checkout
`/Aidata/shared/ldm/default-research-20261024`. Existing experiment sources,
results, environments and credentials were not modified. No paid provider calls
or formal benchmark campaigns were launched.

Command: `python -m pytest -q tests/test_research_tree.py tests/test_research_task_adapters.py tests/test_research_defaults.py tasks/nucleobench/tests tasks/iron_mind/tests tasks/atomworld/tests tasks/reasyn/tests tasks/alphabench/tests tasks/synthonbench/tests --continue-on-collection-errors`.

Result: **855 passed, 118 skipped, 6 warnings**, 77.66 seconds. Warnings are ASE /
NumPy deprecations. Optional external integrations (including Pi images, pinned
AlphaBench source and Qlib/Assay execution) were not qualified in this run.
Task-adapter tests use fixture providers and evaluators; they validate orchestration
and contracts, not scientific quality or real-model compliance with the prompts.

The interpreter was the existing NucleoBench Python 3.12 runtime; test-only packages
were installed into the isolated checkout's `testdeps`, not the experiment env.
SynthonBench unit tests used PyPI `synthonbench==0.2.0`: fetching the configured Git
revision failed, so this is not validation of that source-pinned integration.

Coverage includes all six default task entry points, dynamic main-history parent
selection, inherited cursors, latest authorized feedback, role instructions in
actual branch requests, frozen replay plans, recovery and task budget accounting.
Existing baseline tests now explicitly select their original method. Named legacy
baseline profiles/configurations explicitly pin their method so changing the CLI
default does not silently redefine those experiments.

Implementation: [shared architecture](research-search-architecture.md).

## Pre-merge audit (2026-10-24)

The production-only integration branch is `codex/pi-default-research-main`.
Experiment deployment scripts, result archives and analysis binaries are excluded.
The audit additionally corrected:

- Iron Mind and SynthonBench's default proposal mode: Harness resolves to `none`,
  with full validation tests rather than argument parsing alone.
- Pilot contracts: multi-method matrices remain selectable; direct-LLM profiles
  explicitly select `llm` instead of incorrectly locking them to `ldm`.
- Remaining README, QUICKSTART, contract and prompt references to 2+2 or an
  independent persistent branch lifecycle; task submission schemas remain intact.
- Shared integration tests declare optional SynthonBench and AlphaBench reporting
  dependencies. These cases ran in the task-equipped environment and skip in the
  minimal shared-runner environment.

Evidence in `/Aidata/shared/ldm/pi-default-audit-20261024`:

| Check | Result | Log |
| --- | --- | --- |
| Six task suites + shared research tests (Python 3.12) | 857 passed, 118 skipped | `audit-tests.log` |
| Shared runner + nanoGPT, isolated root dependencies (Python 3.10) | 619 passed, 4 skipped | `final-shared-tests.log` |
| Final default/configuration/matrix regression checks | 63 passed | `final-config-tests.log` |
| Pi native build and tests (Node 24.7.0) | 73 passed, no skips | `native-tests.log` |
| Task registration and mock-suite dry run | Passed | `registry.log`, `mock-dry-run.log` |

The shared environment used a mirror-resolved copy of the root lockfile outside
Git; repository dependency files were not changed. The archived checkout supplied
its actual Git commit via `LDM_PILOT_EVALUATION_COMMIT`; PATH selected the isolated
Python for subprocess tests. The latest functional source tested is `d0c8fdfd`.
No new scientific experiments or paid model requests were launched.
