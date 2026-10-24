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
