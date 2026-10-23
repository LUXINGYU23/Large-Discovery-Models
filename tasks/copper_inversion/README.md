# Copper inversion algorithm discovery

Implementation of [the specification](../../docs/copper-inversion-spec.md).
One persistent Pi researcher proposes complete `solver.py` programs. The shared
LDMEngine owns admission, rounds, budgets, checkpoints and result publication.
This first stage is independent Harness search, without GP or a research tree.
There is no inherited MCMC/controller/comparison runner.

A solver maps public geological data to a floating level-set grid. The host runs
the sealed code in a network-disabled, resource-limited Docker worker. Fixed
small interface inputs are read-only. The controller validates outputs outside
the candidate interpreter. Formal scoring uses a second container with only
frozen geometry and trusted scientific code: it never mounts submitted source.
The geometric field is reinitialized before the fixed logistic/contact/sketch
loss. All example data are synthetic; fitting loss is not geological ground truth.

Use the remote development server under `/mnt/data1/`. No local execution is
required. From repository root:

```sh
uv sync --locked --project tasks/copper_inversion
uv run --locked --project tasks/copper_inversion python scripts/validate_tasks.py --task copper_inversion
uv run --locked --project tasks/copper_inversion python scripts/run_ldm_tts.py config/copper_inversion/mock.yaml
```

Mock uses the real source submission parser and shared campaign, with a fake
Harness transport and synthetic evaluation. It executes no generated Python and
makes no scientific claims. Native Harness traces are retained separately;
`ldm-2.0` collection is not yet applicable to these multi-turn artifact actions.
A future conversion must define the accepted source/context boundary explicitly.

Build evaluator and Pi guest on the remote server:

```sh
docker build -t ldm-copper-evaluator:20261023 -f tasks/copper_inversion/resources/Dockerfile .
npm --prefix harnesses/pi ci
npm --prefix harnesses/pi run build
node harnesses/pi/dist/build-task-guest.js --task copper_inversion --cache-dir "$HARNESS_CACHE_DIR"
node harnesses/pi/dist/smoke-task-guest.js --task copper_inversion --cache-dir "$HARNESS_CACHE_DIR"
```

Build the Pi sidecar from current repository source following its README, tagged
`ldm-pi-harness:copper-20261023`. All subsequent real runs use the official DeepSeek endpoint
`https://api.deepseek.com`, model `deepseek-flash` (V4.1 Flash), thinking `low`;
these values are locked in the real profiles. Supply `LLM_API_KEY` through a
protected environment variable or `--api-key-file`. The existing remote key file
is `/mnt/data1/Large-Discovery-Models/secrets/alphabench-t3/deepseek.key`.
The provider must support the Responses API used by Pi. Never place credentials
in committed YAML or logs.

```sh
uv run --locked --project tasks/copper_inversion python scripts/run_ldm_tts.py config/copper_inversion/harness_2d_smoke.yaml
```

Default formal run: one initial-geometry baseline plus three generated algorithms,
seed 47, 2 CPU/4 GiB, 60s per 2D solver. Use a fresh `--out-dir` per scientific
configuration. Resume the same path with `--resume`; completed jobs are reused.
Unknown dispatched containers pause for reconciliation instead of being retried.
Resume requires the same implementation and image identities. After code changes,
start a fresh run; archived runs retain their source snapshot for exact recovery.
The two 3D cases each have a locked single-researcher, three-round profile,
with 120s per solver and 900s per research turn:

```sh
uv run --locked --project tasks/copper_inversion python scripts/run_ldm_tts.py config/copper_inversion/harness_drillholes_3d.yaml
uv run --locked --project tasks/copper_inversion python scripts/run_ldm_tts.py config/copper_inversion/harness_drillholes_outcrops_3d.yaml
```

Each run has its own history and baseline plus three scored proposals. Do not
override the locked 2D profile to run 3D.

Run integration tests only remotely:

```sh
COPPER_TEST_IMAGE=ldm-copper-evaluator:20261023 uv run --locked --project tasks/copper_inversion python -m pytest tasks/copper_inversion/tests
```

Outputs include authoritative shared checkpoint/events/budget/status, task
`result.json`, `trajectory.csv`, `best_solver.py`, all source snapshots, interface
and execution receipts, and independently scored model arrays. Failed candidates
retain statuses without fabricated scores. The qualification stage is `contract_verified`; experiment qualification remains
`draft`. Remote evidence, final-executor rechecks and the measured best source are
in `resources/evidence/`. See `../../docs/copper-inversion-implementation.md`.
