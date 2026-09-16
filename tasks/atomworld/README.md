# AtomWorld

This task uses the shared `run_campaign → LDMEngine → CampaignRuntime` lifecycle to apply a public natural-language operation to a public input CIF. The official evaluator owns the private target. All scheduled revisions are fixed before execution; the **last scheduled answer** determines the final score.

## Methods and scientific protocol

| Configuration | Method | Visible feedback |
| --- | --- | --- |
| `one_shot.yaml` | One direct CIF response | Public question only |
| `extended_reasoning.yaml` | Four direct CIF revisions | Previous draft and public syntax checks |
| `extended_operations.yaml` | Four bounded JSON operation plans | Previous plan, public tool errors and syntax checks |
| `harness.yaml` | Persistent parallel research sessions | Current public question and public draft history |
| `harness_public_audit.yaml` | Persistent research plus independent compiled public audit | Public drafts and structure features; empty objective-label history |
| `ldm.yaml` | Direct reservoir plus fixed GP/UCB and empirical-q0 sampling | Past same-question scalar correctness |
| `ldm_harness.yaml` | Persistent research reservoir plus the same LDM selector | Past same-question scalar correctness |
| `ldm_harness_compiled.yaml` | LDM plus independently compiled residual prior and alpha/eta | Measured history, fixed-GP diagnostics and current unmeasured candidate features |

Blind baselines never pass `ExpansionRequest.observations`, `parent`, acquisition feedback or correctness to agents. LDM methods expose only already measured scalar correctness for the current question; never target CIFs, judge errors, RMSD, future scores or cross-question history. Public syntax validation uses no target. The Harness starts a separate pool for each question; each research profile has its own persistent session, receives compact incremental draft updates, and can query complete history on demand. Two profiles run independently in parallel, each submitting exactly one CIF answer per revision. The ordinary blind Harness selects a profile by `(revision + campaign_index) % profile_count`; LDM uses the selector described below. Only the selected answer is evaluated.

`ldm_harness_compiled` implements the same algorithmic structure as NucleoBench: empirical proposal frequencies before canonical deduplication, an independently maintained BO pool, a measured-history residual GP, fixed UCB, and Gumbel sampling proportional to `q0**alpha * exp(eta * robust_z(UCB))`. The AtomWorld representation is signed-log public input-to-draft geometry, with a fixed exact RBF kernel. The first round of each question uses proposal frequencies without objective feedback. Later rounds fit the GP to measured scalar correctness and run an independent `PolicyResearchController` session with both `prior_mean@1` and `ldm_weights@1`. Compiled history priors are subtracted before fitting and query priors added afterward; covariance remains fixed. Policy diagnostics compare chronological holdouts and actual selection probabilities. No prior argmax is used in LDM.

The optional `harness_public_audit` remains a separate blind baseline with empty label history and an untrained public prior. It is not the implementation of `ldm_harness_compiled`.

Each Harness session submits a raw `answer.cif` file plus rationale. The host verifies its immutable snapshot digest and size before public CIF parsing. Explicit JSON-pointer rejection reasons return to the same session for repair, up to `--harness-max-submission-attempts`. A missing or rejected session fails the strict barrier; it never silently reduces the requested count. Direct malformed outputs retain the original baseline behavior: the official judge scores them incorrect. Missing final answers remain incorrect; `oracle_any_attempt_accuracy_diagnostic` is an offline diagnostic, never an answer-selection rule.

LDM runs declare `feedback_protocol=measured_correctness_optimization`. This is
an explicit feedback-enabled optimization extension, distinct from the paper's
blind-answer protocol. The official evaluator is unchanged; target structures
remain private. Do not report these runs as official no-feedback paper results.
`--proposal-samples` controls direct LDM breadth; Harness breadth is one draft
per configured research profile. `--bo-pool-size`, `--acquisition-alpha`,
`--acquisition-eta`, `--ucb-beta`, `--z-clip` and the `--gp-*` settings are frozen
on resume. Independent submissions can repeat a CIF in later rounds, with a
new scheduled evaluation identity and charge; within-round duplicates add q0.

Direct and Harness runs expose `--llm-wire-api`, `--llm-reasoning`,
`--llm-temperature` and `--llm-extra-body-json`. Defaults are Responses, off, 0.0
and an empty object. Direct calls also support Chat Completions; Harness requires
Responses. `--harness-thinking` remains an alias that must agree with reasoning.
The schedule freezes these settings and the native manifest records the actual
Harness request options. Rebuild the shared sidecar when using this protocol.
Both paths preflight configured providers before research, including Harness
with mock scoring.

History messages contain a compact draft index and preserve the original
rationale. `get_public_history` accepts `draft_ids` and `detail=detailed` to
retrieve selected CIFs and rationales. Only LDM runs also expose measured past
scalar correctness; unmeasured proposals have no label.

## Installation and data

From the repository root, use the independently locked task environment:

```bash
uv sync --locked --project tasks/atomworld --group dev
uv run --locked --project tasks/atomworld python -m pytest -q tasks/atomworld/tests
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/mock.yaml
```

Obtain the official AtomWorldBench release identified in `resources/source_manifest.json`, place it anywhere, and set `ATOMWORLD_UPSTREAM_ROOT` to its root. The adapter verifies the evaluator's SHA-256 before import and refuses mismatched sources. The pinned source archive does not identify a Git revision. Dataset/model assets are external official resources; all task-owned bounded geometry code ships in this repository and needs no sibling project.

```bash
export ATOMWORLD_UPSTREAM_ROOT=/absolute/path/to/AtomWorldBench
uv run --locked --project tasks/atomworld python -m tasks.atomworld.scripts.prepare_official_data \
  --source-dir "$ATOMWORLD_UPSTREAM_ROOT/src/data" \
  --out-dir /absolute/path/to/prepared-atomworld --per-action 1 --seed 0
export ATOMWORLD_DATA_ROOT=/absolute/path/to/prepared-atomworld
```

Preparation mirrors the official CSV/HDF5 priority with JSON fallback, preserves source precision, and writes hash-checked `public.jsonl`, permission-0600 `private.jsonl`, and `manifest.json`. Use `--per-action 0` for every released row, or `--actions move_atom_action,rotate_around_atom_action` for selected actions. Prepared output cannot overwrite an existing manifest. Released subsets are not a verified paper test split; outputs explicitly identify `local_released_subset` and make no paper-score claim.

The evaluator retains last-CIF-tag extraction, nonprimitive parsing, species-count checks and official StructureMatcher settings. Its RMSD and maximum distance are normalized dimensionless quantities. Geometry code covers all ten paper operations and is shared verbatim between the host adapter and guest image; `resources/harness/geometry_provenance.json` records its delivered source hashes.

## Direct and Harness runs

Configure the existing provider environment (`LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY`). Compatibility aliases are `LDM_LLM_*` and `OPENAI_*`. Keep credentials in the protected environment or configured MCP secret references. Direct methods use the selected wire API and a separate endpoint preflight; the shared Pi Harness uses Responses and records the effective provider request bodies.

```bash
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/extended_operations.yaml
```

For Harness execution, prepare the shared sidecar and task guest on a Linux Docker/KVM host as described in `harnesses/pi/README.md`:

```bash
docker build -t ldm-pi-harness:latest harnesses/pi
npm --prefix harnesses/pi ci
npm --prefix harnesses/pi run build:task-guest -- --task atomworld --cache-dir /absolute/path/to/harness-cache
npm --prefix harnesses/pi run smoke:task-guest -- --task atomworld --cache-dir /absolute/path/to/harness-cache
```

Set `ATOMWORLD_HARNESS_CACHE` to that cache before running the configs:

```bash
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/harness.yaml
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/harness_public_audit.yaml
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/ldm_harness_compiled.yaml
```

The guest includes ASE, NumPy, pymatgen and the delivered `atomworld_tools` package. Agents obtain task context and paginated history through structured tools and execute geometry or scratch analysis themselves using the guest's `bash` tool. Task roles and selected skills are digest-bound resources. Shared network/MCP policies and per-tool budgets are wired through `PiHarnessConfig`; `--harness-mcp-config` loads optional shared MCP configuration. The compiled policy runs in a separate network-disabled, read-only, resource-limited Docker container.

Only the current question's public artifact directory and research resources are mounted into its sidecar. The campaign root, prepared private labels, judge files, other questions and evaluator implementation are never mounted. Closed question pools remain on disk for provenance, without accumulating live processes. A public draft can itself equal the correct answer; isolation forbids access to private labels, not accurate reasoning from public input.

## Budgets, traces and resume

`--attempts-per-sample` fixes scientific answer opportunities; `--iterations` is its pilot alias. Service recovery has a separate finite `--service-retry-allowance` (default 3), which permits failed attempts without increasing scheduled answers. Harness turns, actual provider calls, tools, submission repairs and policy usage are recorded separately. Shared per-tool and wall-time budgets bound each research turn. Changing schedule, source/tool hashes, model or Harness configuration on resume is rejected.

```bash
uv run --locked --project tasks/atomworld python -m tasks.atomworld.ldm_task.procedure \
  --data-dir "$ATOMWORLD_DATA_ROOT" --search-method harness --attempts-per-sample 4 \
  --harness-cache-dir "$ATOMWORLD_HARNESS_CACHE" --resume-from /absolute/path/to/printed/run_dir
```

Resume the actual printed run directory with the same settings. Accepted attempt files are replayed without repeated calls; interrupted Harness turns use their native session journals and immutable turn IDs. A completed run performs no new model or judge work. Do not mount the private campaign root into any agent process.

The run includes shared lifecycle/checkpoint/budget files, `result.json`, `trajectory.csv`, `attempts/*.json`, and private `evaluations/*.json`. Native proposal traces live under `harness/sample_*/`; policy traces under `policy_harness/sample_*/`. Each pool retains the shared sidecar manifest, persistent session events, submission validation and raw provider transport records. Research outputs remain distinct from direct CIF SFT collection; no tool-derived CIF is misrepresented as a direct model response.

`config/pilot_evaluation/atomworld.yaml` compares `ldm`, `ldm_harness` and `ldm_harness_compiled` with matched two-draft breadth, the explicit `final_submission` protocol and one public question per case. It supports `--search-method`, `--campaign-index`, `--initialization-mode`, `--iterations`, `--run-name`, `--proposal-mode` and `--resume-from`. Here `shared_start` means the same public input question, with no free objective measurement; all revisions are counted. Final score always belongs to the last scheduled selection, not the highest observed score. Blind baselines may reuse an earlier identical answer's judge observation; LDM uses distinct per-round submission identities.

## Verification status

Qualification remains **draft/scaffolded** until a real provider → persistent research/tool use → repaired submission → official evaluation loop, including subsequent history updates and native traces, is verified on the required host. Deterministic protocol tests, mock accuracy, and local geometry/judge parity are regression evidence, not live agent benchmark results. Historical local verification records retain their original scope. See [the current repair record](../../docs/reviews/pr9-adapter-repair.md) for validation and environmental limits.
