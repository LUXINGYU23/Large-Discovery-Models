# AtomWorld

This task uses the shared `run_campaign → LDMEngine → CampaignRuntime` lifecycle to apply a public natural-language operation to a public input CIF. The official evaluator owns the private target. All scheduled revisions are fixed before execution; the **last scheduled answer** determines the final score.

## Methods and scientific protocol

| Configuration | Method | Visible feedback |
| --- | --- | --- |
| `one_shot.yaml` | One direct CIF response | Public question only |
| `extended_reasoning.yaml` | Four direct CIF revisions | Previous draft and public syntax checks |
| `extended_operations.yaml` | Four bounded JSON operation plans | Previous plan, public tool errors and syntax checks |
| `harness.yaml` | Persistent parallel research sessions | Current public question and public draft history |
| `blind_harness_compiled.yaml` | Persistent research plus independent compiled public audit | Public drafts and structure features; empty objective-label history |

The task never passes `ExpansionRequest.observations`, `parent`, acquisition feedback, target CIFs, judge errors, RMSD or correctness values to proposal or policy agents. Public syntax validation uses no target. The Harness starts a separate pool for each question; each research profile has its own persistent session, receives compact incremental draft updates, and can query complete public history on demand. Two profiles run independently in parallel, each submitting exactly one CIF answer per revision. The ordinary Harness selects a profile by the declared `(revision + campaign_index) % profile_count` rotation, before any judge call. Only that answer is evaluated.

The compiled extension uses a separate `PolicyResearchController` and persistent policy session per question. It can submit, validate, repair and execute a deterministic NumPy artifact with the shared `prior_mean@1` interface. The interface here represents an **untrained standardized public plausibility prior** over the simultaneous drafts. No GP is fitted and no objective labels are synthesized. The prior ranks only the current revision; ties choose the first configured profile. Zero prior is the explicit fallback. This method is named `blind_harness_compiled`, not LDM/BO. Its scores are hypotheses, not benchmark measurements. It must be reported separately from the paper's direct-answer baselines.

Each proposed Harness answer is checked against its strict submission schema and public CIF parser before acceptance. Explicit JSON-pointer rejection reasons return to the same session for repair, up to `--harness-max-submission-attempts`. A missing or rejected session fails the strict barrier; it never silently reduces the requested count. Direct malformed outputs retain the original baseline behavior: the official judge scores them incorrect. Missing final answers remain incorrect; `oracle_any_attempt_accuracy_diagnostic` is an offline diagnostic, never an answer-selection rule.

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

Configure the existing provider environment (`LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY`). Compatibility aliases are `LDM_LLM_*` and `OPENAI_*`. Keep credentials in the protected environment or configured MCP secret references. Direct methods use Chat Completions and a separate endpoint preflight; the shared Pi Harness uses its Responses provider transport and records original provider messages.

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
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/blind_harness_compiled.yaml
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

`config/pilot_evaluation/atomworld.yaml` uses the explicit `final_submission` pilot protocol and one public question per case. It supports `--search-method`, `--campaign-index`, `--initialization-mode`, `--iterations`, `--run-name`, `--proposal-mode` and `--resume-from`. Here `shared_start` means the same public input question, with no free objective measurement; all revisions are counted. Repeated final CIFs legitimately reuse judge observations, while the trajectory retains every scheduled submission.

## Verification status

Qualification remains **draft/scaffolded** until a real provider → persistent research/tool use → repaired submission → official evaluation loop, including a subsequent public-history update and native traces, is verified on the required host. Deterministic protocol tests, mock accuracy, and local geometry/judge parity are useful regression evidence and are not live agent benchmark results. Historical local verification records are preserved as historical evidence, not current qualification. See `resources/review_verification.json` for the current repair validation and environmental limits.
