# NucleoBench Quick Start

Run every command from the repository root. Real source, data, model, cache,
trace, and run directories should be outside the clone.

## 1. Install and Verify the Task

```bash
uv sync --locked --project tasks/nucleobench

uv run --locked --project tasks/nucleobench \
  python scripts/validate_tasks.py --task nucleobench \
  --require-qualified --require-stage tiny_campaign_verified

uv run --locked --project tasks/nucleobench \
  python -m pytest tasks/nucleobench/tests

uv run --locked --project tasks/nucleobench \
  python scripts/run_ldm_tts.py config/nucleobench/mock.yaml
```

The mock campaign is deterministic. It does not download official artifacts or
call a model endpoint.

## 2. Prepare a Source-Pinned Case

Install the official runtime on Linux and select external roots:

```bash
uv sync --locked --project tasks/nucleobench --extra official

export NUCLEOBENCH_WORK_ROOT=/absolute/path/to/nucleobench-work
export NUCLEOBENCH_SOURCE_ROOT="$NUCLEOBENCH_WORK_ROOT/source"
export NUCLEOBENCH_DATA_ROOT="$NUCLEOBENCH_WORK_ROOT/data"
export NUCLEOBENCH_RUNS_ROOT="$NUCLEOBENCH_WORK_ROOT/runs"
export NUCLEOBENCH_API_KEY_FILE="$NUCLEOBENCH_WORK_ROOT/secrets/api-key"
```

For Malinois or BPNet, provide the published CSV start table and corresponding
official model artifact:

```bash
uv run --locked --project tasks/nucleobench --extra official \
  python -m tasks.nucleobench.scripts.prepare_official_data \
  --case malinois_k562 \
  --source-dir "$NUCLEOBENCH_SOURCE_ROOT" \
  --output-dir "$NUCLEOBENCH_DATA_ROOT/malinois_k562" \
  --starts-file /external/nucleobench/start_sequences_df.csv \
  --model-artifact /external/nucleobench/malinois_artifacts.tar.gz \
  --bending-factor 1.0
```

The preparation command verifies source and artifact digests, selects the
declared 100-row start block, and writes an external
`prepared_manifest.json`. Enformer instead uses the published Parquet starts
and checkpoint. RiNALMo can be prepared only with an explicitly digest-pinned
external paired-start set and is not an official paired-start claim until the
benchmark publishes that set.

Export the digest written by the prepared manifest:

```bash
export NUCLEOBENCH_START_SET_SHA256=<prepared-start-set-sha256>
```

## 3. Configure a Model Endpoint

```bash
export LLM_BASE_URL=https://your-provider.example/v1
export LLM_MODEL_NAME=your-model

install -d -m 700 "$(dirname "$NUCLEOBENCH_API_KEY_FILE")"
read -r -s -p 'API key: ' key_value; printf '\n'
printf '%s' "$key_value" > "$NUCLEOBENCH_API_KEY_FILE"
unset key_value
chmod 600 "$NUCLEOBENCH_API_KEY_FILE"
```

The endpoint and model are not tied to one provider. Direct proposal methods
support both `chat_completions` and `responses` through
`args.llm-wire-api`. The committed real configs use Responses and maximum
reasoning; override them only for a separately labeled experiment. Harness
methods use the same provider identity through the Pi Responses sidecar.

Inspect the resolved task contract before a real run:

```bash
uv run --locked --project tasks/nucleobench --extra official \
  python scripts/run_ldm_tts.py \
  config/nucleobench/malinois_k562_tiny_campaign.yaml --dry-run
```

Dry-run output never includes the key or key-file path.

## 4. Build the Task Guest and Pi Sidecar

Real Harness methods require Linux, Docker, KVM, the host-architecture QEMU
system emulator, `e2fsprogs`, `cpio`, and `lz4`.

```bash
export HARNESS_CACHE_DIR="$NUCLEOBENCH_WORK_ROOT/gondolin-cache"

npm --prefix harnesses/pi ci
npm --prefix harnesses/pi run build:task-guest -- \
  --task nucleobench --cache-dir "$HARNESS_CACHE_DIR"
npm --prefix harnesses/pi run smoke:task-guest -- \
  --task nucleobench --cache-dir "$HARNESS_CACHE_DIR"

docker build -t ldm-pi-harness:latest harnesses/pi
```

The task guest is derived from the committed recipe under
`resources/harness/image/`. The cache stores base images, task build records,
Gondolin sessions, and copy-on-write overlays. Its smoke test verifies the
preinstalled sequence-analysis, statistics (including statsmodels and pyDOE3),
plotting, FASTA, interval, and alignment/RNA-folding tools before a campaign starts.

Candidate sessions load four task-local scientific Skills on demand from
`resources/harness/skills/`. No global skill installation or extra service
credentials are needed. Rebuild the guest after changing its package lock;
use a new campaign after changing profile or Skill content, whose digests are
part of the recorded session configuration. See the [research Harness section](README.md#persistent-research-harness)
for the Skill selection and its source attribution.

Optional MCP tools are loaded with `args.harness-mcp-config`. Proposal-session
network budgets use `args.harness-tool-budget`; the independent compiled
policy session uses `args.policy-tool-budget`. Tools not listed in a budget
remain unlimited. See [the shared Harness guide](../../docs/research-harness.md).

## 5. Run the Qualified Tiny Campaign

```bash
CUDA_VISIBLE_DEVICES='' \
uv run --locked --project tasks/nucleobench --extra official \
  python scripts/check_task_dependencies.py \
  config/nucleobench/malinois_k562_tiny_campaign.yaml

CUDA_VISIBLE_DEVICES='' \
uv run --locked --project tasks/nucleobench --extra official \
  python scripts/run_ldm_tts.py \
  config/nucleobench/malinois_k562_tiny_campaign.yaml
```

The pinned Malinois wrapper uses its official CPU path because its CUDA output
does not match the NumPy values expected by the source-pinned runner. Enformer
uses CUDA when available.

## 6. Run the Six-Method Pilot Evaluation

Review the resolved matrix first:

```bash
uv run --locked --project tasks/nucleobench --extra official \
  python scripts/run_pilot_evaluation.py \
  config/pilot_evaluation/nucleobench.yaml --dry-run
```

The matrix contains direct-API LDM, four-Agent Harness LDM,
Harness-Compiled LDM, task-local BO, direct LLM, and one-Agent direct Harness.
It uses three seeds, one shared initialization evaluation, eleven active rounds,
16 real evaluations per active round, and 64 proposal occurrences for each LDM
method.

Then run:

```bash
uv run --locked --project tasks/nucleobench --extra official \
  python scripts/run_pilot_evaluation.py \
  config/pilot_evaluation/nucleobench.yaml
```

Results are written below `$NUCLEOBENCH_RUNS_ROOT/pilot_evaluation/`.
Use `--resume` only with the same repository revision, prepared inputs, model,
and resolved configuration.

Candidate sessions submit annotated mutation patches through `candidates.json`.
Agents receive compact measurement updates and can query detailed history.
See [the Harness contract](README.md#persistent-research-harness) for submission,
validation, and history tools.

To verify the feedback path, use a separately named tiny run with `args.iterations=3`: one
paired-start evaluation followed by two active rounds, so the second proposal
turn consumes the first batch's measured results and annotations.

## 7. Official Wall-Time Run

The official profile uses the source-pinned runner's wall-time termination:

```bash
uv run --locked --project tasks/nucleobench --extra official \
  python scripts/run_ldm_tts.py \
  config/nucleobench/malinois_k562_official_base.yaml --dry-run

uv run --locked --project tasks/nucleobench --extra official \
  python scripts/run_ldm_tts.py \
  config/nucleobench/malinois_k562_official_base.yaml
```

Pilot Evaluation results are development diagnostics and must not be reported
as official wall-time benchmark results.

For eight independent comprehensive researchers with compiled policy, use the
same base config with overrides:

```bash
uv run --locked --project tasks/nucleobench --extra official \
  python scripts/run_ldm_tts.py \
  config/nucleobench/malinois_k562_official_base.yaml --dry-run \
  --set args.search-method=ldm_harness_compiled \
  --set args.proposal-mode=none \
  --set 'args.harness-profile=["comprehensive_research","comprehensive_research","comprehensive_research","comprehensive_research","comprehensive_research","comprehensive_research","comprehensive_research","comprehensive_research"]' \
  --set args.harness-unique-candidates=true \
  --set args.harness-candidates-per-session=32 \
  --set args.proposal-samples=256 \
  --set args.bo-pool-size=176 \
  --set args.evaluations-per-round=128 \
  --set args.campaign-index=42 \
  --set args.start-index=0
```

Use the same overrides for dependency checks and execution. Confirm the complete
resolved run before removing `--dry-run`, including the actual hardware label,
provider, image, tool budgets, and output directory. This is one official start
with one optimization seed, not the paper's aggregate over 100 starts.
The batch is an upper limit on unique evaluations, while the 256 proposal
occurrences remain fixed. Use `result.json.wall_time_result` for the strict 8h
score; the official runner may complete its last round after the deadline.
For an interrupted campaign, set `args.resume-from=<campaign-directory>`.
Only the original budget's remaining runtime is used. Repair downtime is excluded,
so resumed results are labelled cumulative-runtime runs, not uninterrupted
official comparisons. Keep the original failure artifacts and record runtime changes.

Each session submits 32 distinct, historically unmeasured sequences. Agreement
between independent sessions is preserved as empirical `q0` mass; the BO pool,
compiled policy, and selection rule remain unchanged. The earlier specialist
roles remain selectable through the same `--harness-profile` option. Omit
`--harness-unique-candidates` to permit within-session multiplicity.
