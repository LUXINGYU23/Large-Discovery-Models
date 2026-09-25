# ResearchGym

LDM search over **method programs** for four cases of
[ResearchGym](https://github.com/Anikethh/ResearchGym), pinned to commit
`0adc08e93754b606d8a76055ed2f1c9574504312`. A candidate is one complete Python
module for a case's method slot. Each selected candidate is installed into a
fresh workspace assembled from SHA-256-verified pinned files, trained with the
case's own scripts on the configured devices, and scored by the case's own
official grader. Qualification is `draft`.

## Cases

| Case | Slot | Job matrix per candidate | Objective (maximize) | Protocol identifier |
| --- | --- | --- | --- | --- |
| `time_series_explanation` | `attribute(classifier, test_loader, timesteps, device)` in `ldm_explainer.py` | 5 datasets x 5 folds | mean of 10 real-data CPD cells (dataset x Average/Zeros, top-k 0.1) | `rg_tse_real_state_cpd_topk0.1_5ds_2sub_5fold_v1` |
| `continual_learning` | `class Learner(BaseLearner)` in `models/ldm_method.py` | seeds 1992-1994 | CIFAR100 N=10 final accuracy | `rg_cl_cifar100_n10_vitb16_3seeds_epoch5_v1` |
| `cross_modal_retrieval` | `class LDMTTA(nn.Module)` in `models/tta_baselines/ldm_method.py` | 16 COCO-C corruptions | BLIP ViT-B/16 i2t Recall@1 row average | `rg_cmr_cococ_sub1000_sev5_i2t_blipb_16corr_v1` |
| `improving_replay_buffers` | `class ReplayBuffer` in `ldm_buffer.py` | seeds 0, 1 | DMC cheetah-run average return | `rg_rl_dmc_cheetah_run_100k_2seeds_tasklocal_sac_v1` |

Every case uses its official grader but a declared **sub-protocol**
(`official_protocol: false`); results are not comparable to paper tables. The
deviations are recorded in `resources/cases/catalog.json` and in every
evaluation's metadata: TSE runs only the real-data table; CL runs only CIFAR100;
CMR uses a seed-0 1000-image query subset; the RL case uses a task-local SAC
stack because the pinned upstream task contains no training loop.

**Coverage is exact.** Each case declares its complete cell identity (for
example all 10 TSE cells with exactly 5 folds each, all 16 CMR columns, all
three CL seeds). A crashed job, a missing output, or a missing or extra cell
makes the candidate `failed` or `invalid`; the grader's average over whatever
cells exist is never used.

## Methods

| Method | Candidate generation | Selection |
| --- | --- | --- |
| `llm` | One direct API request per round proposes exactly the evaluation batch. | Submission order. |
| `ldm` | Independent direct requests of `proposal-batch-size` produce `proposal-samples` occurrences. | Empirical q0, BO pool, residual RBF GP-UCB, `q0^alpha * exp(eta * robust_z(UCB))` sampled without replacement. |
| `harness` | One persistent Pi session proposes exactly the evaluation batch. | Submission order. |
| `ldm_harness` | `harness-sessions` independent persistent sessions produce the occurrences. | Same as `ldm`, fixed alpha/eta. |
| `ldm_harness_compiled` | As `ldm_harness`, plus one independent policy session. | Same GP with a compiled residual prior mean and per-round alpha/eta (`prior_mean@1`, `ldm_weights@1`). |
| `bo` | **Unsupported.** No score-blind, LLM-free generator of valid method programs is defined for these cases; the CLI rejects it. | |

Three sizes are distinct: occurrences per round (`proposal-samples`), BO pool
(`bo-pool-size`), and evaluation batch (`evaluations-per-round`).

### Submission rules

- A submission contains exactly the requested number of entries, each with
  `program`, `change_summary`, `rationale`, and optional
  `comparison_candidate_ids` (validated against measured history).
- Entries of one request or session must be distinct canonical programs
  (the AST with comments, formatting and docstrings removed). Independent
  requests or sessions may repeat each other; those repeats are counted in q0
  before the shared reservoir deduplicates them.
- Only authoritative measured candidates (including failed evaluations) are
  excluded. Programs proposed earlier but never evaluated remain eligible.
- Invalid entries get indexed JSON-Pointer errors and are repaired in the same
  exchange (direct) or session (Harness). If repair attempts run out the round
  pauses resumably (`paused_proposal_exhausted`); it is never silently shrunk.
- If agreement between independent requests or sessions leaves fewer distinct
  programs than the evaluation batch, up to `--max-refill-collections` extra
  collections ask for programs that differ from this round's proposals. Their
  occurrences also enter q0. If the batch still cannot be filled, the round
  pauses instead of evaluating fewer candidates.

The static admission rules live in `resources/harness/tools/program_rules.py`,
which the host validator imports unchanged and the guest checker tool runs.

## Research Harness

Proposal sessions load `program_research/AGENTS.md` and the
`researchgym-method-research` and `ml-experiment-analysis` Skills. The guest
has Python 3.11 with NumPy, SciPy, pandas, scikit-learn, matplotlib and CPU
PyTorch (no GPU, data, or checkpoints). Task tools:

- `describe_researchgym_case`: the public case contract plus pinned public
  reference sources, exported to a read-only guest file;
- `get_measured_history`: filtered, sorted, paginated measured candidates with
  source, per-cell metrics, errors and original research notes;
- `check_candidate_programs`: the task's admission rules on a draft
  `candidates.json`.

Sessions write programs as files and build `candidates.json` with the Skill's
script, so code is never hand-escaped into JSON. Each turn carries only a
compact delta of newly measured candidates. Web search/fetch, MCP servers
(`--harness-mcp-config`) and per-tool budgets (`--harness-tool-budget`,
`--policy-tool-budget`) use the shared Pi features.

## Accounting, clock and resume

- `budget.json` counts outer iterations, direct requests (including failed
  transport attempts, whose usage is recorded as unknown), Harness turns,
  provider/tool calls, validation submissions, tokens, and evaluations. Harness
  tokens are read from the sidecar's redacted provider captures, for the
  proposal pool and the compiled-policy pool alike.
- `--api-budget-usd` with `--llm-input-usd-per-mtok` and
  `--llm-output-usd-per-mtok` enforces a spending cap: measured cost is always
  recorded and checked before each new request or turn, so only the request in
  flight can overshoot. Stopping on the cap sets `stopped_api_budget`.
- `--campaign-hours` is a durable wall clock: segments persist across resume
  with heartbeats, per-candidate allowances shrink to the remaining time, and
  a resumed run never regains elapsed time.
- `--resume-from RUN_DIR` requires an identical scientific configuration.
  Cached direct responses and committed Harness turns replay without new
  charges; an unfinished Harness session continues in its existing workspace;
  a session already accepted for a round is not re-run even when another
  session exhausts its repair attempts. A request left in flight by a stopped
  process is counted as failed and never reused. A launched evaluation without
  a terminal record is recorded as failed instead of being repeated.
- Timeouts and interruptions terminate each job's whole process group and wait
  until it is empty before the next candidate starts. Provider credentials are
  removed from candidate process environments.
- After the jobs and before the official grader runs, every file the evaluator
  installed (pinned code, injected slot, support files, generated configs, and
  the candidate) is re-hashed; a candidate that modified any of them is
  `invalid`.
- `config.json` declares `proposal_counting: bounded_minibatches` with the
  per-request size and the maximum extra attempts per round (repairs and
  refills), which the pilot-evaluation integrity check verifies. The mock
  proposal client is a local synthetic source and is not charged as model
  requests.

## Setup

1. A clean checkout of ResearchGym at the pinned commit
   (`git -C ResearchGym checkout 0adc08e9`). The evaluator rejects modified or
   missing tracked files and never copies untracked files.
2. One case environment per case, from the case's pinned
   `requirements.txt`; for the RL case use `environments/improving_replay_buffers.txt`.
3. Case data under a data root (defaults to the upstream case directory):

   ```bash
   python tasks/researchgym/scripts/prepare_case_data.py tse --data-root /data/rg/tse
   python tasks/researchgym/scripts/prepare_case_data.py cl  --data-root /data/rg/cl
   /path/to/cmr-venv/bin/python tasks/researchgym/scripts/prepare_case_data.py cmr \
     --data-root /data/rg/cmr --upstream-root ~/ResearchGym --generate-corruptions
   ```

   TSE also needs the trained state classifiers under `<data-root>/model`
   (upstream `real/main.py --train True`); CMR needs the upstream COCO images,
   Karpathy annotations and BLIP weights under `dataset/` and `weights/`.
   Every download URL is a flag; `cl` accepts any mirror of the official
   archive and verifies its published md5.
4. For Harness methods: build and smoke the guest (Linux with Docker and KVM):

   ```bash
   npm --prefix harnesses/pi ci
   npm --prefix harnesses/pi run build:task-guest -- --task researchgym --cache-dir /path/to/harness-cache
   npm --prefix harnesses/pi run smoke:task-guest -- --task researchgym --cache-dir /path/to/harness-cache
   ```

Real configs read `RESEARCHGYM_ROOT`, `RESEARCHGYM_CASE_DATA_ROOT`,
`RESEARCHGYM_CASE_PYTHON`, `RESEARCHGYM_DEVICES` (comma-separated GPU indices)
and `RESEARCHGYM_RUNS_ROOT`; the endpoint comes from `LLM_BASE_URL`,
`LLM_MODEL_NAME` and `LLM_API_KEY`. Deployment-specific settings such as
`HF_ENDPOINT` belong in the environment or a config `env` block.

## Running

```bash
# Service-free checks
python scripts/run_ldm_tts.py config/researchgym/mock.yaml
python scripts/run_ldm_tts.py config/researchgym/mock_ldm_harness.yaml

# Real: dependency check, then a tiny profile-locked run
python scripts/check_task_dependencies.py config/researchgym/continual_learning_tiny_ldm_harness.yaml
python scripts/run_ldm_tts.py config/researchgym/continual_learning_tiny_ldm_harness.yaml

# Matched method comparison (1 released-seed round + 5 optimization rounds, 3 seeds)
python scripts/run_pilot_evaluation.py config/pilot_evaluation/researchgym.yaml
```

Each case has `*_tiny_{ldm,harness,ldm_harness,ldm_harness_compiled}.yaml` and
`*_pilot_ldm_harness.yaml`, each locked by an `experiment.json` profile.
`initialization-mode shared_start` evaluates the released baseline seed in
round 0 for every method.

## Run artifacts

`result.json` (status, best candidate, protocol, source commit and tree digest,
clock, budget), `evaluations.csv` (every attempt, including failures),
`trajectory.csv` (successful evaluations, the pilot-evaluation input),
`best_program.py`, `evaluations/<candidate>/` (workspace, per-job logs,
`jobs.json`, grader summary, terminal `result.json`), `proposal_requests/`,
`proposal_diagnostics/`, `harness/` and `policy_harness/` (native Pi sessions,
redacted provider traces, turn inputs, submissions, manifests), plus the shared
campaign files. Pilot reporting requires at least one successful evaluation per
round; a round in which every candidate fails is reported by the pilot tool as
incomplete rather than scored.

## Data collection

Direct methods append accepted submissions as `ldm-2.0` complete-design rows
after validation (`LDM_DATA_COLLECTION_ENABLED=1`); run and outcome fields stay
in provenance. Harness sessions and provider traces are raw research records,
not training rows.

## Qualification status

`resources/qualification_evidence.json` records `mock_verified`. Not yet done:
`contract_verified` (real-environment CPU/GPU contract checks per case),
`seed_evaluated` (released seed through the real evaluator on each case), and
`tiny_campaign_verified` (real sidecar capability smoke and one real tiny run
per method). Historical results from the earlier stand-alone package are not
evidence for this implementation.
