# Copper encoding ablation: progress, matched budgets and repair

Snapshot: 2026-10-23 22:51:06 Asia/Shanghai. These are **v2 full campaigns**, not the earlier one-round preflights. Lower loss is better. Times are cumulative process-start to the last independent score in the stated round, including baseline, research, admission, encoding and evaluation. A research round adds four formal evaluations; round zero is the fixed baseline. Pending rounds are excluded from comparisons.

## Progress before versioned restart

| Case | Encoding | Completed rounds | Formal evaluations | Best loss | Minutes to last complete round |
|---|---|---:|---:|---:|---:|
| drillholes_3d | AST | 5 | 21 | 6.518914 | 51.39 |
| drillholes_3d | Qwen | 0 | 1 | 405.079581 | 0.46 |
| drillholes_3d | fusion | 4 | 17 | 6.516138 | 46.96 |
| drillholes_outcrops_3d | AST | 2 | 9 | 33.082900 | 31.49 |
| drillholes_outcrops_3d | Qwen | 2 | 9 | 36.788643 | 35.63 |
| drillholes_outcrops_3d | fusion | 1 | 5 | 35.201883 | 16.36 |

All 62 recorded formal evaluations succeeded. The 0.46 minutes for drillholes/Qwen is baseline completion time, **not** the cost of the failed campaign: its first process ran 19.22 minutes before failure. Time subsequently spent on incomplete branches is also absent from the last-complete-round column and remains recorded in raw traces and resource receipts.

## Matched-round comparisons

| Case / round / evaluations | AST loss / minutes | Qwen loss / minutes | Fusion loss / minutes |
|---|---:|---:|---:|
| drillholes / 1 / 5 | 6.523782 / 8.60 | unavailable | 6.530942 / 12.36 |
| drillholes / 2 / 9 | 6.521003 / 20.77 | unavailable | 6.523612 / 22.22 |
| drillholes / 3 / 13 | 6.520863 / 30.42 | unavailable | 6.522996 / 36.14 |
| drillholes / 4 / 17 | 6.520863 / 41.79 | unavailable | **6.516138** / 46.96 |
| drillholes+outcrops / 1 / 5 | **33.104948 / 14.02** | 42.573995 / 18.38 | 35.201883 / 16.36 |
| drillholes+outcrops / 2 / 9 | **33.082900 / 31.49** | 36.788643 / 35.63 | unavailable |

Fusion has a small late improvement on drillholes: round-four best loss is 0.0725% below AST, with 12.36% more elapsed time. At the only three-way matched outcrop round, fusion is 6.33% worse than AST and takes 16.69% longer; it beats Qwen there. This is not evidence of a consistent fusion advantage. There is one stochastic campaign per arm, distinct online proposal pools, missing observations and no uncertainty estimate. Same seed does not force the hosted LLM to generate identical candidates. A causal encoder comparison also needs the planned offline common-pool analysis or repeated online seeds.

## Actual anomalies

1. Drillholes/Qwen, round 1, researcher 1: rejected after three submission attempts. An immutable interface fixture exited 134. Its native library reported `std::system_error: Resource temporarily unavailable`; receipt reports no OOM and about 46 MB peak memory. This points to native thread/process creation under the 64-PID limit; HiGHS must be explicitly constrained by the algorithm rather than assuming OMP_NUM_THREADS controls it.
2. Outcrops/fusion, round 2, researcher 2: repeatedly submitted an already measured source and exhausted the three-attempt allowance. The other three peers waited at the all-profile static validation barrier until their 1200-second wall-time limits. A simple resume replays the terminal rejection and blocked history, so cannot resolve it.
3. Before stopping, outcrops/AST and outcrops/Qwen also had terminally rejected peers in round 3: candidate zero used `candidates/4/solver.py` and `candidates/00/solver.py`, respectively, rather than the required `candidates/0/solver.py`. Their overall process status still read running while the remaining peers waited. Checking PIDs alone would have missed this.
4. Task feedback always pointed to `/candidates/0/artifact_path`, even when another file failed. Four-file submissions still shared the original single-file allowance of three attempts. These made recoverable candidate mistakes unnecessarily fatal.

## Repair and continuation policy

Code commit: `c5a04718`.

- Errors now carry the actual candidate index; interface errors include the fixture variant and a bounded excerpt of untrusted candidate diagnostics. Signed supervisor envelopes are excluded.
- Validation allowance is three attempts per submitted candidate, hence twelve for the current four-file submission. Wall-time, tool, fixture and formal evaluation limits remain enforced.
- The shared Harness client closes unanswered peers immediately after a terminal peer rejection, including submissions that arrive later. It does not admit invalid candidates or publish a partial pool.
- Research instructions explicitly prohibit reusing measured source and explain native thread limits / HiGHS `threads=1`.
- Existing immutable interface tests, candidate isolation, loss and independent scorer are unchanged.
- Remote checks passed: **145 tests including Docker isolation/recovery**, plus **17 focused regressions** after adding extra diagnostic/exhaustion assertions. The suites overlap; do not add these counts. All twelve configuration/dependency/dry-run gates passed.

The task's AGENTS.md prohibits forcing a modified implementation into an old scientific identity. Therefore the original six campaigns were stopped with their source, checkpoints, requests and budgets preserved. No failed submissions were deleted, no clocks were reset, and no old observations were inserted into new campaigns. The new version runs six native one-round preflights, then automatically launches six fresh 20-round campaigns only if every preflight and completed-resume audit passes. New and old trajectories must be reported separately.

New root: `/Aidata/shared/ldm/copper-encoding-ablation-v3-20261023`.
Old root: `/Aidata/shared/ldm/copper-encoding-ablation-v2-20261023`.
Durable controller: new root `restart.json` (initial PID 1542794); per-stage supervisors and six status files remain authoritative. At restart dispatch the active stage is `tiny`; the 20-round `full` stage is gated, not yet claimed started.

Frozen runtime identity: `6c17bf369c4c9c676878812fe13b508890241b9f814822120d62edf60a57d1d5`.
Source archive: `/mnt/cephfs/xingyu/ldm/archive/copper-inversion/encoding-repair-c5a04718-20261023.tar.gz`, SHA256 `cc340b9af15f60cbf5c644564c20027e5d5a92d07314bdb9aca33c5791cfd9d7`.

Model/provider remain official `https://api.deepseek.com`, `deepseek-flash`, low; Qwen uses the existing frozen local 0.6B cache, GPU 0, AST/fusion weights unchanged. Each full campaign still has 20 rounds, 16 proposals/round, 4 selected evaluations/round, maximum 81 formal evaluation attempts. Provider keys are passed through remote pipes/memory only.

Evidence: [v2 snapshot](../tasks/copper_inversion/resources/evidence/encoding-ablation-20261023/repair/progress-v2.json), [remote integration tests](../tasks/copper_inversion/resources/evidence/encoding-ablation-20261023/repair/repair-integration-tests.log), [focused regressions](../tasks/copper_inversion/resources/evidence/encoding-ablation-20261023/repair/repair-regression-tests.log).
