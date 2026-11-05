# Copper inversion v2

The registered task discovers one inversion program across independent geology
parents. It uses only equal-weight AST/Qwen fusion, the shared LDMEngine,
Campaign, research tree, immutable candidate admission and independent scoring.

Paper codebase navigation: [RSI4Geo guide](../../docs/rsi4geo/README.md).

Scientific contract: [v2 spec](../../docs/copper-inversion-v2-spec.md).
Current evidence and limitations: [implementation log](../../docs/copper-inversion-v2-implementation.md).

## Input and evaluation

`solve(problem, budget, seed)` returns a finite float32/64 3D array. Positive
voxels define the proposed body. Non-active/forbidden regions are hard
constraints. Missing observations are not negative labels. Public physical-unit
geometry, likelihood and priors are shared by LDM and classical baselines.

Only public training cases and aggregate training feedback reach researchers.
Each candidate passes six fixed read-only interface cases. A separate scorer
receives frozen geometry and trusted truth, never submitted source. The current outer objective is `1 - parent-macro confirmation utility`, where
each successful solve has utility `1 / (1 + confirmation_loss_total)`. Failed
solves contribute zero utility and remain in the denominator. IoU is a separate
geometric diagnostic; public fitting loss is also reported separately. The
earlier IoU-based development runs retain their original scientific identities.

## Remote development

Run on the remote server, not the local development machine. The current H20
pilot data, frozen environments and outputs are under `/Aidata/shared/ldm/`.

```sh
python scripts/validate_tasks.py --task copper_inversion
python scripts/run_ldm_tts.py config/copper_inversion/fusion_pilot.yaml
```

Supply `LLM_API_KEY` via protected environment or `--api-key-file`; do not store
credentials in config or logs. Official DeepSeek endpoint, `deepseek-flash`, low
thinking and fusion representation are fixed in the workflow. The pilot config
requires existing remote data, Qwen cache and Docker/Pi images; it is not a data
generator or installation script. Override the output path for every new run.

The service-integration pilot has one training parent with two observation packs, one discovery
round, 8 proposed programs and 4 selections plus the fixed initializer. It uses
10 seconds per solve, 2 CPU and 4 GiB. This verifies the pipeline and does not
constitute the formal 120-second comparison. The earlier eight-parent pilot is
preserved under its frozen implementation. Formal-stage catalogs are rejected
until isolation, calibration and selection/test protocol gates are qualified.

The controller accepts `--public-catalog`, not a private suite manifest. A trusted
operator publishes training observations with `experiments.v2.publish_training`.
The private evaluator owns the full manifest and truth. Its Unix socket,
contract digest and service UID must be pinned in the run configuration; the
client checks the public observation digest, solve time and feedback cap before
starting research. The checked-in pilot configuration identifies one deployed
development instance; a new scientific campaign requires its own service run.
Public interface jobs remain in the controller, while suite solver/scorer jobs
and their detailed resource receipts remain in the private service ledger.

Resume only the same scientific implementation, image, data and configuration:
`--resume` reuses durable receipts and does not double-charge feedback slots.
Infrastructure uncertainty pauses evaluation; invalid candidate outputs are
scored as failures. Each modified scientific configuration needs a fresh run.

## Historical material

Old registration/configuration files are archived in
[registration-configs.tar.gz](../../docs/archive/copper-v1/registration-configs.tar.gz).
Old reports and frozen remote runs remain historical evidence. They do not
qualify v2 and cannot be compared on the same loss scale. Old case IDs and
AST-only/Qwen-only options are not accepted by the registered entry point.


### 2026-10-24 独立合成正式实验

当前标准预算为20轮、每轮32提案选16个，加1次初始化，共321次训练suite反馈。
复用main的1+3树搜索与研究角色提示，仅使用AST/Qwen等权融合。
该阶段的启动、配置、校准及恢复记录保留在[历史实验归档](../../docs/copper-v2-formal-study-r3-archive-20261031.md)。

### 当前论文结果与已知限制

最终交付复用 r3 的 LDM 冻结结果，与修复后的 r4 MCMC/ACQ 比较；
见[正式实验报告](../../docs/copper-v2-formal-study.md)及
[种子42报告](../../docs/copper-seed42-final-20261031/README.md)。
验证实际仅使用执行种子47，测试使用47/48/49；该偏差已接受并明确披露，未补评或重选。
种子42冻结赢家的内部最佳模型记录问题也已披露；历史源码和结果保持原样。
