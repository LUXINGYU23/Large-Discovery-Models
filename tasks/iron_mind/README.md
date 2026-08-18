# Iron Mind：LDM 反应条件优化任务

本目录是 Iron Mind 冻结反应表上的独立 LDM 任务适配器。公开仓库 clone
完成后，用户只需准备固定版本的上游数据、设置三个环境变量，并通过仓库统一
runner 运行配置。这里不复现 Iron Mind 官方方法或其他 baseline；被评估的方法始终
是本仓库的 LDM 闭环。

## 方法与评估边界

每轮实验执行同一条路径：

1. DeepSeek 端点按当前数据集的精确有限 schema 返回 4 个不同候选；
2. task adapter 严格校验字段、类型、选项和冻结表成员关系；
3. 共享 RBF GP-UCB 从 4 个候选中选择 1 个；
4. 冻结 Olympus 反应表返回 `reaction_score`；
5. 观测进入下一轮 GP 和 prompt 上下文。

正式 campaign 固定为 20 轮、20 次模型请求和 20 次冻结表评估。论文 v2 套件包含
6 个数据集；`public_union` 在此基础上加入
`alkylation_deprotection`，共 7 个数据集。每个数据集默认运行 20 个独立
campaign。所有候选、日志、JSON 和 CSV 运行产物使用英文。

## 目录结构

```text
tasks/iron_mind/
├── ldm_task/                 # 统一 runner 的薄适配层
├── core/                     # schema、候选、提案、GP、评估器和 workflow
├── resources/                # 上游版本/哈希契约与本地 mock fixture
├── scripts/                  # 数据快照准备和结果汇总
├── tests/                    # task contract 与行为测试
├── experiment.json           # 指标、预算和锁定 profile
└── pyproject.toml

config/iron_mind/
├── mock.yaml                 # 无网络、无外部数据
├── real_smoke.yaml           # 1 次真实模型请求与评估
├── ldm_20_<dataset>.yaml     # 单数据集 20 轮 campaign
├── paper_v2_ldm_20x20.yaml   # 6 × 20 campaigns
└── public_union_ldm_20x20.yaml # 7 × 20 campaigns
```

运行数据、模型凭据和结果都位于仓库外，不会写入 `tasks/iron_mind/resources/`。

## 1. 安装任务环境

在仓库根目录执行：

```bash
uv sync --locked --project tasks/iron_mind
uv run --locked --project tasks/iron_mind \
  python scripts/validate_tasks.py --task iron_mind --require-qualified \
  --require-stage tiny_campaign_verified
uv run --locked --project tasks/iron_mind \
  python -m pytest tasks/iron_mind/tests
```

## 2. 准备固定版本数据

任选一个仓库外目录作为源代码和数据根目录：

```bash
export IRON_MIND_WORK_ROOT=/absolute/path/to/iron-mind-work
export IRON_MIND_DATA_ROOT="$IRON_MIND_WORK_ROOT/data/official-complete"
export IRON_MIND_RUNS_ROOT="$IRON_MIND_WORK_ROOT/runs"

mkdir -p "$IRON_MIND_WORK_ROOT/sources" "$IRON_MIND_RUNS_ROOT"
git clone https://github.com/gomesgroup/iron-mind-public \
  "$IRON_MIND_WORK_ROOT/sources/iron-mind-public"
git -C "$IRON_MIND_WORK_ROOT/sources/iron-mind-public" checkout \
  476c555e45e2556e2ee4b24c726e774c2bfb7762

git clone https://github.com/gomesgroup/olympus \
  "$IRON_MIND_WORK_ROOT/sources/olympus"
git -C "$IRON_MIND_WORK_ROOT/sources/olympus" checkout \
  7b4bb35c04eb31dc57a8e46cc79a9cab71dee06d

uv run --locked --project tasks/iron_mind python \
  tasks/iron_mind/scripts/prepare_official_data.py \
  --iron-mind-checkout "$IRON_MIND_WORK_ROOT/sources/iron-mind-public" \
  --olympus-checkout "$IRON_MIND_WORK_ROOT/sources/olympus" \
  --output "$IRON_MIND_DATA_ROOT"
```

准备脚本会先核对两个 Git revision，再逐文件核对字节数和 SHA-256，最后写出
`revision_manifest.json`。正式运行的依赖预检会再次验证这些约束。

## 3. 配置模型凭据

正式配置锁定以下实验设置：

- OpenAI-compatible base URL：`https://api.deepseek.com`
- model：`deepseek-v4-flash`
- temperature：`0.7`
- 每轮候选数：`4`
- 每轮评估数：`1`

只通过进程环境提供凭据：

```bash
export LDM_LLM_API_KEY='your-api-key'
```

配置、运行快照和结果不会保存该值。若改动模型、温度、候选数或预算，结果必须按
新的实验设置报告，不能与本任务的锁定 profile 混用。

## 4. 运行验证

无端点 mock：

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/mock.yaml
```

正式依赖预检和单步真实 smoke：

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/check_task_dependencies.py config/iron_mind/real_smoke.yaml \
  --no-optional
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/real_smoke.yaml
```

## 5. 运行 benchmark

单个数据集、单个 20 轮 campaign：

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/ldm_20_buchwald_hartwig.yaml
```

论文 v2 的 6 数据集 × 20 campaigns：

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/paper_v2_ldm_20x20.yaml
```

全部 7 个公开数据集 × 20 campaigns：

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/public_union_ldm_20x20.yaml
```

每个 campaign 使用 `dataset_id + campaign_index` 形成独立运行目录，可通过该目录的
checkpoint 恢复，已经完成的冻结表评估不会重复执行。

## 6. 汇总结果

```bash
uv run --locked --project tasks/iron_mind python \
  tasks/iron_mind/scripts/aggregate_official_results.py \
  --runs-root "$IRON_MIND_RUNS_ROOT/official_complete" \
  --output-dir "$IRON_MIND_RUNS_ROOT/summary/paper_v2" \
  --suite paper_v2
```

汇总器默认严格要求每个数据集 20 个 campaign、每个 campaign 20 次评估，并输出：

- `summary.json`
- `dataset_summary.csv`
- `aggregate_trajectory.csv`

每个原始 campaign 还包含共享运行时的 `campaign.json`、`budget.json`、
`checkpoint.json`、`status.json`、`experiment_contract.json`、`result.json` 和
`trajectory.csv`，以及 task 自己的 search、selection 和 evaluation manifest。

## 可复现性声明

- `resources/upstream_contract.json` 是数据集、suite、Git revision、文件哈希、行数和
  schema 哈希的唯一事实来源。
- `experiment.json` 是指标角色、每轮预算和 profile 锁定参数的唯一事实来源。
- mock fixture 只用于验证软件闭环，不能作为科学 benchmark 数据。
- 本任务只报告 LDM 结果，不执行或包装 Iron Mind 官方优化器。
