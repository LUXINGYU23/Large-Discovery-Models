# Iron Mind：LDM 反应条件优化任务

本任务在 Iron Mind 的冻结反应条件表上运行 LDM 闭环：模型每轮提出四个反应条件，
任务内的 GP-UCB 从中选择一个条件，再由冻结表返回反应得分。它评估的是本仓库的
LDM 方法；Iron Mind 与 Olympus 提供数据、指标和实验范围。

## 目录结构

```text
ldm_task/   统一 runner 的稳定适配层
core/       schema、候选、提案、GP、评估器和 workflow
resources/  上游版本契约、mock fixture 与验证记录
scripts/    数据准备和结果汇总工具
tests/      task-local 测试
```

正式配置位于 `config/iron_mind/`：`real_smoke.yaml` 运行一次真实闭环，
`ldm_20_<dataset>.yaml` 运行一个 20 轮 campaign，两个 suite 配置分别展开论文
范围的 6 个数据集和全部 7 个公开数据集。

## 快速开始

在仓库根目录创建任务环境并运行无网络 mock：

```bash
uv sync --locked --project tasks/iron_mind
uv run --locked --project tasks/iron_mind \
  python scripts/run_ldm_tts.py config/iron_mind/mock.yaml
```

完整的首次运行步骤见 [QUICKSTART.md](QUICKSTART.md)。

## 模型端点配置

真实运行使用 OpenAI-compatible Chat Completions API。提交的 YAML 不绑定某个服务商，
端点、模型和密钥通过环境变量提供：

```bash
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_MODEL_NAME=your-served-model
export LLM_API_KEY=your-api-key
```

也可通过 `--llm-url`、`--llm-model-name` 和 `--api-key` 临时覆盖。为兼容已有运行，
`TTS_LLM_URL` / `TTS_LLM_MODEL` / `TTS_LLM_API_KEY`、`LDM_LLM_URL` /
`LDM_LLM_MODEL` / `LDM_LLM_API_KEY` 和 `OPENAI_API_KEY` 仍然可用。URL 可以是 API
根地址或以 `/v1` 结尾的地址；任务会补全 Chat Completions 路径。

## 准备冻结数据

选择仓库外目录保存上游 checkout、数据和结果：

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

## 运行真实 campaign

先检查本机或服务器上的数据和模型配置，再运行一次 smoke：

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/check_task_dependencies.py config/iron_mind/real_smoke.yaml --no-optional

uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/real_smoke.yaml
```

单数据集 20 轮 campaign：

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/ldm_20_buchwald_hartwig.yaml
```

论文范围的 6 × 20 campaigns：

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/paper_v2_ldm_20x20.yaml
```

## 选择器

候选仍按 schema 顺序编码为 one-hot，保证 task spec 和运行记录稳定。选择器在内部按
反应 factor 计算距离：分类 factor 使用是否相同，离散数值 factor 使用归一化距离；
每个 factor 的权重由当前 campaign 的观测通过正则化边际似然拟合。

每轮的有效探索系数根据当前四候选 reservoir 和历史轮数计算。`acquisition-beta`
是该系数的基础倍率，默认值为 `1.0`。`selection_record.json` 会记录每轮的后验预测、
有效探索系数、核参数、目标标准化尺度和选择结果。

## 结果

每个 campaign 写入 `campaign.json`、`budget.json`、`checkpoint.json`、
`selection_record.json`、`result.json` 和 `trajectory.csv`。汇总完整 suite：

```bash
uv run --locked --project tasks/iron_mind python \
  tasks/iron_mind/scripts/aggregate_official_results.py \
  --runs-root "$IRON_MIND_RUNS_ROOT/official_complete" \
  --output-dir "$IRON_MIND_RUNS_ROOT/summary/paper_v2" \
  --suite paper_v2
```

汇总目录包含 `summary.json`、`dataset_summary.csv` 和
`aggregate_trajectory.csv`。所有运行产物字段使用英文，方便后续统计和绘图。
