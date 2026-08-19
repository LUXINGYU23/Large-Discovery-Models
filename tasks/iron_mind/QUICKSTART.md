# Iron Mind：首次运行指南

本指南从一个干净 checkout 开始，依次完成 mock、冻结数据准备、端点配置和一次真实
smoke。所有命令从仓库根目录执行。

## 1. 安装并验证 mock

```bash
uv sync --locked --project tasks/iron_mind
uv run --locked --project tasks/iron_mind \
  python -m pytest tasks/iron_mind/tests
uv run --locked --project tasks/iron_mind \
  python scripts/run_ldm_tts.py config/iron_mind/mock.yaml
```

mock 不需要外部数据、GPU 或模型端点。

## 2. 配置模型服务

真实运行需要一个 OpenAI-compatible Chat Completions 服务：

```bash
export LLM_BASE_URL=https://your-model-host.example/v1
export LLM_MODEL_NAME=your-served-model
export LLM_API_KEY=your-api-key
```

本地服务如不要求鉴权，可省略 `LLM_API_KEY`。服务地址和模型名会写入运行配置，密钥
不会写入 YAML 或运行产物。

## 3. 准备数据

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

## 4. 运行 smoke

```bash
uv run --locked --project tasks/iron_mind python \
  scripts/check_task_dependencies.py config/iron_mind/real_smoke.yaml --no-optional

uv run --locked --project tasks/iron_mind python \
  scripts/run_ldm_tts.py config/iron_mind/real_smoke.yaml
```

成功后，结果位于 `$IRON_MIND_RUNS_ROOT/smoke/` 下的带时间戳目录。确认端点可用后，
再运行 `config/iron_mind/ldm_20_<dataset>.yaml` 或 suite 配置。
