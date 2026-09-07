# AtomWorld × LDM

AtomWorld 已通过 `task.json` 注册为 LDM Science Benchmark，执行入口使用共享
`run_campaign → LDMEngine → CampaignRuntime`。每道题输入原始 CIF 和自然语言操作，
候选是该题的完整结构答案。所有题目、轮数和最终提交位置在运行前固定；没有另写一个 benchmark agent 循环。

## 科学适配

- **`one_shot.yaml`**：每题一次模型请求，使用与 upstream 完全相同的 CIF prompt；默认最多 8192 输出 token。
- **`extended_reasoning.yaml`**：每题四次请求。后续轮只读取原题、上一轮答案和公开 CIF 语法检查，进行长程推理与自我检查。最后一次提交作为答案，不能挑评分最高的答案。
- **`extended_operations.yaml`**：模型产生有界 JSON 几何操作，调用并列目录 `atomworld-agentic-reproduction/` 的工具生成 CIF；下一轮能看到自身计划和工具错误。它是独立标注的工具辅助扩展，不是论文原版裸 CIF 输出。

官方 correctness、`target_cif`、RMSD、错误类型以及 LDM 默认选出的 oracle parent 都不会进入 proposal prompt。
expander 只拥有公开题目；它不读取 `ExpansionRequest.observations/parent/acquisition_feedback`。
公开语法检查不使用 target。运行目录含离线判分和 checkpoint，未来如改为有文件系统访问能力的 Harness，必须只挂载 `public.jsonl` 和目标隔离的工具工作区；当前 direct ProposalClient 没有文件系统工具。

`result.json` 独立报告 `one_shot_accuracy`、`extended_final_accuracy`、各动作 accuracy。
`oracle_any_attempt_accuracy_diagnostic` 只是事后覆盖率诊断，不能当可部署的准确率或 one-shot 分数。
缺失、格式错误以及被拒绝的最终提交按错误计入固定分母；重复答案仍计模型请求，复用已有 evaluator 结果。
`summary.json` 是共享引擎的原始统计，可能包含逐候选最佳分数；对外 benchmark 分数读取 `result.json`。
本适配采用 target-blind 顺序 refinement，未启用基于隐藏标签的 GP/acquisition 优化。

## 数据和 evaluator

本地源码 `../../../atomworld-main` 完整提供 `src/atomworld/evaluate.py`，因此直接调用原实现，无需复写 judge。
运行时验证其 SHA-256；`resources/source_manifest.json` 同时记录 evaluator、prompt、动作实现、loader 和论文哈希。
源码快照没有 `.git`，不伪造 commit。

论文 AtomMotor-2K 的十类操作是 change、remove、add、move、move_towards、insert_between、swap、delete_below、rotate_around、super_cell，共 2500 题。
本地发布数据每动作约 1000 行，不等于已验证的论文 250×10 test split。数据脚本默认从这十类按固定 seed 各抽一题；显式记录源行号、CIF 文件名、输入/目标哈希及全部源文件哈希。
**任何本地子集结果都标记 `local_released_subset`，不声称论文分数可比。** 其余五种工具箱动作仅作为扩展动作显式选择。

准备脚本按 upstream 的 CSV+HDF5 优先、JSON 次选规则读取，不重新生成题目、不舍入数值、不读取上游模型结果文件。
输出 `public.jsonl`、权限 0600 的 `private.jsonl` 及 `manifest.json`，运行前再次核验哈希和 ID 对齐。

Judge 保留最后 `<cif>…</cif>` 标签提取、非 primitive cell 解析、元素数量检查与
`StructureMatcher(primitive_cell=False, stol=0.5)`；`move_all_action` 扩展保持 upstream 特殊 positional matching。
`StructureMatcher` 返回的 RMSD/max distance 是**归一化无量纲量**，虽然上游部分 docstring 标了 Å。
模型回答 CIF 错误是一次成功执行的 judge 调用、correctness=0；不是基础设施失败。

## 运行

在仓库根目录，建议使用 Python 3.11：

```bash
uv sync --locked --project tasks/atomworld --group dev
uv run --locked --project tasks/atomworld python scripts/validate_tasks.py --task atomworld
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/mock.yaml
uv run --locked --project tasks/atomworld python -m pytest -q tasks/atomworld/tests

uv run --locked --project tasks/atomworld python -m tasks.atomworld.scripts.prepare_official_data \
  --source-dir ../atomworld-main/src/data \
  --out-dir tasks/atomworld/runs/data/released_seed0 --per-action 1 --seed 0
```

`--per-action 0` 使用全部发布行；`--actions move_atom_action,rotate_around_atom_action` 选择动作。
先准备一个小子集核对，再扩展。不得覆盖已有 manifest，换新输出目录即可。

```bash
export ATOMWORLD_DATA_ROOT="$PWD/tasks/atomworld/runs/data/released_seed0"
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL_NAME="your-served-model"
# LLM_API_KEY 从已有受保护环境读取；不写入 yaml 或命令参数。
uv run --locked --project tasks/atomworld python scripts/check_task_dependencies.py config/atomworld/one_shot.yaml
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/one_shot.yaml --dry-run
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/one_shot.yaml
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/extended_reasoning.yaml
```

兼容 `LDM_LLM_URL/MODEL/API_KEY` 和 `OPENAI_BASE_URL/MODEL/API_KEY`，优先使用仓库标准 `LLM_*`。
`ATOMWORLD_UPSTREAM_ROOT` 可指向搬迁后的源码。`--llm-extra-body-json` 可设置 provider 支持的 reasoning 参数；参数原样记录，不对不支持的模型虚构 reasoning 能力。
真实模式先执行同一 Chat Completions 后端的短 preflight；mock 不访问网络。模型请求不开隐藏自动重试。
真实预算额外预留一次服务失败恢复请求，不增加有效答案数；preflight 单独计数（最多两次）。

工具模式的隔离 task 环境已包含 ASE；保留并列目录后执行：

```bash
uv run --locked --project tasks/atomworld python scripts/run_ldm_tts.py config/atomworld/extended_operations.yaml
```

工具只支持十种有界操作，没有任意 Python、文件访问或网络访问。输入、schema 和操作实现哈希进入 `schedule.json`。
论文提到 Python/RAG agent，但本地没有完整 agent 执行代码；并列工具是有限功能复现，未声称恢复论文全部系统。

## 产物、恢复和验证状态

运行输出共享 `campaign.json/events.jsonl/checkpoint.json/status.json/budget.json/summary.json`，以及
`result.json/trajectory.csv/attempts/*.json/evaluations/*.json/dataset_manifest.json/schedule.json`。
预算分别计 outer rounds、请求、proposal attempts、有效候选、selected candidates、判分调用和 geometry tool calls，含零值。
调用数和输出 token cap 是延长推理比较的额外计算预算；报告基线/延长的预算差异。

恢复请使用终端打印的**实际 run_dir**，不要用自动加后缀之前的路径：

```bash
uv run --locked --project tasks/atomworld python -m tasks.atomworld.ldm_task.procedure \
  --data-dir "$ATOMWORLD_DATA_ROOT" --attempts-per-sample 4 \
  --out-dir /absolute/path/to/printed/run_dir --resume
```

恢复要求数据、model、token/temperature/extra body、proposal format 和工具源码哈希一致。已完成模型回答会从 attempt 记录恢复，已完成 campaign 不重跑。
服务失败写 `paused_endpoint`；恢复前修好 endpoint。共享引擎负责预算、checkpoint 与 dedup，不由 adapter 重写生命周期。

启用 `LDM_DATA_COLLECTION_ENABLED=1` 后，只在 CIF 解析通过的 accepted-answer 边界收集 `ldm-2.0` IR；不收错误输出。
collection provenance 和 judge 结果不进入 SFT prompt。operations 模式保留原始计划/工具轨迹，尚未定义将工具衍生 CIF 当模型动作的 IR 转换，因此不冒充 CIF 模型训练样本。

当前 `experiment.json` 保持 **draft**；注册与本地验证已完成，但源 archive 无 Git tracking、正式 paper split 和真实 endpoint campaign 证据不足。
正式 qualification 文件暂留 scaffolded，实际完成的本地测试记在 `resources/local_verification.json`，与正式追踪证据门槛分开。
mock 和确定性客户端+真实 judge 测试都不是 LLM benchmark 分数。SSH 调试结果及总体阻塞见仓库 Science Benchmark 接入报告。
