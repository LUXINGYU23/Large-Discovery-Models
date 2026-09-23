# AlphaBench T3

目标是完整 T3，包括全部初始化、双方言、三种原生算法、五种 LDM/Agent 方法、私有 validation、冻结后完整 test、组合分析和全市场验收。目前处于实施中，`qualification=draft`；未完成的能力不会映射到其他算法或后端。

[数据准备](DATA.md) 包含官方数据资产核查、本机代理下载、固定来源/哈希、SSH 传输和服务器离线审计。所有数值测试和实验在指定服务器的 `/mnt/data1/` 下执行。

## 已实现的路径

`llm` 与 `ldm` 的共享 Campaign 生命周期已接线：独立初始化、完整表达式准入、最多五次修复、真实请求预算/receipt、search→private validation、固定选择→逐因子 full test、独立组合、质量审计及 accepted-action 训练数据导出。Qlib worker 复用固定 FFO 指标和 TopkDropout 回测。[Assay worker](ASSAY.md) 使用固定官方算子和独立回测器，补齐历史成分、逐日分类、实际基准及离线资产校验。

原生 CoT/ToT/EA 的 [workflow](NATIVE.md) 已接入固定算法、原生 cold/修复、整批预算、并发重放、私有 validation 和终局报告。Qlib/Assay searcher source 与 example source 使用实际原生最终池，matched 使用完整已测池；source 保留原生轮数，预算不足即暂停。example 原入口没有 validation/test，本 task 明确冻结独立的后续区间。Assay source 固定原始面板、开盘标签、复权及诊断规则，完整独立 portfolio 在初始化前校验。

`harness` 与 `ldm_harness` 已接入持久 Pi session、发送前模型请求授权、Host 工具服务和隔离 guest。前者按提交顺序评价，后者合并各 session 的候选并用冻结 GP 选择；GP 查询只在显式启用的 LDM Harness 变体开放。`ldm_harness_compiled` 另启独立 policy session，在受限 Docker runner 中执行其提交的 prior/weight 策略；Host 仍固定 GP、选择、Oracle 和预算。实际 DeepSeek Responses + synthetic Oracle 已分别完成两轮 direct、LDM 和 compiled smoke，证据见 `resources/evidence/`。这些运行不代表真实市场或完整方法矩阵资格。真实数据、Assay portfolio REST 接口、故障矩阵和完整资格矩阵仍未关闭，因此 `complete_t3=false`。

## 服务器验证

从实际实现仓库根执行：

```bash
export UV_CACHE_DIR=/mnt/data1/Large-Discovery-Models/cache/uv
uv sync --locked --project tasks/alphabench --group dev --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv run --locked --project tasks/alphabench python -m pytest tasks/alphabench/tests -q
uv run --locked --project tasks/alphabench python scripts/run_ldm_tts.py config/alphabench/mock.yaml
```

真实运行要求完整冻结的 `--protocol-file`、已复核的 `--data-manifest`、固定 `--upstream-root` 和受控 `--oracle-url`。
模型固定为 DeepSeek Responses API 的 `deepseek-flash`、`reasoning.effort=max`；Host 从 `--api-key-file` 或 `DEEPSEEK_API_KEY` 读取凭证。凭证不写入配置、报告或 guest。Harness 还需要本地构建的 Pi sidecar 和指定的 KVM guest 镜像；`--harness-sidecar-image` 默认 `ldm-pi-t3:local`。

`--resume-run` 仅恢复同一不可变协议；`--init-mode import_pool` 导入表达式并重新评价，不能替代恢复。
未知请求结果暂停并对账，不能伪造失败分数或隐式重发。

## 产物

- `protocol.json`、`campaign.json`、`budget.json`、`events.jsonl`、`checkpoint.json`：冻结身份、预算、共享事件和恢复视图。
- `initialization/`：独立初始化预算、种子清单和观测。
- `private/`：Host 私有 model/oracle receipts、validation 和终局阶段记录。
- `selection_frozen.json`：任何 test 前冻结的 validation 排名、因子集合、方向和持仓参数。
- `result.json`、`report.md`、`trajectory.csv`：search、full test、组合、质量/多样性、预算与逐阶段完整性。
- `generation/`、`generation_attempts/`：直接/原生生成原始项与 Harness 每次提交的不可变账本；后者单独记录格式失败和修复项。
- `accepted_actions/`：不可变 accepted action；IR/SFT 成对原子发布在数据收集目录，`current.json` 指向完整 generation。
- `harness/`、`policy_harness/`：分离的 proposal 和 compiled-policy 会话、不可变输入、策略 epoch、预测、trace 与 usage。

原始行情、环境、运行目录和完整 trace 不提交 Git。紧凑的来源与开发验证证据保留在 `resources/evidence/`。

`protocol.validation_metric` 独立于 search objective，可冻结为 `rank_ic/ic/icir/rank_icir`，默认 `rank_ic`。最终池与 test selection 分别报告结构和信号多样性：前者复用全部已测成员的私有 validation scores，后者使用冻结集合的 test scores。每个信号 pair 保留有效样本数、样本索引 digest 和未定义原因；成员缺 scores 时整个集合的信号多样性不可用。AST 距离去常数、保留实际运算符，归一化使用该集合的最大成对距离。

`result.json` 的 `test[].raw` 与 `independent_combination` 原样保留后端 daily、scores、portfolio、holdings、actions 和未规范化的 turnover；`test[].derived` 另记无年化、ddof=1 的 ICIR、固定方向 WinRate 与中心矩 skewness。`search.ea_update` 从每轮冻结的 top pool 计算 `U_t` 和实际 `T`，与 best-factor 更新事件分开。`quality_audit` 的分母为 raw occurrence，格式失败单列；覆盖不足时通过率为 null。`completeness` 每阶段记录状态和原因，`complete_t3=false` 不因 mock 成功而改变。`report.md` 仅从 `result.json` 渲染。

`trajectory.csv` 以新 search evaluation attempt 为主轴；`physical_search_jobs` 是该次 Oracle search 响应的实际 jobs，`oracle_elapsed_seconds` 是该响应耗时，累计耗时是响应耗时之和，不是墙钟时间。`model_requests_run_total` 和 `tool_calls_run_total` 是整次 run 的成本，每行重复展示而非逐候选归因；sidecar 缺 usage 时保留 null。Harness 的 Search Cost 用每个已提交 session 的实际 provider calls 表示，与直接/原生方法每步最多五次修复的单位分开。

CoE/ToT 多 run 的 FracSuccess 只从预先保存的 roster 聚合。roster 的 `schema_version=1`，包含 `method`、`backend`、`market`、`profile`、`invalid_rule`（`count_as_failure` 或 `exclude_with_evidence`），以及 `runs` 数组；每项含唯一 `run_id`、整数 `seed`、冻结的 `protocol_digest` 和 `run_dir`。不同 run 除随机种子外须使用相同协议。缺失结果和模型生成失败留在分母；基础设施无效须在独立的 invalidations JSON 中以 run ID 指向非空证据文件。相对运行目录相对于 roster 所在目录，相对证据文件相对于 invalidations 所在目录。产物记录原始清单、结果与证据的 SHA-256、逐 run 纳入决定；`aggregation_qualified=false` 的比率仅用于诊断。

```bash
tasks/alphabench/.venv/bin/python -m tasks.alphabench.aggregate \
  --roster /mnt/data1/alphabench-t3/roster.json \
  --invalidations /mnt/data1/alphabench-t3/invalidations.json \
  --output /mnt/data1/alphabench-t3/aggregate.json
```
