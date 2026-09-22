# AlphaBench T3

目标是完整 T3，包括全部初始化、双方言、三种原生算法、五种 LDM/Agent 方法、私有 validation、冻结后完整 test、组合分析和全市场验收。目前处于实施中，`qualification=draft`；未完成的能力不会映射到其他算法或后端。

[数据准备](DATA.md) 包含官方数据资产核查、本机代理下载、固定来源/哈希、SSH 传输和服务器离线审计。所有数值测试和实验在指定服务器的 `/mnt/data1/` 下执行。

## 已实现的路径

`llm` 与 `ldm` 的共享 Campaign 生命周期已接线：独立初始化、完整表达式准入、最多五次修复、真实请求预算/receipt、search→private validation、固定选择→逐因子 full test、独立组合、质量审计及 accepted-action 训练数据导出。Qlib worker 复用固定 FFO 指标和 TopkDropout 回测。[Assay worker](ASSAY.md) 使用固定官方算子和独立回测器，补齐历史成分、逐日分类、实际基准及离线资产校验。

原生 CoT/ToT/EA 已有[固定源码加载、并发日志重放和进程恢复组件](NATIVE.md)，正式 workflow 接线仍未完成。当前 synthetic 检查不代表真实市场资格。Assay 真实数据和市场资格、持久 Harness、compiled policy、原生 source profiles 与完整资格矩阵仍未关闭。数据缺口及未完成能力使 `complete_t3=false`。

## 服务器验证

从实际实现仓库根执行：

```bash
export UV_CACHE_DIR=/mnt/data1/Large-Discovery-Models/cache/uv
uv sync --locked --project tasks/alphabench --group dev --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv run --locked --project tasks/alphabench python -m pytest tasks/alphabench/tests -q
uv run --locked --project tasks/alphabench python scripts/run_ldm_tts.py config/alphabench/mock.yaml
```

真实运行要求完整冻结的 `--protocol-file`、已复核的 `--data-manifest`、固定 `--upstream-root` 和受控 `--oracle-url`。
模型固定为 DeepSeek Responses API 的 `deepseek-flash`、`reasoning.effort=max`；Host 从 `DEEPSEEK_API_KEY` 读取凭证。凭证不写入配置、报告或 guest。

`--resume-run` 仅恢复同一不可变协议；`--init-mode import_pool` 导入表达式并重新评价，不能替代恢复。
未知请求结果暂停并对账，不能伪造失败分数或隐式重发。

## 产物

- `protocol.json`、`campaign.json`、`budget.json`、`events.jsonl`、`checkpoint.json`：冻结身份、预算、共享事件和恢复视图。
- `initialization/`：独立初始化预算、种子清单和观测。
- `private/`：Host 私有 model/oracle receipts、validation 和终局阶段记录。
- `selection_frozen.json`：任何 test 前冻结的 validation 排名、因子集合、方向和持仓参数。
- `result.json`、`trajectory.csv`：search、full test、组合、质量/多样性、预算与资格缺口。
- `accepted_actions/`：不可变 accepted action；IR/SFT 成对原子发布在数据收集目录，`current.json` 指向完整 generation。

原始行情、环境、运行目录和完整 trace 不提交 Git。紧凑的来源与开发验证证据保留在 `resources/evidence/`。

`protocol.validation_metric` 独立于 search objective，可冻结为 `rank_ic/ic/icir/rank_icir`，默认 `rank_ic`。最终池与 test selection 分别报告结构和信号多样性：前者复用全部已测成员的私有 validation scores，后者使用冻结集合的 test scores。每个信号 pair 保留有效样本数、样本索引 digest 和未定义原因；成员缺 scores 时整个集合的信号多样性不可用。AST 距离去常数、保留实际运算符，归一化使用该集合的最大成对距离。
