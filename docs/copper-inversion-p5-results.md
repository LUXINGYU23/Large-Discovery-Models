# P5：真实共享研究树与 GP 选点

2026-10-23，在远程 `111.2.199.31:5320` 执行。代码为 `39a2789c` 阶段冻结实现，完整身份见 [full.json](../tasks/copper_inversion/resources/evidence/research-tree/full.json)。后续资源诊断与矿体切片改动没有写入本次活动目录。

## 配置与结果

- `ldm_drillholes_3d` 命名契约；官方 DeepSeek `deepseek-flash`、low，Pi Responses。
- 弱基线加三轮；每轮主研究会话及四个共享 Coverage 分支，每分支一个 solver，四选一。
- AST 256 维代码特征，共享 RBF GP，最小化 acquisition；不使用自报损失作为训练标签。
- 单次正式 solver：2 CPU、4 GiB、120 秒；独立 scorer。所有实验在远程执行。

| 正式测量 | loss_total |
| --- | ---: |
| 基线 | 405.0795810830953 |
| 第 1 轮 | 6.522992427679737 |
| 第 2 轮 | 6.519173431804898 |
| 第 3 轮 | 6.516789436652854 |

总墙钟 1406.006 秒（23.43 分钟）。实际 3 个主研究 turn、12 个分支 turn、12 份通过准入的不同源码，连同基线共 4 次正式测量。每轮预测数为 4，仅选中项进入正式 evaluator；基线直接测量。全部 232 条原生请求均为 `deepseek-flash` / low，完成响应也报告该模型。

三轮 panel、fork plan、会话 cursor、预测与选点、负 loss 的研究 utility、末轮树快照均保存在机器可读记录中。末轮测量的 tree credit 需下一个研究 step 才刷新，这是共享树既有语义；没有额外增加一轮来伪造收尾归因。完成态原样恢复后计数不变、无新增模型请求。

这证明共享研究树和 GP 选点的工程路径可运行，不证明 GP 提升了算法发现效果。与单 agent 对照的研究预算不同，不将两者损失差异解释为策略优劣。

## 归档与限制

原始目录保留在远程 `copper-inversion-20261023/runs/p5-{tiny,full}-v1`；完整压缩归档存入本地忽略目录 `runs/research-tree-20261023/`，复制前检查没有受保护 API key 明文。

| 归档 | SHA-256 |
| --- | --- |
| p5-tiny-v1.tar.gz | `624f2d4659f2319efcd621d7a84d58197ac6793b6cf6f13d71a187212034023b` |
| p5-full-v1.tar.gz | `7df7be8816430dd90946bb8c7e211016dbb7950c82a2d33d5129b35157a64acb` |

恢复源码时须同时解压 run 内 `source-snapshot.tar.gz` 和 `bootstrap-source.tar.gz`：前者遗漏根 pyproject.toml 与 tasks/__init__.py，后者补齐，原档未改写。新的独立测试目录还需 sidecar package.json/package-lock.json 用于版本一致性测试。

后续 runtime 的 deadline、宿主 cgroup 采样与矿体反馈已在独立远程目录完成 [217 项回归](../tasks/copper_inversion/resources/evidence/runtime/verification.json)。完整 spec 尚需逐项审计；当前不能将各分支分别准入等同于跨分支整批先语法后执行，也不能把采样内存最大值称为生命周期峰值。

## 最终协议的三轮验收

上述记录保留旧实现身份。当前数组提交、全池静态准入和可信资源监督执行器的三轮运行已完成：`405.079581 → 6.521733 → 6.519578 → 6.519067`，15 个研究 turn、12 份提案、4 次正式评分、182 次官方 Flash/low 请求，完成态恢复新增请求 0。

最后增加的 runner 输入核对在独立目录验证，并对本次四份实际评分算法及历史算法共 28 份复评，损失差值全部为 0。详细边界、归档与证据见 [最终结果](copper-inversion-final-results.md)。
