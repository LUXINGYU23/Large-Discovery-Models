# Copper inversion spec 验收台账

依据：[完整 spec](copper-inversion-spec.md)，工作分支 `codex/copper-inversion`。
本台账区分已有证据、待补证据和未实现能力；只有全部必需项得到直接证据后才宣告完成。

## 已有证据与剩余验收

| spec 要求 | 当前实现/证据 | 状态与下一步 |
| --- | --- | --- |
| §1–3 / P0：取消旧 Monte Carlo 路径，独立数据、坐标、科学公式和来源 | `core/data.py`、`geometry.py`、`objective.py`，来源 manifest、`tests/test_science.py`，implementation 中科学对照 | 已有远程证据；最终回归重新覆盖 |
| §4–6 / P1：封存代码、只读准入、无网络执行、独立评分 | `candidate.py`、`interface_checks.py`、`execution.py`、`evaluator.py`，`test_submission.py`、`test_isolation.py`、`test_recovery.py` | 隔离、子进程/OOM、评分回执损坏已验证；采样内存/CPU 已实现；完整生命周期峰值尚未证明 |
| §7 / P2：一个共享 runtime/engine、注册和 mock | `workflow.py`、`test_procedure.py`、注册 contract_verified | 已有远程证据；P5 改造后需回归单 agent 路径 |
| §7 / P3：真实代码设计→评分→反馈→修改 | 2D baseline＋三轮记录；3D 两 case 的两种 provider 记录 | 已验证单 agent 路径；原生轨迹为研究记录，不宣称已转为 SFT |
| §7.3、§8、§14 / P4：solver 后、scorer 后、evaluator 返回后、发布后 checkpoint 前恢复 | `tests/test_recovery.py` 在两个真实 3D case 注入四个中断点；[远程证据](../tasks/copper_inversion/resources/evidence/recovery/verification.json) | 已验证；每组恰好 8 个实际容器、2 次正式评估和 1 个研究 turn，恢复和完成态重放不增加 |
| §7.2 / P5：Coverage 2+2 / independent / main_forks | `harness_runtime.open_search` 复用共享策略、协调器和 `MeasuredResearchExpander` | 已实现，三种策略远程 mock 通过；真实 Pi 三轮已完成，见 P5 结果记录 |
| §7.2 / P5：四份源码整批准入、跨分支去重、部分屏障恢复 | `Expander` 按分支 artifact root 封存，持久准入回执保留源码归属，整批返回 sessions/lineage | 远程验证重复拒绝、部分屏障恢复不重跑已接受分支；真实 native fork 已完成；整批先语法后接口执行的顺序仍需补齐 |
| §7.2 / P5：代码编码、共享 GP、最小化 acquisition | `core/surrogate.py` 复用 `RBFGPSurrogate` 和 `make_acquisition`；[组件证据](../tasks/copper_inversion/resources/evidence/surrogate/verification.json) | 已接入 `ldm_harness` workflow；mock 验证三轮 12 份候选仅选中 3 份，连同基线共评分 4 份；真实三轮已完成，四次正式测量 |
| §7.2、§14 / P5：utility 符号与历史归因 | 共享 wrapper 使用 `research_utility`；history 序列导出测量源码，分支采用继承 cursor | mock 覆盖负 utility、旧节点增量历史；真实归因已核对；末轮 credit 按共享语义待下一 step |
| §8–10：版本化契约、预算、最佳代码及实际评分矿体 | 新增 `ldm_mock`、`ldm_3d_tiny`、`ldm_drillholes_3d`，P5 锁定 Coverage 和模型预算 | pool=4/select=1，正式 4 次、研究 turn 15 次；真实执行与完成态恢复已验证 |
| §11–14：远程、注册 gates、完整产物和资格声明 | 现有远程归档与 evidence；qualification 为 contract_verified，experiment 为 draft | 最终验收需补 P4/P5 证据并保持工程结果与方法有效性分开 |

## P5 实施顺序

1. 使用固定版本的代码特征，直接调用共享 RBF GP 与最小化 acquisition；增加方向、非有限值、冷启动和无标签泄漏测试。
2. 将单会话提交适配器扩展为共享研究树可调用的分支适配器。每分支仍只提交一个 solver，保持原封存/准入/正式评分边界。
3. 在唯一 runtime 内装配共享协调器和 `MeasuredResearchExpander`，基线 round 0，active round 从 1 开始；关闭 measurement policy。
4. 远程 mock/故障注入覆盖策略、增量历史、去重和部分屏障恢复；通过后用官方 DeepSeek Flash / low 做真实四分支工程运行。
5. 三轮共生成 12 个候选，但只正式评分 3 个选中候选及 1 个基线。记录全部准入和选点证据，不把未测候选的预测当成正式成绩。

本台账不新增 SFT 训练或官方 benchmark 认证要求；spec 要求保留轨迹收集边界，并明确工程验收不能证明 surrogate 有效性。

## 2026-10-23 最终审计进展

- [P5 真实三轮及恢复证据](copper-inversion-p5-results.md)：12 份准入候选、每轮四选一、共 4 次正式测量；232 条官方 Flash/low 请求，完成态恢复新增请求 0。
- 独立远程 runtime 回归 217 项通过，覆盖控制器离线后容器内硬 deadline、宿主采样、已评分矿体切片、反馈历史和既有恢复路径。见 `resources/evidence/runtime/verification.json`。
- [BGRSI 固定版本复核](evidence/copper-inversion/bgrsi-reference.json) 已记录 commit 与文件 blob 身份。
- 物理面积/体积与空气占用已实现并通过远程 79 项任务测试；无 active_mask 时不虚构空气指标，修改诊断不改变 loss。见 [几何诊断证据](../tasks/copper_inversion/resources/evidence/runtime/geometry-diagnostics.json)。
- 当前执行器对既有 2D 四份、两组单 agent 3D 八份和 P5 四份正式源码完成复评，16 次损失均与历史一致，结果重读不重跑。见 [复评记录](copper-inversion-geometry-diagnostics.md)。
- 跨分支整批语法/身份检查先于全部接口执行已实现；共享客户端增量批量 validator 不改变原生协议，远程 225 项回归通过。见 [批量准入记录](copper-inversion-batch-admission.md)。真实多分支运行尚须使用新冻结实现另行验证。
- 未完成：完整生命周期峰值内存；最终版本真实端到端验收、完整复评及注册资格适用性审计。
- §4 提交契约已统一为严格 candidates 数组，每 researcher 恰好一项，artifact pointer 为 `/candidates/0/artifact_path`；旧扁平对象被拒绝。远程 220 项 Python 回归通过，另用真实 Pi SubmissionController 与 Python parser 验证拒绝、重开 turn 修复、封存和摘要。见 [契约证据](../tasks/copper_inversion/resources/evidence/runtime/array-submission.json)。这不代表跨分支整批语法优先屏障已完成，也未把历史真实运行改标为新契约。
