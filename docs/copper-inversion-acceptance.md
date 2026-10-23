# Copper inversion spec 验收台账

依据：[完整 spec](copper-inversion-spec.md)，工作分支 `codex/copper-inversion`。
本台账区分已有证据、待补证据和未实现能力；只有全部必需项得到直接证据后才宣告完成。

## 已有证据与剩余验收

| spec 要求 | 当前实现/证据 | 状态与下一步 |
| --- | --- | --- |
| §1–3 / P0：取消旧 Monte Carlo 路径，独立数据、坐标、科学公式和来源 | `core/data.py`、`geometry.py`、`objective.py`，来源 manifest、`tests/test_science.py`，implementation 中科学对照 | 已有远程证据；最终回归重新覆盖 |
| §4–6 / P1：封存代码、只读准入、无网络执行、独立评分 | `candidate.py`、`interface_checks.py`、`execution.py`、`evaluator.py`，`test_submission.py`、`test_isolation.py`、`test_recovery.py` | 隔离、子进程/OOM、评分回执损坏已验证；峰值内存等资源诊断仍需补齐 |
| §7 / P2：一个共享 runtime/engine、注册和 mock | `workflow.py`、`test_procedure.py`、注册 contract_verified | 已有远程证据；P5 改造后需回归单 agent 路径 |
| §7 / P3：真实代码设计→评分→反馈→修改 | 2D baseline＋三轮记录；3D 两 case 的两种 provider 记录 | 已验证单 agent 路径；原生轨迹为研究记录，不宣称已转为 SFT |
| §7.3、§8、§14 / P4：solver 后、scorer 后、evaluator 返回后、发布后 checkpoint 前恢复 | `tests/test_recovery.py` 在两个真实 3D case 注入四个中断点；[远程证据](../tasks/copper_inversion/resources/evidence/recovery/verification.json) | 已验证；每组恰好 8 个实际容器、2 次正式评估和 1 个研究 turn，恢复和完成态重放不增加 |
| §7.2 / P5：Coverage 2+2 / independent / main_forks | 共享 `configure_research_strategy`、`ResearchCoordinator`、`MeasuredResearchExpander` 已存在，铜矿尚未接入 | 未实现；复用共享协调器，不新增任务调度状态机 |
| §7.2 / P5：四份源码整批准入、跨分支去重、部分屏障恢复 | 当前 `Expander` 仅单持久会话、一份源码 | 未实现；适配 branch artifact root、session metadata 和 inherited history cursor |
| §7.2 / P5：代码编码、共享 GP、最小化 acquisition | `core/surrogate.py` 复用 `RBFGPSurrogate` 和 `make_acquisition`；[组件证据](../tasks/copper_inversion/resources/evidence/surrogate/verification.json) | 组件及共享 engine 集成已验证；256 维版本固定，基线单样本拟合，异常显式失败。尚待接入四分支生产 workflow |
| §7.2、§14 / P5：utility 符号与历史归因 | `objective.py` 成功结果含 `research_utility=-loss_total` | 需接入共享树并验证失败/未测项不归因、旧节点补全历史及末轮 credit 语义 |
| §8–10：版本化契约、预算、最佳代码及实际评分矿体 | 已有 task/experiment/config，输出含模型路径和冻结文件 | P5 需新增明确 pool=4/select=1、baseline＋3轮 profile，记录主干额外 turn 预算 |
| §11–14：远程、注册 gates、完整产物和资格声明 | 现有远程归档与 evidence；qualification 为 contract_verified，experiment 为 draft | 最终验收需补 P4/P5 证据并保持工程结果与方法有效性分开 |

## P5 实施顺序

1. 使用固定版本的代码特征，直接调用共享 RBF GP 与最小化 acquisition；增加方向、非有限值、冷启动和无标签泄漏测试。
2. 将单会话提交适配器扩展为共享研究树可调用的分支适配器。每分支仍只提交一个 solver，保持原封存/准入/正式评分边界。
3. 在唯一 runtime 内装配共享协调器和 `MeasuredResearchExpander`，基线 round 0，active round 从 1 开始；关闭 measurement policy。
4. 远程 mock/故障注入覆盖策略、增量历史、去重和部分屏障恢复；通过后用官方 DeepSeek Flash / low 做真实四分支工程运行。
5. 三轮共生成 12 个候选，但只正式评分 3 个选中候选及 1 个基线。记录全部准入和选点证据，不把未测候选的预测当成正式成绩。

本台账不新增 SFT 训练或官方 benchmark 认证要求；spec 要求保留轨迹收集边界，并明确工程验收不能证明 surrogate 有效性。
