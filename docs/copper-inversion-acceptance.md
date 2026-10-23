# Copper inversion spec 验收台账

依据：[spec v1.2](copper-inversion-spec.md)，独立工作分支 `codex/copper-inversion`。

**P0–P5 工程验收完成。** 每项要求的实现、测试、真实运行和适用限制见 [最终逐项审计](copper-inversion-final-audit.md)；数值及原始归档见 [最终结果](copper-inversion-final-results.md)。注册为 `contract_verified`，experiment 为 `draft`，不把工程闭环宣称为官方 benchmark 资格或 surrogate 效果证明。

| 要求 | 最终直接证据 | 结论 |
| --- | --- | --- |
| P0 数据、坐标、科学公式、来源 | provenance、原式对照、解析距离场、手写非对称 UBC fixture、真实 mask/接触计数 | 完成 |
| P1 封存、只读准入、无网络 worker、独立 scorer | 非法类型/形状/NaN/Inf、输入修改、import 退出、伪造分数、链接和超时/OOM 故障测试 | 完成 |
| P2 共享 runtime/engine、注册与 mock | 最终注册 gate、两种 mock 完整运行及完成态恢复 | 完成 |
| P3 2D 持久单 agent RSI | 基线＋三轮真实源码和反馈轨迹；最终 runner 复评四份损失一致 | 完成 |
| P4 两个 3D case | 当前 Flash/low 各三轮，一个持久 agent，全部独立评分与恢复核对 | 完成 |
| P4 四处故障恢复 | 两个真实 3D case 在 solver/scorer/evaluator/publication 边界注入中断；无重复派发/计数/turn | 完成 |
| P5 研究树及三种策略 | 共享 Coverage/independent/latest-main 测试，禁 measurement policy | 完成 |
| P5 整批准入、重复拒绝与部分恢复 | 当前增量批量 validator；静态错误零接口执行；接受两份后故障恢复；原生 tiny 和三轮快照顺序 | 完成 |
| P5 代码特征、GP/LCB 四选一 | 代码-only 256 维 AST，固定参数，无隐式 fallback；每轮四个 acquisition 逐项核对 | 完成 |
| P5 历史与归因 | 当前第三轮旧节点 cursor 1→3，实际补两条权威测量；credit 与负 loss 一致 | 完成 |
| 资源计量 | 可信 PID1 回收所有后代后读取/认证 memory.peak、CPU；离线 deadline、伪造 MAC 和短时峰值测试 | 完成 |
| 物理诊断与切片 | 非单位 spacing、无 mask 不伪造空气指标；实际评分矿体派生切片；不改变 loss | 完成 |
| 最终源码验证 | 98 项完整任务回归通过；此前相关共享回归 228 项通过；28 份算法最终复评 loss 差值均为 0 | 完成 |
| 完整产物、来源身份、资格边界 | 原生轨迹、源码、独立结果、模型、预算、恢复、归档 SHA256；所有 compact evidence 入库 | 完成 |

## 关键记录

- [完整 P5 三轮](../tasks/copper_inversion/resources/evidence/research-tree/final-full.json)：15 turn、12 提案、26 接口 job、4 次正式评分，182 条官方 Flash/low 请求，恢复新增请求 0。
- [最终 98 项回归](../tasks/copper_inversion/resources/evidence/runtime/final-acceptance.json)及[日志](../tasks/copper_inversion/resources/evidence/runtime/final-acceptance.log)。
- [最终 28 份算法复评](../tasks/copper_inversion/resources/evidence/runtime/final-input-replay.json)：覆盖历史 2D、历史 P5、历史与最新两个单 agent 3D case、本次 P5。
- [单 agent 最新比较](copper-inversion-flash-low-rerun.md)、[批量准入](copper-inversion-batch-admission.md)、[资源监督](copper-inversion-resource-accounting.md)、[输入完整性](copper-inversion-input-integrity.md)。
- [注册资格适用性](copper-inversion-qualification-scope.md)：不新增官方认证、SFT、多 seed 或效果优势作为本 spec 的完成条件。

P5 原生三轮在冻结源码上完成。其后 runner 增加输入核对和明确 masked 输出拒绝；没有修改活动实验或重标原轨迹。最终 runner 单独完成全部 28 份源码复评，[运行源码差异](../tasks/copper_inversion/resources/evidence/runtime/final-runtime-diff.json)与最终实现身份已保存。没有开放的必需工程验收项。
