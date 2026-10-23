# 铜矿任务最终逐项审计

审计对象是 spec v1.2 的 P0–P5 工程交付；正式 benchmark 资格、SFT 训练、多 seed 和 surrogate 效果优势不属于本次完成声明。资格适用性见 [资格边界](copper-inversion-qualification-scope.md)。

本轮发现并修复了 runner 缺少输入前后核对的问题，新增直接故障用例。当前状态为**P0–P5 工程验收完成**。最终任务回归 98 项通过；当前协议 P5 三轮、完成态恢复、旧节点历史归因和最终 runner 的 28 份算法复评均已有直接证据。注册资格仍按独立规则保留 draft。

## 需求与直接证据

下表的测试路径均相对 `tasks/copper_inversion/tests/`；实现路径相对 `tasks/copper_inversion/core/`。测试在授权 H20 执行，本机仅编辑和静态检查。

| spec 项 | 必须成立的事实 | 已检查的实现和证据 | 当前结论 |
| --- | --- | --- | --- |
| §1 | Candidate 是完整算法，执行产物才是矿体；取消原 Monte Carlo 链和 controller | `candidate.py`、`runner.py`、`evaluator.py`；可信 core/adapter 搜索无 shpmc、GaussianField、Metropolis、ridge、rescue 调用 | 已实现 |
| §2、§13.1 | 必需数据可独立部署、来源 ZIP 和逐文件摘要、代码许可与数据声明 | resources/provenance.json、LICENSE.upstream、H20 从仓库资源部署并实际运行；BGRSI 固定参考 commit/blob 记录 | 已验证；官方 benchmark 资格不适用 |
| §3 | 三个 case、网格、标签 mask、接触与观测数量正确 | `data.py` 使用 discretize UBC loader；`test_handwritten_ubc_coordinate_fixture` 手写非对称网格验证 x/y/z 与 z 反向，`test_actual_data` 验证真实原点/空气/接触计数 | 已验证 |
| §3.1、§13.2 | logistic/contact 和 Procrustes 科学公式与参考一致 | `objective.py`、`test_binary_contacts`、固定数值回归、docs/evidence/copper-inversion/reference-check.json | 已验证；历史 C/F 存储错误的修复与旧数值分开记录 |
| §3.2 | 有限浮点、正负区域、无 masked/object/complex；统一 C-contiguous 距离重建 | `geometry.py`；平面解析解、实际数据 C/F 一致性、椭圆/椭球缩放测试 | 已验证 |
| §4 | 单文件 UTF-8、1 MiB、solve 入口、源码字节 SHA256；禁止语义去重承诺 | `validate_source`、`seal`、`parse_snapshot`；`test_snapshot_contract`、原生数组提交证据 | 已验证 |
| §4 | problem 数组只读且执行前后检查 | `read_problem` 设置只读；新 `runner.input_digest` 检查数组内容、shape/dtype、只读标志和嵌套元数据 | 已修复，98 项远程任务回归通过 |
| §4 | budget 提示与实际强制限制分离；自报分数不可信 | `runner.py` 只写 model；execution 强制限制；伪造 result.json 仍由 scorer 得到真实 loss 的故障测试 | 已验证 |
| §4 | 严格 candidates 数组、数量、路径、父身份、重复和摘要检查 | `test_snapshot_contract`、`test_domain_rejects_link_and_tampering`、真实 Pi SubmissionController/Python parser 记录 | 已验证 |
| §5.1 | 公开自检独立于正式准入，研究资源不计正式评分槽 | public/selfcheck 与工具扩展；Pi `/workspace/.ldm-resources` ReadonlyProvider 与 sidecar `/resources` 只读挂载；研究 bash 用量和正式 jobs 分账 | 已验证 |
| §5.2 | 全池静态检查先于接口；错误分支可修复；不发布半批 | `validate_batch`、两个静态错误测试、部分屏障恢复测试、当前原生 tiny 的快照/派发时间 | 当前 tiny 与完整三轮均验证通过 |
| §5.2 | 准入父进程不 import 候选，回执随源码/实现/数据/镜像/预算变化 | `InterfaceChecks`、`Execution.run`、source_identity；封存修改、Docker 超时暂停和回执恢复测试 | 已验证；依赖以镜像内容身份及受跟踪 recipe/lock 固定 |
| §5.3 | 干净、禁网、只读源码/输入、无凭证/socket，CPU/内存/PID/deadline 限制 | execution 的精确 Docker 参数、core-only 挂载测试、supervisor UID/capability 与断连测试 | 已验证 |
| §5.3 | 停止后代后收取普通 model.npy；拒绝链接、超限、异常 header/shape/dtype/pickle | `supervisor.py`、`freeze_model`、`read_model`、`test_bounded_npy`、链接/非法输出测试 | 已验证 |
| §5.3 | scorer 无候选源码，仅冻结矿体、观测与科学代码；由宿主发布 | evaluator/execution 分离挂载；伪造评分及 scorer receipt 故障测试；16 份历史源码监督器复评 | 已验证 |
| §6 | 入口、2D/3D、稀疏标签、无接触点、异常几何、距离场、loss、只读、执行故障、恢复 | test_science、test_isolation、test_submission、test_recovery、test_supervisor；新增 import 时退出/Inf/全负/复数/错误类型形状用例 | 98 项任务回归已覆盖新增用例 |
| §7.1 | 一个 runtime/engine，注册 seam，退出清理，无第二套框架状态 | workflow 一次 open_runtime、ExitStack、LDMEngine；procedure 浅层导出；共享库无 copper 任务导入 | 已检查；共享 Campaign/Engine 回归已有 228 项证据 |
| §7 单 agent | 一个持久 researcher，基线＋三轮、权威历史/源码/切片反馈 | 原 2D campaign 与当前两个 3D Flash/low 三轮；全部成功，完成态恢复零请求 | 已验证 |
| §7 surrogate | 256 维 AST 代码特征，不读分数/理由；共享 RBF GP/LCB，最小 loss 与最大 acquisition 对齐 | surrogate.py 固定参数；test_surrogate 覆盖无执行、无泄漏、冷启动、拟合错误、无 fallback、四选一 | 已验证；真实 tiny acquisition 逐项核对 |
| §7.2 | Coverage 2+2、independent/latest-main、禁 measurement policy | test_research 三种策略；共享 configure_research_strategy/ResearchCoordinator/MeasuredResearchExpander | 已验证 |
| §7.2 | 12 份提案、每轮选 1、4 次正式测量、15 turn；未测量者无假分数 | 当前完整三轮：12 提案、15 turn、26 接口 job、4 次正式测量；final-full.json | 当前版本三轮已验证 |
| §7.2、§14 | history 前缀稳定、旧节点增量历史、profile/source 归属及 utility 符号 | test_research、共享 tree 回归、旧版真实三轮 panel/provenance；末轮不追加预算外总结 | 当前版本三轮已验证 |
| §7.3、§13.5 | 共享评估预留/发布；任务仅持久子作业回执；未知结果暂停不重派 | evaluator usage_key；consume_many 稳定 job key；两 3D case 在 solver/scorer/evaluator/publication 四处中断恢复 | 已验证 |
| §8 | 子作业、正式尝试、主/分支 turn 分账，失败用量保留、恢复不重复 | test_recovery 精确 8 容器和一次 turn 断言；单 agent/tiny 实账与完成态恢复 | 已验证；三轮实账与完成态恢复核对通过 |
| §8 | 真实峰值、CPU、墙钟，与候选 stdout 和宿主采样分离 | H20 228 项回归；96 MiB 瞬时后代、禁止读取 PID1 key、伪造 MAC、断连 deadline 测试 | 已验证 |
| §8 | 面积/体积/空气诊断不改变 loss；无 mask 不伪造 0 | test_physical_measure_and_air_diagnostics_do_not_change_loss；16 份历史复评 loss 差值 0 | 已验证 |
| §8 | 切片来自实际评分矿体、保留坐标/索引/seed/hash，不作为评分 | feedback.py、test_feedback；实际独立评估→serialize_history 测试 | 已验证 |
| §8–10 | 配置/契约/镜像/源码/准入/selection/模型/日志/预算/结果/最佳算法可追溯 | run 源码归档、原生记录、result.json/trajectory.csv/模型相对路径；runner 锁定 named profile | 完整三轮和最终 28 份复评归档均已保存 SHA256 |
| §10、§13.4 | 工程预算强制执行，baseline 计预算，single-seed 47，非 official | experiment.json profiles、workflow budget、配置 dry-run、真实运行 counters | 已验证 |
| §11、§13.3 | 授权远程、Docker/KVM/依赖、实际 Responses preflight、无密钥泄露 | H20 预检、镜像内容 digest、原生请求核对、归档前密钥扫描 | 完整三轮原生请求和归档凭证扫描已核对 |
| §12 P0–P5、§14 | 逐阶段科学/契约/mock/真实 RSI/恢复/研究树证据，不超范围宣称 | 上述分项、验收台账、注册资格边界 | P0–P5 工程证据已闭合；不升级官方 benchmark 资格 |

## 证据索引与时间边界

- [历史 2D/3D 与最终监督执行器复评](copper-inversion-geometry-diagnostics.md)
- [监督器及 228 项远程回归](copper-inversion-resource-accounting.md)
- [当前单 agent 两个 3D case 各三轮](copper-inversion-flash-low-rerun.md)
- [当前数组与整批准入协议、真实 tiny](copper-inversion-batch-admission.md)
- [旧版真实 P5 三轮](copper-inversion-p5-results.md)
- [注册资格适用性](copper-inversion-qualification-scope.md)

P5 full 在冻结目录 `copper-inversion-20261023` 完成；输入前后核对在独立 `copper-final-audit-20261023` 验证。后者未混入前者运行。最终 runner 对全部 28 份历史及本次实际评分源码复评，损失全部一致；原模型轨迹保留其原身份。运行相关文件仅 runner.py 的输入核对与 masked 输出拒绝发生变化，完整差异、最终源码身份与复评结果均已存证。

最终直接证据：[三轮 P5](../tasks/copper_inversion/resources/evidence/research-tree/final-full.json)、[98 项回归](../tasks/copper_inversion/resources/evidence/runtime/final-acceptance.json)、[测试日志](../tasks/copper_inversion/resources/evidence/runtime/final-acceptance.log)、[28 份复评](../tasks/copper_inversion/resources/evidence/runtime/final-input-replay.json)、[最终结果](copper-inversion-final-results.md)。
