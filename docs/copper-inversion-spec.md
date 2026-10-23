# Copper Inversion：Agent 反演算法发现任务规格

日期：2026-10-23。版本：1.1，已对齐 main PR #14。状态：第一阶段已实现并完成远程 2D 单 agent 三轮闭环；验收详情见 [实现记录](copper-inversion-implementation.md)。后续 surrogate/GP 阶段未实现。

本规格规定新增 `copper_inversion` 任务的科学问题、候选接口、可信执行边界和交付门槛。第一阶段完成持久 Agent 设计、提交、执行、读取反馈并修改反演算法的闭环；第二阶段在相同科学契约上接入 LDM surrogate/acquisition。各阶段独立记录证据，不能把注册成功、mock 运行或分数下降当成完整方法有效性证明。

## 1. 目标与范围

给定观测 D、网格 G 和计算预算 B，候选算法 A 输出矿体几何 m：

`m = A(D, G, B, seed)`；外层目标为最小化 `J(A) = L(m, D)`。

LDM 的 Candidate 是反演算法源码，矿体是其执行产物。Agent 可以设计初始化、内部表示、优化器、尺度策略和停止条件。评分公式、输入数据、几何规范化、测试、运行预算和正式结果发布由任务维护。

用户要求取消原包的 Monte Carlo 流程。新任务不得导入或调用原 MCMC 链、Metropolis 接受规则、GaussianField 驱动、LLM 参数 controller、ridge surrogate、ACQ/rescue 或 comparison/API runner。不保留旧算法兼容入口。Agent 自主编写的搜索算法不受原方法限制；本规格不额外禁止随机算法。

本次先完成 spec。后续实现和调试按第 12 节推进。本机只做开发、文档和静态检查，实际测试、候选运行、模型调用和实验放到远程。

## 2. 来源与参考边界

- LDM 实现基线：`37ebd33d`（PR #14）。使用现有 [Campaign](../ldm_tts/campaign.py)、[任务注册规范](../tasks/README.md)、[Harness 规范](research-harness.md)及[共享研究树](research-search-architecture.md)。架构收敛文档用于理解职责，下面的复用表以当前实际代码为准。
- 铜矿来源：工作区原始 `copper_ldm_3d_skill_portable/Knowledge-driven-stochastic-modeling/` 中的 examples 和 `shpmc/loss_functions.py`。迁入时生成逐文件 SHA-256 manifest，记录原 ZIP 摘要；不得依赖开发机相邻目录运行。原始包保留为来源档案，新任务仅迁入所需数据和科学计算。沿用代码的 MIT 声明及 Xiaolong Wei 版权信息；数据来源与许可单独列明，不能只凭代码许可证推断所有外部资产许可。
- BGRSI 参考为读取时的 `main`：`runtime/src/bgrsi/tasks/rsi_exam/{cases,research,interface_checks,programs,verifier}.py`。参考机制为公开自检、全批接口准入、源码封存、选后评估与回执。实现时记录所参考 commit，当前未把远程分支称为不可变源码基线。
- BGRSI 私有源码不直接复制到此仓库；按已核对的行为实现任务适配，复用本仓库共享能力。其 ControlStore、调度器和研究循环不整体迁移。

可核对的参考入口：[BGRSI 接口检查](https://github.com/LUXINGYU23/BGRSI/blob/main/runtime/src/bgrsi/tasks/rsi_exam/interface_checks.py)、[批次验证](https://github.com/LUXINGYU23/BGRSI/blob/main/runtime/src/bgrsi/tasks/rsi_exam/research.py)、[方法封存](https://github.com/LUXINGYU23/BGRSI/blob/main/runtime/src/bgrsi/tasks/rsi_exam/programs.py)。参考测试的只读挂载不等价于抗任意代码干扰；本任务增加进程外输出验证和独立评分。

## 3. 数据与科学定义

原示例说明数据为合成地质数据，标签描述侵入体/非侵入体/接触点。此任务评估二值目标体几何拟合，不声称铜品位反演、重磁正演或真实矿体恢复。

| case_id | 网格 | 数据 | loss 权重 |
| --- | --- | --- | --- |
| `drillholes_sketch_2d` | 70×50 | 35 个钻孔观测，含 2 接触点；485 个草图正像素 | drillhole 1，sketch 5 |
| `drillholes_3d` | 40×50×30 | 70 个钻孔观测，含 4 接触点 | drillhole 1 |
| `drillholes_outcrops_3d` | 40×50×30 | 同上钻孔；3530 个露头观测，含 26 接触点 | drillhole 1，outcrop 0.1 |

一个 campaign 固定一个 case。两个 3D case 共用钻孔，不作为独立泛化样本；不同 case 不混合原始 loss 排名。

输入约定：0 表示外部、1 表示内部、0.5 表示接触点，NaN 表示未观测。加载器显式输出观测 mask，禁止把缺失值转换为负标签。内部数组使用物理轴顺序：2D 为 x/z，3D 为 x/y/z；磁盘内存排列不等于物理轴方向。原 UBC 文件坐标原点、z 方向和 Fortran reshape 必须通过固定坐标 fixture 验证，不能只复制旧 runner 的 reshape 注释。

原 3D `ind_active` 有 55517 个有效单元和 4483 个空气单元。`objective_v1` 保留旧评分使用的完整网格，地形 mask 作为公开元数据和诊断，不新增裁剪或空气体素惩罚。若后续要求地形成为硬约束，创建新 objective 版本，重新评估全部候选。

### 3.1 固定 objective_v1

对标准距离场 phi，正值为内部、负值为外部。每类二值观测的 loss 为：

`c * [sum(softplus(phi[y=0])) + sum(softplus(-phi[y>0])) + mean(phi_contact)^2 + mean(phi_contact^2)]`。

这忠实保留原实现把接触点同时计为正标签的行为；`np.log2(2)` 在原实现中为 1。实现使用 `logaddexp(0, x)` 避免溢出。接触集为空时两个接触项为 0；非有限输入属于无效输出，不能吞掉并返回伪造的大 loss。

2D 草图项保留原坐标提取次序、中心化、范数归一化、补零和 SVD Procrustes 计算，权重为 5。它是形状相似约束，不是逐像素重合度。零范数或空几何须明确返回 invalid。不得静默替换为 Chamfer/IoU 或另一个点匹配算法。

### 3.2 固定几何规范化

候选返回有限浮点 level-set 场 f；几何为 f 的零等值面，要求至少一个严格正单元和一个严格负单元。允许零值，禁止 masked/object/复数数组。评估器统一以锁定版本 `skfmm.distance(f, dx=spacing)` 重建 phi；第一版正式案例均为单位网格。重建异常、非有限结果或不可用等值面标记 invalid。

重建保留离散场插值定义的界面位置，不将 f 的幅值直接用于 loss。测试正比例缩放不改变规范化评分（在固定容差内）；不得宣称任意逐点非线性变换保持界面插值。原 runner 使用已重初始化的场，迁移后再次规范化可能存在离散误差：记录固定旧模型的差异，当前任务分数不直接与旧表格混用。

几何、loss、容差和依赖版本共同组成 objective digest。客观修复导致评分变化时升版本，不能在同一观测池内修改。

## 4. 候选与执行契约

第一版候选为 UTF-8 单文件 `solver.py`，最多 1 MiB，无外部权重或伴随文件。开放完整源码，不把算法限制为预定义参数表。

```python
def solve(problem: dict, budget: dict, seed: int) -> numpy.ndarray:
    ...
```

`problem` 由可信 loader 构建，字段固定：`schema_version`、`case_id`、`shape`、`axis_names`、`spacing`、`origin`、`observations`（name/values/mask/weight）、可空 `sketch`、`initial_level_set`、可空 `active_mask`、`objective_version`。数组不可写；修改输入的检查在候选执行前后进行，但内存内保护不作为评分信任根。

`budget` 含 `wall_time_seconds`、`cpu_count`、`memory_mb`；只是算法可读取的提示，实际限制由执行器强制。初版 CPU 环境提供锁定的 Python、NumPy、SciPy、scikit-fmm；其他包必须先进入版本化镜像。研究环境可编辑和分析，正式候选不可联网安装依赖。

可信 runner 调用 solve，将结果写入 `model.npy`；禁用 pickle。标准输出只是日志。候选自报 loss、测试通过或运行时间均不作为权威数据。算法内部可以计算公开 loss，无法据此可靠计数所有实际计算，所以主预算采用固定 CPU/内存/墙钟，loss 调用数仅为可选诊断。

Harness 提交严格 JSON：`candidates` 数组，每项含 `artifact_path`、`parent_sha256`（可空）、`change_summary`、`rationale`。路径固定为 `candidates/<index>/solver.py`，拒绝额外字段、错误数量、路径越界、符号链接、摘要不符。父版本必须是当前 campaign 已知的已评估算法。父信息和理由只作研究元数据，不进入代码表示或 oracle。

候选 ID 为实际源码字节 SHA-256；不声称语义去重。相同批次及已正式评估源码禁止重复提交，未被评估的历史提案仍可再次出现。同一候选的多种子评估属于内部固定 panel，不作为多个候选。首阶段每轮恰好一份候选，第二阶段明确批次数量与去重规则后启用多候选。

## 5. 三个验证与执行边界

### 5.1 公开开发自检

Agent 可读数据、loss 规范、固定 fixture 和测试，可对当前草稿执行自检。测试目录独立只读挂载。自检记录计入开发资源，不消耗正式算法评估槽，也不直接写入 BO 观测。

### 5.2 控制器强制准入

先对整批完成语法、文件、身份及重复检查，再对全部候选运行小型接口案例。任何候选不通过则返回带 JSON Pointer 的错误，保持同一 Harness turn，最多三次提交验证尝试。通过部分的控制器回执可以缓存，但不发布半批 reservoir。

准入测试父进程不 import 候选。它在隔离 worker 中运行候选，父进程读取结果文件验证结构与几何。单靠 pytest 退出码、候选打印 PASS 或只读测试文件不构成通过。

回执 key 包含源码、fixture、测试实现、runner、镜像内容 digest、依赖锁、资源限制、seed 和协议版本。相同 key 的已完成控制器回执可复用；Agent 自检不能代替。修改任一 key 分量重新检查。基础设施故障不伪装成候选错误。

### 5.3 正式执行与独立评分

选中候选使用新的干净 worker。源码封存只读，可拷贝到临时工作目录运行；输入只读，scratch/output 可写，禁网、无凭证、无宿主 Docker socket。限制 CPU、内存、进程数、输出大小和硬超时，结束后回收整个进程组/容器。

执行结束并停止所有候选进程后，控制器收取普通文件 `model.npy`，拒绝链接、超出大小或异常 header/shape/dtype 的文件，保存内容哈希至控制器拥有的快照。读取前检查声明元素数与网格一致，再以 `allow_pickle=False` 加载。默认输出上限 2 MiB，日志上限 1 MiB；实际预算写入 profile。

独立 scorer 只挂载受控矿体快照、固定观测和固定科学代码，不挂载 solver 源码或研究工作区，不导入候选模块。它验证矿体、重建 phi、计算分项和总 loss，并由控制器发布正式回执。候选无正式评分目录写权限。

当前所有地质数据均公开，正式评分是受控复评，不是隐藏测试集。正式分数的可信性与对新地质案例的泛化是两个独立问题。

## 6. 固定测试清单

| 测试 | 输入或故障 | 通过标准 |
| --- | --- | --- |
| 入口/协议 | 缺失 solve、异常返回类型、错误 schema | 稳定错误码，拒绝准入 |
| 2D 接口 | 小型非方形 x/z 网格 | 正确 shape、有限实数、有效界面 |
| 3D 接口 | 小型 x/y/z 网格 | 同上，无轴置换 |
| 稀疏标签 | NaN、mask、0/0.5/1 | 输入被正确传递；合法输出，无缺失值传播 |
| 无接触点 | 有正负观测、无 0.5 | 可执行；评分接触项为 0 |
| 几何异常 | 全正/全负、NaN/Inf、错误维度 | 可信验证器拒绝 |
| 距离场 | 固定平面、椭圆/椭球、正比例缩放 | 符号/单位/缩放不变性在固定容差内 |
| loss 回归 | 固定场与标签、小型草图 | 总项等于分项和，符合手算/冻结参考结果 |
| 输入与输出边界 | 写只读路径、链接输出、伪造评分文件 | 输入不变，评分不采信伪造内容 |
| 执行边界 | import 时退出、无限循环、子进程、OOM | 无假通过，资源及时回收，状态正确 |
| 回执/恢复 | 提交后改草稿、重复恢复、更换测试版本 | 实际执行绑定快照；不重派已完成任务 |

候选只跑对应 case 维度的接口 fixture，不要求所有算法同时支持 2D 和 3D。通用几何、loss 与故障注入测试验证环境实现，不在每次准入时运行整个环境回归套件。测试不设质量阈值，不要求与另一算法输出相同；随机算法不强制逐字节确定性。标量回归默认 `atol=1e-8, rtol=1e-6`，距离场默认 `atol=1e-6, rtol=1e-6`，远程建立 reference 后锁定，不能为通过候选临时放宽。

## 7. LDM 装配和 RSI 反馈

`ldm_task/procedure.py` 只解析配置并调用 task workflow。当前 Harness/评估适配器需要先取得 runtime 才能装配客户端、回执和计账，因此采用注册规范允许的单 runtime、直接 LDMEngine 路径，参考 Iron Mind 的当前装配方式：

1. 构造 `CampaignRequest`、`CampaignBudget`、task spec 和冻结配置；调用 `request.open_runtime(task_spec)` 恰好一次。
2. 在 `ExitStack` 内创建任务执行资源和 Harness 客户端，构造领域 expander/domain/evaluator 及可选 encoder/selector。
3. 使用共享 `LDMEngine`，恢复 `LDMEngineState.from_checkpoint`，调用 `engine.run(request.budget.engine_config(), ...)`。
4. 从 engine/runtime 导出任务报告，在所有退出路径关闭资源。

这条路径用于 mock 和真实模式。不得随后再调用会重新打开 runtime 的 `run_campaign`，不得复制其状态、预算或评估实现。无需新增 recipe factory、bind-runtime 中转类或第二套工作流框架。task workflow 仅保留装配、加载恢复状态和报告胶水。

| 现有接口 | 任务实现 |
| --- | --- |
| ReservoirExpander | 持久 Harness 会话，加载已发布历史，提交封存代码 |
| CandidateDomainAdapter | 准入快照与回执校验，生成稳定 Candidate |
| CandidateEvaluator | 调度固定 seed panel，独立评分并返回 EvaluationResult |
| SurrogateEncoder | 第二阶段提供代码 token/AST 固定维度哈希特征 |
| AcquisitionSelector | 第二阶段复用共享 GP/acquisition，按最小化目标选择 |

第一阶段 `harness`：通过 `configure_research_strategy(..., enabled=False)` 固定 independent；拒绝要求 fork 的配置。一个持久会话，每轮一个候选，按 reservoir 顺序评估，不实例化 GP。Agent 获得最佳/上一轮源码引用、正式分项 loss、矿体切片和失败信息，能基于同一版本代码继续编辑。只能将正式已发布 observation 投影为权威历史。原生多轮 Harness 轨迹先保存为研究记录，不擅自转成单轮 SFT 样本；任务 README 说明收集边界。

第二阶段 `ldm_harness`：使用第 7.2 节共享研究树，同一接口生成多个代码候选，整批准入后选择有限数量评估。初始代码权重为均匀；代码重复频率不作为有效的概率估计承诺。编码只用代码，不含结果、理由或自报质量；编码维度、版本及核参数写入契约。代码 surrogate 效果另行比较，不能以工程连通证明有效。此阶段不引入 compiled policy/Guide 自我修改评分规则。

### 7.1 当前框架复用清单

| 已有实现 | 铜矿的使用方式 | 不再实现的内容 |
| --- | --- | --- |
| `campaign.CampaignRequest/CampaignBudget` | 创建唯一 runtime，使用统一预算与恢复配置校验 | 独立 run/config 管理器 |
| `engine/state.py` | 使用 LDMEngineConfig/State 与 observation checkpoint | 自定义轮次状态机 |
| `engine/evaluation.EvaluationExecutor` | 由 LDMEngine 调用；任务实现 CandidateEvaluator | 选中预留、结果发布、事件/checkpoint 重放 |
| `engine/run_store.CampaignRuntime` | `consume_many(..., usage_key=...)`、原子 JSON、status/events | 第二套账本、SQLite/ControlStore |
| `harness.container.pi_sidecar_command` | 装配 Pi sidecar，复用用户、缓存、KVM 和挂载规则 | 任务手写 Pi Docker 启动命令 |
| `harness.pi.PiHarnessConfig`、HarnessClient | 配置资源、提交 schema、fork_sources 和持久会话 | JSONL 通信、provider retry、原生会话保存 |
| `HarnessArtifactRule` / submission validator | 封存每份 solver 后执行任务专属准入 | 重写 snapshot/提交事务协议 |
| `research_strategy` / `research_runtime` / `research_store` | 策略解析、协调主干与分支、快照恢复和 provenance | 铜矿自己的 fork/archive/研究调度器 |
| `optimization.gp/acquisition` | 第二阶段的共享 GP 和最小化选点 | 铜矿包旧 ridge/ACQ 或复制其他任务 selector |
| `transport.openai.OpenAIProviderSettings` | 统一 provider 配置解析；实际 preflight 对齐 Pi 协议 | 铜矿专属模型凭证/HTTP 客户端 |

职责参考 `tasks/iron_mind/core/harness_runtime.py`，只复用模式和共享 API，不 import 其化学 schema、q0、特征或 oracle。

**现有框架没有替本任务完成的部分**：`EvaluationExecutor` 是评估预留/发布层，不是可执行任意 solver 的容器调度后端；`pi_sidecar_command` 是研究 sidecar 启动器，不是 solver/scorer runner。独立求解与评分的容器执行、矿体输出验证、科学 loss、准入 fixture 及子阶段回执仍需任务实现。已有 compiled-policy executor 的协议针对 policy，不以伪装 policy 的方式运行 solver。只写一个任务内 execution 模块服务准入、开发自检与正式运行，不引入通用作业平台。

### 7.2 第二阶段的共享研究树

通过 `configure_research_strategy(..., enabled=True, measurement_supported=False)` 使用共享默认 Coverage 2+2，支持已有 JSON 配置的 independent/main_forks 对照，保留参数名 `harness-research-strategy-config`。不新增策略 DSL。默认四个 proposal identity；每步一个主研究 turn，随后两个最新主干 fork、一个旧主干 restart、一个归档 lineage continuation。缺少归档时的行为由共享策略决定。

复用 `ResearchCoordinator.prepare/commit`、`main_research_profile`、`research_contract`、`PiForkSource`、`write_fork_provenance`。优先用 `MeasuredResearchExpander` 包裹铜矿的分支提交适配器，其现有契约要求：

- `serialize_history` 输出稳定、按发布顺序增长的 history，含 candidate_id、round_index 和指标；`write_history` 导出已测源码引用与结果。
- 成功结果除 `loss_total` 外添加诊断 `research_utility = -loss_total`，wrapper 的 `objective_name` 指向 `research_utility`。它按最大化给研究树归因，正式 objective/selector 仍最小化 loss_total；失败和未测量候选不提供虚构 utility。
- `proposal_key` 返回候选源码 digest，与 observation.canonical_key 一致。
- 分支 expander 保留 `harness_lineage.profile_id`，在 `metadata.sessions` 返回真实稳定 turn/session/submission 信息，供共享 wrapper 提交 panel。
- 将 fork source 的 `history_cursor` 传给分支适配器；旧节点收到全部缺失证据，不只上一物理轮。父算法 metadata 与研究树父节点分别记录，两者不是同一身份。
- 全部分支提交接受后才 commit；部分成功时沿用共享屏障和回放，禁止重建计划或再次调用已提交的分支。

初始化独占 round 0，active rounds 从 1 开始，设置 `first_active_round=1`，保持共享树 step 从 1 开始。第一阶段共四个 engine rounds：一个基线加三个研究轮。弱基线通过已有 `InitialRoundReservoirExpander` 发出，并用 `selection_mode=reservoir_order`，不占研究 turn。

铜矿提交的是多个源码 artifact，不能直接使用只接受单个 candidates.json 的共享 `read_candidate_file`。保留一个小型任务解析器，复用 Harness 的快照/大小/摘要字段；不复制整个 MeasuredResearchExpander。第一版跨分支同源码按批次重复拒绝修复，使用共享提交 validator 回调执行任务去重；同源码已通过的准入仍可复用回执。

Coverage 2+2 只选择研究父节点，不强制某个分支获得正式评估槽。`optimization.measurement.MeasurementPolicy` 是另一项测量配额机制，当前只在部分任务支持；铜矿明确禁用并拒绝相关配置。P5 使用四个分支各一份代码、总池 4、选 1 的工程 profile，先测弱基线再做 3 轮；不把 2+2 理解为四次正式测量。GP 冷启动用已测弱基线，特征/拟合异常显式失败，不能默默切换 random。

### 7.3 评估回执与计账接点

任务 evaluator 实现 `evaluation_attempt_usage_key(candidate)`。key 由 run/case、源码、完整科学 context、固定 seed panel 与执行预算确定，只有存在可恢复的任务执行 manifest 才能承诺重放。返回结果 metadata 携带共享 `EVALUATION_ATTEMPT_RECEIPT_KEY`，与预留 key 一致。有 dispatch 而无法确认结果时抛 `EvaluationPaused`，不返回 failed 后重派。可恢复基础设施异常也用明确的暂停状态保留已选槽。

LDMEngine 自动使用 `EvaluationExecutor`：持久化 selection、预留正式次数、保存结果，再发布 observation/checkpoint。铜矿不直接写 `engine_active_selection`，不重复扣 `selected_candidates/external_evaluations/expensive_evaluation_attempts/successful_evaluations`。

任务 execution manifest 只补共享层看不到的子阶段：每 seed 的 solver/scorer 容器 ID、输入摘要、dispatch/terminal 状态、输出摘要和累计资源。用共享 `atomic_json_write` 写入 run 目录；创建固定容器名/ID 的回执必须先持久化，再启动并对账，防止重连后重复执行。它不拥有 observation、选点或 campaign 状态。

`EvaluationResult.resource_usage.benchmark_jobs` 只报告正式 panel 的 solver+scorer 实际作业数，由共享 executor 在结果发布时计账；准入/开发 jobs 分别记入任务额外计数。共享 executor 不会自动消费所有 resource_usage 项：CPU秒、峰值/墙钟等由任务调用 `runtime.consume_many`，稳定 usage_key 对应 job，传累计值而非重复传增量。累计预算预留/硬超时在启动前检查；失败与暂停的已用资源同样结算。正式 job 上限在 `CampaignBudget.extra_limits` 显式按 panel 大小覆盖默认每评价一个 job 的假设，避免 solver+scorer 超额。

每 R 个树研究轮最多 5R 个逻辑研究 turn；主干计入 harness_turns/main_research_turns，不计候选数或分支提案次数。独立四会话为 4R，首阶段单会话为 R。失败调用与恢复 usage 使用共享累计去重，不能同时由 task 和 coordinator 对同一主干计费。末轮结果留在 checkpoint，不额外追加预算外的研究总结 turn。

## 8. 评价、失败、预算与产物

正式优化/报告指标：`loss_total`，方向 minimize。单次候选评价为固定种子列表的完整 panel，最终取算术均值；保存每 seed 的 loss、分项和矿体。若 panel 任一运行无可信结果，本次评价不生成聚合 loss，保留已完成子回执供诊断。第一轮工程闭环使用单 seed，后续多 seed profile 独立冻结。

诊断：钻孔/露头/草图分项、排除接触点后的标签不匹配数、接触点距离误差、体积、空气占用、运行时间、峰值内存。最优算法是 aggregate loss 最小的合法候选；多 seed 保存全部矿体，展示某个代表矿体需标明 seed。

| 情况 | 记录与预算 |
| --- | --- |
| 语法或准入不合格 | 拒绝并允许修复；消耗准入资源，不占正式槽 |
| 合法但 loss 差 | succeeded；真实值进入历史 |
| 正式输出无效 | invalid；占一次正式尝试，无伪造 loss |
| 超时/OOM/算法异常 | timed_out 或 failed + 原因；占正式尝试 |
| 镜像/服务/调度故障 | 明确 infrastructure 分类；无科学分数，按实际派发事实计账 |
| 外部任务状态未知 | 保留 execution handle，先对账，禁止盲目重派 |

初版用 attempts 作为正式预算，不自动无限替换失败。共享计数区分 turns、提交尝试、合法候选、正式选中、external evaluations、benchmark jobs 和 successful evaluations；新增准入/开发/solver/scorer 的资源计数时沿用共享账本，所有计数含零值。一个 panel 是一个候选评估，内部 jobs 单独计数。

每次运行保存契约/数据/镜像摘要、候选源码、准入回执、selection、worker/scorer 回执、输出模型、日志、分项结果及 shared checkpoint/status/budget。引用采用 run-relative 路径。完成时输出 `result.json`、`trajectory.csv`、最佳 `solver.py` 及其实际评分矿体；开发产物不写入 Git。

复用条件：候选源码 + objective/data + runner/image + seed + budget 全部一致。恢复先核对回执：solver 完成则只继续 scorer，scorer 完成则只发布结果；不能重复计数或再次推进已提交的 Harness turn。任何科学契约、研究策略或候选单次资源配置变化启动新 campaign，不将新分数追加到旧池。共享框架允许总 campaign 预算单调扩展：只有显式变更运行计划时通过其既有校验记录 old/new limits，不能为恢复失败静默加预算；config.json 保留初始快照，budget.json 记录当前限额。

## 9. 文件布局

```text
tasks/copper_inversion/
  task.json, experiment.json, README.md, pyproject.toml, __init__.py
  ldm_task/{__init__,procedure,dependencies}.py
  core/
    data.py, geometry.py, objective.py, candidate.py
    interface_checks.py, execution.py, runner.py, scorer.py, evaluator.py
    harness.py, harness_runtime.py, history.py, task_spec.py, workflow.py
    surrogate.py                         # 第二阶段
  resources/
    cases/<case_id>/                      # 原始必要数据、manifest
    contracts/v1/                        # fixtures、参考值、schema
    seeds/solver.py                      # 直接返回初始几何的弱基线
    harness/profiles/researcher/AGENTS.md
    provenance.json, qualification_evidence.json, LICENSE.upstream
  tests/                                 # 接口、科学回归、隔离与恢复
config/copper_inversion/
  mock.yaml, harness_2d_smoke.yaml, harness_3d_smoke.yaml
```

文件名为目标设计，当前尚未存在。只通过 task.json 注册，不添加中央 task-name 分支。`harness_runtime.py` 只负责客户端/工具资源装配，使用 ExitStack 关闭；`execution.py` 是任务内唯一容器与子回执实现。参数/报告复杂到需要独立职责时再提取 workflow_args/reporting，不机械复制其他任务全部模块。首阶段预计无需修改共享引擎、Pi 或 strategy；确有共享缺口时用定向测试证明并最小修改，领域评分与验证始终留在任务内。

## 10. 初版工程 profile

以下为拟定工程预算，不是原论文官方预算，也不是效果实验协议。远程 preflight 核定后写入 experiment.json；改动须在开跑前记录。

| 项目 | 首版值 |
| --- | --- |
| case 顺序 | 2D 钻孔+草图，再 3D 钻孔+露头；最后检查钻孔-only |
| 研究会话 | 1 个持久 Agent，每轮 1 份算法，3 轮 |
| seed 基线 | 初始几何返回算法，正式评估 1 次，计入预算 |
| 总正式尝试 | 基线 1 + 生成候选 3 = 4 |
| seed panel | `[47]`，single-seed 工程验证 |
| solver | 2 CPU，4 GiB；2D 60 秒、3D 120 秒 |
| 单次接口 job | 1 CPU，2 GiB，30 秒；并发 1 |
| scorer | 1 CPU，2 GiB，30 秒；超时归为评分基础设施问题 |
| Harness turn | 15 分钟，最多 3 次提交验证；记录实际模型与工具用量 |

并发和资源先核实再运行；不自动占用全部服务器。第一阶段完成时仍保持 `qualification: draft`，直到源码来源、固定评分和真实评估证据满足注册规范；自定义预算必须明确为 task-defined，不伪装为 official。

## 11. 远程环境与运行约束

主开发服务器：`ssh zsgpu@111.2.199.31 -p 5320`，建议任务根 `/mnt/data1/ldm/copper_inversion/`。本地不启动模型调用、容器调试或候选测试。

备选 H20：`ssh zsgpu@120.133.37.228 -p 10522`，活动代码/环境/缓存/结果使用 `/Aidata/shared/ldm/copper_inversion/`；归档使用 `/mnt/cephfs/xingyu/ldm/archive/`。使用前核查可用资源。Pi 的 Linux/KVM、Docker、锁定 guest 及实际 wire API 能力按现有 Harness 规范 preflight，不只验证 Chat Completions。

部署包包含本仓库所需源码、任务资产和锁文件，不依赖相邻铜矿/BGRSI checkout。凭证通过现有 Harness bootstrap 或受保护环境注入，solver/scorer 无凭证。正式运行使用唯一目录和可恢复执行句柄；远程中断不等于作业已结束。

## 12. 工作包与验收

| 阶段 | 实现范围 | 必须提供的验收证据 |
| --- | --- | --- |
| P0 科学内核 | 来源 manifest、loader、geometry、objective、弱基线 | 远程坐标/标签计数、手算 loss、旧实现固定模型对照、缩放测试 |
| P1 执行契约 | 候选快照、独立 worker/scorer、只读接口检查 | 正常案例及 NaN/超时/伪造评分/修改草稿等故障均正确分类 |
| P2 注册与 mock | task、experiment、共享 Campaign、假 sidecar | 完整 mock 轮次、拒绝修复、正确计数与 checkpoint |
| P3 真实 RSI | 持久 Harness、历史/源码反馈、2D 三轮 | 每轮真实新代码执行；下一轮实际读取反馈，提交修改及理由 |
| P4 3D 与恢复 | 两个 3D case、阶段恢复、产物整理 | 3D 真运行；solver 后/scorer 后恢复不重复执行、计数或推进 turn |
| P5 LDM 选点 | 共享 Coverage 2+2、MeasuredResearchExpander、encoder、GP/acquisition | 四分支准入、仅选中项评分、utility 符号正确、旧节点增量历史及部分屏障恢复 |

P3 验收不要求分数一定改进：必须能证明 agent 完成代码设计→正式执行→反馈→代码修改。若无提升如实报告。P5 才能宣称 surrogate-guided LDM 路径可运行；此前明确称为在 LDM Campaign 上运行的 Harness 算法发现。

所有测试在远程执行，从定向科学/接口测试到注册 mock，再到一个真实 seed、tiny campaign。新增测试覆盖此规格边界；共享模块变化才扩大相应回归。按 register-ldm-task 的 qualification stages 保存证据，不将 ignored runs 路径直接作为已入库验收证据。待证据文件被 Git 跟踪后才能推进对应阶段。

## 13. 实施前必须关闭的具体事项

1. 冻结迁入数据/源码/ZIP 摘要及许可记录；确认 UBC 物理坐标映射。
2. 建立 objective_v1 的参考数值，量化重新距离化相对旧模型 loss 的差异。
3. 核实远程 Docker/KVM、资源和 provider 协议；确定实际镜像内容 digest。
4. 将本规格预算写成 runner-enforced profile；不存在的 CLI/config 不提前当作已可执行命令。
5. 按第 7.3 节实现现有 evaluation_attempt_usage_key/consume_many 接点，补齐任务 solver/scorer 子回执；不得重新实现共享评估提交协议。

以上事项由实现阶段检查解决，未完成时保留 draft 状态，不阻止开始 P0/P1。本规格没有启动任何远程实验，也不声明候选或系统已通过验收。

## 14. v1.1 架构更新验收增量

本轮只修订 spec，未执行候选或远程测试。实现时额外覆盖：

- 唯一 runtime、启动失败清理、正式计数与子 job 计数不重复。
- 在 evaluator 返回前后、结果发布后 checkpoint 前注入中断；参考现有 `tests/test_ldm_engine.py` / `tests/test_campaign.py` 的真实边界。
- 默认 2+2、显式 independent/latest-main 策略解析及 measurement 配置拒绝，参考 `tests/test_research_tree.py` / `tests/test_research_task_adapters.py`。
- 最小 loss 候选具有最大 research_utility；非有限/失败结果不进入研究归因。
- 研究树恢复不改变历史前缀、父快照、strategy identity 或 panel；最后一轮未刷新树 credit 不被误读为测量失败。
- 共有库无铜矿 task import，未新增第二个 Harness 客户端协议、研究树或预算数据库。


## 15. 第一阶段实现结果（2026-10-23）

实现与证据：[实现记录](copper-inversion-implementation.md)、
[机器可读证据](../tasks/copper_inversion/resources/evidence/campaign.json)。

已复用共享 Campaign/Engine/Harness 完成 baseline + 三轮独立算法设计；
真实损失依次为 293.2197608、3.9504788、3.7968566、3.7338023。
只读接口、独立评分、原式对照、恢复幂等性和最终执行器复评均已验证。
最终远程回归 162 项通过。主循环源码快照随远程 run 归档；其后收窄容器代码挂载、
补充 CLI 超时暂停、preflight 用量恢复和 sidecar digest 固定；最终执行器复评四个源码，
损失与主运行逐项一致。

资格阶段为 contract_verified，experiment 保持 draft：ZIP 来源有内容摘要，但尚未完成
官方 benchmark 来源/预算认证；2D 闭环工程证据不自动升级为方法有效性证明。
P4 中的 3D agent 搜索和 P5 的 surrogate 引导留在后续阶段。
