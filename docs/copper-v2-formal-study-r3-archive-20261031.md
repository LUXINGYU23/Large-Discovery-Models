# r3 及此前历史报告归档

本页为修复后 r4 交付前的历史记录，不代表新基线结果。

## 最终交付（2026-10-27）

**后续基线质量审计：** 在开发数据上应用r3公开先验后，冻结种子43参数复现了约94.5%的提议因超过四个连通体被拒绝，ACQ的16次开发求解均未启动采集；这是已定位的主要适配问题。距离重初始化的符号翻转是另一个真实数值问题，但仅改一阶格式未解决主要拒绝。可行域投影加一阶重初始化通过开发验收，保留硬约束与原配置；投影改变提议分布，因此修复版称为随机优化适配器，不主张严格后验MCMC。独立的同实际时间开发比较见[审计记录](copper-baseline-audit-20261027/README.md)。r3原数值保留，只能解释为优于当时的适配实现，不能据此断言优于充分适配的强基线；已揭示测试集不参与修复或调参。

正式 r3 九组训练、验证选型和固定赢家 ID/OOD 测试已全部完成。见[中文最终技术报告](copper-measurement-study-r3/report.html)，包含 PNG/PDF 图、冻结算法解析、同坐标真值与预测的3D及三正交切片、1296次求解明细、54组执行种子统计、预算对账和成本披露。

ID损失：LDM **0.361726±0.032651**，MCMC **0.415236±0.000979**，ACQ **0.415201±0.000595**；LDM三个发现种子均领先。OOD：**0.395486±0.083478 / 0.417936±0.001172 / 0.418215±0.001086**；LDM43不领先，因此未证明稳定OOD优势。均值±样本标准差跨三个发现种子。LDM求解约112.56秒，基线约91.1秒，约慢24%；所有测试合法率100%、失败0。

原账本、收费拒绝、源码和收据保留，预算上限未提高。API有17份响应缺usage，部分早期中断及开发恢复成本无法完整量化，报告明确披露而不记零。旧r1/r2作废；以下为按时间保留的历史记录，不代表当前状态。已揭示的测试数据不得回流研究。

---

2026-10-24。本文件记录实验设计、发布检查和启动状态，不是完成结果。

**当前独立测量主指标 r2 已因新发现的基线数值异常与接口超时再次暂停：见[新数据、闭环验收与暂停整改记录](copper-measurement-study-r2/README.md)。以下旧 r1 的 IoU 结果只保留作历史诊断，不能作为公平优势结论。**


**2026-10-25 02:25：128 并发完整验收 1308/1308 通过，实测整机 CPU 中位 79.33%。正式 r3 九组已自动启动，20×16＋初始化、321 收费源上限，预算与约束一致。目前处于初始化/第一轮研究；私有执行收据暂未见失败，LDM provider 传输错误正在框架内续接，尚无方法优势结论。见[完整验收记录](copper-measurement-r3-qualification.md)。**

## 早晨阶段报告（2026-10-25 06:25）

正式九组仍在训练，尚未完成20轮或验证测试。[打开阶段可视化报告](copper-measurement-study-r3/report.html)。共同33个收费源下，LDM/MCMC/ACQ训练最佳损失三种子均值分别为0.436986/0.422021/0.421975；当前LDM均值和稳定性仍落后。报告图表使用05:25快照，包含收费拒绝，不能据此宣称最终或泛化优势。

## 框架与预算

已将最新 main `3580273b`（PR #15）合入铜矿分支，合并提交 `d5c304cc`，原有未提交工作保留。沿用共享 MainHistoryForks 1+3、主线反思及侧枝竞争假设提示；修复铜矿适配器没有传递 branch_instructions 的遗漏，将历史工具映射为 get_inversion_context。研究种子从固定47改为明确配置。

| 项目 | 固定设置 |
|---|---|
| 方法 | LDM 融合、MCMC-v2、ACQ-v2 |
| 发现重复 | 42、43、3407，各自独立搜索、封存和选型 |
| LDM | 20轮；每轮4侧枝各8提案，融合BO选择16个评估；加1次初始化 |
| 每方法每发现种子额度 | 321个完整训练suite反馈槽位；失败/拒绝不得免费换源 |
| 基线 | 各自321个预定Sobol参数配置，包含默认配置 |
| 求解 | 120秒硬上限，2 CPU，4 GiB；所有方法相同 |
| 执行种子 | 训练/验证47；最终测试47、48、49 |
| 并发 | 每套件固定4个case作业，账本写入加锁；九组同时运行 |
| 模型及表征 | 官方DeepSeek V4.1 Flash/low；AST/Qwen等权融合；既有Qwen3-Embedding-0.6B缓存 |
| 验证 | 每发现运行训练封存前3名，验证后固定1名 |
| 测试 | 所有训练结束封存，所有验证结束选型后，才运行ID/OOD测试 |

“评估一个候选”指评估完整48包训练suite，不是单个case。每发现运行最多15,408次训练求解；九组共138,672次，120秒上限总和4,622.4小时。按4路case并发，每组求解墙钟上限总和为128.4小时，实际耗时取决于算法是否提前退出，另有容器、评分、研究和编码开销。不得将最终程序耗时当作发现总成本。

## 新数据与私有发布

全新24训练/8验证/16 ID测试/8 OOD测试母体，每母体两个观测包，共56母体、112包。使用预先封存的新随机流，未复用旧开发母体。生成、确认观测及发布均在`ldm-copper-eval`账户中完成。研究端仅获得公开训练catalog，验证/测试及生成参数不进入guest。

数据来自连续GemPy结构查询，测量不从评分网格SDF抽样。每个母体另外生成独立钻孔/露头位置及噪声流的确认观测；其文件仅与真值一起挂载给独立scorer，求解容器不可见。确认观测误差为辅助指标，不进入公开目标，也不用于改变固定验证选型规则。

发布资格包括：

- 18个解析校准组合：球、薄层、斜层、错断层、开放边界和地形裁切，三个spacing设置；接触误差、远离界面的分类、单位换算、非活动区违规拒绝均通过。
- 56母体的接触误差及非活动区检查通过；公开目标排除未经校准的轴向表面积惩罚。
- 跨split按归一化网格、轴置换与反射筛查近重复，预定阈值IoU<0.98；通过。另与旧私有开发集做规范化几何指纹排除。
- 独立确认观测全覆盖112个包，几何与mask一致，测量点未复用；发布清单绑定其文件摘要。
- 新Pi镜像真实guest只读、路径/符号链接、私有文件、凭证和socket探针通过；一次HTTP出站请求失败。此探针不是穷尽网络或虚拟机逃逸证明。
- 正式服务检查私有发布资格与文件摘要，固定321槽位/120秒；只有指定评估账户可验证发布身份，单改stage标签不能启动正式服务。

这是**通过当前数值与隔离检查的独立合成基准**，不等于真实矿区验证或地质专家认证。解析薄层校准不覆盖任意亚体素矿体，接触阈值为经验分辨率容差，不是误差上界定理；这些限制必须进入最终报告。宿主管理员和可信Docker操作员不在隔离威胁模型内。

## 验证与修复记录

- main整合及新预算/资格检查的远程v2回归：76通过、2跳过（依赖旧开发集成路径）。
- 隐藏确认观测的真实solver/scorer容器集成：1通过；solver不可读truth，scorer返回confirmation指标。
- 数据部署曾因个人目录解释器权限、部署包装脚本、控制器依赖环境和GemPy共享临时缓存权限失败。均发生在算法搜索前；保留失败现场与原始生成随机流。最终使用私有运行环境、独立控制器和专属TMPDIR，r5生成完成。
- Pi使用原锁定依赖镜像加最新编译代码形成新镜像。首次构建器将SHA误解为远程标签，改为已核验本地基础镜像标签构建；没有更换依赖或覆盖旧镜像。

## 当前运行与恢复边界

运行根：`/Aidata/shared/ldm/copper-formal-study-20261024-r1/`。
私有发布：`/Aidata/shared/ldm/copper-v2-evaluator/datasets/formal-20261024-r1/`。
私有服务及选型：`/Aidata/shared/ldm/copper-v2-evaluator/runs/copper-formal-study-20261024-r1/`。
冻结服务源码：`/Aidata/shared/ldm/copper-v2-evaluator/code/formal-study-r1/`。
监督器启动PID：1641770；九个客户端PID、服务身份及预算见deployment.json。PID仅供定位，恢复前必须核对进程命令及存活状态，防止PID复用。

监督器目前执行训练。训练全完成后核对并封存321槽位，接着调用原有freeze/selection状态机：全部冻结→全部验证→全部测试→重放核对收据。失败不新建免费搜索；保留原目录、源码身份、账本、已完成case收据和日志，写入needs-reconciliation.json并停止后续阶段。不得重复运行launch_formal；恢复应核对原进程、同源码及同契约后使用原目录resume。

本轮尚无正式胜负结论。最终按同预算绘制逐轮曲线，ID/OOD分别报告三次发现的均值与离散程度；执行种子和观测包先在母体内聚合，配对统计以地质母体为单位。确认观测误差、失败比例及全部发现成本同时报告，不能仅取三个种子中的最好结果。

## 本地证据

[启动配置](../tasks/copper_inversion/resources/evidence/formal-20261024/deployment.json)、[启动核查](../tasks/copper_inversion/resources/evidence/formal-20261024/startup-audit.json)、[解析校准](../tasks/copper_inversion/resources/evidence/formal-20261024/calibration-r1.json)、[回归](../tasks/copper_inversion/resources/evidence/formal-20261024/full-v2-regression-r2.log)、[容器确认观测检查](../tasks/copper_inversion/resources/evidence/formal-20261024/confirmation-container-r1.log)、[guest探针](../tasks/copper_inversion/resources/evidence/formal-20261024/guest-isolation.json)。本地未复制新验证/测试的真值与确认观测数组。

## 并发提速：独立调度纪元（2026-10-24）

九组运行已切到 `copper-formal-scheduling-20261024-r1`。冻结科学源码、镜像、数据、服务契约、候选与已完成收据保持原样；另行记录调度文件摘要和策略摘要，当前进程以 `scheduling-epoch.json` 为准，原 `deployment.json` 保留启动历史。

- 复用共享引擎 `BatchCandidateEvaluator`：同轮最多16候选并发，按原候选顺序返回和提交。轮际依赖保持不变。
- 两个基线的预先生成参数表以16线程并发评估，参数、随机流和最终排序不变。
- 私有评分服务接收并发请求，同源请求互斥；共享输入的接口检查串行，候选的完整训练套件并行。槽位和资源账本写入有锁。
- 全局80个容器作业令牌、每进程16个；单候选仍为4个case worker、每次求解120秒/2 CPU/4 GiB。排队耗时另记，不进入求解时限。按求解容器CPU上限计算最多160 CPU，保留服务器余量。
- 迁移冻结控制器后接管原Docker作业，沿用稳定容器身份、原始终止时间及认证资源计数；不重新执行已完成求解。核对8252份旧收据、计划和服务身份，摘要全部不变。
- 一次迁移后快照实际观测80个求解容器并发；多个基线同时运行3–5个不同候选，九个客户端均存活。此为并发实测，不是端到端加速倍数；最终耗时报告应分开标注旧串行阶段、新并发阶段和切换开销。

迁移触发42和3407种子的未完成研究轮次摘要检查。确认原因是恢复后嵌套JSON对象字段顺序改变，消息语义和历史摘要未改变。独立恢复适配器逐字段核对历史/轮次身份、解析后的完整消息相等、原始消息摘要正确后，重放原消息字节；不删除原turn、不放宽摘要验证、不重置已计费用。已恢复两组，原错误及恢复事件保留在日志中。新调度测试28项通过，精确输入重放与拒绝语义变更测试1项通过，均在远程执行。

监督器仍执行训练完成→核对321槽位封存→全部验证→全部测试的原流程。验证和测试未提前开放给研究端。正式结果未完成，不据本次吞吐快照推断方法优势。

## 并发运行复查（2026-10-24，约18:06 CST快照）

三个发现种子的顺序均为42、43、3407。当前仍仅为训练阶段，验证/测试尚未开始。

| 方法 | 已完成的候选记录 | 原序完整前缀 | 当前最佳训练损失 |
|---|---|---|---|
| LDM | 17 / 17 / 17 | 17 / 17 / 17 | 0.694900 / 0.653017 / 0.664051 |
| MCMC | 5 / 4 / 4 | 5 / 2 / 3 | 0.718356 / 0.718356 / 0.718356 |
| ACQ | 6 / 4 / 5 | 6 / 3 / 4 | 0.717201 / 0.717201 / 0.717201 |

“已完成记录”包括接口拒绝，拒绝消耗预算但没有有效矿体。三组LDM均完成初始化和第1轮（20轮中的1轮）；下一轮研究进行中。并发提交/占用槽位不是已完成评分，乱序返回的候选不提前进入预算前缀比较。

### 共同前缀：前2次评估

| 方法 | 最佳损失，三种子均值±样本标准差 | 所选算法每case求解秒数，均值±标准差 | 前缀累计求解容器秒数，三种子均值 | 到达前缀的实际经过分钟，均值±标准差 |
|---|---:|---:|---:|---:|
| LDM | 0.715703 ± 0.000494 | 1.245 ± 0.231 | 122.69 | 11.62 ± 1.91 |
| MCMC | 0.718356 ± 0 | 91.414 ± 0.036 | 8785.53 | 63.33 ± 2.42 |
| ACQ | 0.717201 ± 0 | 91.368 ± 0.085 | 8782.91 | 62.00 ± 3.45 |

所有算法单case上限均120秒，每个完整候选48个训练case。累计容器秒数是求解时间之和，不是经过时间；不包括研究、接口检查和独立scorer。经过时间分别从campaign创建/基线计划写入到原序前缀完成收据或提交事件，包含旧调度、排队和切换影响；并发还会同时计算前缀之外的候选，因此不能将此表解释为隔离运行的“两次评估基准”、纯调度加速倍数或完整搜索总成本。

前2次评估时，LDM只领先MCMC约0.00265、领先ACQ约0.00150。基线标准差为0，是因为三组此时都选中了相同的固定初始配置，不是成熟搜索稳定性的证明。LDM在各自17次评分下的最佳损失均值0.670656、标准差0.021709，但其他方法尚无相同完整前缀，不能据此宣称公平预算下的显著优势。

### 异常与处理

- 43、3407号LDM在第2轮研究因官方provider请求超时退出。已用原候选、原输入字节、原账本恢复，保留失败日志和费用记录；已观测恢复后的新provider响应。监督器也已接管新PID。没有修改模型、端点、low推理设置或扩大预算。
- 42号LDM进程仍在，但本轮provider调用等待较长；进程存活不代表轮次已推进。该组尚无第2轮完成记录。
- 六组基线各有一次5秒公开接口测试超时拒绝，42号LDM也有一次接口拒绝。当前保留为计费拒绝，不删除、不补送免费候选；不能直接归因为调度故障。接口上限与120秒正式求解上限不同。
- 复查时全局80令牌占满，76个求解容器处于运行态，其余令牌可能在启动/结束阶段。8252份迁移前收据和身份摘要仍全部一致。
- 客户端预算包含本地接口拒绝，私有服务预算仅含送评源，两者可相差拒绝数。最终封存必须核对两端唯一源并集，不能把私有计数单独当完整预算；已修复监督器的终态断言：核对唯一源总数321、私有源是客户端源的子集、差集全部有已计费接口拒绝；私有封存计数必须匹配实际私有源数。远程4项检查通过，预算不足、未完成送评和未知私有源均被拒绝。没有扩大预算。尚未进入封存阶段，不能声称整个正式流程已经完成。

原始汇总见 `tasks/copper_inversion/resources/evidence/formal-20261024/progress-comparison-20261024.json`。此次快照用于进度判断；最终稳定性结论仍需完整预算和独立验证/测试。

### 18:40监控：重复超时与接口排队保护

43/3407再次因provider timeout停止，42长期未完成第2轮。进一步检查发现部分provider请求文件生成后长期没有授权记录，同时研究接口检查停止推进；共享80令牌长期由基线求解占满。原flock随机抢占没有公平性保证，接口验证同步阻塞控制器时，其他研究分支的provider授权也会等待。因此此前将所有timeout直接归因为官方API不够准确：已确认存在本地排队/授权阻塞风险，不能排除上游超时。

部署独立调度r2：总上限仍80，其中8个令牌专供公开接口检查，72个用于求解/评分；进程上限16不变。远程检查证明计算令牌全部占满时接口任务仍能立即获得预留名额，计算任务不能占用接口预留。原科学源码、120秒求解、5秒接口、候选/预算均不变；继续接管原Docker作业。修复不会取消已计费的接口拒绝。

本次发现的基线完成记录已达13–14个，但尚未全部原序提交，不能按新记录数直接宣布相同预算胜负。三组LDM仍17次评分，第2轮未完成；优先修复研究推进瓶颈，后续必须确认新接口收据和provider授权持续出现，而不仅是PID存活。

### 19:10监控：首个共同17槽位检查点

九个客户端均存活，未发现本次新增退出。接口预留之后，三个LDM均完成第2轮提案并送出下一批16候选，私有完整套件分别已完成25/24/24个；公开checkpoint仍17个，因为本轮要收齐后按原序提交。私有job收据在本次检查前0–2秒仍更新，不能因一次Docker快照恰好没有短时LDM容器而认定停滞。六组基线均已有前17个候选记录，继续评估后续参数。复核22433份迁移前身份与收据，摘要未变。

共同前缀为17个计费唯一源，包含基线接口拒绝，不补评；LDM-42的本地拒绝发生在下一轮，不计入此前17槽位。训练损失为1−母体聚合IoU，越低越好。

| 方法 | 种子42 / 43 / 3407最佳损失 | 均值±样本标准差 | 最佳算法每case求解秒数（均值±标准差） |
|---|---|---|---|
| LDM | 0.694900 / 0.653017 / 0.664051 | 0.670656 ± 0.021709 | 1.889 ± 0.627 |
| MCMC | 0.716541 / 0.715261 / 0.715461 | 0.715754 ± 0.000689 | 91.341 ± 0.026 |
| ACQ | 0.715036 / 0.715925 / 0.715697 | 0.715553 ± 0.000462 | 91.285 ± 0.043 |

在这个早期共同预算检查点，三个发现种子的LDM训练损失均低于对应基线，平均绝对降低约0.045（相对约6.3%），但LDM种子间波动明显更大。这里仅支持“前17槽位训练表现占优”，不能替代完整321预算、独立验证/测试和最终稳定性结论。

到达前17槽位的实际经过时间（混合串行、并行、迁移和排队）分别为LDM 26.19±6.47分钟、MCMC 114.14±2.89分钟、ACQ 115.40±1.89分钟；不应解释为纯调度加速比。前缀累计求解容器时间均值分别为2416.31、65783.66、65777.89秒，接口检查、独立scorer、API研究费用和前缀外在途作业不在该累计值内，必须另外报告。后续研究超时不能从整个发现成本中扣除。

证据：`tasks/copper_inversion/resources/evidence/formal-20261024/progress-comparison-20261024-1910.json`。本次未重启健康任务，保留半小时监控。

### 19:40例行核查

九组均正常推进，最近私有作业收据年龄0–8秒；没有新增退出或摘要变化（22433份旧身份/收据）。三个LDM公开checkpoint均33次评分，已完成第2轮、正在评估第3轮；六组基线完成记录20–21个，但共同原序前缀仍17，因此同预算比较与19:10记录一致。不重启健康任务、不修改预算。聚合证据保存为 `progress-comparison-20261024-1940.json`。

### 20:10异常修复与比较口径更正

**此前“前17参数记录=17计费源”的基线比较需更正，暂不能作为完整同预算结论。** 发现四组基线退出，错误为 `receipt source changed`。根因是并发恢复适配器从JSON计划读取参数时，键顺序被排序；`source_for`使用`repr(parameters)`生成CONFIG，参数语义相同但源码字节/hash改变。六组均存在参数索引1的两个已计费字节版本，其中早期版本的套件未完成。旧记录/身份检查不变并不意味着没有重复提交，这是本次恢复适配器的问题。

修复：新增独立 `baseline_runtime_r3.py` 和策略摘要；按已完成收据或已计费源恢复原始源码字节，逐个验证完整模板一致且CONFIG语义与预定参数表一致，不绕过原始源码摘要。六个基线客户端已恢复、复用已有结果，三个健康LDM未重启。总预算不退款：每组1个历史重复源仍计费，后续最多320个参数条目加该重复源，总计321。新监督器为 `recover_baselines_r3.py`，迁移脚本不得再次运行。

六个早期已计费中断源另用同源RPC幂等续跑，接管原case收据而非新建免费评估；输出记录在各组 `serialization-interrupted-recovery/`。这部分在本次记录时尚未完成，不计作新成功结果。重算17计费预算应使用基线前16参数条目加历史重复源；目前可用结果的最佳损失数值暂未改变，但重复源完整结果及历史开销必须补齐，先前17条目耗时不再作为精确同预算耗时。完整结果出齐前不强调方法优势。

验证：六组全部已收费源码均与冻结模板及预定CONFIG逐字生成一致，仅CONFIG键序不同；修复后成功重放原索引0至27–29收据，九组再次运行。操作日志、历史收据和费用保留，不修改已计费拒绝。监控已更新为新的恢复入口与真实预算口径。

### 20:40例行核查

九组持续推进，最近评分/provider活动年龄0–53秒，无新增进程退出；22433份旧身份/收据摘要未变。三个LDM各完成65次评分（初始化+4轮），正在第5轮研究/评分。基线完成记录28–33个，客户端已收费45–50源（含在途、接口拒绝及历史重复）。六个中断重复源中MCMC-3407已恢复完成，其余仍在同源续跑，没有重复启动恢复进程；等待全部结果后再更新完整同预算比较。公开接口超时拒绝继续计费并保留，未增加科学预算。本轮不重启健康任务。

### 21:10核查：六个历史中断源均完成

历史重复字节版本的六个已计费源均按原source/receipt恢复成功，核对回复来源SHA一致，不新增槽位、不退款。修正后的17计费源检查点已完整：LDM前17个已评分源；每组基线前16个参数条目（包含拒绝）加1个历史重复源。补入这六个原始版本评分后，最佳损失均值/标准差仍为LDM 0.670656±0.021709、MCMC 0.715754±0.000689、ACQ 0.715553±0.000462。

该结果是含序列化故障成本的实际运行比较，不是无故障算法能力基准；历史重复源占用了基线探索名额，不能将其不利影响隐去。旧17参数条目的耗时未重算，仍不作为精确同预算耗时。训练优势仅限当前早期检查点；验证测试未开始。

九组均存活，基线完成记录31–36个，LDM各65次已提交评分、下一批16仍在评分；没有新的退出，22433份旧身份/收据未变。证据：`corrected-17-budget-complete-2110.json`。

### 最新可视化快照

同31个实际计费源的均值/种子曲线、算法求解时间及完整限制说明见 [阶段性可视化报告](copper-progress-20261024/README.md)。三方法平均最佳训练损失为0.63894、0.71501、0.71524。基线历史重复源和LDM接口拒绝均保留在预算轴，不使用参数记录数冒充计费数；仅为训练阶段的含故障成本比较。

## 2026-10-24 用户暂停与算法审计

按用户要求停止全部铜矿搜索、基线、私有训练服务、中期验证/测试和自动监控。保留账本及收据。发现基线适配每步占据规范化抹掉亚体素位移；正式数据未启用地质先验；IoU 榜单优势不能等同观测损失优势。原结果现仅用于诊断，不再作为公平战胜原论文方法的证据。

已修开发代码的连续搜索状态、源码序列化、验证期间 provider 授权、并发记账及拒绝计费终态，并增设正式地质约束校准准入。独立科学校准及新镜像全闭环验收尚待完成，因此没有恢复实验。

详见 [算法审计、真值对比与整改清单](copper-algorithm-audit-20261024/README.md)。中期独立评估停止时只有部分验证作业，测试收据为 0；不得报告测试完成。

### 用户确认主指标与修订验收

用户确认独立测量误差为主、IoU 为辅。已统一私有聚合和验证选型，启用固定公开体积/连通性先验及发布约束门；独立开发 10 个新父体、18 个数值检查通过。首次 GemPy 共享临时缓存权限错误已保留并修复为作业独立缓存。旧/新转移各 MCMC/ACQ 的 24 次 120 秒容器诊断全部成功且结果不劣于初始化；这不是新正式实验。新的数据版本尚未发布，搜索与监控保持暂停。详见审计报告末节。


## 2026-10-25 09:55 监控：LDM43 研究会话阻塞

- LDM43 在第5轮 researcher_2 触发1800秒 session wall-time limit，客户端退出1；history-recovery.json 明确要求显式修复，普通 resume 无法解除。最后三次 provider exchange 均返回HTTP200，不能据历史502就认定此次是API超时。尚需分解响应流、工具和验证等待耗时。
- 其余八组仍有新评分收据或研究提交；全局 queue/PAUSED.json 不存在。LDM43已收费67源、私有65源、完整65源，已有两次收费拒绝原样保留。其他各组最新快照见 progress-20261025-0955.json。
- ACQ43本地收费拒绝计数增加至2，需要在后续同预算前缀扩展中保留该槽位；不得免费补评。共同65源结果仍以09:25已核对报告为准。
- 已保留退出日志、账本和历史阻塞文件摘要：incident-20261025-0955-ldm43.json。未删除历史失败标记，未重置预算，未变更冻结源码，未重启一次性控制器。
- 尚未修复/续跑LDM43。原监督器持有旧子进程退出码，最终会要求 reconciliation，不能再宣称现有监督器会自动完成全部私有流程。恢复需同时解决有审计记录的会话修复与监督器接管；不得仅启动第二个客户端或篡改状态宣称恢复。


### 2026-10-25 10:25：LDM43 超时定位到提交后校验等待

researcher_2 的 Pi 日志显示：01:14:09 UTC 调用 submit_solver，01:41:28 才返回“harness sidecar is closing”；随后历史标记1800秒超时。宿主校验线程继续完成，并于01:48:31 UTC写入 researcher_2 的 admission。四个研究分支共32个候选均已有 admission，第五轮共192个接口收据完成。说明本次故障发生在提交后等待宿主校验期间，不能归为模型一直没有产出或仅凭200判定传输正常。

代码路径：client.py 的 validate_without_blocking_provider 串行执行提交回调，其他 submission 帧暂存 deferred；core/harness.py 每分支8候选，每候选6接口顺序校验。侧车1800秒整轮计时仍包含这段等待。超时关闭后宿主线程按既有设计 join 完成，故出现“宿主已接受，但侧车未提交且历史被阻塞”的不一致终态。尚未精确分离每个作业的队列、启动和求解时间，不能声称192个接口均在纯计算中耗尽限额。

恢复所需：保留32个原候选字节、输入摘要和所有收据；以宿主 admission 与原提交摘要进行严格对账，设计能完成原提交协议的恢复路径，不能伪造 turn_committed、删除 history-recovery 或重新生成替代候选。并在开发副本覆盖“校验晚于侧车期限完成”的回归测试后，再确定正式身份兼容的恢复方式。当前冻结代码不具备此恢复入口，未原地修改或强行续跑。原监督器仍记录失败客户端，后续也需显式接管完成流程。

其他八组收据均在10秒内更新；LDM42收费97/结果81，LDM3407收费113/结果96（另1拒绝），六基线收费97、完整结果82–92；ACQ43另有2个收费拒绝。LDM43仍收费67/结果65。无全局暂停，正式验证测试尚未开始。证据：progress-20261025-1025.json（含四个admission摘要）。


### 2026-10-25 11:00–11:05：接受记录恢复及原输入重放

恢复路径已具体实现为独立运维脚本，不修改冻结源码/侧车镜像：从 researcher_2 原 Pi submit_solver 参数、8个只读artifact快照和宿主admission重建缺失的submission.json。重建后的完整submissionDigest必须与既有admission完全一致；工具费用从原tool-budget恢复，history failure、Pi历史及所有收据保持原样。原侧车已有savedSubmission分支，会验证快照并自行执行commit；运维脚本不生成turn_committed。

远程检查：3个已正常提交分支重建journal逐字段完全相等；目标摘要匹配；篡改源码/接收摘要均拒绝。首次恢复暴露原框架resume消息JSON键排序造成partial turn digest mismatch，已退出并保留失败记录；没有进入新候选评估。新增外部exact-input重放包装，只在全部metadata及解析后message完全相等时恢复原字节，并重算输入摘要；远程验证等价序列化通过、metadata与语义变更拒绝。首次测试调用因私有Python权限失败，随后使用sudo在远程执行通过，未在本机测试。

第一次恢复监督器1860444/客户端1861748已退出。第二次一次性恢复监督器2438552、客户端2439449，以monitoring/ldm43-recovery-controller.json为准；原监督器2665407继续负责其他八组，恢复监督器等待原监督器退出后负责严格对账、封存、验证和测试。禁止重复启动两个resume脚本。原progress.json的ldm43退出1代表旧子进程，恢复状态看monitoring/recovery-progress.json。是否恢复成功必须以原生commit、新选中事件和新收据确认。

新快照progress-20261025-1100.json：其他八组继续推进，无新增私有执行错误或共享暂停；LDM42结果81、LDM3407结果108，六基线结果93–97。共同可核实比较仍用65收费源结果，未按不同预算当前最优值混比。


11:05恢复确认：原冻结侧车已自行生成researcher_2_round_5的turn_committed.json，摘要fec69e079eb4daece78d5f14d8f69ad8dfa929c1a23b9a866710fea6018d3e2e；恢复后新增第5轮reservoir_expanded/reservoir_built，32个原候选进入原框架选型。provider exchange仍为20次，history failure文件保留，恢复提交时收费仍67源。证据monitoring/ldm43-restored-native-commit.json。这是原提交状态修复成功，不表示第5轮评分完成，也不表示串行校验长等待机制已经消除；后续仍需监测新轮次并按相同摘要校验原则恢复。


### 2026-10-25 11:25 监督：恢复评分推进，新增基线截止信号竞态

LDM43恢复后已经出现第5轮candidates_selected，并有新私有评分收据（最新约5秒）；收费83、私有81、完整结果68，原有2个研究拒绝保持。恢复监督器2438552及替代客户端2439449均存活，其余八组持续推进。LDM42完整93；LDM3407完整112加1私有拒绝，进入第8轮研究/选型；六基线完整97–98。无全局PAUSED或新增私有执行错误。证据progress-20261025-1125.json。正式验证测试未开始，共同预算结论仍以已核实65源为准。

新异常：MCMC42参数索引99（原序第100收费源），源码c696710029e839b9a426dbeca9eca166a8cbe76b6d2720a6e7c06ed81c689c81，在public-interface-4退出1，已按协议占用预算。作业2cc97879aef53a8c0227210a7dfd35a9821ec7c8b407b946d9c4a7bcca05fc12的日志显示SearchDeadline在solve第122行signal.setitimer(signal.ITIMER_REAL,0)期间触发，逃逸到runner；不是OOM或Docker启动超时，worker耗时4.3156秒。说明冻结基线的截止信号清理存在竞态，不能把这次拒绝仅归类为资源压力或非法地质参数。

此失败保留原费用与收据，不免费补评。修复信号处理会改变冻结基线源码/科学身份，因此本次没有原地修改基线或静默替换候选；需要另行科学版本验证。当前运行可继续作为该冻结实现的结果，但应在最终方法比较中明确披露此实现缺陷及失败率，不能声称这是已修正基线的最终性能。


### 2026-10-25 11:55 监督记录

九组继续推进，无新增私有执行错误、收费拒绝或全局暂停。LDM43恢复后的完整结果从68增至73，最新评分收据约2秒；收费83（含2个历史研究拒绝），第五轮仍在评分。LDM42完整97，已进入第7轮研究；LDM3407收费129、完整112加1历史私有拒绝，第8轮评分活跃。六基线均收费113、完整97–108，MCMC42与ACQ43分别保留1/2个收费接口拒绝，其余无新增。

原监督器和恢复监督器均按既定职责运行，没有启动重复控制器或修改冻结配置。阶段证据progress-20261025-1155.json记录实际PID、账本、收据及最新事件。尚未全部训练完成，正式验证测试未启动。共同预算方法比较仍采用已严格核对的65源报告，不把乱序完整数量作为预算前缀。


### 2026-10-25 12:25：共同81源阶段比较

九组仍在推进，没有新增收费拒绝、私有执行错误或共享暂停。LDM43第五轮已完成81份评分结果（83收费源含2拒绝），第六轮研究有新provider活动；LDM42第七轮评分，LDM3407完成第八轮并进入第九轮研究。六基线完整结果109–113。恢复监督器继续正常运行，正式验证测试尚未开始。

根据原序选择事件、LDM43第34/35收费位置拒绝及客户端/私有账本，重算共同81收费源前缀，未将乱序完成数当成预算。最佳训练损失均值±样本标准差：LDM 0.421487±0.028543，MCMC 0.414481±0.001785，ACQ 0.414724±0.001805。LDM相较65源有改善，但平均和种子稳定性仍落后于基线；LDM42单种子最好0.392534。赢家求解均时LDM三种子83.08/81.33/25.97秒，基线约92.2秒；不包括研究、API、排队和恢复成本，不能当作整体速度优势。

MCMC42第100收费源的已知信号竞态不在本次81源前缀内，后续必须保留该拒绝及实现缺陷说明。阶段HTML、PNG/PDF已更新至81源，保持“训练阶段，非最终结果”标注；证据prefix-20261025-1225.json与progress-20261025-1225.json。


### 2026-10-25 12:55：第六轮提交恢复及第二次基线信号竞态

LDM43第六轮再次触发1800秒会话上限，恢复监督器及客户端均退出。此次四个分支原生submission.json已全部存在；严格核对各自inputDigest、admission的submissionDigest及8个artifact实际源码摘要，均一致。未修改这些原生记录；确认旧写入者已退出后，以原消息字节重放恢复。一次性第三次恢复监督器2663389/客户端2664661，沿用冻结源码和全部历史预算；当前状态看monitoring/recovery-progress.json，禁止重复运行resume脚本。原监督器2665407继续管理其他八组。

MCMC42新增私有接口失败：参数索引121（第122收费源），源码853c06828fbbe01dd9915e8f520a304cec5030081877f3e5458e34aa2e8ad873，作业603d2e6f5345fc17dd390c8571facccbf9bfbdd9a3276639dc9e535412c6fa84同样在取消ITIMER_REAL时SearchDeadline逃逸，worker耗时4.4077秒。私有账本记录interface_rejected_sources；外层tuning收据的status=succeeded表示评分请求完成，但loss=1/legal_rate=0，不是合法求解成功。后续失败率统计必须结合私有账本，不能只按status计数。保留收费和原始收据，未修改冻结科学实现或补评。此前索引99是另一个本地拒绝，两个失败不能合并成一个。

其他八组仍有新评分收据；LDM42完整102，LDM3407完整128并已选中第九轮，六基线完整112–117。全局暂停不存在，验证测试尚未开始；本次无需改动共同81源报告，其预算前缀不包含上述两次MCMC42失败。证据progress-20261025-1255.json及round6-recovery-proof.json。

12:57恢复确认：第六轮四个原生turn_committed均已存在，新增reservoir_expanded(iteration=6)。原提交恢复成功，尚不代表第六轮评分完成；未重新生成候选或增加原提交的模型调用。

### 永久修复验收中（2026-10-25）

用户要求停止依赖逐轮恢复。r3 九组已暂停新派发，正在验收并行只读接口、分离验证等待计时与基线 SIGALRM 清理竞态修复。已有结果及收费预算保留，不重跑收费拒绝。完整回归 217 项通过、41 项跳过；全容器压测及开发私有闭环完成前不恢复正式实验。操作和续跑版本边界见 `copper-measurement-r3-qualification.md`，当前结果仍为阶段结果，不包含正式验证/测试结论。

永久修复验收已通过并续跑：217 Python 回归、14 sidecar 容器测试、1,308 满负载作业、192 并行接口及 102 私有开发闭环收据通过。监督器 993900 接管原九组，118,858 历史文件摘要核对不变；预算未重置。此次开发闭环不是正式验证/测试，正式结果仍待训练完成。

### 2026-10-25 13:54（北京时间）：修复后同预算曲线

修复后九组持续推进，快照新增3,347份作业收据，无新作业超时、信号失败或共享暂停。仍在补完暂停前收费批次，三组LDM尚无新的provider响应，不能把此检查称为新提交计时修复的正式实战验证。六基线仍各129收费源，尚未进入使用修复模板的未收费参数段；已收费源码保持原字节。

各组连续已结算/已收费：LDM种子42、43、3407分别109/113、83/99、129/145；MCMC为118/129、124/129、113/129；ACQ为113/129、123/129、113/129。连续已结算含收费拒绝，评分记录数可能乱序，二者不能混用。LDM分别处于第7/6/9轮评分；六基线第8批处理中。

共同完整83收费源下，损失均值±样本SD：LDM **0.419540±0.031546**，MCMC **0.414481±0.001785**，ACQ **0.414724±0.001805**。LDM平均损失比MCMC高约1.22%，比ACQ高约1.16%，三个种子的稳定性仍明显较差，尚无稳定优势。LDM三种子损失为0.386694、0.422326、0.449601，仅42优于对应基线。

同前缀赢家平均求解时间（跨发现种子均值±样本SD，秒/case）：LDM **63.48±32.50**，MCMC **92.27±0.07**，ACQ **92.23±0.11**。三种子平均累计已完成solver作业墙钟之和分别49.88/102.02/101.61小时；这些是并行作业时间相加，不是实际经过时间或CPU小时，均不含API、接口、排队及恢复，因此不能认定端到端发现速度优势。

旧81源曲线逐点核对全部保持不变，83源的改善不能归因于修复重评。HTML、四面板PNG/PDF已更新，聚合证据为 `copper-measurement-study-r3/prefix-20261025-post-repair.json`。当前仍是训练阶段，正式验证/测试未开始。

### 2026-10-25 14:36：共同99源前缀

修复后新sidecar已成功完成LDM42第8轮四profile的原生提交并选中16候选进入评分，没有再次手工恢复。九组无新作业或provider错误，六基线进入修复模板段。共同99收费源下，LDM损失0.417507±0.029904，MCMC0.414481±0.001785，ACQ0.414724±0.001805；赢家求解均时67.74/92.27/92.23秒。累计已完成solver作业墙钟之和（三发现种子均值）64.95/121.71/120.89小时，不含API、接口、排队和恢复。LDM仍无稳定优势；正式验证测试尚未开始。99源对比曲线和报告已更新，原83源证据保留。

### 2026-10-25 16:36：共同115源前缀

九组健康推进，无新超时/信号或provider错误。同115收费源损失均值±样本SD：LDM0.415582±0.032033，MCMC0.414481±0.001785，ACQ0.414724±0.001805。赢家平均求解68.17/92.27/92.23秒；累计已完成solver作业时长之和均值80.00/140.97/140.57小时，均不代表含API、排队、接口和恢复的端到端耗时。LDM平均损失接近基线但波动仍明显更大，未体现稳定优势。当前前缀已包含MCMC42第100收费源历史拒绝，严格保留预算；第122源失败尚未纳入。115源曲线/HTML已更新，99源历史曲线核对不变，正式验证测试未开始。

### 2026-10-25 18:36：共同131源前缀

九组正常推进，无新增异常，已更新131源曲线及报告。LDM平均训练损失0.412735±0.034595，MCMC0.414481±0.001785，ACQ0.414724±0.001805；LDM均值略低0.42%/0.48%，但种子间波动远大于基线，仅种子42胜出，不构成稳定优势或泛化结论。赢家平均求解79.78/92.27/92.23秒，累计已完成solver作业墙钟时长之和的三种子均值92.60/160.23/160.24小时，不包含API、接口、排队和恢复。前缀含MCMC42两次历史拒绝以及基线130–131源修复模板，严格保留预算和版本说明；旧115源曲线不变。正式验证测试未开始。

### 2026-10-25 20:06：共同133源前缀

九组继续正常训练，修复后无新增超时或信号失败。共同完整收费前缀133，损失与赢家耗时结论不变：LDM0.412735±0.034595、MCMC0.414481±0.001785、ACQ0.414724±0.001805；赢家平均求解79.78/92.27/92.23秒。累计已完成solver作业墙钟之和的三种子均值94.75/162.69/162.70小时，不含API、接口、排队、恢复成本，不等于实验实际历时。LDM仅种子42优于对应基线，稳定性仍弱，未完成独立验证测试，不能据此宣称泛化优势。133源报告与曲线已更新，131源历史数据不变，历史收费拒绝和修复模板版本边界继续保留。

### 2026-10-25 20:38：共同147源前缀

阶段主损失（均值±样本SD）：LDM0.402241±0.048386，MCMC0.414481±0.001785，ACQ0.414724±0.001805。LDM均值约低2.95%/3.01%，优势仍集中在种子42，稳定性弱于基线。赢家平均求解98.02/92.27/92.23秒，LDM此预算点不再有平均求解速度优势。累计完成solver墙钟之和均值110.83/179.94/179.95小时，不能解释为包含API、接口、排队与恢复的发现成本。

新增两份LDM42候选被接口拒绝：类型错误、5秒真实求解超限，共6份失败接口收据；均保留收费，原序162/163，尚在147源比较前缀之外。没有发现之前宿主串行校验超时或基线信号竞态复发。研究流程已自行继续，监控汇总补齐新拒绝原序并验证旧曲线未变；九组继续训练，未开展正式验证测试。报告和147源曲线已更新。

### 2026-10-25 22:36：共同163源前缀

同预算损失均值±样本SD：LDM0.400637±0.046384，MCMC0.413679±0.001084，ACQ0.413912±0.001359。LDM均值较低但仍只有种子42领先，赢家平均求解98.93秒，基线92.31/92.32秒。累计完整solver作业墙钟之和均值128.68/199.65/199.66小时，不含API、接口、排队和恢复开销。曲线已更新，147源历史数据未变；当前前缀已计入LDM42两个新收费拒绝。

ACQ43第221源新增一次私有接口真实执行超限（5.65秒、约5.83 CPU秒、无OOM），源码摘要匹配信号修复版，未发现SIGALRM异常栈，具体计算热点尚未确定。按协议保留收费拒绝，不改冻结代码或免费重跑；它位于当前163源比较范围之外。九组继续推进，尚无正式验证测试结论。

### 2026-10-26 00:36：基础设施暂停，正式未恢复

Docker启动命令40秒超时触发全局暂停，九客户端已退出，历史结果及收费保留。已盘点36个未收据作业，拟降低容器启动并发4→1，在独立完整1308作业/128峰值验收通过后按原容器、原源码接管；12次控制面探针通过不代表完整验收通过。当前163源阶段报告仍是最后核实的同预算比较，不代表正在正常训练，也不包含正式验证测试。详见资格文档与docker-incident-20261026-0036.json。

2026-10-26 01:08更新：启动并发降1的128作业完整验收因2个接口硬超时未通过，正式保持暂停。低负载12次诊断未复现；现进行96作业（88+8）完整1308容器验收，全部通过后才恢复，科学预算及历史收费不变。


2026-10-26 01:40：96并发（88求解/评分+8接口、startup=1）完整1308作业全部通过。首个续跑控制器因遗漏queue/startup/PAUSED.json退出；子队列标记仍是00:21事故（PID2797426，time1792945284.101233），不是新候选超时。主队列故障文件又被监督器通用错误覆盖，已修正后续控制器不覆盖已有首故障。新恢复逻辑严格核对全部暂停标记原字节，未知/变化标记拒绝恢复，先归档子队列再归档主队列；三项远程回归通过。确认旧九客户端和监督器退出，再重新核对36旧容器和全部账本/收据/结果摘要后接管，未更改科学源码、数据或预算。唯一新监督器continue_docker_control96_reconciled_20261026.py（启动PID3919838），操作目录controller-epochs/docker-control96-reconciled-20261026；所有旧门控/续跑入口不可重复执行。96完整验收不重复消耗；本次属于恢复状态协调修复。


2026-10-26 02:06监控：新监督器3919838及九组命令正常，主/启动队列均无暂停，恢复后新增2787作业收据，各组最新收据0–23秒。没有新增Docker、接口硬超时或信号异常。LDM42新增36份失败接口收据均为候选自身float.copy类型错误，保持收费拒绝，不人工修候选或免费重评。原序聚合器仍严格拒绝输出：LDM42目前有9个收费但未入selected的拒绝源，旧聚合只登记其中2个，新增7个须结合验证journal定位原序；不删除断言，不把预留计为完成。故本次不替换已核实163源比较曲线，独立验证测试未开始。证据copper-measurement-study-r3/health-20261026-0206.json。


2026-10-26 02:36监控：九组及监督器3919838正常，全部队列无暂停，各组新收据3–13秒内；相对02:06无新增失败。已根据第13轮验证journal错误指针与artifact摘要、时间确认LDM42新增7拒绝的原序196–202，聚合严格集合断言保留且通过。共同完整收费前缀179：LDM0.400263±0.045928、MCMC0.413679±0.001084、ACQ0.413912±0.001359；赢家平均求解103.89/92.31/92.32秒。LDM仍仅42领先，稳定优势未证实。连续前缀LDM202/179/225，六基线225–236。HTML及PNG/PDF曲线已更新、检查，独立验证测试未开始。ACQ3407接口超时收据mtime为00:21:40，属于旧Docker事故期间，不能误作本次新增异常；收费保留。证据monitor-20261026-0236.json。


2026-10-26 03:06监控：新增3059份作业收据，九组最新收据2–23秒；监督器3919838、九服务及客户端、固定epoch模块摘要正常，主队列和启动队列均未暂停。失败集合逐项不变，无新增超时、信号或provider错误。连续前缀LDM202/179/225；六基线225–241，MCMC43已完成241源。共同179源的损失、赢家求解时间及全部历史曲线逐点不变，无需重复绘图。仍在训练，正式封存验证测试尚未开始。证据copper-measurement-study-r3/monitor-20261026-0306.json。


2026-10-26 03:36监控：新增3165份作业收据，九组最新收据1–16秒；监督器3919838、九服务及客户端、固定epoch模块摘要正常，主/启动队列无暂停。失败集合逐项不变，无新增超时、信号或provider错误。LDM第13/12/15轮评分持续推进，完整前缀202/179/225；六基线225–241，MCMC42/43与ACQ43均完成241源。共同179源损失、赢家求解耗时及历史曲线逐点不变，无需重复绘图。正式封存验证测试尚未开始。证据copper-measurement-study-r3/monitor-20261026-0336.json。


2026-10-26 06:06 CST update: combined async Unix submit/status plus real-container qualification passed 1308/1308 jobs, peak 64, zero failures. The one-shot gate exec-transferred PID 3294205 to OPS/continue_async64_20261026.py. All nine clients and nine services are alive; scheduling-epoch confirms 64 total slots (56 solver/scoring + 8 interface), startup 1, per-process 32. Both pause markers were reconciled and archived; no queue pause remains. Continuation checked 233177 retained accounting/receipt/result files. Post-start audit observed 413 new private receipts, all exit_code 0, with new receipts in every group. This is evidence of resumed execution, not merely process liveness. Scientific identity and per-candidate budgets remain unchanged. Snapshot monitor-20261026-0606.json still supports only common charged prefix 179; comparative losses/timing remain unchanged and formal validation/testing has not started.


2026-10-26 07:36 CST: supervisor3294205 and all nine clients/services continue, no queue PAUSED. Common complete charged prefix advanced to195 (LDM prefixes218/195/241; six baselines241). LDM loss0.399281 +/-0.044755; MCMC0.413679 +/-0.001084; ACQ0.413912 +/-0.001359. Winner mean solver seconds102.82/92.31/92.32; cumulative completed solver hours171.25/239.10/239.10 exclude API/interface/queue/recovery. Only seed42 leads; no stable superiority established. HTML and PNG/PDF curves regenerated from monitor-20261026-0736.json and PNG visually checked. New LDM3407 private interface receipt69d7ec1e... failed its5-second execution budget: worker5.012s, CPU5.090s, exit137, noOOM, no signal traceback. Paid rejection retained (group now2); specific hotspot unproven. No new infrastructure/transport failure; no restart or free reevaluation. Formal validation/testing remains unstarted.


2026-10-26 09:08 CST throughput audit: LDM43 round13 selected at07:30:57, all16 candidates actively progressing; each has20-27 of48 case receipts, total384/768, with135 cases completed in last30min. Docker snapshot found12 active private LDM43 containers. Prefix195 waits for candidate196 to complete all48 cases; partial/other unordered completions do not advance it. Recent LDM43 rate270/266/204 cases per hour over30/60/120min; current batch remaining384 cases implies about1.4-1.9h at these rates, allow1.5-2.5h for tails. Across nine groups remaining training workload31737 case-equivalents (future sources assumed admitted, known rejections excluded); aggregate rates1994/2044/2010.5 cases/hour imply15.5-15.9h uninterrupted workload floor. Group-specific rates and LDM sequential research/admission barriers justify planning20-28h remaining training (Oct27 about05:00-13:00 CST), not a completion guarantee. Faster groups releasing slots may shorten this; new incidents extend it. Formal validation/test and report delivery are excluded. Evidence: throughput-20261026-0910.json.


2026-10-26 10:37 CST: nine clients/services remain healthy under supervisor3294205, no PAUSED and no new failed receipts versus10:06. LDM43 round13 progressed to738/768 case receipts (96.1%); the shared complete charged prefix has advanced from195 to196. LDM prefixes232/196/257; six baselines257. Updated report.html and training-prefix-196 PNG/PDF from monitor-20261026-1036.json. Training remains in progress; no formal validation/testing yet.


2026-10-26 11:07 CST: LDM43 round13 has all768/768 case receipts and checkpoint completed_rounds13; round14 selected and recent provider response confirms native continuation. LDM42 completed14/selected15, LDM3407 completed16/selected17. Common complete prefix211, LDM234/211/257, baselines257-258. No new failed receipts or PAUSED, all nine clients/services active under3294205. Updated HTML/PNG/PDF at211 sources: LDM0.399074 +/-0.045104, MCMC0.413679 +/-0.001084, ACQ0.413912 +/-0.001359; winner seconds102.51/92.31/92.32. Only LDM seed42 leads; training-only result, formal validation/test not started. Snapshot monitor-20261026-1106.json.


2026-10-26 14:36 CST: LDM43 completed round14 (225 private results plus2 charged research rejections), advancing the common complete charged prefix to227. LDM42 completed round15, prefix250; LDM3407 prefix273, baseline prefixes273-288. All nine clients/services and supervisor3294205 remain active, no PAUSED and no new failed receipts versus14:06. LDM training loss0.398862 +/-0.044970; MCMC0.413679 +/-0.001084; ACQ0.413912 +/-0.001359. Winner mean solver seconds110.21/92.31/92.32. Only seed42 leads; no stable superiority established. HTML and PNG/PDF regenerated from monitor-20261026-1436.json. Formal validation/testing has not started.


2026-10-26 18:06 CST: LDM43 completed round15; common complete charged prefix243. Snapshot prefixes LDM261/243/289, baselines289-305. All nine clients/services remain active under supervisor3294205, no PAUSED, no new failed receipts or provider errors versus17:36. LDM loss0.397510 +/-0.046646; MCMC0.413679 +/-0.001084; ACQ0.413912 +/-0.001359. Winner mean solver seconds106.29/92.31/92.32. Only LDM seed42 leads; no stable superiority established. HTML and PNG/PDF updated from monitor-20261026-1806.json. Formal validation/testing remains unstarted.


2026-10-26 21:36 CST: first two training groups completed: MCMC42 and MCMC43 each have complete charged prefix321 and client exit_code0. MCMC42 retains two charged rejections (319 private results), MCMC43 has321 results; no replacement evaluations. Remaining seven groups continue under supervisor3294205; all nine private services remain available, no PAUSED or new failed receipts. Common prefix244; LDM prefixes282/244/305. LDM42 completed round17, LDM43 round16 scoring, LDM3407 final round20 scoring. Training losses and winner times unchanged from243-source snapshot. Report regenerated at244 sources from monitor-20261026-2136.json. Wait for all groups before reconciliation/sealing/validation; formal validation and testing have not started.


2026-10-26 22:06 CST: five groups now completed321 charged sources and exited0: all three MCMC groups plus ACQ43 and ACQ3407. Paid rejections retained; private result counts respectively319/321/321 and318/320. ACQ42 and three LDM groups still train, no PAUSED or new failures. LDM rounds completed/selected:17/18,16/17,19/20. Common complete prefix259; LDM282/259/305, ACQ42 prefix305. Training loss LDM0.396781 +/-0.045885, MCMC0.413679 +/-0.001084, ACQ0.413912 +/-0.001359; winner solver seconds111.59/92.31/92.32. Only LDM seed42 leads. Report HTML/PNG/PDF updated from monitor-20261026-2206.json. Formal validation/testing still awaits all-group training completion and reconciliation.


2026-10-26 22:36 CST: all six baseline groups completed321 charged sources and exited0; ACQ42 now has321 private results. Three LDM groups continue, no PAUSED or new failed receipts. LDM42 round18 has462/768 case receipts, LDM43 round17 has387/768, LDM3407 final round20 has615/768 at live audit. Common complete prefix remains259; comparative report unchanged. Snapshot monitor-20261026-2236.json saved. Formal validation/testing waits for all LDM training and all-group reconciliation/sealing.


2026-10-26 23:06 CST: LDM3407 completed round20, complete charged prefix321,319 private results and client exit0. Together with all six baselines,7/9 groups have completed training. LDM42 completed round18 (prefix298), LDM43 completed round17 (prefix275); both continue native research. No PAUSED or new failures. Common complete prefix275; LDM loss0.385336 +/-0.035978, MCMC0.413679 +/-0.001084, ACQ0.413912 +/-0.001359; winner solver seconds112.00/92.31/92.32. Training-only improvement does not establish generalization. HTML/PNG/PDF updated from monitor-20261026-2306.json; formal validation/testing remains unstarted pending all-group completion/reconciliation.


2026-10-26 23:36 CST: seven groups remain complete; LDM42 round19 and LDM43 round18 continue. Collector initially refused an order mismatch caused by three new paid LDM42 research rejections. Round19 validation error pointers, artifact digests and journal times identify charged positions299-301, all preceding round19 selection. Collector now inserts these proven positions and retains strict uniqueness/set assertions; remote collection passes. One inspected candidate traceback is IndexError in _remove (3D mask indexed with26 indices), not a transport or signal failure. Three new exit1 receipts retained, no refunds or reevaluation. Evidence OPS/round19-rejection-order-20261026-2336.json. No PAUSED, formal execution continues. Complete prefixes LDM301/275/321, all baselines321; common275 metrics/report remain unchanged. Formal validation/testing not started.


2026-10-27 00:06 CST: seven groups remain complete; LDM42 completed round19 (317 charged sources), LDM43 completed round18 (291). No PAUSED or new failed receipts/provider errors versus23:36. Common complete prefix291; training loss LDM0.378665 +/-0.034140, MCMC0.412551 +/-0.001178, ACQ0.412577 +/-0.001159; winner solver seconds112.58/91.95/91.99. HTML/PNG/PDF updated from monitor-20261027-0006.json. These are training-only comparisons, not formal validation/test results. LDM42 has4 remaining charge slots, LDM43 has30; preserve the321 cap and paid rejections.


2026-10-27 00:36 CST: LDM42 completed round20 and exited0, reaching321 charged sources (308 private results plus13 paid research rejections). Eight of nine training groups are complete. Final round admitted three sources after one new paid rejection at position318; validation error pointer/artifact SHA and journal time prove its original position. Collector includes round20 evidence with strict uniqueness/set assertions retained; remote collection passes. Six new exit1 interface receipts belong to the rejected candidate; no PAUSED or provider errors. LDM43 continues round19 scoring,307 charged and common complete prefix291, with1302 new job receipts since00:06. Report curves remain at291, formal validation/test not started. Evidence monitor-20261027-0036.json and OPS/round20-rejection-order-20261027-0036.json.


2026-10-27 01:06 CST incident: LDM43 completed round19,307 charged sources fully accounted (305 private results plus2 paid rejections), no pending private evaluations. Before round20 researcher execution, harness_turns100+1 exceeded frozen limit100; client exited1 and supervisor3294205 exited after asserting all exit codes zero. Main PAUSED retained. No new failed receipts/provider errors versus00:36; this is budget exhaustion, not a new socket/Docker failure. Ledger has80 copper turn keys and20 main turn keys; whether recovery changed turn identity requires further audit, not assumed. Do not reset counters, increase limits or blindly restart. Evidence OPS/harness-budget-incident-20261027-0106.json pins pause/budget/checkpoint/event/status digests. Common complete prefix307; LDM0.376954 +/-0.036712, MCMC0.411590 +/-0.000895, ACQ0.411598 +/-0.000993. Stage report updated; formal validation/testing unstarted. Eight groups completed321; LDM43 has14 unused source slots but no remaining harness turns.


2026-10-27 01:36 CST audit: formal pause remains. Exactly76 of80 charged researcher input digests match persisted inputs across19 rounds x4; four keys are unmatched. Three have only zero-validation usage and one has no usage record. Historical snapshots already show three excess turns by round6, so the discrepancy predates async recovery; the fourth may be partial round20 reservation, not yet cryptographically mapped. Historical round5 logs record partial digest mismatch and exact-input serialization replay, but this does not yet prove which unmatched keys are duplicates. No counter correction, limit increase, restart or formal holdout access performed. Evidence turn-identity-audit-20261027-0136.json; continue evidence mapping and a development-only regression before any qualified recovery. Common prefix307 and report unchanged.


2026-10-27 02:06 CST: exact digest reconstruction now proves three extra harness charges are round5 researcher2/3/4 serialization aliases. Sorting only nested metrics keys reproduces fcadd072...,8952a20e...,4d601cac...; metadata and parsed messages equal persisted inputs, both charge keys exist, and alias usage has zero validation submissions. This identifies duplicate input accounting; it does not authorize budget resets. Remote in-memory identity regression passed33 checks (equivalent serialization accepted; metadata/history/message mutations rejected), not full recovery acceptance. Formal ledger, scientific source, results and PAUSED unchanged. Remaining unmatched918838c8... still requires exact round20 reservation mapping; then an append-only audited operational accounting repair and recovery qualification are needed before continuation. Evidence OPS/turn-identity-audit-20261027-0206.json and turn-identity-regression-20261027-0206.json.


2026-10-27 02:36 CST: remaining unmatched918838c8... reconstructed exactly as researcher_1_round_20 from saved round20 measured history, fork plan/snapshot cursor305 and frozen branch instructions; evidence OPS/round20-first-reservation-reconstructed.json. All80 raw researcher keys now accounted:76 completed-round inputs,3 proven round5 serialization aliases,1 legitimate round20 pre-reservation. Logical total including20 main turns is97, not100. Development-only copied ledger reconciliation retains every raw cumulative_usage entry and records original hash/raw100/alias evidence while normalizing unique-turn counter97, limit100 unchanged. Eight remote checks passed: first reservation replay unchanged, three remaining branches reach100,101st rejected, source cap321 enforced, original usage retained, formal bytes unchanged. Evidence turn-budget-development-20261027-0236.json; this is not full recovery acceptance. Production canonical-input accounting, crash/replay regression and guarded one-shot takeover remain to implement. Formal PAUSED, source/data/ledger/results unchanged, no holdout access.


2026-10-27 03:06 CST: implemented external durable_turn_identity_20261027.py, SHA256 c5e4fb564de5f47babfdd2ab5c6d9a9ffc744b74ec4c5e3489cd6590f5c95bef. It persists exact researcher input before existing budget accounting; replays require equal metadata and parsed message, native/journal disagreement or corrupt digest fail closed. Existing frozen runtime already handles persisted native inputs; this adds protection for the pre-sidecar reservation crash window. Seven remote development checks passed including reload after reservation, reordered JSON replay charged once, changed metadata/semantics rejection, corrupt journal rejection, unchanged budget-limit rejection. Module is NOT deployed. Ledger-copy eight checks and identity33 checks remain separate; native resume integration and guarded takeover still required. Formal pause/budget/results unchanged. Evidence OPS/durable-turn-regression-20261027-0306.json.


2026-10-27 03:36 CST: implemented evidence-bound reconcile_turn_budget_20261027.py (eb5c7853601863f5a94028979c021ae40ee6adee78c2b63bade23ccb54c17d77) and layered turn_repaired_runtime_20261027.py (43aad2269249e75f25323b4a674506e28c469a114a000ad7a717de28d7e5438a). Correction validates all original input hashes and serialization aliases, retains byte-exact before image and all cumulative_usage, records correction metadata, leaves limits unchanged, refuses repeated correction after further progress. Eight remote copied-ledger recovery tests passed (idempotency, before-image crash, prepaid replay,101 rejection, progressed-ledger and tampered-evidence rejection). Real wrapper/epoch/module installation integration passed with dispatch stubbed: original digest preserved, layered replay charged once, qualified async transport installed. No formal mutation or native research dispatched. Inventory: all old clients/supervisor exited, zero pending private evaluations or receipt-less jobs;22 running shared-host containers still require mount/ownership classification. Evidence turn-recovery-regression-20261027-0336.json, turn-runtime-integration-20261027-0336.json, turn-recovery-inventory-20261027-0336.json. Next: finish container classification, pin and implement guarded one-shot takeover, preserve all receipts/results hashes and existing science policy; formal still PAUSED.


2026-10-27 04:06–04:10 CST: guarded one-shot continue_turn_identity_20261027.py started supervisor1334269/client1338181. Seven orphaned early-round native sidecars identified by exact mounts/image/command and stopped; unrelated shared containers untouched. Old clients and services exited, zero unfinished jobs/evaluations verified,319613 existing ledger/receipt/result files hashed and retained. Original LDM43 budget bytes archived under controller-epochs/turn-identity-20261027/budget-correction; three proven serialization duplicates reconciled to97 unique turns with all raw usage retained and limits unchanged. Seeded the exact prepaid round20 first input; main turn20 replay did not charge again. New round20 has4 native inputs and28+ fresh provider responses, unique turns now100/100; no PAUSED. Actual native research resumed, not yet new private scoring or complete source prefix. Common307/report unchanged; formal validation/test still pending. Existing controller now owns remaining LDM43 training, all-group reconciliation/seal/finish; never rerun it or old gates. Evidence OPS/turn-recovery-monitor-20261027-0406.json.


2026-10-27 04:36 CST: LDM43 final round20 selected14 sources, reaching321 charged and319 private charged (two historical paid rejections). All14 are scoring,324/672 private case receipts at live audit; no new failed job receipts or provider errors since01:06.940 new job receipts, latest11 seconds; supervisor1334269/client1338181 and nine services active, no PAUSED. Unique harness100/100, main20/20 unchanged. Common complete prefix307 and curves unchanged. Collector command recognition updated for explicitly versioned turn_repaired_runtime_20261027.py; previous expected_command false was filename mismatch, not absent execution. Snapshot monitor-20261027-0436.json otherwise retained as captured. Formal validation/testing still awaits final14 results and all-group reconciliation.


2026-10-27 05:06 CST milestone: all nine training groups completed321 charged sources; LDM43 exited0 with319 private results and2 preserved rejections. All nine reconcile_discovery entries passed, all nine training-sealed.json records exist before finish dispatch. Supervisor1334269 now private_validation_test, finish PID2258205; no PAUSED or new training failures. All nine validation directories have fresh receipts (42–60 per group sampled), state frozen; winner selection and ID/OOD tests not yet confirmed. Common321 training report HTML/PNG/PDF updated: LDM0.374606 +/-0.036835, MCMC0.411590 +/-0.000895, ACQ0.411598 +/-0.000993; winner seconds112.59/91.56/91.55. Training-only result; seed43 still does not outperform baselines, no generalization claim. Evidence monitor-20261027-0506.json, OPS/private-stage-20261027-0506.json and P/budget-reconciliation.json. Do not restart training or duplicate finish.


2026-10-27 05:36 CST: all nine private validation selections completed; all state phases selected and canonical selection SHA256 checks passed. Each validation directory has162 receipts, no failed status/exit codes. Fixed-winner tests are progressing (66–90 receipts per group sampled, latest9–68 seconds), no failed receipts. All nine selection timestamps precede the earliest test receipt. Supervisor1334269 and finish2258205 remain active, no main PAUSED or completion.json. Evidence OPS/private-stage-20261027-0536.json. Final ID/OOD evaluation and report remain pending; do not rerun finish or feed heldout scores back into research. Training report unchanged at321 sources.


2026-10-27 07:06 CST: formal evaluation completed; completion.status completed, replay_receipts_unchanged true,5454 receipts. All nine phases tested, no main PAUSED; supervisor/finish exited. ID96 and OOD48 solves per group, legal_rate1 and failed_solves0. Across discovery seeds, ID loss LDM0.361726±0.032651 / MCMC0.415236±0.000979 / ACQ0.415201±0.000595; OOD LDM0.395486±0.083478 / MCMC0.417936±0.001172 / ACQ0.418215±0.001086. LDM all seeds lead ID, only42/3407 lead OOD;43 OOD0.491752. Solver112.56s vs91.1s. No stable OOD superiority claim. Test data now seen and must not feed research. Saved formal-test-summary-20261027-0706.json, formal-test-metrics.csv, formal-test-results.html and formal-test-loss.png/pdf in report directory. Full technical delivery remains unfinished: algorithm analysis, same-coordinate truth/prediction3D and slices, complete case/execution-seed tables and cost breakdown. Keep monitor active for report delivery, do not rerun experiments.


2026-10-27 07:36 CST report work: exported all1296 test case/execution rows,54 parent-equal execution-seed aggregates and nine hash-verified frozen solver sources. Added algorithm-analysis.html; generated same-coordinate ID/OOD geometry and central orthogonal overlap/miss/false-positive slices (PNG/PDF). Case selection fixed to lexicographic minimum case_id per split, discovery42/execution47. ID3D and OODslices visually inspected;3D labels need more margin, inactive-region overlays still missing. Script OPS/export_report_geometry_20261027.py, remote report-geometry-20261027. Full cost reconciliation, complete visual QA and final merged report remain pending. No experiment rerun.


2026-10-27 08:36 CST report work: reconciled3726 private validation/test container jobs (270 interfaces,1728 solvers,1728 scorers), zero failures and no missing workload resources. Completion5454 counts1728 additional case-level receipts, not additional executions. Saved holdout-cost-summary-20261027-0836.json. Summed research/training/holdout admission waits from jobs/scheduling records into queue-cost-summary-20261027-0836.json; parallel wait sums are not elapsed delay and interrupted jobs may lack records. Both cost tables integrated into cost-accounting.html. Remaining: recovery/qualification costs and chronology, Chinese merged final report, full visual/link/data QA and delivery; automation remains active.
