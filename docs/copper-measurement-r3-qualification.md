# 铜矿 r3：计时修复、饱和负载与九组试跑门槛

## 2026-10-27 后续基线适配审计（独立开发任务）

r3已完成并保持封存。按用户要求，在独立开发副本对齐r3公开先验后，定位到约94.5%提议因超过4连通体被拒绝，导致ACQ有效样本不足以启动；一阶距离只解决独立数值符号翻转问题。可行投影＋一阶格式通过32次开发验收、14项数值检查和10项pytest，未放宽硬约束或修改冻结参数。投影改变提议分布，修复版不主张严格后验MCMC。

432次同实际时间开发验证全部完成，0失败、0基线invalid、ACQ 144/144实际多候选决策；29份r3控制文件摘要不变。30秒档公开损失LDM/MCMC/ACQ为0.122503/0.557041/0.560510，90秒档0.092483/0.547681/0.556719；90秒IoU为0.182236/0.133772/0.134205。无独立确认测量，不能当作新泛化结果。早期未对齐公开先验的开发比较已撤销并保留INVALID-POLICY记录。全部协议、源码、逐次数据和限制见[后续审计报告](copper-baseline-audit-20261027/report.html)。不重跑r3，不用已揭示测试调参。

## 为什么 r2 的小样本验收不足

r2 在低负载容器验收中通过，但正式并发后出现 7 个接口超时及一个 ACQ 连通性异常。旧结果和收费记录完整保留，九组停止，没有原地修订算法、退款或免费补评。

计时审计发现：接口容器内的 5 秒硬限覆盖解释器、输入读取、候选导入和求解；传给候选的预算却仍为完整 5 秒。宿主端还从 `docker start` 请求前计时，仅留 3 秒启动/回收余量。并发时启动开销和实际可用求解时间不一致。旧失败收据没有完整可信工作进程计时，因此不能将每次超时都精确归因于其中一种机制。

另一个确定缺口是连续距离场重建可能使接近零的数值改变符号。此前只验证重建前占据，重建后可能违反连通性约束。固定随机开发样本已复现符号漂移，补充回归保留正常亚体素位移累积，同时拒绝重建造成的占据变化。

## 本次修订

- PID 1 设置单调时钟截止时间；候选收到导入和读取后真正剩余的预算。核心 `timeout` 仍限制整个工作进程，求解上限没有从 120 秒放宽。
- 宿主 watchdog 改从 Docker 实际启动时间计时，为启动/可信回收留有界的 30 秒基础设施余量；这不增加候选的工作进程预算。超过该余量按基础设施暂停处理，不伪装成科学接口拒绝。资源收据增加工作进程耗时和硬预算字段。
- 源码封存改为先写完、同步并只读化临时 inode，再原子链接到摘要路径；已存在且内容不同的快照仍拒绝，不覆盖。旧版对照压测实际触发了并发封存竞态，已保留失败记录，不能把该对照轮算作通过。
- 参数表在启动并发线程前固定生成，避免压测工具并发首次初始化 SciPy Sobol 表。正式参数搜索本来就是先固定计划再并发，此项属于验收工具修正。
- 基线计时起点同步移到 `solve` 入口，避免运行器扣除导入开销后又从模块导入前计时、重复扣除开销。验收门槛同时核对运行器科学摘要和基线模板摘要；修改基线后必须重新验收。

第一轮修订压测（`load-full-r4`）828 作业全部通过，峰值 80；612 次接口工作进程最大耗时 3.83 秒，而容器额外启动/回收开销最高 11.66 秒，直接证明 3 秒基础设施余量不足。该轮的基线仍有重复扣时，因此仅保留作诊断，不能批准最终版本启动。预算适配版 `load-full-r5` 已完成 828 作业、峰值 80，出现 2 次真实 ACQ 接口硬超时（工作进程分别 5.024/5.035 秒，无 OOM），因此验收失败，自动后续门槛已经阻止私有验收及试跑。低负载诊断重放两例均在约 4.15 秒结束，说明小样本通过不足以证明并发可靠性。

基线原先仅在提案之间检查截止时间，无法中断已开始的随机场生成/代理拟合。新增基线软截止中断，保留并返回已验证的最佳解，容器硬上限不变；专门的慢提案回归通过。注意：旧失败没有函数级栈，尚不能确定每个失败具体卡在哪个库调用，不能把“越界风险已确认”表述为每例调用栈都已定位。修订版将重新完整压测（`load-full-r6`），不得复用失败轮作验收证明。

## 恢复门槛

1. 小样本真实容器接口检查全部通过。
2. 饱和负载：72 个求解槽加 8 个接口槽，8 个独立控制器、每控制器不超过 16 作业。覆盖两种基线、三个种子、首轮全部 17 个参数配置、六个接口样例，共 612 次接口检查；216 个 120 秒上限的背景求解维持压力。要求全部合法、无执行失败，实测活跃作业峰值达到 80。
3. 上次 ACQ 失败的训练 case、原参数，在修订版中以执行种子 47/48/49 重做独立容器评分。这是额外开发回归，不能改写原正式失败收据。
4. 重新通过私有封存→验证选型→ID/OOD 测试和幂等重放验收。
5. 九组开发试跑：LDM/MCMC/ACQ ×42/43/3407，初始化加一个 16 源预算轮，各 17 源；每次求解仍是 120 秒/2 CPU/4 GiB，使用开发地质体。检查模型 API、源码生成、并发预算、拒绝、完整评分、私有封存、选型和测试。开发成绩不用于向正式研究 agent 推荐算法。
6. 以上全部通过后才启动新版本正式九组：初始化＋20×16，共 321 源；相同地质约束、相同独立测量主目标。正式验证/测试数据继续保持未评价状态，训练结束后统一执行私有流程。

启动器已增加饱和负载和九组试跑的强制门槛，未通过不能启动全量。半小时监控在正式新版本启动后恢复。当前验收日志与原始证据位于远程 `/Aidata/shared/ldm/copper-audit-fixes-20261024/`，私有资格代码位于 `/Aidata/shared/ldm/copper-v2-evaluator/code/measurement-r5-qualification`。

## 最终候选版本的验收记录（持续更新）

- `load-small-r6`：12/12 真实容器接口通过。
- `full-container-r5-regression.log`：205 项通过；两个依赖开发数据的项目最初跳过。补齐先前独立开发地质体后，`export-integration-r5.log` 中相关两个测试文件 11 项全部通过，包括这两项。没有使用正式验证/测试数据。
- 新增慢提案中断回归与既有基线回归共 8 项通过。
- `load-full-r6`：828/828 通过，峰值 80；612 次接口工作进程中位 4.513 秒、P95 4.731 秒、最大 4.818 秒。216 个长时背景作业全部通过。原始完成记录摘要见 `copper-measurement-r3-qualification/load-summary.json`。
- `failure-case-acceptance-r5b`：原 ACQ 失败训练样例，执行种子 47/48/49 全部独立评分成功。
- 私有资格部署首轮 `r5` 因新复制代码目录权限 750/文件 640，容器降权后无法导入运行器，已保留失败记录；公开科学代码调整为只读可遍历（755/644），私有数据与密钥权限不变，源码内容摘要不变，使用新验收目录 `r5b` 重新运行。正式启动器本来即使用 755/644，此问题属于本轮资格部署脚本。
- `protocol-acceptance-20261024-r5b`：通过；3 个合法源和 1 个收费拒绝源，97 收据重放不变，超预算、提前测试及封存后训练均被拒绝。正式验证/测试未访问。
- `copper-development-pilot-20261024-r5`：九组一轮开发试跑已启动，每组 17 个收费源上限；等全部训练、预算对账及私有选型/测试完成后，才启动全量。

## 资源目标上调至约 80%（2026-10-25）

服务器实测 224 个逻辑 CPU，MCMC/ACQ 求解的 CPU 秒/工作进程秒中位数约 1.001。用户授权提高资源目标后，新正式调度拟为 180 作业（172 求解/评分＋8 接口），每进程 32；每轮仍 16 候选，每次仍 120 秒/2 CPU/4 GiB。原九组开发试跑继续按已冻结的 80 作业配置完成，不原地改写其策略。

- 新调度相关回归 8 项通过；`load-small-r7-180` 12/12 接口通过。
- 九组试跑训练已完成，进入私有验证/测试。
- 随后执行 `load-full-r7-180`：612 接口＋696 长时背景求解，共 1308 作业，要求峰值 180、全部合法。背景每作业一个忙线程，匹配实测基线负载；两核上限保持不变。这是新并发的独立饱和验收，不能冒称九组端到端试跑本身采用了 180 作业。
- 新启动门槛必须同时匹配科学代码、基线模板和通过验收的 180/8/32 调度配置。

### 180 作业首轮失败与第二轮验收（00:55 更新）

九组开发试跑已经完成：各组 17 个收费源均有完整结果，无本地或私有拒绝；私有封存、验证选型、ID/OOD 测试通过，540 个选型收据幂等重放不变，选型测试合法率均为 1。此证明对应 80 作业策略。

`load-full-r7-180` 未通过。Docker 日志出现 systemd 创建容器 scope 超时；Docker 命令 40 秒超时后，压测控制器仍释放槽位并派发后续作业，造成启动积压。停止时有 710 个 job 记录、669 个本次压测容器，远超同时作业目标；这不能算作有效饱和验收。已停止派发，保留 job 摘要、容器状态和清理记录，仅清理该压测的 669 个容器。没有重启共享 Docker 或改动其他任务。全部成本保留为开发失败成本。

调度层新增独立的 4 槽容器启动许可，并在基础设施结果不确定时原子写入共享 `queue/PAUSED.json`，停止新派发，等待人工/监督器核对；已经运行的作业保留原预算和收据。科学源码、测量目标、基线模板与 120 秒硬上限未修改。启动容量与活跃作业容量分开计数。

- 调度回归：`scheduling-startup-r8-regression.log`，7 项通过，包括故障停止派发、4 槽启动限制和 321 唯一收费源并发幂等。
- `load-small-r8-180`：12/12 接口通过。
- `load-full-r8-180`：1308 作业完整验收已启动，尚未通过。单一控制器为 `qualified_full_r8_controller.py`；只有成功才进入正式启动脚本 `launch_qualified_full_r8.py`。禁止重复执行。
- 新冻结资格目录：`/Aidata/shared/ldm/copper-v2-evaluator/code/measurement-r7-qualification`。
- 正式目标目录：`/Aidata/shared/ldm/copper-measurement-study-20261025-r3`；目录未创建之前不得报告正式实验已开始。
- 每半小时监控已恢复，覆盖验收、正式实验、私有评估和可视化报告。80% 是依据实测 CPU 消耗的资源目标，并非 180 容器必然等于 80% 实际利用率；持续检查 CPU、内存及容器控制面容量。

### 01:25 监督检查：180 作业未获准启动正式实验

`load-full-r8-180` 完成 1308 作业、峰值 180：1298 合法，10 个接口真实触发 5 秒工作进程硬超时，均无 OOM。696 个背景求解全部成功；本轮没有 Docker 命令不确定或宿主监督超时，未触发共享基础设施暂停。启动限流修复了上一轮的容器积压，但尚未消除高负载下短接口超时。因此验收仍失败，控制器已经退出，正式 r3 目录尚未创建。

不放宽接口或求解预算、不修改冻结科学版本。下一轮调度容量改为 **144（136 求解/评分＋8 接口）**，每进程 32，启动同时最多 4；为控制面与短接口保留更多余量。80% 为资源目标，不能只按工作进程数量估算总 CPU，新增宿主 CPU 利用率采样 `load-r9-host-resources.jsonl`。这项容量调整尚待验收，不能提前宣称解决了接口超时。

- 新调度回归 8 项通过，覆盖 144/180 两种容量、本地并发限额、共享故障暂停和收费幂等。
- 单一控制器 `qualified_full_r9_controller.py`（PID 见同目录 `qualified-full-r9-controller.json`）依次执行 `load-small-r9-144` 和 `load-full-r9-144`；完整验收仍为 612 接口＋696 长求解，共 1308 作业，要求全部合法、峰值 144。
- 冻结资格源码 `measurement-r8-qualification`；仅修改调度/验收容量，科学实现与基线模板不变。
- 只有全部通过才调用 `launch_qualified_full_r9.py`，新正式目标目录仍为 `copper-measurement-study-20261025-r3`。旧 r8 控制器和失败目录不得重启。
- 已更新半小时监控。136 求解槽下六组基线训练的理想满载下界约 17 小时，未含其他开销，不承诺清晨完成。

### 01:55 监督检查：实测整机 CPU 超出目标

`load-full-r9-144` 完成 1308 作业、峰值 144，1307 合法，1 个接口（索引 196）触发真实 5 秒硬超时，工作进程 5.015 秒，无 OOM。没有 Docker 基础设施失败，正式启动继续被拦截。整机采样 CPU 忙碌率中位 **85.62%**、峰值 **91.62%**，失败附近 **91.02%**；采样包含其他任务与控制面开销，不能全部归因于求解器，也不能直接证明单个超时的函数级原因。它明确说明按单求解器一核估算会低估总负载。

因此将验收容量调到 **128（120 求解/评分＋8 接口）**，每进程 32、启动限流 4 均不变；接口 5 秒和求解 120 秒硬上限不放宽，科学代码不变。9 项调度回归通过。新控制器 `qualified_full_r10_controller.py` 依次执行 `load-small-r10-128` 和 `load-full-r10-128`，仍需 1308 作业全部合法与峰值 128；宿主采样在 `load-r10-host-resources.jsonl`。冻结资格目录为 `measurement-r9-qualification`，只有完整验收通过才执行 `launch_qualified_full_r10.py`。旧控制器确认已退出，没有重启旧失败目录。监控配置已同步更新。

正式 r3 尚未启动。120 求解槽下，六组基线训练理想满载下界约 19.3 小时，未含其他开销。

### 02:25 监督检查：128 并发验收通过，正式九组已启动

`load-full-r10-128` **1308/1308 全部通过，峰值 128**；整机 CPU 采样中位 **79.33%**、峰值 **82.67%**，符合约 80% 的目标。完整验收收据 SHA-256 为 `04cfc92e7717634b0f181ea638020b424747d8c13d4317e8968957096ce3f2bb`。这不保证以后永不超时，仍持续监控。

验收门槛通过后，唯一控制器已自动启动正式 `copper-measurement-study-20261025-r3`，正式监督器 PID 2665407，约 02:08 开始。已核实部署为非开发模式、20 轮、每轮 16、321 收费源上限、120 秒求解预算；策略为 128/8/32/4，调度源码摘要与策略记录相符，九个客户端和九个私有服务均存活。启动器已经执行，禁止重复运行。

初始检查各组均已收费 1 源；三个 LDM 初始化独立测量损失均为 0.553368（固定公开种子算法），正在第一轮研究；六个基线初始化套件的私有作业收据持续增加。私有执行收据暂未发现失败。尚无九组共同完成预算前缀，不能比较方法优势。

LDM 日志有 provider 502 transport/terminated；框架正在续接未完成会话，最新响应与 turn 提交仍在推进，未人工重启或重置预算。这与容器接口硬超时分开记录，后续继续检查是否停滞。正式验证/测试尚未执行，当前不是最终结果。

### 02:55 监督记录

九组仍在训练，共享基础设施暂停标记不存在。六基线已完成初始化，均预留至 17 源，第一批套件作业持续产出；完整候选结果仍各 1。LDM-42 已完成 17 源结果并进入第二轮研究；LDM-3407 预留 17、完成 7 个候选结果；LDM-43 完成初始化，仍在首轮研究，最近 turn 提交约 7 分钟前，既有会话恢复继续观察，不因历史 provider 报错重复启动。

私有 LDM-3407 出现一个 5 秒接口超时（收据 `fe3dce7d1060f15e0d3aaaee586fc8544b672f85d9f8dd198c3eb1a771456d20`，无 OOM），保留原收费和失败收据，不免费补评、不改预算；其他作业继续推进。此为候选执行结果，尚不足以认定共享基础设施故障，后续核对最终拒绝归账。九组共同完整预算目前只有初始化，暂不作算法优势判断。远程聚合证据：`monitoring/progress-20261025-0255.json`。

### 03:25 监督记录

九组各已预留 17 个收费源，进程和私有服务均存活，调度源码摘要匹配，未触发共享暂停。LDM-42 完成 17 个结果，LDM-3407 完成 16 个结果＋1 个已计费私有接口拒绝，两组均在第二轮研究；LDM-43 已结束首轮研究，正在首批候选评分（完整 1 个，底层作业持续增加）。六基线完整结果分别为 ACQ-42/43/3407：5/5/6，MCMC-42/43/3407：7/8/7；完成顺序未必等于原序前缀。

ACQ-43 的参数索引 13 在客户端接口检查超时，保留拒绝收据及 `interface_rejected_sources=1`，因此客户端收费 17、私有 16，差额有明确证据，不补候选。此前 LDM-3407 的接口失败也已核实计入私有拒绝账本。所有组仍正常推进，未重启或改动冻结科学版本；provider 的历史传输失败没有导致目前研究/评分整体停滞。正式验证测试仍未启动。聚合证据：`monitoring/progress-20261025-0325.json`。

### 03:55 监督记录与共同预算 7 源比较

九组持续推进，未新增私有作业错误或共享基础设施暂停。LDM-42/3407 已预留 33 源，LDM-43 预留 17 源；六基线大部分进入第二批。预留和乱序完成不用于公平比较。

结合 LDM `candidates_selected` 原序事件、私有完整训练结果，以及基线原序 `tuning-receipts`，可核实的九组共同连续前缀为 **7 个收费源**。遇未映射私有拒绝或缺失结果即停止前缀，不跳过失败。三种子最佳训练独立测量损失如下（均值±样本标准差）：

| 方法 | 最佳损失均值 | 三种子标准差 |
|---|---:|---:|
| LDM | 0.460257 | 0.049579 |
| MCMC | 0.507022 | 0.058200 |
| ACQ | 0.506837 | 0.055868 |

LDM 在此早期前缀较低，但不足以判断最终稳定优势或泛化。最佳 LDM 的每 case 求解均时随种子为 91.10、73.10、4.33 秒，基线约 91–92 秒；因此当前优势也不能概括为统一“2 秒胜过 90 秒”。这些是求解时长，不含研究 API、队列和恢复等发现成本。详细源前缀与耗时见 `monitoring/prefix-20261025-0355.json`，进度见同目录 `progress-20261025-0355.json`。正式验证测试未执行。

### 04:25 监督记录与共同预算 15 源比较

九组持续推进，无新增私有执行错误，共享暂停标记不存在。LDM-3407 已完成第二轮（32 完整结果＋1 收费拒绝），第三轮有新 provider 授权；LDM-42 第二批评分推进，LDM-43 第二轮研究活跃。六基线均进入第二批，收据不断增加。客户端与私有账本的 ACQ-43 差额仍为已记录的 1 个收费拒绝。

保守共同原序前缀扩展至 **15 个收费源**（包含 ACQ-43 的一个拒绝槽位；LDM-3407 第 16 位私有拒绝尚未纳入聚合映射，故此处停止，不跳过）。训练最佳损失三种子均值±样本标准差：LDM **0.457556±0.046772**，MCMC **0.422454±0.005501**，ACQ **0.422631±0.005642**。在这个前缀基线更低且种子间更稳定，与 7 源时排序不同；不能用早期单点宣称 LDM 稳定优势。LDM 最佳算法均时按种子为 91.10、94.83、6.44 秒，基线约 92.2 秒，不含发现/API/排队开销。证据为 `monitoring/prefix-20261025-0425.json` 和 `progress-20261025-0425.json`，正式验证测试仍未执行。

### 04:55 监督记录

九组继续推进，未新增私有作业失败或共享暂停。LDM-42 已完成 33 个结果并在第三轮研究；LDM-3407 已预留 49 源、第三批评分活跃；LDM-43 第二批评分活跃。基线第二批继续完成，最近底层收据均有更新。

已按私有账本的 `copper:v2:scientific-source:` 源标识，将 LDM-3407 拒绝源 `e5cedffb32091e31b1a0595b77c9779c50f8b2acbc17a01dec7550ada3f2329c` 映射回候选原序。九组现在都具有 **17 个收费源的完整前缀（初始化＋首轮）**，其中 ACQ-43 和 LDM-3407 各包含 1 个收费拒绝，不免费替换。此预算下各组最佳损失与上次 15 源比较相同：LDM 0.457556±0.046772、MCMC 0.422454±0.005501、ACQ 0.422631±0.005642。当前训练前缀尚未体现 LDM 的稳定优势，继续按固定协议运行。聚合证据为 `monitoring/prefix-20261025-0455.json`、`progress-20261025-0455.json`；正式验证测试保持隔离且尚未执行。

### 05:25 监督记录

九组已具备 **33 个收费源完整前缀（初始化＋两轮）**，拒绝占位保持计费。底层作业或研究会话均继续更新，没有新增执行失败或共享暂停。六基线已预留第三批至 49 源；LDM-3407 第三批评分接近结束，另两组第三轮研究活跃。

同 33 源预算训练最佳损失均值±样本标准差：LDM **0.436986±0.032789**，MCMC **0.422021±0.005219**，ACQ **0.421975±0.005170**。LDM 的差距缩小，但当前均值仍落后且种子波动较大。LDM 各种子赢家求解均时为 91.10、101.90、1.68 秒，基线约 92.2 秒；不能据其中最快的种子代表整个方法。正式验证测试未开始。聚合证据 `monitoring/prefix-20261025-0525.json` 已包含 1–33 源的九组原序最佳损失曲线，供阶段图表使用；进度证据为 `progress-20261025-0525.json`。

### 05:55 监督记录

九组仍正常推进，无新增私有执行错误或共享暂停。LDM-3407 已完成第三轮（48 完整结果＋1 私有拒绝），第四轮研究正在提交 turn；LDM-42 第三批评分，LDM-43 第三轮研究仍有新产物。LDM-43 新增 1 个研究阶段客户端收费拒绝，客户端收费 34、私有 33，拒绝保留、不补评。后续原序预算聚合必须将该研究阶段拒绝的实际发生位置合并，不能再仅串接 selected_candidate_ids；此前 33 源快照位于该拒绝之前，仍有效。

六基线第三批继续评分，底层收据持续更新；共同已完成预算目前仍至少为 33 源。进度证据为 `monitoring/progress-20261025-0555.json`。已使用独立的远程绘图脚本准备 05:25 快照的训练损失曲线、三种子稳定性和赢家求解耗时阶段报告；不读取正式验证测试矿体，不向研究 agent 提供报告反馈。

绘图脚本已保存为 `docs/copper-measurement-study-r3/render_stage.py`，远程位于整改目录。现有训练和 GemPy 环境没有 Matplotlib；独立创建 `copper-report-env-v2` 安装绘图依赖，未修改实验环境。镜像下载曾超时后重试，日志为整改目录 `report-env-install.log`，当前安装/随后绘图命令仍在运行。输出目标为正式公共根 `monitoring/stage-report-0525/`；后续检查成功生成并查看 PNG 后才能交付，不将脚本已写好称为报告已完成。

### 06:25 监督及早晨阶段报告

实验仍在训练，九组均有新作业收据，无新增私有失败或共享暂停。LDM-3407 已预留 65 源，LDM-42 49 源；LDM-43 已收费 51 源，其中 2 个研究阶段客户端拒绝，私有 49 源，差额维持计费。六基线预留 49 源，完整结果约 32–45。后续超过 33 源的公平比较需把两次研究拒绝按实际发生顺序归入预算。

独立绘图依赖安装因网络超时未成功；随后找到既有 `copper-technical-report-20261024/plot-env`，复用该远程环境成功生成全新的 r3 阶段图表，停止冗余安装进程，未修改训练环境。已取回 PNG/PDF/HTML 并查看 PNG：三联图展示共同 33 源预算的训练损失曲线、种子稳定性和赢家求解耗时，无明显裁切或标注遮挡。

阶段报告：`docs/copper-measurement-study-r3/report.html`（05:25 数据快照，明确标注训练阶段、非最终）；图表 `training-prefix-33.png` 与 `.pdf`。当前仍不能宣称 LDM 的平均效果或稳定性优于基线。发现/API/排队等成本尚未完成汇总，图表求解耗时不代表总成本。矿体真值与正式测试可视化等独立评估结束后再发布，避免测试泄露。

### 06:55 监督记录

九组继续推进，没有新增执行错误或共享暂停。六基线均已完成 49 源前缀（ACQ-43 含 1 拒绝），进入第四批；LDM-42 完成 49 个结果并研究第四轮，LDM-3407 第四批评分接近完成；LDM-43 第三批仍有底层新收据，未停滞。完整结果数与预留数的区别保留在 `monitoring/progress-20261025-0655.json`。

LDM-43 两个研究拒绝已核对到同一 researcher 的第三轮 attempts/0001 和 0002：两次验证均在第二轮 33 源之后、第三轮批次选择之前，验证 artifact 哈希与收费账本一致，顺序为收费位置 **34、35**。证据另存 `monitoring/ldm-43-rejection-order-round3.json`。后续合并曲线应在第三轮选中候选前插入这两个失败槽位，不能按 49 个选中源当作总收费 49。早晨报告继续保持明确的 33 源快照，正式验证测试尚未执行。

### 07:25 监督记录

九组继续推进，未发现新增私有执行错误、共享暂停或失活控制器。LDM-3407 已预留 81 源，LDM-42 65 源，LDM-43 51 源；基线均预留 65 源。私有结果与底层收据仍在更新，未重启实验。

将两次研究拒绝按核实位置插入，并断言合并后源集合等于客户端收费源集合，九组可比较的共同前缀为 **49 个收费源**。最佳训练损失均值±样本标准差：LDM **0.432219±0.025737**，MCMC **0.416382±0.001784**，ACQ **0.416600±0.001528**。当前仍是基线平均损失更低、种子波动更小；仅 LDM-42 的单种子赢家更优。LDM 赢家求解均时按种子为 91.10、101.90、2.64 秒，基线约 92.2 秒；发现成本尚未合并，不能以最快种子替代整体耗时。聚合证据 `monitoring/prefix-20261025-0725.json` 包含原序曲线和拒绝占位，正式验证测试尚未开始。

### 07:55 监督记录

九组仍在训练，底层收据或研究产物持续更新，无新增私有执行失败或共享暂停。LDM-42/43/3407 完整结果数为 50/49/66，收费数为 65/51/81；LDM-43 正在第四轮研究，其余两组评分继续推进。六基线完整结果约 55–63，收费均为 65。结果数不代表完整原序前缀，也不替代拒绝计费。

早晨阶段报告已更新到 07:25 的共同 49 源快照（`training-prefix-49.png/.pdf`），并完成图像检查；当前方法结论仍以该已核实前缀为准，尚无最终或验证测试成绩。新进度证据为 `monitoring/progress-20261025-0755.json`，本次无需重启或修改运行配置。

### 08:25 监督记录

九组正常推进，共享暂停不存在，没有新增私有执行错误。六基线均已完成 65 个收费源（ACQ-43 包含 1 个拒绝），进入第五批；LDM-3407 完成 81 个收费源（80 结果＋1 拒绝）并在第六轮研究，LDM-42 完成 65 个结果进入第五轮，LDM-43 第四批仍有新评分收据，已收费 67 源包含 2 个研究拒绝。未修改配置或重启进程。

最新进度证据 `monitoring/progress-20261025-0825.json` 保留完整/预留/拒绝差异。阶段报告仍展示已核实的共同 49 源原序比较；不将较快组的 65/81 源最优值与较慢组混比。全部训练尚未完成，正式验证测试继续隔离。

### 08:55 监督记录

九组仍正常推进，私有作业收据持续增加，没有新增执行错误或共享暂停。LDM-42/43/3407 收费数为 81/67/97，完整结果数为 65/61/80；分别处于第五、第四、第六批评分。六基线收费均为 81，完整结果为 65–73（另保留 ACQ-43 的收费拒绝）。最近底层收据均在数秒内更新，没有仅凭进程存活判断状态，也未重复启动控制器。

新证据 `monitoring/progress-20261025-0855.json` 已保存；没有恢复操作或预算调整。当前仍为训练阶段，正式验证测试尚未触发。方法优势判断继续以已核实共同原序预算快照为准，不按各组当前最好值直接比较。

### 09:25 监督记录与共同 65 源比较

九组继续推进，没有新增私有执行错误或共享暂停。LDM-43 已完成第四批（65 结果＋2 个研究拒绝）并在第五轮研究；其他组的评分收据仍不断增加。未重启控制器或改动预算。进度证据为 `monitoring/progress-20261025-0925.json`。

按照原序选择、两次研究拒绝位置及私有拒绝账本，九组共同完整前缀为 **65 个收费源**。训练最佳损失三种子均值±样本标准差：LDM **0.429135±0.030690**，MCMC **0.416258±0.001588**，ACQ **0.416485±0.001336**。LDM-42 改善至 0.395848，但其余种子尚未追上基线，因此平均优势和稳定优势仍未出现。LDM 各种子赢家求解均时为 83.04、101.90、2.64 秒，基线约 92.2 秒；此比较不包含研究/API/队列成本。证据 `monitoring/prefix-20261025-0925.json` 包含原序曲线，阶段图表据此更新；正式验证测试未执行。


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

## 2026-10-25：用户要求永久修复后保留结果续跑

正式九组已通过共享 `queue/PAUSED.json` 停止新派发，旧客户端退出，原结果和收费账本保留。禁止再次运行旧恢复脚本。

根因与修复：
- 研究提交后的六个只读接口逐个执行，宿主等待占用了研究计时；改为每候选六接口并行，仍受全局 8 个接口令牌限制，按原 fixture 顺序收集错误，输入文件先初始化并复用。
- Pi 研究计时与宿主请求总计时均改为扣除宿主验证等待；验证保留独立、有界超时，provider 授权仍持续处理。
- 基线一次性 SIGALRM 可能落在清理函数入口或关定时器瞬间。用嵌套 `try/finally` 将安装 `SIG_IGN` 纳入外层 `SearchDeadline` 捕获范围，再清除定时器并恢复处理器。第一版 helper 修复被压力回归证伪，已废弃；不能引用其压测作为验收。

远程 `/Aidata/shared/ldm/copper-permanent-repair-20261025`：当前第二版 Python 完整回归 **217 passed / 41 skipped**，信号边界及 10,000 次实际单次定时器测试 4 项通过；新 sidecar 实际容器 14 项通过。小负载第二版 12/12 通过。`load-full-v2` 进行 1,308 个真实容器验收，`parallel-admission-v2` 额外验证四个 profile 各八候选、192 接口及收据幂等，不能只凭进程存活判通过。

续跑 epoch `controller-epochs/admission-deadline-r1/epoch.json` 单独钉住宿主模块、新 sidecar 和源码映射。六组各 129 个已收费源（共 774）保留精确原字节；后续未收费参数才使用信号修复模板。1,926 个映射均逐字节验证通过。原冻结科学代码、数据、评分、候选和收据不改，预算不重置，已收费拒绝不补评。新建研究 session 使用新 sidecar，现存 session 保留原 manifest 和镜像，继续使用修复后的并行宿主验证；这项边界必须披露，不能称历史 session 已换成新镜像。

单一门控器 `qualify_and_continue_repair.py` 依次等待全部负载、并行接口验收，再执行独立开发地质私有协议闭环。全部通过后写 `repair-acceptance.json`，才接管为 `continue_permanent_repair.py`；任何失败均维持正式暂停。禁止重复启动该门控器。正式验证测试不参与本次修复选择。

第二版满负载已完成：**1,308/1,308，通过，峰值 128，无失败**；额外四 profile × 八候选的 **192 接口通过**，重放收据完全不变，含满负载排队总耗时 563.04 秒。整机 CPU 抽样 81.16%。已对原正式账本、结果与收据共 118,858 文件建立摘要快照，接管前再核对。门控配置自检修正了 Node 测试日志格式和开发观测生成器解释器路径，未变更候选、预算或负载测试；目前唯一门控为 `qualify_and_continue_repair_v3.py`，PID 993900，以 `qualification-controller.json` 为准。旧门控均停止并保留记录。开发私有闭环正在运行，正式仍暂停，不能将满负载通过写成正式已续跑。

### 永久修复正式续跑已启动

开发私有闭环 **102 收据通过且重放不变**（并行接口会完整执行无效候选的六个 fixture，因此比此前串行提前退出的 97 多 5 份；仍只收费一个拒绝源）。超预算、提前测试、封存后继续训练均拒绝。门控写入 `repair-acceptance.json` 后接管为 `continue_permanent_repair.py`，监督器 PID **993900**，以正式 `scheduling-epoch.json` 为准。旧服务已停止，新服务使用原 socket、身份及预算；118,858 文件摘要核对通过，PAUSED 归档到 epoch 目录。九组进程已启动并开始复用旧结果、补完未完成评分，不是从头重跑。后续仍应据新收据/provider 活动判断健康，不能把历史暂停异常视为新失败。正式验证/测试仍须等待全部训练结束。

续跑后首次监控：九组命令均指向固定 epoch 的 `repaired_runtime.py`，各组相较接管验收快照均新增作业收据（17–35份），无新增失败，共享队列未暂停。完整训练结果数：LDM42/43/3407为107/81/128；MCMC为112/119/113；ACQ为113/117/113。这些是完整结果计数，不是共同原序预算前缀。当前仍在补完暂停前评分，因此旧checkpoint/provider时间不能单独判定停滞。再次核对118,840份历史非预算文件摘要，全部不变；18份预算文件允许正常累计。尚未进入正式封存、验证和测试。证据 `copper-measurement-study-r3/permanent-repair-monitor-latest.json`。

13:54用户请求进度检查：九组新增3,347份作业收据，无新超时/信号失败；仍补完旧收费批次，尚无新provider响应或未收费基线参数，故不夸大永久修复在正式后续研究的实战覆盖。共同83源曲线已更新，旧81源曲线逐点不变。LDM0.419540±0.031546，MCMC0.414481±0.001785，ACQ0.414724±0.001805；赢家求解均时63.48/92.27/92.23秒，API/排队/恢复不含其中。详情见正式研究文档和prefix-20261025-post-repair.json。

14:06监控：九组正常推进，无新增作业失败或队列暂停，共同完整收费前缀仍83，损失曲线结论不变。LDM42已完成第7轮并进入第8轮研究，确认新分支使用修复后的sidecar镜像；81份新provider响应按SSE事件解析无error/failed/incomplete。监控器不再把生成内容中出现数字502当作传输错误（仅修正观测脚本，正式运行未修改）。MCMC43已完成原129源批次并预留至145，确认第130收费源使用signal-cleanup-r2模板且源码摘要匹配；其余组继续评分。新分支尚未完整提交，不能提前宣称全部新计时路径已验收于正式后续轮。证据progress-20261025-1406.json。

14:36监控：LDM42第8轮首次在修复后新sidecar完成完整正式提交，四个原生turn_committed.json与四个submission.json均存在，已选中16候选进入评分（收费129）。没有依赖手工补提交或恢复脚本。LDM43已完成第6轮，LDM3407已完成第9轮，继续研究/评分；六基线均完成原129源批次并预留145，进入修复模板段。所有九组存活且有新活动，无新增作业错误、provider错误或共享暂停。

共同完整收费前缀提高到99。损失均值±样本SD：LDM0.417507±0.029904，MCMC0.414481±0.001785，ACQ0.414724±0.001805；赢家求解均时67.74/92.27/92.23秒。LDM尚无稳定优势。已更新99源HTML/PNG/PDF，证据prefix-20261025-1436.json；仍是训练阶段，正式验证测试未开始。

15:06监控：九组均继续推进，当前监督器及九私有服务正常；修复后累计15,140份作业收据，比14:36增加4,958份，无新增作业失败、provider错误或共享暂停。LDM43第7轮和LDM3407第10轮也已各有四个原生turn_committed，均使用新sidecar，不需手工恢复；当前三组LDM分别第8/7/10轮评分。六基线均在145收费源批次，MCMC43已有一个新模板候选完成整套评分。118,840份历史非预算文件摘要再次核对不变，整机CPU抽样68.9%，没有据单次采样擅自调高并发。共同完整收费前缀仍99，已有对比曲线数值不变；正式验证测试未开始。证据progress-20261025-1506.json。

15:36监控：九组新增4,719份作业收据，无新增超时、信号或provider错误；九私有服务、单一监督器与固定epoch模块摘要均核对正常。MCMC43的修复模板第130–145收费源共16个已完成完整套件，逐条核对源码映射，legal_rate均为1；MCMC43和ACQ43已进入161源批次。其余批次继续评分，最新作业收据年龄1–12秒，不把旧provider时间误判为停滞。共同完整前缀仍99，曲线及同预算结论不变，未进入正式验证测试。证据progress-20261025-1536.json。

16:06监控：九组持续推进，修复后累计24,636份作业收据，较上次新增4,777份，无新增作业超时/信号异常、provider错误或共享暂停；单一监督器、九服务、固定epoch模块摘要核对正常。LDM42完成第8轮评分并进入第9轮研究，新增provider响应正常；LDM43/3407仍在第7/10轮评分，均持续新增收据。六基线均已完成原序145收费源并进入161源批次。共同完整前缀仍受LDM43限制为99，同预算曲线与结论不变；正式验证测试未开始。证据progress-20261025-1606.json。

16:36监控：九组新增4,739份作业收据，无新作业超时/信号异常或provider错误，单一监督器、九服务及固定epoch摘要正常。LDM42已选第9轮评分，LDM43完成第7轮并研究第8轮，LDM3407完成第10轮并研究第11轮；基线均处于161源批次。共同完整收费前缀提高至115，旧99源曲线逐点不变。更新HTML/PNG/PDF，并修正报告生成器的历史拒绝范围说明：115前缀已包含MCMC42第100源旧信号失败，第122源仍在范围外，拒绝没有退款。LDM均值0.415582±0.032033，MCMC0.414481±0.001785，ACQ0.414724±0.001805；LDM仍无稳定优势。正式验证测试尚未开始，证据prefix-20261025-1636.json。

17:06监控：九组新增4,847份作业收据，无新超时、信号或provider异常；九服务、单一监督器、epoch摘要及118,840份历史非预算文件摘要核对正常。LDM当前第9/8/11轮评分，MCMC43/ACQ43已完成161源前缀并进入177源批次，其余基线继续161源批次。各组最新作业收据年龄2–12秒，旧provider时间与评分阶段一致。共同完整收费前缀仍115，同预算曲线与结论不变，正式验证测试未开始。证据progress-20261025-1706.json。

17:36监控：九组新增4,529份作业收据，无新超时/信号/provider异常，共享队列未暂停；监督器、九服务和epoch摘要核对正常。LDM仍在第9/8/11轮评分，各组最近作业收据1–10秒内新增，因此没有将provider静默判定为故障。MCMC42完成161源前缀并进入177源批次；ACQ42、MCMC3407、ACQ3407连续前缀分别152、149、152。共同完整前缀仍115，报告曲线结论不变，正式验证测试未开始。证据progress-20261025-1736.json。

18:06监控：九组新增4,575份作业收据，无新超时/信号/provider异常；九服务、单一监督器和epoch摘要正常。六基线均完成161源前缀并进入177源批次；LDM42完整前缀140，LDM43完整115（另有乱序完成），LDM3407完整163。当前三组仍第9/8/11轮评分，各组收据持续推进，没有因provider静默盲重启。共同115源前缀及对比曲线不变，正式验证测试未开始。证据progress-20261025-1806.json。

18:36监控：九组新增4,926份作业收据，无新增作业/provider错误，监督器、九服务和epoch摘要正常。LDM已分别完成第9/8/11轮并进入第10/9/12轮评分，MCMC43进入193源批次。共同完整收费前缀提高至131，115源历史曲线逐点不变；当前前缀已包含MCMC42第100与122源两个历史收费拒绝，以及基线第130–131源修复模板，均保留版本边界与预算。

131源损失均值±样本SD：LDM0.412735±0.034595，MCMC0.414481±0.001785，ACQ0.414724±0.001805。LDM均值略低约0.42%/0.48%，但仅种子42优于对应基线，另两个种子仍落后，不能宣称稳定优势。LDM三种子损失0.376580/0.416099/0.445524，赢家求解83.21/114.41/41.71秒；方法求解均值79.78秒，基线约92.27/92.23秒，均不含API/排队/恢复成本。131源HTML/PNG/PDF已更新，正式验证测试仍未开始。证据prefix-20261025-1836.json。

19:06监控：九组新增4,655份作业收据，无新作业超时、信号或provider异常；九私有服务、单一监督器及固定epoch摘要核对正常。MCMC42与ACQ43完成177源前缀并进入193源批次；LDM继续第10/9/12轮评分，所有组均有持续新收据，不因provider静默盲重启。共同完整前缀仍131，同预算对比结论不变，尚未进入正式封存、验证或测试。证据progress-20261025-1906.json。

19:36监控：九组新增4,473份作业收据，无新超时/信号/provider异常；九服务、单一监督器和epoch摘要正常。六基线均已完成177源连续前缀并进入193源批次；LDM继续第10/9/12轮评分，各组最新收据1–15秒内新增。共同完整前缀仍131，同预算曲线与结论不变，正式封存、验证和测试未开始。证据progress-20261025-1936.json。

20:06监控：九组新增4,197份作业收据，修复后累计61,577份，无新作业超时、信号失败或provider错误；单一监督器993900、九服务及epoch模块摘要正常，共享队列未暂停。LDM42完成第10轮，开始新研究响应；LDM43/3407继续第9/12轮评分，连续已结算前缀分别161/133/179。六基线连续前缀177，均已收费193源，乱序完成不计入完整前缀。共同预算提高至133，之前131源损失和耗时曲线逐点不变。已远程更新HTML/PNG/PDF并检查图像，证据prefix-20261025-2006.json。正式验证和测试尚未开始。

20:36–20:40监控：新增4,988作业收据（累计66,565），九组、九私有服务与单一监督器正常推进，epoch摘要不变，无provider错误或基线信号异常。LDM42第11轮出现两个新收费拒绝：dacffa…调用float.copy导致4个接口退出1；b5a8b7…两个接口用满5秒硬限被终止，实测约5.2 CPU秒，非此前宿主串行验证等待。原生研究流程已重提并选定第11轮候选，不手工修改候选，不退款或重置预算。validation.json与artifact摘要证据保存candidate-rejections-2036.json。两个拒绝在原序162/163；监控器旧固定拒绝列表因缺这两项触发严格断言，已按验证journal时间早于第11轮selection的证据补入并远程通过，未改正式运行代码。

共同完整收费前缀147，历史133源曲线核对不变。LDM连续163/147/193，已完成研究评分轮10/9/12，42在第11轮评分，43/3407新研究有响应；六基线前缀177–193。147源主损失LDM0.402241±0.048386、MCMC0.414481±0.001785、ACQ0.414724±0.001805，赢家平均求解98.02/92.27/92.23秒。LDM平均约好2.95%/3.01%，但仍只有种子42胜出且波动更大，不能宣称稳定优势。HTML/PNG/PDF已更新并检查，正式验证测试尚未开始。

21:06监控：九组新增4,711份作业收据，六基线均连续完成193源并进入209源批次；LDM分别在第11/10/13轮评分，连续前缀163/147/193。九服务、单一监督器993900、epoch摘要及训练进度正常，共享队列未暂停，各组最新收据1–10秒内新增。失败集合与20:38逐项完全相同，仅含已诊断的LDM42两个候选的6份接口失败收据，无新超时、信号或provider错误。共同147源比较曲线、损失及耗时逐点不变，无需重新绘图，正式封存验证测试未开始。证据progress-20261025-2106.json。

21:36监控：九组新增4,119份作业收据，各组均持续获得新收据（最新1–10秒），无新增失败；历史6份候选接口失败逐项与上次相同，未发现新超时、信号或provider错误。九服务、监督器993900、固定epoch摘要及训练阶段正常，共享队列未暂停。LDM第11/10/13轮评分持续推进，六基线209源批次评分中；完整前缀暂未增加，不能将provider静默或整批尚未结束误判为停滞。共同147源损失、耗时、曲线核对不变，正式封存验证测试尚未开始。证据progress-20261025-2136.json。

22:06监控：九组新增4,314份作业收据，无新增失败，历史6份候选接口失败集合不变，无新信号或provider错误。九服务、单一监督器993900、epoch摘要正常，队列未暂停。LDM42完成第11轮、连续179收费源，新研究响应继续增长；43/3407仍在第10/13轮评分且持续新增收据。MCMC43连续209源并进入225源批次，其余基线连续193–200。共同147源损失、耗时与曲线逐点核对不变，正式封存验证测试尚未开始。证据progress-20261025-2206.json。

22:36监控：新增4,497收据，九组和九服务、监督器993900、epoch摘要正常，未暂停。新增ACQ43参数索引220（第221收费源）私有接口超时，source=9041248c3d173c0421062db428906d9f6aece17eb1d43ce63d5c995033a00ae2，摘要匹配signal-cleanup-r2映射。worker硬限5秒、实际5.65秒，累计CPU5.83秒，退出137、无OOM，没有SIGALRM异常栈。现有证据是实际执行超限，不是宿主串行校验等待；未定位具体计算热点，不能断言根因。源码仍采用75%软期限和嵌套finally清理；不修改冻结候选或免费重评，拒绝正常收费。其余新增收据无失败，旧LDM42六失败保持原样。证据acq43-interface-timeout-2236.json。

LDM完成第11/10/13轮，连续179/163/209；六基线均连续209源、225源批次进行中。共同163源主损失LDM0.400637±0.046384、MCMC0.413679±0.001084、ACQ0.413912±0.001359；赢家平均求解98.93/92.31/92.32秒。LDM仅种子42领先，仍无稳定优势证据。当前前缀包含LDM42第162/163源拒绝，尚不含ACQ43第221源拒绝；此前147源曲线不变。HTML/PNG/PDF已更新并检查，正式验证测试未开始。

23:06监控：新增4,685份作业收据，各组最新收据2–9秒；九组、九服务及单一监督器993900持续推进，epoch摘要不变、队列未暂停。失败集合与22:36逐项相同（LDM42六份、ACQ43一份），无新增超时、信号或provider错误。LDM第12/11/14轮评分中，连续179/163/209源；六基线225源批次中，连续209–210源。共同163源损失、求解耗时及曲线逐点核对不变，正式封存验证测试尚未开始。证据progress-20261025-2306.json。

23:36监控：九组新增4,233份作业收据，各组最新收据1–9秒；监督器993900、九服务、epoch摘要及训练进度正常，队列未暂停。失败集合与23:06逐项不变，无新增超时、信号或provider异常。MCMC43连续完成225源、进入241源批次；ACQ43连续214源，其余基线连续209源。LDM第12/11/14轮评分持续推进，连续179/163/209源。共同163源对比数据和曲线逐点不变，正式封存验证测试尚未开始。证据progress-20261025-2336.json。

2026-10-26 00:06监控：九组新增4,684份作业收据，失败集合与23:36逐项不变，无新增超时、信号或provider异常。单一监督器993900、九服务、epoch摘要正常，队列未暂停。LDM42完成第12轮、连续195源并启动新研究响应；43/3407仍在第11/14轮评分，连续163/209源且持续新增收据。三组MCMC及ACQ43连续完成225源，ACQ42/3407连续216/209源。共同163源损失、求解耗时及曲线逐点不变，正式封存验证测试尚未开始。证据progress-20261026-0006.json。

2026-10-26 00:36–00:40重大异常：00:21 LDM42的docker start命令超过40秒，触发共享暂停，监督器993900和九客户端均已退出；九私有服务仍在。Docker/containerd同时间存在DeadlineExceeded及cleanup失败日志，具体控制面阻塞根因仍待进一步定位，不能归因于候选5秒限额或已修复的SIGALRM竞态。禁止重启共享Docker。已核对36个未收据作业：17未启动且容器不存在、14created、5已退出0；后者必须按原容器采集结果，不能再次执行。证据docker-incident-20261026-0036.json。已有结果与预算保留，当前collector因LDM42研究未完成的收费/选中顺序不一致而严格拒绝汇总，没有绕过断言或发布新曲线。

12次串行容器create/start/回收控制面探针全部通过，最长0.57秒。拟仅把create/start并发4降为1，保留128总作业和全部科学预算；这是控制面降压措施，不宣称根因已彻底修复。新门控control_gate_20261026.py（PID2568030，以control-gate-20261026.json为准）已启动独立1308作业、128峰值的load-control-20261026完整验收；正式仍暂停，不能仅据12探针放行。只有全部通过才exec一次性continue_docker_control_20261026.py。续跑前再核对旧客户端已退出、未完成容器状态、保存全部现存账本/收据/结果摘要，停止旧服务，更新已记录调度策略，归档PAUSED后重建服务接管原作业；科学epoch和源码、原收费源字节不改。新操作目录controller-epochs/docker-control-20261026，门控和旧脚本均不得重复启动。新续跑脚本已远程语法检查通过，完整验收尚未结束。监控指令已更新。

2026-10-26 01:06–01:08：load-control-20261026完成1308作业、峰128，无Docker基础设施失败，但interface-0索引19与interface-1索引14真实5秒超限（约5.2 CPU秒），门控2568030按要求退出，正式没有恢复。对这两个配置/fixture进行12次独立诊断（只增加stderr时间点和faulthandler，不参与正式评估），全部退出0：候选导入约0.73秒、初始化毫秒级，2秒时采样栈主要在GSTools调用emcee，SIGALRM可正常递交并在约4.1秒内返回。低负载没有复现硬超时，不能用这些采样证明原失败的具体热点或宣称根因解决。证据interface-diagnostic-20261026.json及远程diagnose-interface-20261026。

已启动新独立完整load-control96-20261026，仍1308真实作业、但峰值目标96（88求解/评分+8接口），startup1、每进程32，全部科学预算不变；临时牺牲资源使用率以验证稳定性。唯一门控control_gate96_20261026.py，PID3060422（以control-gate96-20261026.json为准），通过全部门槛后才exec continue_docker_control96_20261026.py。新操作目录controller-epochs/docker-control96-20261026，继续核对原36容器及历史账本摘要、停止旧服务后接管。正式PAUSED保留，旧128门控不得重跑，监控指令已更新。当前163源对比仍仅为最后核实阶段结果，尚无正式验证测试。


2026-10-26 01:40：96并发（88求解/评分+8接口、startup=1）完整1308作业全部通过。首个续跑控制器因遗漏queue/startup/PAUSED.json退出；子队列标记仍是00:21事故（PID2797426，time1792945284.101233），不是新候选超时。主队列故障文件又被监督器通用错误覆盖，已修正后续控制器不覆盖已有首故障。新恢复逻辑严格核对全部暂停标记原字节，未知/变化标记拒绝恢复，先归档子队列再归档主队列；三项远程回归通过。确认旧九客户端和监督器退出，再重新核对36旧容器和全部账本/收据/结果摘要后接管，未更改科学源码、数据或预算。唯一新监督器continue_docker_control96_reconciled_20261026.py（启动PID3919838），操作目录controller-epochs/docker-control96-reconciled-20261026；所有旧门控/续跑入口不可重复执行。96完整验收不重复消耗；本次属于恢复状态协调修复。


2026-10-26 02:06监控：新监督器3919838及九组命令正常，主/启动队列均无暂停，恢复后新增2787作业收据，各组最新收据0–23秒。没有新增Docker、接口硬超时或信号异常。LDM42新增36份失败接口收据均为候选自身float.copy类型错误，保持收费拒绝，不人工修候选或免费重评。原序聚合器仍严格拒绝输出：LDM42目前有9个收费但未入selected的拒绝源，旧聚合只登记其中2个，新增7个须结合验证journal定位原序；不删除断言，不把预留计为完成。故本次不替换已核实163源比较曲线，独立验证测试未开始。证据copper-measurement-study-r3/health-20261026-0206.json。


2026-10-26 02:36监控：九组及监督器3919838正常，全部队列无暂停，各组新收据3–13秒内；相对02:06无新增失败。已根据第13轮验证journal错误指针与artifact摘要、时间确认LDM42新增7拒绝的原序196–202，聚合严格集合断言保留且通过。共同完整收费前缀179：LDM0.400263±0.045928、MCMC0.413679±0.001084、ACQ0.413912±0.001359；赢家平均求解103.89/92.31/92.32秒。LDM仍仅42领先，稳定优势未证实。连续前缀LDM202/179/225，六基线225–236。HTML及PNG/PDF曲线已更新、检查，独立验证测试未开始。ACQ3407接口超时收据mtime为00:21:40，属于旧Docker事故期间，不能误作本次新增异常；收费保留。证据monitor-20261026-0236.json。


2026-10-26 03:06监控：新增3059份作业收据，九组最新收据2–23秒；监督器3919838、九服务及客户端、固定epoch模块摘要正常，主队列和启动队列均未暂停。失败集合逐项不变，无新增超时、信号或provider错误。连续前缀LDM202/179/225；六基线225–241，MCMC43已完成241源。共同179源的损失、赢家求解时间及全部历史曲线逐点不变，无需重复绘图。仍在训练，正式封存验证测试尚未开始。证据copper-measurement-study-r3/monitor-20261026-0306.json。


2026-10-26 03:36监控：新增3165份作业收据，九组最新收据1–16秒；监督器3919838、九服务及客户端、固定epoch模块摘要正常，主/启动队列无暂停。失败集合逐项不变，无新增超时、信号或provider错误。LDM第13/12/15轮评分持续推进，完整前缀202/179/225；六基线225–241，MCMC42/43与ACQ43均完成241源。共同179源损失、赢家求解耗时及历史曲线逐点不变，无需重复绘图。正式封存验证测试尚未开始。证据copper-measurement-study-r3/monitor-20261026-0336.json。

2026-10-26 05:06: Async transport private development closure passed (102 receipts, budget/sealing/early-test rejection, replay unchanged; no formal holdouts). Independent 1308-job / peak-96 container load also passed. Combined authenticated async socket + saturated real-container qualification is now running in OPS/async-joint-load-20261026 (PID1751335 at launch). Gate OPS/gate_async_continue_20261026.py PID1932047 waits for all 1308 jobs legal and peak96, then execs the guarded continue_async_20261026.py; it is one-shot and must not be rerun. The controller pins transport/runtime digests, checks old clients exited, reconciles ten unfinished jobs (two created containers, eight not started/missing), snapshots ledger/receipt/results, stops old services, and archives both pause markers before restarting. Science epoch and paid source identities remain unchanged. Formal training is still paused until these gates pass. See async-continuation-gate.json and logs for actual status.

2026-10-26 05:36: Combined async socket/container saturation completed 1308 jobs at peak96 but failed five real 5-second interface budgets (exit137, no OOM); gate1932047 correctly refused continuation. There was no reported async socket failure. Worker elapsed times 5.02–8.32 seconds do not identify the computation hotspot or prove a specific host-contention cause. Independent qualification passes do not override this combined failure. Evidence: async-joint96-failure-summary.json.

A fresh guarded qualification now tests 64 total jobs (56 solver/scorer + 8 interface), startup1, unchanged per-candidate resources and paid-source budget. OPS/gate_async64_20261026.py runs 12 small jobs then all1308 combined protocol/container jobs; samples host CPU every10 seconds to async64-host-resources.jsonl. Only a full pass at peak64 may exec continue_async64_20261026.py. New operation directory: controller-epochs/async-transport64-20261026. Previous gates/controllers must not be rerun. Formal training remains paused pending acceptance and immutable-state reconciliation.


2026-10-26 06:06 CST update: combined async Unix submit/status plus real-container qualification passed 1308/1308 jobs, peak 64, zero failures. The one-shot gate exec-transferred PID 3294205 to OPS/continue_async64_20261026.py. All nine clients and nine services are alive; scheduling-epoch confirms 64 total slots (56 solver/scoring + 8 interface), startup 1, per-process 32. Both pause markers were reconciled and archived; no queue pause remains. Continuation checked 233177 retained accounting/receipt/result files. Post-start audit observed 413 new private receipts, all exit_code 0, with new receipts in every group. This is evidence of resumed execution, not merely process liveness. Scientific identity and per-candidate budgets remain unchanged. Snapshot monitor-20261026-0606.json still supports only common charged prefix 179; comparative losses/timing remain unchanged and formal validation/testing has not started.


2026-10-26 07:36 CST: supervisor3294205 and all nine clients/services continue, no queue PAUSED. Common complete charged prefix advanced to195 (LDM prefixes218/195/241; six baselines241). LDM loss0.399281 +/-0.044755; MCMC0.413679 +/-0.001084; ACQ0.413912 +/-0.001359. Winner mean solver seconds102.82/92.31/92.32; cumulative completed solver hours171.25/239.10/239.10 exclude API/interface/queue/recovery. Only seed42 leads; no stable superiority established. HTML and PNG/PDF curves regenerated from monitor-20261026-0736.json and PNG visually checked. New LDM3407 private interface receipt69d7ec1e... failed its5-second execution budget: worker5.012s, CPU5.090s, exit137, noOOM, no signal traceback. Paid rejection retained (group now2); specific hotspot unproven. No new infrastructure/transport failure; no restart or free reevaluation. Formal validation/testing remains unstarted.


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


2026-10-27 06:06 CST monitor: fixed-winner testing continues in all nine groups,222–299 test receipt files per group, latest0–87 seconds, no failed receipt statuses/exit codes. Supervisor1334269 and finish2258205 remain active; no main PAUSED or completion.json. Evidence OPS/private-stage-20261027-0606.json. No restart or research feedback; final evaluation/report still pending.


2026-10-27 06:36 CST monitor: all six baseline selections now phase tested,444 test receipt files each. Three LDM groups still selected and progressing,396/408/402 receipts for42/43/3407, latest104/67/2 seconds. No failed receipt statuses/exit codes; supervisor1334269 and finish2258205 active, no main PAUSED or completion.json. Evidence OPS/private-stage-20261027-0636.json. Await all nine completed evaluations before final comparisons/report; no duplicate finish or research feedback.


2026-10-27 07:06 CST: formal evaluation completed; completion.status completed, replay_receipts_unchanged true,5454 receipts. All nine phases tested, no main PAUSED; supervisor/finish exited. ID96 and OOD48 solves per group, legal_rate1 and failed_solves0. Across discovery seeds, ID loss LDM0.361726±0.032651 / MCMC0.415236±0.000979 / ACQ0.415201±0.000595; OOD LDM0.395486±0.083478 / MCMC0.417936±0.001172 / ACQ0.418215±0.001086. LDM all seeds lead ID, only42/3407 lead OOD;43 OOD0.491752. Solver112.56s vs91.1s. No stable OOD superiority claim. Test data now seen and must not feed research. Saved formal-test-summary-20261027-0706.json, formal-test-metrics.csv, formal-test-results.html and formal-test-loss.png/pdf in report directory. Full technical delivery remains unfinished: algorithm analysis, same-coordinate truth/prediction3D and slices, complete case/execution-seed tables and cost breakdown. Keep monitor active for report delivery, do not rerun experiments.


2026-10-27 07:36 CST report work: exported all1296 test case/execution rows,54 parent-equal execution-seed aggregates and nine hash-verified frozen solver sources. Added algorithm-analysis.html; generated same-coordinate ID/OOD geometry and central orthogonal overlap/miss/false-positive slices (PNG/PDF). Case selection fixed to lexicographic minimum case_id per split, discovery42/execution47. ID3D and OODslices visually inspected;3D labels need more margin, inactive-region overlays still missing. Script OPS/export_report_geometry_20261027.py, remote report-geometry-20261027. Full cost reconciliation, complete visual QA and final merged report remain pending. No experiment rerun.


2026-10-27 08:06 CST report work: regenerated fixed ID/OOD geometry and slices with grey inactive-domain overlays and expanded margins. OOD3D and IDslices visually checked. Exported nine research/private-training ledger counters and provider usage in cost-ledger-summary-20261027-0806.json; cost-accounting.html distinguishes summed job hours from elapsed/CPU hours. Provider response usage missing in3/6/8 files for LDM42/43/3407; totals explicitly incomplete, not a financial invoice. Evaluation/test/queue/recovery cost reconciliation, Chinese merged report and final QA remain pending. No scientific rerun.


2026-10-27 08:36 CST report work: reconciled3726 private validation/test container jobs (270 interfaces,1728 solvers,1728 scorers), zero failures and no missing workload resources. Completion5454 counts1728 additional case-level receipts, not additional executions. Saved holdout-cost-summary-20261027-0836.json. Summed research/training/holdout admission waits from jobs/scheduling records into queue-cost-summary-20261027-0836.json; parallel wait sums are not elapsed delay and interrupted jobs may lack records. Both cost tables integrated into cost-accounting.html. Remaining: recovery/qualification costs and chronology, Chinese merged final report, full visual/link/data QA and delivery; automation remains active.


2026-10-27 09:07 CST final report delivery: merged Chinese report at copper-measurement-study-r3/report.html now covers formal ID/OOD results, frozen algorithms, all1296 case rows,54 execution-seed aggregates,9 source hashes, same-coordinate fixed-case geometry and orthogonal slices, research/training/holdout/queue costs, qualification runs and recovery chronology. Revised3D panels to2x2 to keep all axes readable; final ID/OOD3D and slices visually inspected. Remote data QA reaggregated1296 normalized confirmation losses by parent and execution seed, matching all54 aggregates and18 group/split metrics. Artifact QA verified33 links (zero broken),9 frozen source SHA256 values and linked PDF headers. Evidence final-report-data-qa.json and final-report-artifact-qa.json; qualification-cost-inventory-20261027-0906.json preserves located load-run costs, including failures. Missing provider usage17 responses and incompletely reconstructable interrupted/development recovery costs are explicitly disclosed, not assumed zero; no exact all-in monetary claim. Final result: all3 LDM discovery seeds lead ID, only42/3407 lead OOD, roughly24% longer solver wall time. No experiments rerun or heldout feedback into research. Full requested report package delivered with these evidence limits; recurring monitoring can now stop.
