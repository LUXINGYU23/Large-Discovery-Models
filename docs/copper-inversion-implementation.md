# 铜矿反演接入：第一阶段实现与远程验收

## 范围

独立分支 `codex/copper-inversion`，基于 main `37ebd33d`。任务位于
`tasks/copper_inversion/`，配置位于 `config/copper_inversion/`。

复用共享 `CampaignRequest`、`CampaignBudget`、`LDMEngine`、
`InitialRoundReservoirExpander`、`HarnessClient`、Pi/Gondolin 和原有任务注册入口。
没有修改共享框架，也没有迁入原包的 MCMC、调度器或比较运行器。

当前算法策略是单个持久 researcher：初始几何基线 + 三轮完整程序设计。
每轮候选是 `solve(problem, budget, seed)` 的源码快照，按 SHA-256 去重。
下一轮能读取此前实际测量的源码、损失分量、状态及修改说明。
尚未接入 Coverage 2+2、编码器、GP 或 acquisition；这是 spec 的后续阶段。

## 代码边界

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 科学定义 | `core/data.py`、`geometry.py`、`objective.py` | UBC 坐标解析、有符号距离重建、冻结损失 |
| 候选 | `core/candidate.py`、`harness.py` | 源码封存、父关系、快照校验、同轮拒绝修复 |
| 准入 | `core/interface_checks.py` | 两个小网格，含/不含接触点；只读输入；宿主校验输出 |
| 隔离执行 | `core/execution.py`、`runner.py` | 无网络、只读根目录、资源/时间限制、持久子作业回执 |
| 独立评分 | `core/evaluator.py`、`scorer.py` | 冻结 model.npy；另一容器评分，不挂载候选源码 |
| 生命周期 | `core/workflow.py` | 唯一 Campaign runtime、共享引擎、恢复、最终产物 |
| 注册 | `ldm_task/` | 薄适配器和依赖检查 hook |

源码的 Python 解释执行只发生在 worker 内。宿主仅解析 AST、验证摘要和有界
NPY 文件。进程退出后才复制并冻结矿体文件，评分容器只读取可信输入和冻结矿体。
候选写出的评分文件不参与正式结果。输入不可写；正式 solver 不持有 API 凭证。

## 科学对照

输入来自用户提供的 ZIP；每个数据文件和原始 ZIP 的摘要在
`resources/provenance.json`，原始许可证随任务保存。
独立参考检查记录见 [reference-check.json](evidence/copper-inversion/reference-check.json)。

在相同重新距离化的模型上，原包 binary loss 和 Procrustes loss 与新实现一致
（容差 `1e-12`）。测试同时覆盖没有接触点时的有限损失，以及正比例缩放的几何不变性。
3D 使用 `discretize` 官方 UBC reader，轴为 x/y/z，z 原点 -25，网格 40×50×30。

| 初始几何 | loss_total |
| --- | ---: |
| 2D 钻孔 + 草图 | 293.2197607509265 |
| 3D 钻孔 | 405.0795810830953 |
| 3D 钻孔 + 露头 | 1248.8385761871953 |

3D 基线已按下述内存顺序修复更正；旧 reference-check.json 保留为历史记录，
其中 3D 距离重建结果已失效，不能作为当前评分基准。二值 loss 公式本身未改变。

这些是合成教学案例的几何拟合损失，不是铜品位、储量或真实矿体恢复精度。

## 远程环境

所有调试、测试、容器和模型调用均在 `111.2.199.31:5320` 执行。
代码和活动结果：`/mnt/data1/Large-Discovery-Models/worktrees/copper-inversion-20261023`。
Gondolin 缓存：`/mnt/data1/Large-Discovery-Models/cache/gondolin`。
本地仅编辑和同步文件。

npm 使用 `https://registry.npmmirror.com`，PyPI 使用
`https://pypi.tuna.tsinghua.edu.cn/simple`。GitHub 下载超时的官方 release 资产通过
本机下载并传入远程缓存：sandbox helpers 0.12.0 与 libkrunfw 5.2.1。
没有改动仓库的 Gondolin 实现或取消哈希校验。

研究 guest 和正式评估使用 NumPy 1.26.4、SciPy 1.14.1、scikit-fmm 2025.6.23、
discretize 0.11.3。基础 Python 镜像固定 digest，运行记录保存实际 evaluator/sidecar digest。
CPU 评分使用 2 CPU/4 GiB/60s（2D），接口检查使用 1 CPU/2 GiB/30s。

## 验收状态

具体远程测试、真实闭环和恢复的最终证据记录在本目录的 evidence 子目录。
任务保持 `qualification: draft`；这些是工程验收，自定义预算不声明为官方 benchmark。
两个 3D case 已完成单 agent 三轮真实算法设计和恢复验收，见文末；surrogate 引导阶段另行验收。


## 实测闭环结果

| 轮次 | 设计变化 | 独立 loss_total |
| --- | --- | ---: |
| 基线 | 返回初始几何 | 293.2197608 |
| 1 | 草图平移搜索 + 边界单点修正 | 3.9504788 |
| 2 | 成对边界修正，跨越单点局部最优 | 3.7968566 |
| 3 | 小角度旋转假设 + 扩大的成对修正 | 3.7338023 |

同一个 researcher 会话完成三轮，每轮读取已测量代码和反馈。
真实 Responses 启动探针验证模型身份、Python 依赖和只读资源。
预算记录：3 个研究 turn，4 次正式评估，4 个 solver、4 个独立 scorer、8 个接口 job。
共 66 次模型请求（含启动探针），完整恢复后所有计数不变。

最终远程回归 **162 passed**。主运行后补充的挂载边界/超时/恢复修正也已测试，
全部四个真实源码在最终执行器中复评，损失与主运行**逐项完全一致**。
主运行的源码 tar 快照、配置、日志、模型、会话和预算保留在远程 run 内，
摘要及代码版本差异均在 [campaign.json](../tasks/copper_inversion/resources/evidence/campaign.json)。
最佳实测源码见 [best_solver.py](../tasks/copper_inversion/resources/evidence/best_solver.py)，
它是验收产物，不替换默认弱基线。

一次早期运行在第二轮因历史游标接法错误中止，随后按现有 Harness 增量历史语义修复，
并增加 mock 游标回归；最终证据来自完整重跑。远程对空闲镜像的清理通过实验期间的临时
镜像保留容器处理，验收结束后移除保留容器；重新运行前应按 README 确认或重建镜像。

本次资格推进到 `contract_verified`，并保留 experiment 的 `draft` 状态。
上述 2D 验收未包含 3D agent 搜索；3D 的后续实测见下文。Coverage 2+2、GP/acquisition、SFT 转换和正式 benchmark 认证仍未包含。


## 3D 距离重建修正（2026-10-23）

首次真实 3D campaign 暴露了旧 reference 与 solver 基线的差异。远程最小复现表明，
锁定的 scikit-fmm 2025.6.23 对 Fortran-strided 输入会错误解释 3D 网格：
9×8×7 的解析平面 `phi=x-3.25` 出现 248 个符号错误、最大距离误差 5.6273。
相同数据的 C-contiguous 输入误差仅 3.55e-15。UBC reader 输出的物理轴顺序正确，
问题发生在距离重建库的输入存储处理。

`geometry.signed_distance` 现在先统一为 C-contiguous float64，再调用锁定版本的库。
没有改变坐标轴、观测、loss 公式或权重。新增解析平面（2D/3D、float32/float64、C/F）
及三个实际 case 的存储顺序一致性测试；远程回归 **167 passed**。
修复后的两个 3D 基线为 405.0795811 和 1248.8385762。
旧 reference-check.json 中的 3D 数值仅保留为历史记录，其几何结果已被此次修复取代。

证据：[geometry-order-check.json](evidence/copper-inversion/geometry-order-check.json)。
上次 2D 四个冻结矿体使用新代码重评分逐项不变，见
[2d-geometry-fix-recheck.json](evidence/copper-inversion/2d-geometry-fix-recheck.json)。
两个 `real-*-v1` 试跑在第一研究轮中停止，保留 `ABORTED.md`；完整三轮从 `real-*-v2`
新目录开始，不跨实现版本恢复。运行前封存源码快照，运行中不修改评分代码。

新增 `harness_drillholes_3d`、`harness_drillholes_outcrops_3d` 两个锁定 profile：
各自一个持久 researcher，基线加三轮设计，seed 47；solver 2 CPU、4 GiB、120 秒，
每研究轮 900 秒。使用与 2D 相同的 gpt-5.6-luna / Pi Responses / high 配置。
两个 campaign 可并行，但不共享候选、反馈或 researcher 会话。


## 两个 3D case 的完整三轮结果

本节结果使用切换要求到达前已启动的 `gpt-5.6-luna / high` 运行；
两组均已完整结束，不能重标为 DeepSeek 结果。两个 case 共用钻孔，不能当作两个独立泛化样本。

| case | 基线 | 第 1 轮 | 第 2 轮 | 第 3 轮（最佳） | 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3D 钻孔 | 405.079581 | 6.526027 | 6.518132 | 6.517446 | 98.39% |
| 3D 钻孔＋露头 | 1248.838576 | 33.048998 | 33.005478 | 32.948273 | 97.36% |

钻孔算法依次演进：接触点拟合仿射曲面＋局部高斯修正 → 趋势正则化与接触偏移调整 →
按观测间距自适应局部修正半径。最终内部／外部标签误判为 0，接触点 RMSE 为 0.268844，
正体素 26405。原始体积只有 1655，说明稀疏钻孔约束允许非常不同的未观测几何。

钻孔＋露头算法依次演进：双尺度高斯观测插值＋最近正负观测距离修正 → 各向异性平滑 →
保符号幂变换，改进体素内零交叉位置。最终损失分项为钻孔 7.591369、露头 25.356904；
标签误判 34，接触点 RMSE 为 1.001730，正体素 6776。agent 早期自检曾声称全部标签保留，
但正式独立评分并非零误判，本报告以独立评分为准。

最后两轮的改进均较小；这是三轮已测到的最佳值，不能称为全局最优或真实矿体恢复精度。
两组各有 4 次成功正式评估、8 个只读接口 job、4 个 solver、4 个独立 scorer；
researcher turn 均为 3。模型请求分别为 53 和 71（含预检），研究耗时分别约 993 秒和 1043 秒。
完整恢复后全部计数逐项不变，新增模型请求为 0。

结果、源码摘要、每轮分项、预算、预检、恢复证据和远程产物路径：

- [钻孔 campaign](../tasks/copper_inversion/resources/evidence/3d/drillholes_3d/campaign.json)，[最佳算法](../tasks/copper_inversion/resources/evidence/3d/drillholes_3d/best_solver.py)
- [钻孔＋露头 campaign](../tasks/copper_inversion/resources/evidence/3d/drillholes_outcrops_3d/campaign.json)，[最佳算法](../tasks/copper_inversion/resources/evidence/3d/drillholes_outcrops_3d/best_solver.py)

## 后续运行默认值

按用户新要求，任务 CLI 默认值和三个真实 YAML/profile 均切换为
`https://api.deepseek.com`、`deepseek-flash`（V4.1 Flash）、`low`。
[官方模型说明](https://api-docs.deepseek.com/news/news260910/)与
[思考强度文档](https://api-docs.deepseek.com/guides/thinking_mode/)。
凭证仅从远程受保护文件读取；任务 AGENTS.md 已记录后续运行规则。
预检现在使用与研究相同的思考强度，避免只验证非思考模式。
DeepSeek 思考模式拒绝 `tool_choice=required`，因此复用 PiHarnessConfig 的
`force_first_tool_call=False`；仍要求实际工具执行和合法提交，不放宽源码／接口校验。
历史运行已保存源码快照；切换模型或实现必须使用新目录。

最终远程验证：**168 passed**；三个 case 的共 12 个历史冻结矿体在最终实现下重评分，
全部分项精确不变。DeepSeek 官方真实 Pi 预检通过，5 次请求逐条核对
`model=deepseek-flash`、`reasoning.effort=low`，只读 guest 校验及 researcher 启动成功。
首次预检的强制 tool choice 400 和一次镜像 tag 清理引起的启动失败均保留在远程记录；
最终按共享 workflow 的方式使用已解析 image digest 验证通过。
详见 [3D 最终验证](../tasks/copper_inversion/resources/evidence/3d/verification.json)。
此项验证是新端点连通与 Harness 工具链预检，尚未在 DeepSeek 下重跑三轮算法设计。
