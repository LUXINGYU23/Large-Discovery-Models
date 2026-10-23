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
| 3D 钻孔 | 514.0113798896817 |
| 3D 钻孔 + 露头 | 1060.7747438220217 |

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
3D 已完成数据加载、原式对照和基线隔离评估；3D agent 搜索与 surrogate 引导阶段另行验收。


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
3D agent 搜索、Coverage 2+2、GP/acquisition、SFT 转换和正式 benchmark 认证未包含在本次验收中。
