# 两个 3D case：Flash / low 单 agent 三轮重跑

日期：2026-10-23。使用当前铜矿任务实现，在授权 H20 `120.133.37.228:10522` 完成。两组均首次启动成功，无失败重试。

## 配置与执行边界

- 官方 `https://api.deepseek.com/responses`，`deepseek-flash`（V4.1 Flash），`reasoning.effort=low`。逐条检查全部 101 次原生请求均一致。
- 每个 case 一个独立持久 researcher，`search-method=harness`，三轮算法设计；无主 agent、分支 agent 或 GP。两个 campaign 同时运行，各自独立计时。
- 固定 seed 47、相同初始几何和数据；基线加三次候选评分。没有注入历史最佳算法。
- 每次正式 solver：2 CPU、4 GiB、120 秒；每研究轮 900 秒、最多 8 次 bash，每次显式 timeout 不超过 60 秒。
- 固定只读接口输入，宿主检查输出合法性；冻结候选源码后执行；独立 scorer 仅接收冻结几何与可信科学代码，不挂载候选源码。
- 每组实际 8 个接口 job、4 个 solver job、4 个 scorer job，均成功。签名工作负载资源记录保留在评估证据中。
- 完成态恢复核对：两组预算计数不变、新增模型请求均为 0。

## 损失轨迹

以独立 scorer 的 `loss_total` 为准，越低越好。

| case | 基线 | 第 1 轮 | 第 2 轮 | 第 3 轮／最佳 |
| --- | ---: | ---: | ---: | ---: |
| 钻孔 | 405.079581 | 6.519107 | 6.517357 | **6.516471** |
| 钻孔＋露头 | 1248.838576 | 336.935814 | 226.696404 | **36.169527** |

## 历史对比

总时间从首次 `campaign_started` 到首次 `campaign_finished`，包含预检、研究、接口和正式评估；不包含事后恢复检查和归档。

| case | 运行 | 最佳损失 | 三轮总时间 |
| --- | --- | ---: | ---: |
| 钻孔 | 早期 Luna / high | 6.517446 | 16.98 分钟 |
| 钻孔 | 上次 Flash / low | 6.516604 | 18.08 分钟 |
| 钻孔 | 本次 Flash / low | **6.516471** | **11.02 分钟** |
| 钻孔＋露头 | 早期 Luna / high | **32.948273** | 17.80 分钟 |
| 钻孔＋露头 | 上次 Flash / low | 35.599488 | **9.08 分钟** |
| 钻孔＋露头 | 本次 Flash / low | 36.169527 | 10.63 分钟 |

相对上次成功的 Flash 运行：钻孔损失下降约 **0.0020%**，总时间缩短约 **39.1%**；露头损失上升约 **1.60%**，总时间增加约 **17.0%**。本次钻孔优势很小，露头仍未超过历史最好。

历史 Flash 曾有失败尝试，全部尝试累计分别为 39.05、39.85 分钟；上表只比较各自成功完成三轮的 campaign。本次没有额外失败尝试。

这些是单次工程运行，不是受控模型排名：本次换到 H20，加入了新的提交数组契约、诊断反馈及可信资源监督，历史钻孔研究工具预算也有差异。相同模型的算法选择有随机性，不能把耗时变化完全归因于模型或框架。

## 计算成本与算法

| case | 三轮研究时间 | 四次正式 solver 合计（含基线） | 最佳 solver 单次 | 最佳 solver 峰值内存 | 模型请求（含预检） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 钻孔 | 605.90 秒 | 27.89 秒 | 12.60 秒 | 84.58 MiB | 51 |
| 钻孔＋露头 | 567.71 秒 | 44.95 秒 | 8.82 秒 | 91.52 MiB | 50 |

峰值口径为容器工作负载及 supervisor 至最终子进程回收；不是宿主采样最大值。求解墙钟包含容器启动开销。

钻孔采用隐式曲面组合、接触约束和有界局部搜索。露头最终算法扩展到 2–4 阶多项式 logistic/最小二乘候选，按公开重初始化损失选择，并保留图拉普拉斯正则化 IRLS 改进路径。三轮结果体现算法设计路径的差异；当前损失衡量合成观测拟合，不代表真实矿体恢复精度。

## 可复核产物

- [完整评估证据](../tasks/copper_inversion/resources/evidence/flash-low-rerun-20261023/evidence.json)：配置、实现身份、四次正式分数、资源数据、会话数、请求核对与源码快照 SHA256。
- [完成态恢复](../tasks/copper_inversion/resources/evidence/flash-low-rerun-20261023/resume-verification.json)、[凭证扫描](../tasks/copper_inversion/resources/evidence/flash-low-rerun-20261023/credential-scan.json)。
- [钻孔最佳算法](../tasks/copper_inversion/resources/evidence/flash-low-rerun-20261023/drillholes_3d/best_solver.py)、[露头最佳算法](../tasks/copper_inversion/resources/evidence/flash-low-rerun-20261023/drillholes_outcrops_3d/best_solver.py)。
- [历史比较及失败尝试](copper-inversion-deepseek-comparison.md)。
- 活动运行目录：`/Aidata/shared/ldm/copper-inversion-20261023/runs/flash-low-single-20261023/`。
- 完整归档：`/mnt/cephfs/xingyu/ldm/archive/copper-inversion/flash-low-single-20261023.tar.gz`。
- 归档 SHA256：`a4b508fbfd0703ad05500a00c2aa826847c216a3835a85db2777205ba5a361ec`。

实现与原生轨迹保存在原目录及归档中；后续代码修改必须使用新运行身份。本次没有修改现有算法框架或覆盖历史结果。
