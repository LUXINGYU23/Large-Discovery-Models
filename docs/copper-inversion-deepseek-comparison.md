# 3D 铜矿反演：DeepSeek Flash / low 三轮对照

## 实验设置

- 远程：`111.2.199.31:5320`；代码和结果位于 `/mnt/data1/Large-Discovery-Models/worktrees/copper-inversion-20261023`。
- 本次：DeepSeek 官方 `https://api.deepseek.com`，`deepseek-flash`（V4.1 Flash），`low`，Pi Responses。
- 上次：`gpt-5.6-luna`，`high`，Pi Responses，运行目录 `real-drillholes_3d-v2`、`real-drillholes_outcrops_3d-v2`。
- 首次尝试目录：`deepseek-drillholes_3d-v1`、`deepseek-drillholes_outcrops_3d-v1`；均未完成三轮，详见失败与重跑记录。
- 每 case 一个独立持久 researcher，从相同初始几何开始，基线加三轮算法设计；不提供上次模型的最佳源码或历史。
- 固定 seed 47；solver 2 CPU、4 GiB、120 秒；每研究轮 900 秒；各 4 次正式评估。
- 相同只读接口输入、宿主输出验证、无网络 solver 容器与独立 scorer。科学与执行相关的 9 个关键文件和上次源码快照逐字节相同。
- 新旧端点的首条工具选择策略不同：DeepSeek 使用现有 `force_first_tool_call=False`，因其思考模式拒绝 `tool_choice=required`。本次预检也使用 low；上次预检关闭思考。

## 时间口径

- **总墙钟时间**：事件日志首次 `campaign_finished` 减 `campaign_started`，包含预检、研究、接口和正式执行。恢复检查产生的时间不计入。
- **研究时间**：三次 Harness turn 的实际耗时，包含 agent 的模型请求、数值试验和提交检查。
- **solver 时间**：四个正式 solver（含基线）的容器墙钟时间之和，包含启动开销；不是研究时间。
- 两组同时启动，但各自独立计时。新旧均为单次运行，机器负载、模型随机性及端点延迟影响时间，不能据此作稳定的模型排名。

## 结果

结果以独立 scorer 的正式记录为准，研究中的自报损失不计入。

| case | 模型 | 基线 | 第 1 轮 | 第 2 轮 | 第 3 轮（最佳） |
| --- | --- | ---: | ---: | ---: | ---: |
| 钻孔 | 上次 gpt-5.6-luna / high | 405.079581 | 6.526027 | 6.518132 | 6.517446 |
| 钻孔 | DeepSeek Flash / low | 405.079581 | 6.637266 | 6.517088 | 6.516604 |
| 钻孔＋露头 | 上次 gpt-5.6-luna / high | 1248.838576 | 33.048998 | 33.005478 | 32.948273 |
| 钻孔＋露头 | DeepSeek Flash / low | 1248.838576 | 129.958071 | 35.645079 | 35.599488 |

钻孔最佳损失改善 **0.0129%**，数值变化很小。露头最佳损失比上次高 **8.0466%**。
这两组只衡量当前合成观测的拟合，不是实际矿体恢复精度。

### 算法演进

- 钻孔：高斯 RBF＋仿射接触面拟合 → 接触点邻域局部平坦化 → 对局部高度做连续 Nelder–Mead 和精细坐标搜索。最终零标签误判、接触 RMSE 0.268407，但每次求解都需在线数值优化。
- 钻孔＋露头：图上调和势场 → 加入平滑的正负观测距离场与核密度候选 → 对仍有较高 logistic 损失的观测做局部高斯修正。最终钻孔损失 8.118387、露头损失 27.481100，标签误判 9，接触 RMSE 0。第一轮虽零误判但损失 129.958，表明标签误判数不能替代固定距离场损失。

### 耗时比较

| case | 模型 | 完成三轮总时间 | 三轮研究时间 | 四次正式 solver 合计 | 模型请求（含预检） |
| --- | --- | ---: | ---: | ---: | ---: |
| 钻孔 | 上次 | 16.98 分钟 | 992.76 秒 | 2.14 秒 | 53 |
| 钻孔 | DeepSeek | 18.08 分钟 | 971.80 秒 | 91.40 秒 | 60 |
| 钻孔＋露头 | 上次 | 17.80 分钟 | 1043.01 秒 | 2.21 秒 | 71 |
| 钻孔＋露头 | DeepSeek | 9.08 分钟 | 512.50 秒 | 8.75 秒 | 43 |

DeepSeek 钻孔最终算法单次正式执行 **34.06 秒**，上次最佳算法 **0.493 秒**。
新算法在运行时进行数值优化，旧算法主要使用已选定参数直接构造曲面；微小损失优势伴随明显求解开销。
露头最后算法单次正式执行 **4.209 秒**，上次 **0.520 秒**。
露头成功重跑总时间缩短约 **49.0%**，但研究执行预算已收紧，不能把加速直接归因于模型。

| case | DeepSeek 失败尝试时间 | DeepSeek 完成三轮时间 | DeepSeek 所有尝试运行时间之和 |
| --- | ---: | ---: | ---: |
| 钻孔 | 20.96 分钟（1 次） | 18.08 分钟 | **39.05 分钟** |
| 钻孔＋露头 | 30.76 分钟（2 次） | 9.08 分钟 | **39.85 分钟** |

这些总投入未包含人工排查、源码同步和运行之间的等待；两组曾并行运行，因此不能简单相加作为用户等待时间。


## 失败与重跑记录

1. 原提示下，钻孔第二轮超时，露头第一轮超时。框架状态标记为 `paused_endpoint_unavailable`，但真实异常是 `Session wall-time limit reached: 900s`，并非凭证或端点故障。没有重置或伪造原运行账本。
2. 在新代码目录 `copper-deepseek-bounded-20261023` 暴露每轮 `research_wall_time_seconds`，要求先保存候选、限制实验时长并预留提交时间。钻孔在 `deepseek-drillholes_3d-v2` 完成三轮；露头 `deepseek-drillholes_outcrops_3d-v2` 第一轮再次超时。
3. 仅靠提示不足以约束露头 agent 的探索。后续在 `copper-deepseek-budgeted-20261023` 复用 `HarnessLimits.tool_call_budgets`，每研究轮最多 8 次 bash；task extension 拒绝没有显式 timeout 或 timeout 超过 60 秒的研究命令。露头使用 `deepseek-drillholes_outcrops_3d-v3`。读写文件和提交不受 bash 次数限制；正式 solver 仍为 120 秒。

所以成功运行之间也存在研究提示/工具预算差异，不能把损失和时间差异完全归因于模型。
首次失败投入：钻孔约 21.0 分钟，露头约 15.4 分钟；露头另有第二次失败。
总投入表会加上全部失败运行时间，不含人工排查与拷贝等待时间。


## 验证与产物

- 每个成功 campaign 恰有一个持久研究会话、三次研究 turn、4 次成功正式评估、8 个接口 job、4 个 solver 和4个独立 scorer。
- 两个成功 campaign 均检查完成态恢复：所有预算计数不变，新增模型请求为 0。
- 原生请求逐条检查 `model=deepseek-flash`、`reasoning.effort=low`；完成响应均报告 `deepseek-flash`。
- 露头三个成功 turn 的共享策略账本均为 bash limit=8、used=8；未放宽正式 solver 的 120 秒限制。
- 最终远程回归：**169 个 Python 测试通过，3 个 Node 研究命令边界测试通过**。
- 失败与成功的完整原始运行均归档到本地忽略目录 `runs/deepseek-comparison-20261023/`；远程原目录继续保留。归档复制前检查未包含 API key 明文。

详细记录：

- [完整比较及全部失败尝试](../tasks/copper_inversion/resources/evidence/deepseek-3d/comparison.json)
- [钻孔运行](../tasks/copper_inversion/resources/evidence/deepseek-3d/drillholes_3d/campaign.json)、[最佳算法](../tasks/copper_inversion/resources/evidence/deepseek-3d/drillholes_3d/best_solver.py)
- [露头运行](../tasks/copper_inversion/resources/evidence/deepseek-3d/drillholes_outcrops_3d/campaign.json)、[最佳算法](../tasks/copper_inversion/resources/evidence/deepseek-3d/drillholes_outcrops_3d/best_solver.py)
