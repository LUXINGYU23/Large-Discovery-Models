# 几何诊断与封存算法复评

## 定义

固定 `objective_v1` 损失不变。评分进程在重新距离化后统计正值区域：

- 2D：`positive_area` = 正值网格数 × dx × dz。
- 3D：`positive_volume` = 正值网格数 × dx × dy × dz。
- 有活动掩码时：`air_positive_voxels` = 正值区域中 active_mask 为 false 的网格数；对应面积/体积使用相同网格测度。`air_fraction_of_positive` 的分母为全部正值网格数。
- 没有活动掩码时省略空气指标，表示无法判断。物理单位沿用输入网格，不另行假定米制。

这些是网格近似和诊断，不参与 loss 或选点，不对零等值面做额外几何积分。正值面积/体积没有预设优劣方向。

## 远程验证

在 `copper-final-runtime-20261023` 独立目录执行，原 campaign 不变：

- 79 项任务测试通过，包括非单位 spacing、2D/3D、缺失掩码、不影响 loss 和真实隔离/恢复路径。
- 注册 `contract_verified` 检查通过，`ldm_drillholes_3d` runner dry-run 通过；资格仍为 draft，不冒充官方 benchmark 认证。
- 三组 3D campaign 的 12 份封存源码全部重新通过只读接口测试、正式 solver 和独立 scorer，loss 与原记录逐项一致（容差 1e-10）；完成结果重复读取无新增执行计数。没有新模型请求。
- 原 2D campaign 的 4 份封存源码也全部复评一致：293.2197607509、3.9504787504、3.7968566398、3.7338023458，重复读取不重跑。见 [2D 复评记录](../tasks/copper_inversion/resources/evidence/runtime/two-d-replay.json)。

| campaign 最佳候选 | loss_total | 正值体积（网格单位³） | 空气正值网格数 |
| --- | ---: | ---: | ---: |
| P5 Coverage/GP 钻孔 | 6.5167894367 | 27452 | 1996 |
| 单 agent DeepSeek 钻孔 | 6.5166041269 | 27756 | 1633 |
| 单 agent DeepSeek 钻孔＋露头 | 35.5994879770 | 5727 | 0 |

钻孔案例的低拟合损失不保证矿体位于活动区域。`active_mask` 在 v1 中仅供诊断，这次没有擅自增加空气惩罚或改变评分目标。

机器证据：[测试与契约](../tasks/copper_inversion/resources/evidence/runtime/geometry-diagnostics.json)、[3D 复评记录](../tasks/copper_inversion/resources/evidence/runtime/three-d-replay.json)。远程完整结果在 `copper-final-runtime-20261023/runs/diagnostics-replay-v1`；完整归档 SHA-256 为 `04cb029c35b87375dc8c607a2416b396c1a13559bfcbe292ef0e9482545e6856`，本地保存于忽略目录 `runs/research-tree-20261023/`，复制前完成凭证扫描。

这些证据覆盖当前评分与执行实现，不替代尚未完成的整批准入顺序、提交契约及生命周期峰值内存验收。

2D 完整归档 `diagnostics-replay-2d-v1.tar.gz` SHA-256：`7634fa4582e8d25d1c33d8bac197d2b5fa03eefba294c5e14276b3caed1bdc2d`，同样保存于远程 runs 和本地忽略归档目录。

## 最终监督执行器复评（H20）

在当前可信 PID 1 执行器上重新执行上述全部 16 份历史源码：2D 四份、P5 四份、两个单 agent 3D case 各四份。全部通过两个只读接口输入、正式 solver 和独立 scorer；与历史 loss 的差值均为 0（断言容差 1e-10）。每次正式结果均具有完整工作负载资源记录及正的内核峰值；重复读取结果不增加执行用量。

这次没有调用模型，也没有改变历史候选。运行目录为 H20 `/Aidata/shared/ldm/copper-inversion-20261023/runs/final-replay-20261023`。

- [完整机器证据](../tasks/copper_inversion/resources/evidence/runtime/final-supervised-replay.json)
- [执行日志](../tasks/copper_inversion/resources/evidence/runtime/final-supervised-replay.log)
- 完整归档：`/mnt/cephfs/xingyu/ldm/archive/copper-inversion/final-replay-20261023.tar.gz`
- SHA256：`76daa200c97368a38b9dbec6cfaa0d1d003375ca8d698a1f8422802833a5b358`

前面的 `369ea6dd…` 证据对应旧采样执行器；本节证据记录最终监督执行器身份，二者没有混用。P5 新批量提交协议的原生运行属于单独验收项。
