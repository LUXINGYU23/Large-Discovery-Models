# ESMFold2 / Boltz-2 构建与结构对照校准

本轮检查抗体构建、MSA 与真实界面恢复。预测器仍是辅助工具，没有新增硬筛选阈值，也没有改变生产默认配置。

## 协议与参考

PD-1 使用 [5GGR](https://www.rcsb.org/structure/5GGR) 的 nivolumab 阳性与 trastuzumab 计算错配；HER2 使用 [1N8Z](https://www.rcsb.org/structure/1N8Z) 的 trastuzumab 阳性与 nivolumab 计算错配。错配不是实验认证阴性。这些老结构可能出现在模型训练数据中，本轮是已知阳性控制与流程诊断，不是排除训练重叠的泛化基准。

每个条件固定 seeds 19/43/101，每 seed 一个样本，不挑最佳 seed、不为失败样本补抽。ESMFold2 使用 20 loops / 200 sampling steps；Boltz-2 保持此前原生默认循环、200 diffusion steps。两者结构、序列、样本数量与 seed 控制一致，但模型循环和计算量不相等，不进行速度或等 FLOP 优劣比较。没有用原生复合物结构、接触约束或模板引导预测；原生结构只用于事后评估。

PD-1 的 Boltz-2 做 Fab/Fv × 原有 MSA/无 MSA 四种条件；ESMFold2 做 Fab/Fv 无 MSA。保留完整抗原序列 125 aa。Fv 明确按重链 `ASTKGPS`、轻链 `RTVAAPSV` 常量区连接处截取完整变量域：nivolumab VH/VL 113/107 aa，trastuzumab 120/107 aa；这是声明的分子构建变化，不是按缺失 ATOM 裁剪。Fab→Fv 同时移除了常量区及原始 Fab 的末端标签，不能据此单独认定哪个因素造成改善。

Fv 的“MSA”是原有真实完整 Fab unpaired A3M 的对应列裁剪，不是新搜索的 Fv MSA。裁剪保留插入列语义，校验完整 query 和所有行的对齐长度，去掉裁后全空行，记录源 SHA256 与行数。抗原 MSA 与序列不变。无 MSA 条件显式指定 `msa: empty`，不联系远程搜索服务。

## 结构审计

5GGR 的正确 author-chain 参考是 **A 重链 / B 轻链 / Y 抗原**；另一组是 H/L/Z。Y 与 H/L 不接触，不能混用晶体内的不同拷贝。1N8Z 的参考是 **B 重链 / A 轻链 / C 抗原**。所有预测统一为 A 重链 / B 轻链 / Y 抗原。

`pose_audit.py` 对阳性做序列匹配和原生结构对照，报告：

- 仅对齐抗原 CA 后的变量域与界面抗体 CA RMSD。
- 原生接触恢复比例：映射到共同已观察残基后，任意重原子距离 <5 Å 的 Fv–抗原残基对中，有多少在预测中也接触。
- 共同残基数量、接触数量、预测接触精度及原生/预测文件 SHA256。

这些不是 DockQ 或亲和力。多结构域 HER2 的全抗原对齐 RMSD 还会受到域间相对运动影响，需同时阅读接触恢复。没有为错配构造虚假的“原生结合姿态”。没有用实验结构 B-factor 伪造 pLDDT，也没有为原生结构合成 PAE。结构/PAE 对没有进行固定框架投影。

两份原生结构自比接触恢复均为 100%、RMSD 约为数值零。额外测试验证全局刚体运动不改变结果、错误抗体位置会丢失原生接触、错误晶体拷贝被拒绝。

## PD-1 完整结果

每格按 seed **19 / 43 / 101** 排列。ipSAE 是重链与抗原两个方向中的较小值，原始文件保留全部方向性分数；这些数值没有被转换成通过/拒绝规则。

| 模型与条件 | 阳性 ipSAE | 阳性原生接触恢复 | 错配 ipSAE |
|---|---|---|---|
| Boltz-2 Fab＋MSA | .011 / .011 / 0 | 18.8% / 12.5% / 12.5% | .140 / .045 / .013 |
| Boltz-2 Fab，无 MSA | 0 / 0 / .010 | 12.5% / 8.3% / 0% | 0 / 0 / .023 |
| Boltz-2 Fv＋裁剪 MSA | 0 / .014 / 0 | 0% / 2.1% / 6.2% | .115 / 0 / 0 |
| Boltz-2 Fv，无 MSA | .205 / 0 / 0 | 45.8% / 0% / 2.1% | .014 / 0 / .129 |
| ESMFold2 Fab，无 MSA | .447 / 0 / .422 | 77.1% / 0% / 81.2% | 0 / 0 / 0 |
| ESMFold2 Fv，无 MSA | .542 / .515 / .547 | 79.2% / 81.2% / 83.3% | 0 / 0 / 0 |

ESMFold2 的 Fv 构建在该 PD-1 控制上改善了三 seed 的一致性，包括原 Fab seed 43 的姿态失败。Boltz-2 的失败不能简单归因于一个 MSA 开关或常量区长度；当前条件中仅 Fv/无 MSA/seed 19 有部分界面恢复，仍不足以作为 PD-1 筛选依据。

## HER2 复核与置信度诊断

HER2 使用完整胞外实体序列 607 aa，阳性为 trastuzumab Fv 120/107 aa，错配为 nivolumab Fv 113/107 aa；均无 MSA、无模板或接触约束，蛋白输入不包含糖基化修饰。

| 模型，Fv／无 MSA | 阳性 ipSAE，PAE cutoff 10 | 阳性原生接触恢复 | 错配 ipSAE |
|---|---|---|---|
| Boltz-2 | .010 / .111 / 0 | 80.9% / 91.5% / 91.5% | 0 / 0 / 0 |
| ESMFold2 | 0 / 0 / 0 | 0% / 0% / 0% | 0 / 0 / 0 |

因此，PD-1 的 ESMFold2/Fv 改善不能推广成“该模型在所有抗原上稳定”；Boltz-2 的 PD-1 失败也不代表它无法恢复 HER2 界面。已知阳性上呈现目标依赖的互补现象，尚不足以证明新 H3 候选的优劣排序有效。

针对 HER2 Boltz-2 seed 19，进一步逐 CA 对照原生 pLDDT NPZ 与 CIF B-factor，归一化最大差约 0.000005；PAE 为 834×834，与 834 个蛋白 CA 对应，没有发现尺度或顺序异常。47 个共同原生接触中恢复了 38 个，但这些原生接触的 PAE 中位数约 11.50/11.65 Å，两方向在 10 Å 以下的比例仅 0%/2.1%。低 ipSAE 可由原生置信度与 10 Å 过滤共同解释，并不要求存在解析错误。

在观察到该现象后，另开展 **探索性** PAE cutoff 15/20 Å 敏感性检查：对两抗原、两模型、所有三 seed 的 Fv／无 MSA 阳性和错配统一评分，距离参数保持 10 Å。没有只放宽阳性，没有重跑或挑选结构，没有把事后参数探索称为独立阈值验证。默认 cutoff 仍为 10。

敏感性检查的 48 次评分已全部完成。以 PAE cutoff **15 Å** 为例，重链/抗原最小方向性 ipSAE：

| 模型与目标，Fv／无 MSA | 阳性，19/43/101 | 错配，19/43/101 |
|---|---|---|
| ESMFold2，PD-1 | .541 / .514 / .545 | 0 / .005 / 0 |
| ESMFold2，HER2 | 0 / 0 / 0 | 0 / 0 / 0 |
| Boltz-2，PD-1 | .229 / 0 / 0 | .103 / .006 / .167 |
| Boltz-2，HER2 | .104 / .153 / .086 | 0 / 0 / 0 |

20 Å 的完整结果也已保留：没有挽救 ESMFold2/HER2 的姿态失败，也没有挽救 Boltz-2/PD-1 的稳定性。HER2 Boltz-2 的正确结构在 15 Å 下得到较高的辅助分数，同时该组错配维持低分；这支持继续检查该路径，但不构成经过验证的普适阈值。

本轮建议：**PD-1 优先研究 ESMFold2/Fv，HER2 优先研究 Boltz-2 的结构恢复与置信度校准。** 两者在这些已知阳性上有互补价值；新 H3 的亲和力排序、假阴性率和假阳性率仍需更大、具有实际测量的数据集。目标特定选择和 PAE 参数探索已经使用了这些对照的结果，不能再把同一批数据称作独立验证集。当前继续保持辅助用途和原有默认配置。

## 路径与复现

服务器根目录 `/Aidata/shared/ldm/antibody-poc`：

- `runs/complex-construct-calibration-001/plan.json`：事前确定的 24 个新增 PD-1 作业。
- 同目录 `inputs/manifest.json`：完整序列→Fv 与 A3M 裁剪审计。
- 同目录 `summary.json`：24 个新增作业＋12 个既有 Fab 作业，共 36 个条件结果；阳性附真实结构审计。
- `runs/her2-complex-calibration-001/plan.json`：第二抗原完整 HER2 胞外实体序列、Fv、无 MSA、两模型、三 seed 阳性/错配计划。
- `runs/complex-pae-sensitivity-001/{plan.json,summary.json}`：两抗原、两模型、所有 Fv／无 MSA 控制的 48 次额外阈值评分。
- `logs/complex-construct-calibration-001.log`、`logs/her2-complex-calibration-001.log`：完整实验驱动日志；每个作业另有 native 工具日志、运行状态与 ipSAE。

仓库工具 `construct_controls.py` 与 `pose_audit.py` 可通过模块 `--help` 查看接口。历史 H20 驱动脚本保存在 `tasks/antibody/tools/complex_screening/repro/h20/`，固定本次数据路径；重新运行时改用新实验目录，保留既有证据。

仓库 `validation/h20-pd1-construct-calibration.json` 保存 PD-1 的计划、输入谱系、原生自检、全部方向性分数和阳性结构审计。`h20-her2-complex-calibration.json` 保存第二抗原的完整结果；`h20-boltz-confidence-audit.json` 保存原生置信度/结构对照；`h20-pae-cutoff-sensitivity.json` 保存全部探索性评分。`validation/protenix-v2-access-review.json` 保存官方权重访问复查记录。

## Protenix v2

再次核对了官方 README、模型列表和下载 URL 注册表。源码支持 `protenix-v2`，但 H20 对官方权重 URL 的 HEAD 请求仍返回 HTTP 403。官方 README 另明确说明 v2 权重为专有资料，第三方转交受权利人书面同意条件限制。没有下载第三方镜像，没有运行 v2，也没有把 v1 当作 v2。

下一步依赖权利人认可的权重获取方式，或具有合适访问条件的官方服务；当前不能通过增加安装命令解决该问题。[官方 README](https://github.com/bytedance/Protenix)、[官方 URL 注册表](https://github.com/bytedance/Protenix/blob/main/protenix/web_service/dependency_url.py)。

本轮新增 24 次 PD-1、12 次 HER2 真实预测及其原生 ipSAE，并完成 48 次额外敏感性评分；预测与评分作业均成功、无缺失。12 个相关单元/回归测试通过，包括 A3M 插入列语义、刚体不变性、错误姿态与错误晶体拷贝。服务器主工作树保持未改动。
