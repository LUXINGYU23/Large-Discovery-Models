# 蛋白设计工具阶段交付

交付分支：`codex/expanded-protein-tools`。代码与校准基线提交：`43aa4ad49b2be0d5f01bdbe0b43fa954da7d5206`；本说明随后作为文档提交追加。交付定位为可选生成工具及实验性辅助预测/评分，当前阶段收尾。

## 交付清单

| 工具 | 交付范围 | 使用边界 |
|---|---|---|
| SolubleMPNN | 可选 H3 逆折叠工具，含真实 LLM/Oracle 与同预算对照记录 | vanilla 继续默认；不把模型名视为可溶性证明 |
| ipSAE | 独立 CLI 与可选 Pi 评分入口，原生置信度转换和 Boltz 附件处理 | 需要匹配的真实结构/PAE；不替代 Absolut |
| Protenix v1 | 完整序列、MSA、多 seed、真实预测与 ipSAE 衔接 | 阳性漏检；独立实验性辅助入口 |
| BoltzGen 0.3.2 | 固定框架 H3 原生骨架生成＋逆折叠；双抗原真实 LDM/Absolut 闭环 | `--enable-boltzgen` 显式启用；默认关闭；完整原生排名流程未验证 |
| Boltz-2 | 隔离环境、真实复合物预测、原生置信度、ipSAE、双抗原校准 | HER2 结构恢复较好；PD-1 不稳；评分阈值敏感 |
| ESMFold2 | 官方完整权重、原生预测、严格置信度转换、ipSAE、双抗原校准 | PD-1/Fv 控制较好；HER2 控制失败 |
| PXDesign | PD-1/HER2 的真实完整蛋白 binder 生成入口 | 独立 binder 域；不接入 11 aa H3；完整后筛选流程未验证 |
| Protenix v2 | 官方来源与访问条件核对记录 | 未运行；官方权重 HTTP 403，授权来源未解决 |

## 验收证据

本轮校准新增 36 次真实预测及对应 ipSAE，另完成 48 次 PAE 阈值敏感性评分；全部完成且无缺失。12 个相关工具测试通过。此前 BoltzGen 双抗原真实 LLM 闭环和可选工具/共享流程验证保留在证据目录。

当前建议按目标选择辅助复核路径：PD-1 优先研究 ESMFold2/Fv，HER2 优先研究 Boltz-2 的结构与置信度。数据来自少量已知阳性及计算错配，未建立新 H3 的可靠亲和力排序、假阳性率或假阴性率。没有新增自动拒绝阈值，也没有更改默认预测/筛选行为。

## 交接入口

- [配置、路径、实现与运行命令](protein-tools-expanded.md)
- [Fab/Fv、MSA、真实界面与跨抗原校准报告](protein-tools-calibration.md)
- [各工具 CLI 与输入/输出说明](../tasks/antibody/tools/complex_screening/README.md)
- [小型验证记录与模型校验清单](../tasks/antibody/tools/complex_screening/validation/)
- [历史 H20 实验驱动与复现顺序](../tasks/antibody/tools/complex_screening/repro/h20/README.md)

服务器共同根目录：`/Aidata/shared/ldm/antibody-poc`。隔离环境在 `env/`，权重和化学数据在 `cache/`，原始结果在 `runs/`，日志在 `logs/`。完整独立验证代码目录为 `tmp/expanded-protein-tools`。后续新增的校准说明与历史驱动以本交付 Git 分支为准；该临时验证目录不是完整的 Git 交付副本。

本次未切换或修改服务器主工作树 `code/`，未合并到 `main`。权重和大型原始产物留在 H20，不包含在 Git；共享 PyTorch 库依赖及迁机要求已在配置说明中记录。调用时使用对应隔离解释器，并选择空闲 GPU 和新的输出目录。

## 阶段结束状态

本轮计算已全部完成，不再继续调参或启动新实验。后续自动筛选验证应使用具有真实测量的独立阳性、阴性及突变体数据。Protenix v2 等待权利人认可的获取渠道；该项不计入已运行工具。
