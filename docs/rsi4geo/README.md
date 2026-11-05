# RSI4Geo codebase

本分支整理铜矿三维反演的代码、实验协议和现有结果，作为 RSI4Geo 论文的 codebase。
原 LDM 框架和其他任务保留在原目录；论文手稿由外层工作区的 `paper/` 独立仓库管理。

## 阅读入口

| 内容 | 入口 |
| --- | --- |
| 当前任务、输入输出及运行前提 | [Copper inversion README](../../tasks/copper_inversion/README.md) |
| 最终实验结果、种子与限制 | [正式实验报告](../copper-v2-formal-study.md) |
| 发现种子42的算法和效果 | [报告说明](../copper-seed42-final-20261031/README.md)、[完整报告](../copper-seed42-final-20261031/index.html) |
| 算法演进与原始源码 | [交互轨迹](../copper-seed42-final-20261031/trajectory.html)、[算法审计](../copper-seed42-final-20261031/algorithm-notes.html) |
| 全部种子42案例的几何比较 | [案例浏览器](../copper-seed42-final-20261031/browser.html) |
| 全部发现种子的案例比较 | [432套案例](../copper-baseline-rerun-20261027/all-case-visual-draft-20261031/index.html) |
| 科学协议与实施历史 | [v2 specification](../copper-inversion-v2-spec.md)、[implementation log](../copper-inversion-v2-implementation.md) |
| 早期实验与旧基线 | [r3 历史报告](../copper-v2-formal-study-r3-archive-20261031.md) |

## 代码布局

| 路径 | 职责 |
| --- | --- |
| `tasks/copper_inversion/v2/` | 公开问题契约、科学目标、接口检查、私有评分服务、预算和验证选型 |
| `tasks/copper_inversion/core/` | 候选冻结、容器执行、资源计量、harness 接入 |
| `tasks/copper_inversion/experiments/v2/` | 数据发布、资格检查、正式实验编排、并发调度与恢复工具 |
| `tasks/copper_inversion/experiments/baselines/` | 早期基线和报告生成工具；按对应实验身份解释 |
| `tasks/copper_inversion/resources/v2/` | 研究者可见任务说明、提示和自检资源 |
| `tasks/copper_inversion/resources/evidence/` | 历史资格检查和实施证据 |
| `tasks/copper_inversion/tests/` | 科学契约、服务、预算、恢复、并发和隔离回归测试 |
| `ldm_tts/` | 共享 LDM 引擎、预算账本和 harness 客户端 |
| `harnesses/pi/` | Pi sidecar、候选提交及研究计时 |
| `config/copper_inversion/fusion_pilot.yaml` | 已部署开发环境的示例配置；新实验需单独配置 |

实验编排和恢复脚本包含原运行路径与身份约束。它们是相应实验的操作记录，
执行前需阅读其参数和来源；不能在历史实验目录中直接启动新实验或覆盖冻结快照。

## 结果口径

当前主指标为 `1 - parent-macro confirmation utility`。每次成功求解的效用为
`1 / (1 + confirmation_loss_total)`；失败效用为零，所有计划案例保留在分母中，
先按母体内案例平均，再按地质母体等权平均。IoU 是单独报告的几何指标。
实现见 [suite.py](../../tasks/copper_inversion/v2/suite.py)。

当前最终比较复用 r3 的 LDM 冻结结果和修复后的 r4 MCMC/ACQ。
验证只使用执行种子47，测试使用47/48/49；不能表述为已完成三执行种子验证。
种子42冻结赢家的 `best_pair` 内部最佳模型记录问题已在算法审计中披露。
此次仓库整理不修改冻结源码、历史模型或评分收据，也不重跑、重选。

## 运行与验证

调试、测试和实验在授权远程服务器执行。本机用于编辑和 Git 管理。
按任务 README 准备 Python 依赖、Docker 镜像和私有数据服务；示例配置不是一键复现脚本。
私有数据、模型缓存、密钥及活动运行目录不随此 Git 仓库分发。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/validate_tasks.py --task copper_inversion
python -m pytest -q tasks/copper_inversion/tests \
  tests/test_ldm_engine.py tests/test_harness_client.py \
  tests/test_harness_batch_validation.py
cd harnesses/pi
npm ci
npm test
```

Pi 使用 Node.js 24 或更高版本。容器集成测试还需要相应镜像和 Docker 权限；
可选集成测试的跳过条件以测试文件为准。整理后的实际验证记录见
[2026-11-05 verification](verification-20261105.md)。

## 归档约定

报告源码、JSON 数据、冻结求解器、图片、PDF 和浏览器资源保留原路径，
便于核对历史链接及摘要。已有 `draft` 文件名是历史标识，最终状态以正式报告为准。
两个重复 ZIP 导出包仅保留在本地并被 Git 忽略，展开后的报告文件纳入版本控制：

- `docs/copper-technical-report.zip`
- `docs/copper-seed42-final-20261031/copper-seed42-offline-package.zip`

旧 v1 注册配置归档 `docs/archive/copper-v1/registration-configs.tar.gz` 继续跟踪。
运行所需的 `tasks/copper_inversion/resources/v2/AGENTS.md` 明确纳入 Git；
本机操作指令 `tasks/copper_inversion/AGENTS.md` 仍是本地文件。
