# 架构收敛实施计划

日期：2026-10-21。状态：P1—P5 已完成；外部资产限制见最终验收。

代码基线：`31f923efe71e75f7b517039a16b9d0d609321ded`，分支：`codex/default-coverage-tree`。
执行对照基线为计划提交 165f048f，其代码与上述基线相同。下方记录已实施的范围、提交与验收结果。

## 目标与范围

在保持现有功能和用户行为的前提下，完成六个 Pi Harness 任务及其共享执行链路的职责整理。
目标是让配置、科学逻辑、资源生命周期、持久化和恢复各有明确归属，减少一次修改需要跨越的模块。
文件长度和新增模块数量不作为完成指标。

直接范围为 NucleoBench、Iron Mind、SynthonBench、ReaSyn、AtomWorld、AlphaBench，
以及 `ldm_tts/campaign.py`、`ldm_tts/engine/`、`ldm_tts/harness/`、`harnesses/pi/src/`
中与这些任务执行有关的部分。相关入口、配置、测试和文档随实现一起更新。
共享引擎的其他调用方，包括 Antibody、Small molecule、nanoGPT，纳入导入更新与回归验证，
不扩展为新的 Pi 接入迁移。

本轮不新增搜索策略、不恢复 JointGain、不调整模型或科学预算，也不启动性能对照实验。
现有独立 session、最新主干 4fork 和默认 Coverage 2+2 都是当前功能，继续保留；
compiled policy 是现有优化能力，与已删除的 JointGain 决策策略分开处理。

## 基线与约束

已完成的整理：

| 部分 | 当前基础 |
| --- | --- |
| 通用启动 | 六个任务均使用 `pi_sidecar_command`，Docker 装配规则集中管理 |
| 树搜索 | 策略选择、研究协调、快照存储已有独立职责，继续使用现有接口 |
| Compiled policy | 契约、持久化与编排已分离 |
| NucleoBench | 参数解析、官方执行和 Harness 装配已从主流程归位 |
| ReaSyn | 配置、科学身份、结果报告和客户端生命周期已完成整理 |

最近一轮 ReaSyn 验证记录为 **1328 通过、116 跳过**，164 个配置行为案例、13 个
确定性 mock campaign 及完成后恢复案例一致。证据位于远程
`/mnt/data1/Large-Discovery-Models/research/reasyn-workflow-refactor-20261021/verification.json`。
这些是此前验证结果，不代表本计划已执行或未来改动已经通过。

实施约束：

- 使用已有依赖、普通函数、数据类和 `ExitStack` 等现有模式；不引入任务基类、插件框架或通用工作流框架。
- 先归位明确的职责，再合并已确认相同且有实际使用者的逻辑。任务差异通过现有适配器表达。
- 移动实现时同步修改全部调用方，删除原定义、旧导出和转发 wrapper，不建立兼容路径。
- 保持 CLI、错误语义、默认策略、候选身份、q0、预算、评估顺序和结果报告语义。
- 保持现有持久化格式、回执键及提交顺序；不增加迁移分支或绕过身份校验。
- 源码和编译产物改变会产生新的实现哈希，必须如实记录。不得伪装成旧运行环境或改写历史实验 provenance。
- 本机只编辑和静态检查；调试、构建、测试均通过 `ssh zsgpu@111.2.199.31 -p5320` 在 `/mnt/data1/` 下执行。
- 使用独立验证目录，不覆盖冻结实验、已有 session、原始 trace 或共享运行环境。

## 目标职责

| 位置 | 应承担的职责 |
| --- | --- |
| 任务配置与契约 | 解析、规范化和验证输入，定义任务科学约束与运行身份 |
| 任务 `workflow` | 准备输入、恢复运行、装配组件、执行 campaign、输出结果 |
| 任务 `harness_runtime` | 创建客户端与任务工具服务，绑定配置，管理资源生命周期 |
| 任务科学适配器 | 候选合法性、公开证据、评分、投影、q0、surrogate 和 evaluator |
| Campaign / Engine | 统一轮次推进、候选准入、选择、评估与完成判定 |
| 执行持久化 | 预算、评估回执、结果落盘和 checkpoint 的一致性 |
| 树搜索模块 | 选择父节点、冻结计划、传递证据、保存分支快照与结果 |
| Python Harness / Pi | 进程通信、协议校验、原生会话、提交、历史压缩和恢复保护 |

共享层不导入任务实现，任务报告不负责启动客户端。文件按实际职责组织，
不要求六个任务具有完全相同的目录，也不为单个转发函数创建模块。
现有接口和任务边界详见[共享研究树](research-search-architecture.md)与
[Harness 接入说明](research-harness.md)。

## 实施顺序

### P1：完成 AtomWorld 职责分离

涉及 `tasks/atomworld/core/workflow.py`、`harness.py`、相关入口和测试。

- [x] 将参数解析及规范化归入 `workflow_args.py`，报告投影归入 `reporting.py`。
- [x] 将科学运行身份的构造和校验集中归位，保留 `schedule.json` 的内容与校验语义。
- [x] 将客户端工厂、provider 预检和装配归入 `harness_runtime.py`。
- [x] 理清样本级客户端与研究树的生命周期，保留任务专属提交校验和证据导出。
- [x] 更新实际入口和测试导入，删除原位置的实现与间接转发。

验收：配置与报告对照一致；每个样本使用自己的研究树；公开模式不暴露私有答案，
LDM 模式仅增加既有授权反馈；结果仍按最终提交报告，不使用隐藏评分挑选最佳草稿。
覆盖启动失败、部分执行和完成后恢复，核对预算与服务关闭。

建议两个提交：配置、身份与报告；客户端装配与生命周期。

配置、身份与报告已归位；远程 AtomWorld 测试 101 通过、5 跳过。
本轮证据目录为 `/mnt/data1/Large-Discovery-Models/research/architecture-convergence-20261021`，
该批记录为 `p1-config.xml` 与 `p1-config.log`。
运行时整理后 102 通过、5 跳过，新增启动失败后关闭客户端并保持恢复预算的验证；
记录为 `p1-runtime.xml` 与 `p1-runtime.log`。

### P2：完成其余任务的装配边界

涉及 Iron Mind、SynthonBench、AlphaBench 的 `workflow.py`、`harness_runtime.py`
及相应适配器；NucleoBench、ReaSyn 做一致性收尾。

- [x] Iron Mind 将主研究、侧枝及 policy 装配移入已有 runtime 模块，主流程仅组合科学组件。
- [x] SynthonBench 完成同类整理，保留其特征编码、合法反应空间及产品身份规则。
- [x] 检查两者重复逻辑；只合并稳定且任务无关的部分，优先复用现有 `MeasuredResearchExpander`。
- [x] AlphaBench 集中管理 Harness、policy 客户端和 Host 工具服务的启动、绑定与关闭。
- [x] 减少 AlphaBench 主流程中通过可变字典和转发对象连接组件的逻辑，保持 Host 授权及 native 执行路径。
- [x] 核对 NucleoBench、ReaSyn 的入口、装配和已完成恢复路径，移除遗留绕行。

验收：三个任务分别完成默认 2+2 的端到端合成 Harness 验证；检查独立 session、4fork
及 compiled policy 路径；启动中途失败时关闭已创建资源。AlphaBench 验证／测试反馈保持私有，
ReaSyn 保留补充 minibatch 和投影结果到原提案的归因，NucleoBench 保留官方执行与测量规则。

按任务分别提交；跨任务共享代码只在调用方同时完成迁移时提交。

Iron Mind、SynthonBench 与共享树测试 206 通过，记录为 `p2-measured.xml`。
装配复用已有协调器，任务特有 adapter 参数仍留在各自 runtime 中。
AlphaBench、Campaign 和六任务研究树适配测试 265 通过、110 跳过，记录为 `p2-alpha.xml`。
AlphaBench 直接装配共享 LDMEngine，复用 `CampaignRequest.open_runtime` 的原有预算／身份逻辑，
已删除延迟绑定字典与 evaluator 转发类；故障测试验证客户端与 Host socket 一起关闭。

### P3：收敛共享执行与恢复逻辑

涉及 `ldm_tts/campaign.py`、`engine/runtime.py`、`engine/run_store.py` 及全部调用方。

- [x] 将引擎配置、运行状态与 checkpoint 编解码从轮次编排中分离，保留当前数据格式和验证规则。
- [x] 将评估回执、预算预留、dispatch、结果保存和恢复集中为一个明确的执行单元。
- [x] 使主循环负责阶段推进；存储层继续负责持久化，不让任务自行操作引擎恢复状态。
- [x] 保持已有持久化位置与顺序，包括 `engine_active_selection`、usage key、批量返回结果先整体保存。
- [x] 同步更新六个 Harness 任务及其他共享引擎调用方，不增加新的 Campaign 或恢复框架。

验收：在请求发出后、结果保存后、事件写入后、checkpoint 前等现有故障边界恢复；
已保存结果不重调 evaluator、不重复扣费；未知结果且没有可靠回执时继续暂停。
同时验证单条／批量评估、失败补充、预算耗尽、空候选池和完成后恢复。

建议两个提交：状态与配置归位；评估执行与恢复归位。保持回执操作整体可读，
不把紧密相关的保存步骤拆成跨多个层的回调链。

状态和配置已归入 `engine/state.py`，共享引擎、Campaign、AtomWorld、ReaSyn 测试
319 通过、6 跳过，记录为 `p3-state.xml`。

评估执行已归入 `engine/evaluation.py`；预算预留、批量结果保存、事件和 checkpoint 顺序不变。
恢复故障测试改为在实际 `CampaignRuntime.checkpoint` 边界注入，329 通过、6 跳过，
记录为 `p3-evaluation.xml`。

### P4：收敛 Python Harness 与 Pi 会话边界

涉及 `ldm_tts/harness/client.py`、`protocol.py` 和 `harnesses/pi/src/`。

- [x] 分离 Python 响应解析、结构校验与进程通信，保留请求身份、提交摘要和 usage 检查。
- [x] 将 Pi `session.ts` 中的 `SubmissionController` 归入现有提交模块，避免双向依赖。
- [x] 集中 fork 初始化、快照验证与原子落盘逻辑，让会话模块调用明确的初始化操作。
- [x] 保留现有 `HistoryController` 和 provider 代理边界，检查清理、重放和初始化身份的调用关系。
- [x] 迁移测试到对应职责，删除旧导出与仅验证旧组织方式的测试。

验收：执行 TypeScript 构建与测试、Python 客户端回归以及具备资产的隔离 sidecar 测试。
真实代理实现必须验证研究请求仍应用研究参数、原生摘要保留自己的 low 与输出预算。
覆盖 fork 篡改、部分分支提交、重放、拒绝提交、超时、length-stop 和失败调用 usage。
所有关键身份校验和恢复保护继续生效。

Python 响应校验已归入 `harness/responses.py`，错误契约归入 `protocol.py`；
进程通信与重试仍由客户端负责。客户端、policy 和任务适配测试 67 通过，记录为 `p4-python.xml`。

Python 协议整理、Pi 提交模块归位、Pi fork／会话整理分别提交。
提交控制器及修复／重提测试已归入 `submission.ts` 和 `submission.test.ts`，
远程 TypeScript 构建与 45 项测试通过，记录为 `p4-submission.log`。
fork 原子初始化、资源摘要与运行身份校验归入 `session-store.ts`，会话模块不再导出这些实现。
构建与 45 项测试再次通过，记录为 `p4-session-store.log`；真实代理路径的摘要参数隔离、
length-stop、分支排空及提交重放测试均保留并通过。
新构建记录独立的源码与镜像身份，不部署覆盖历史实验运行环境。

### P5：全仓调用方和文档验收

- [x] 检查入口、配置、脚本、测试及文档中对移动实现的引用，清除死代码与旧接口。
- [x] 整理重复或实现细节测试，保留核心行为、科学边界和恢复故障测试。
- [x] 更新六任务 README、`research-harness.md`、`research-search-architecture.md` 和 `testing.md`。
- [x] 补齐任务装配、资源所有权、持久化提交顺序和恢复限制的说明，注明可运行验证命令。
- [x] 完成任务注册、六任务配置与合成运行、共享引擎调用方、Pi 运行时的最终回归矩阵。
- [x] 保存最终验证记录，逐项列出通过、失败和跳过原因；更新本计划状态及各阶段提交。

文档随每批实现同步更新；本阶段负责汇总与查漏，不让文档长期描述尚未实现的结构。

## 验证与提交方式

每批采用相同流程：记录基线与输入 → 修改实现及全部调用方 → 删除旧路径 →
远程针对性验证 → 检查行为差异 → commit → 更新本计划的状态与证据。
出现未解释的行为差异时先修复本批，不叠加下一层重构。

| 验证面 | 主要核对内容 |
| --- | --- |
| 配置与入口 | CLI 帮助、合法／非法参数、冻结配置、默认策略和显式策略、任务注册 |
| 科学行为 | 候选身份与数量、顺序、q0、实际测量、报告、补充请求及预算 |
| 树与权限 | 父快照、历史游标、冻结计划、分支提交、样本隔离及任务授权证据 |
| 恢复与计费 | 部分完成、可靠回执、未知结果暂停、完成重放、失败／摘要／缓存 usage |
| 生命周期 | 主枝、侧枝、policy 与 Host 服务的正常及异常关闭 |
| 构建与集成 | Python 分环境测试、TypeScript `npm test`、可用 sidecar／guest 与上游资产验证 |

确定性行为对照只排除预先列明的时间戳、耗时和运行路径等非科学字段。
实现哈希变化单独记录，不能通过扩大忽略范围掩盖预算、身份或结果变化。
既有测试优先复用，只为缺失的关键行为添加测试。

116 个基线跳过项中包含未就绪的 AlphaBench 固定上游源码和 Pi／policy runner 资产。
实施时先核实远程环境可用性，能在隔离目录补齐的验证就补齐；其余逐项标记为未验证，
不把 mock 通过等同于真实服务或上游集成通过。

## 完成标准

- [x] P1—P5 全部完成，每批均有独立提交和对应验证证据。
- [x] 六个任务的工作流、装配与科学逻辑职责明确，所有受影响调用方完成更新。
- [x] 共享层保持任务无关；没有因重构新增无实际用途的接口、依赖或兼容层。
- [x] 默认策略、科学预算、数据权限、报告与恢复行为满足上述约束。
- [x] 被替代的代码、wrapper、旧测试和失效注释已删除。
- [x] 文档与实际实现一致，未验证的集成边界有明确说明。
- [x] 历史实验、原始 trace 和 provenance 保留；工作树无遗漏改动。

## 最终验收

全部调试、构建、运行及行为对照在独立远程目录完成：
/mnt/data1/Large-Discovery-Models/research/architecture-convergence-20261021。
机器记录为 verification.json，逐文件校验为 source-manifest.json，构建产物校验为 dist-manifest.json。
历史实验、session、trace 和 provenance 未部署覆盖；没有新增策略、模型调用或科学对照实验。

| 验证 | 结果 | 记录 |
| --- | --- | --- |
| 根测试与全部非 Antibody 任务 | 1778 通过、128 跳过 | final-python-recheck.xml |
| Antibody | 212 通过 | final-antibody.xml |
| 固定 AlphaBench 上游（额外启用原生算法资产） | 345 通过、9 跳过 | final-alphabench-upstream.xml |
| 可用 sidecar：Host MCP 和隔离 policy 执行 | 16 通过、1 项因缺 guest 未执行 | final-sidecar-available.xml |
| Pi 构建／测试 | 45 通过 | final-pi.log |
| TypeScript 未使用符号检查 | 通过 | final-typescript-unused.log |
| 任务注册 | 0 错误 | task-registration.json |
| 配置与行为对照 | 356 组配置、23 个 campaign 及完成后恢复一致 | behavior-*-final/behavior.json、reasyn-*/behavior.json |

这些测试集合有重叠，不将表中数字相加。合成运行核对了预算、观察、选择、报告与恢复；
不是官方得分复现。配置对照包含默认 2+2、独立 session、4fork、compiled policy、帮助和非法参数。
AlphaBench 的两个对照 campaign 在两版代码中使用同一组固定 UUID 输入，
以保证运行身份及其派生回执可逐字段比较；未忽略回执键或科学字段。
其他归一化仅覆盖记录中列出的时间戳、耗时和运行路径。

环境处理与剩余限制：

- 首轮全量测试暴露了 NucleoBench 中仍引用旧评估方法的故障注入测试，已更新到评估执行模块并通过。
- 归档不含 Git 元数据，pilot 测试显式使用本次归档 SHA-256 作为来源身份；没有削弱产品身份检查。
- 通用环境的新版 BoTorch 与 Antibody 不匹配，基线也复现。单独测试时使用任务声明的
  BoTorch 0.5.0／GPyTorch 1.5.0；212 项通过。这不是完整的 Python 3.9 锁定生产环境验收。
- 已找到固定 AlphaBench 源码并启用相应测试，补齐原先因源码缺失跳过的 101 项。
- 本次 sidecar 使用新 dist 和 policy_runner 构建；其 package.json、package-lock.json 与基镜像逐字节一致。
  新镜像为 sha256:1dd015066e37330bb051e87089a88f5e3cf23ca70c57b4ff872c36eb0819f3ad，
  标签 ldm-pi-harness:architecture-convergence-20261021。只用于隔离验证；镜像内 dist 已与通过测试的构建逐文件核对。
- 完整 Pi session 初始化与 guest 私有文件隔离未验证：匹配的 guest 资产缺失。
  独立重建在获取 Gondolin sandbox helpers 时受网络下载失败阻塞，证据为 guest-build.log。
  容器 MCP 和 policy 执行可独立验证，已通过；未把它们视作 guest 验收。
- AtomWorld／ReaSyn 官方资产、真实模型及部分科学后端环境的剩余跳过原因逐项保存在 verification.json。
  没有为通过测试添加产品 fallback、放宽预算或绕过校验。

| 阶段 | 已提交实现 |
| --- | --- |
| P1 AtomWorld | 8616344c、892d9efc |
| P2 Iron Mind／SynthonBench／AlphaBench | f8f6cdab、a77dc5da、3d3b1a22 |
| P3 状态、评估和调用方收尾 | 48833c37、8c846fae、8a7975cc |
| P4 Python 协议、Pi 提交和 fork 存储 | 08d97a87、78d2a25d、a9c3ab2e |
| P5 文档、测试归位和验收 | 与本计划最终状态一并提交 |

NucleoBench 和 ReaSyn 的既有装配、完成后恢复、候选测量及投影归因路径已在全量测试中核验。
现有共享协调器足够表达任务差异；没有增加任务基类、插件注册层、兼容 wrapper 或新依赖。
