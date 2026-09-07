# AtomWorld / ReaSyn 接入与调试记录

本次新增 `tasks/atomworld/`、`tasks/reasyn/` 两个 Science Benchmark，使用与
SynthonBench、Iron Mind 相同的 manifest 注册入口及 LDM 共享 campaign 生命周期。
原始代码和论文目录保留。两个实验合同保持 `draft`：接口测试、评分器校验和真实
LDM benchmark 分数是不同的证据，不能相互替代。

## 适配范围

| 任务 | LDM 的操作 | 评分与公平比较 |
| --- | --- | --- |
| AtomWorld | 根据 input CIF 和操作题目产生 CIF；多轮自查，或调用结构化几何操作工具 | 直接调用原版 evaluator；分别报告第一次和最后一次提交的正确率。目标 CIF、correct、RMSD 均不反馈给下一轮模型 |
| ReaSyn reconstruction | LDM 提议投影目标，冻结的 ReaSyn AR/EB 模型产生可执行合成路线，再评价末端产物 | 保留重建率、Morgan 相似度及路线产物/BB 多样性；失败样本仍计入完整分母。与固定原目标的 ReaSyn baseline 比较时匹配投影预算 |
| ReaSyn TDC | LDM 提议目标 → ReaSyn 生成可合成候选 → 共享 GP-UCB 选择 → TDC oracle → 历史反馈 | 按唯一 canonical SMILES 计 oracle 调用，最多 10,000；单独计 LLM 请求、ReaSyn 投影和 oracle。报告 top-10 AUC |

ReaSyn 的“主 evaluation”存在指代空间。这里将论文首先报告的分子重建作为主模式，
并接入适合 LDM 搜索的 TDC 模式。JNK3 hit expansion、sEH docking 和模型训练不属于
本次主评测入口。

ReaSyn 三个数据集各提供四轮 LDM、相同投影预算的固定目标重启基线、原版单次投影
基线。四轮内部 cycle 上限分别为 Enamine 4×3、ChEMBL 4×6、ZINC250k 4×4；
原版为单次 12/24/16 个连续 cycle。重启方式不同，不能只凭总 cycle 相等声称计算
过程相同。采样、投影和 oracle 的预算都单独记录。

详细参数、安装和启动说明见 [AtomWorld](../tasks/atomworld/README.md) 与
[ReaSyn](../tasks/reasyn/README.md)。服务无关的两个任务检查可从仓库根运行：

```bash
python scripts/validate_tasks.py --task atomworld
python scripts/validate_tasks.py --task reasyn
python scripts/check_task_dependencies.py config/suites/science_benchmarks_mock.yaml --no-optional
python scripts/run_ldm_tts.py config/suites/science_benchmarks_mock.yaml --dry-run
python scripts/run_ldm_tts.py config/suites/science_benchmarks_mock.yaml
```

## 原论文与发布代码的差异

- AtomWorld 论文 AtomMotor-2K 为 10 类动作 × 250，共 2,500 题。toolkit 有更多动作，
  本地 CSV 每类约 1,000 题，没有显式的发布测试集索引。适配器保存来源和抽样清单，
  本地子集结果不能自动等同于论文完整测试集结果。
- AtomWorld 官方 evaluator 用 `StructureMatcher(primitive_cell=False, stol=0.5)`；
  `move_all_action` 使用专门的周期坐标匹配。其 RMSD/max_dist 实际经过归一化，
  上游 docstring 的 Å 标注不应原样用于报告。
- ReaSyn TDC 正文写 15 项，发布的表格、CLI 和参数实际一致为 13 项。本次遵循已发布
  的 13 项。原版 oracle 的 `>` 截止判断允许预算多用一次，适配采用严格的 `>=`。
- ReaSyn 原 `eval_recon.py` 固定分母为 1,000。适配支持明确的完整请求目标清单，
  子集使用真实分母，失败目标贡献零。零步 BB 路线只有在产物属于冻结 stock 时有效。
- 本地文件来自源码归档，没有 `.git`。来源以本地文件 SHA-256 记录，没有编造上游
  commit；正式 qualification 的 Git 跟踪证据门槛尚未满足。

依据为本地论文及实现：
[ReaSyn 实验章节](../../reasyn-arxiv/5_experiments.tex)、
[TDC 结果表](../../reasyn-arxiv/tables/pmo.tex)、
[TDC 实现](../../ReaSyn-reasyn_v2/scripts/optimize_tdc.py)、
[重建实现](../../ReaSyn-reasyn_v2/scripts/eval_recon.py)、
[AtomWorld 论文](../../atomworld-arxiv/icml_atomworld_rebuttal.tex)、
[AtomWorld evaluator](../../atomworld-main/src/atomworld/evaluate.py)。

## 缺失实现的并列复现

AtomWorld 论文附录描述了 RAG/API 文档检索、Python 生成和执行的工具方案，本地没有
完整 runner。本次在并列目录 `atomworld-agentic-reproduction/` 实现适合 LDM 调用的
结构化几何执行组件。它只读取输入 CIF 和模型给出的操作计划，不读取答案，也不执行
任意 Python。该组件是工具执行部分的适配复现，并非原论文 RAG 系统的逐项复现。

ReaSyn reconstruction/TDC 已有采样器和评分代码，直接复用并适配其调用边界。
未下载的权重和 stock 数据属于缺失资源，不用伪造路线替代。

## SSH 调试情况（2026-09-06）

目标：`ssh zsgpu@111.2.199.31 -p 5320`。

首次非交互连接因未知 host key 返回 `Host key verification failed.`；接受首次
ED25519 host key 后，服务器返回：

```text
zsgpu@111.2.199.31: Permission denied (publickey).
```

本机默认 RSA key 文件存在，但服务器没有接受；SSH agent 显示 `The agent has no
identities.`。因此网络和 SSH 服务可达，阻碍在身份认证。没有执行任何远程 shell
命令、上传文件或启动 GPU 作业。

恢复条件是在本机配置该账号允许的 SSH key，例如：

```bash
ssh -i /path/to/authorized_key -o IdentitiesOnly=yes -p 5320 zsgpu@111.2.199.31
```

认证恢复后，先检查环境与可用 GPU：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 -p 5320 zsgpu@111.2.199.31 \
  'hostname; python3 --version; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader'
```

上传到独立工作目录，保留 LDM、两个 upstream 和工具复现目录的并列关系；不要覆盖
服务器已有 checkout。按各任务 README 安装独立环境、配置模型 endpoint，先运行 mock
和依赖检查，再运行 tiny 配置。模型地址、模型名、密钥只放环境变量。下载完整 run
目录，保留 `status.json`、`budget.json`、`events.jsonl`、checkpoint 和任务结果。

## 当前真实运行所缺资源

- 本机没有配置 `LLM_BASE_URL`、`LLM_MODEL_NAME`、`LLM_API_KEY`。
- ReaSyn 缺少 AR/EB checkpoint、`data/processed/comp_2048/{fpindex,matrix}.pkl`、
  115 reaction templates，以及 Enamine/ChEMBL 目标文件。ZINC250k 目标与扩展 BB
  文件在本地，但仅这些文件不足以运行神经投影。
- GPU 服务器认证失败，无法检查其是否已有上述模型或数据。

因此本次不能给出真实 LDM 分数，也不能宣称复现论文结果。具体通过的检查与诊断结果
列在下面，并保存在 [机器可读验证记录](validation/science_benchmarks_20260906.json)。

## 已执行的真实评分器和工具校验

| 校验 | 范围 | 结果 | 含义 |
| --- | --- | --- | --- |
| 原版 AtomWorld evaluator 正例 | 10 类动作，各取本地前 2 题，把标准输出重新提交给评分器 | 20 / 20 通过 | 确认数据可读取、原版评分器可执行 |
| 同一评分器格式反例 | 上述 20 题，提交无 CIF 标签的文本 | 20 / 20 被判错误 | 确认格式错误没有被当作成功 |
| 新增几何工具实际数据校验 | 10 类动作，各取本地首题；根据公开题目手工转写操作计划 | 10 / 10 通过原版评分器 | 确认执行组件覆盖 10 类动作；没有调用模型 |

这些是诊断校验，不是 LDM 解题正确率。特别是第三项只验证“正确计划能否被执行”，
不能用来推断模型能否生成正确计划。

## 接口与回归验证

- 最终联合测试：**359 passed，2 skipped**。覆盖共享 LDM、两个新任务和并列工具。
  跳过的是已有共享 Torch 测试和工具可选 JSON Schema 测试；本环境分别未安装
  `torch` 与 `jsonschema`，不影响已执行的真实 RDKit 和 AtomWorld 几何校验。
- AtomWorld 锁定独立环境：11 / 11 tests passed，无跳过。
- ReaSyn 锁定轻量环境：9 passed、3 个可选 RDKit 测试 skipped；这 3 项已在包含
  RDKit 的联合环境中通过。真实神经投影只验证了模拟依赖的接口，尚未加载官方权重。
- 两个任务注册及布局检查通过；18 份任务配置的 dry-run/profile 约束通过；生成的
  27 份重建配置和 39 份 TDC 配置均通过约束检查，没有启动真实作业。
- AtomWorld、ReaSyn reconstruction、ReaSyn TDC 三条 mock 流程均完成，共享
  campaign/status/budget/events/checkpoint/summary 及任务结果文件齐全。
- 测试覆盖了隐藏答案隔离、原版评分接口、工具错误修正、路线回放、TDC 重复去重与
  预算边界、AUC 原版公式一致性、Tanimoto GP 的相似分子预测、完整分母、短数据文件
  拒绝、模型格式错误处理、断点恢复和收集数据的元数据隔离。

源码归档没有 Git 元数据，已有 pilot 测试要求显式归档标记，因此联合测试使用了
`LDM_PILOT_EVALUATION_COMMIT=local-source-archive-test-only`。这是测试标记，
不是上游 commit，也没有据此提升 qualification。

原始记录位于 `runs/science_benchmarks_20260906/`。可携带的
[验证记录](validation/science_benchmarks_20260906.json) 保存了逐题诊断、完整预算计数、
SSH 错误、环境版本、文件与原始产物 SHA-256；各任务也有独立的资源验证记录。
