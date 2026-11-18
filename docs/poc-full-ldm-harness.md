# PoC 多角色与动态策略

`ldm_harness` 使用异步控制器、独立 HTTP 实验服务和独立持久化的研究会话。纯 `harness`、`fixed`、`antbo` 仍用于各自的对照实验。新的 LDM 选择器使用 UCB 与动态先验，不再是旧 PoC 的固定 categorical EI；比较结果时必须注明算法变化。

## 一代研究

1. `explorer` 提出探索候选；`refiner` 根据历史与探索候选寻找改进方向。两者分别拥有生成预算，共同受本代总预算约束。
2. `evidence_reviewer` 阅读冻结的候选清单、来源和真实历史，提交审选结果、证据与假设。它没有生成额度；读到的候选不会被当成已测结果。
3. `policy_architect` 根据冻结的特征、观测和此前决策反馈，编写或保留优化策略。策略通过独立 Docker 执行器校验，在无网络、只读容器和资源限制下运行。
4. 控制器合并候选及其 lineage，应用策略、计算待评估点的分布，再通过 HTTP 提交给实验服务。评估结果进入持久化历史，供后续研究工具查询。

首代保留共同初始化：仅 explorer 生成，另外两个角色审读同一个首候选。这一代提供可比起点，并不宣称已经有三种独立的搜索策略。后续两名生成角色各拥有独立会话和候选来源。

研究技能分别覆盖生成工具、实验设计、证据审查和策略编译。历史查询支持 observations、pending、proposals、summary；提交沿用主机侧候选校验、历史水位和 CAS 机制。候选清单保留内容散列；重放使用原始回合的配额、历史窗口与消息，生成调用按角色与 call ID 去重。

## 数值与调度约定

- 原始 `absolut_energy` 越低越好；选择器使用 `energy_utility = -absolut_energy`，越高越好。两者同时保存，策略输入和预测反馈统一使用 utility。
- 11 残基序列使用固定氨基酸顺序的 220 维 one-hot 编码，不对氨基酸字符施加任意的整数距离。
- 策略可提供标准化先验均值及 LDM 的 alpha、eta 权重。GP 在先验残差上拟合，并将查询先验加回；不足两条观测时不会宣称已使用拟合后的残差 GP。
- pending 使用 kriging believer；选择分布结合来源权重与标准化 UCB。策略轮次、不可变产物、应用和实际 decision 有独立关联记录。
- 研究代数与评估预算分别配置。任务调度为每个研究代预留至少一个评估额度；三代、三次评估时，每代等上一代结果后继续。更大评估预算仍可流式并发，HTTP 隔离与异步控制器保持有效。

## H20 复现

配置为 `config/antibody_rfd3/her2_ldm_research.yaml`：HER2 `1N8Z_C`、三代研究、三次真实评估、并发 1、候选池 6；每代最多两个骨架、32 条序列，调用采样数 16。生成器使用 RFD3/ProteinMPNN，评分使用实际 Absolut。模型从 H20 环境加载，思考强度为 `low`。

在干净且已提交的 H20 PoC 工作树中加载现有凭据，然后运行：

```sh
PYTHONPATH=. /Aidata/shared/ldm/antibody-poc/env/antibody_campaign/bin/python \
  tasks/antibody_rfd3/repro/h20/run_full_ldm.py \
  --output-dir /Aidata/shared/ldm/antibody-poc/runs/<新的运行目录> \
  --absolut-source /Aidata/shared/ldm/antibody-poc/runs/her2-research-3r-e484a217/absolut
```

启动器按已记录的资产清单复制独立评分目录、核对 SHA-256，记录提交、模型名、镜像 ID 和完整有效参数；凭据不写入报告。`audit_full_ldm.py` 对完成后的 campaign 做只读验收。

三次计算仅用于验证闭环运行，不能证明一般优化优势或真实实验亲和力。非空 pending、CAS 过期拒绝和崩溃恢复应由受控测试单独验证，不能从三轮顺序研究成功推断。这套代码、文档及结果只进入 PoC 工作区，不同步到 Claude 工作区。
