# 完整 LDM Harness：H20 三轮 HER2 验收

2026-11-18，主代理负责规划、源码审阅、H20 执行和验收；三个 `gpt-6.1-terra`、`max` 从代理分别实现多研究角色、动态策略及独立审计。代码仅同步到 H20 抗体 PoC 工作树，未同步到 Claude/us-la 工作区，未接真实设备或湿实验。

## 正式结果

真实三轮运行提交为 **`4694e362`**，配置为 `config/antibody_rfd3/her2_ldm_research.yaml`：`ldm_harness`、HER2 `1N8Z_C`、三代研究、三次评估、并发 1、候选池 6、GPU 2。模型为 `deepseek-flash`，思考强度 `low`，已核对所有实际 Harness manifest。

生成使用真实 RFD3/ProteinMPNN，评分使用真实 Absolut，没有 mock。

| 研究代 | 评分序列 | Absolut energy | 策略动作 |
| --- | --- | ---: | --- |
| 1 | TPGPPGSPGVI | -89.05 | replace |
| 2 | TSGGPGSPGWV | -87.26 | keep |
| 3 | SNDASGSNRFD | -80.91 | keep |

能量越低越好，最佳仍是第一条 **-89.05**。本次运行验证闭环工作，未证明优化性能提升，更不能据此推断真实实验亲和力或疗效。

增强审计 `passed=true`、`problems=[]`：

- 三代全部 ready，每代都关联一次实际评估决策；三个实验、三个 Absolut attempt、三条成功观测，无失败或重复评估。
- explorer、refiner、evidence_reviewer 各完成三个有效回合，合计九次研究提交；另有三个独立 policy_architect 回合。
- 三个研究角色实际读取了各自注册的技能；显式历史查询分别为 5、5、10 次。评审角色读取了冻结的真实候选目录。
- 九个真实生成任务、五个新骨架、80 条申请序列，所有角色及共享预算均符合限制。
- 三次策略应用的冻结观测数和反馈数分别为 0/0、1/1、2/2。原始能量、负能量 utility、220 维 one-hot 和全部关联标识均一致。
- 第三次决策实际应用了自定义先验，GP 已拟合；三次决策均可由冻结数据重放，无策略降级。

## 实际研究和算法表现

首代只有 explorer 生成共享初始化，refiner 和 reviewer 审读并提交同一首候选。后两代两个生成角色均结合历史进行新骨架探索和已有骨架重采样；评审角色区分观测和未测候选，但两代均保留全部六个候选，未证明筛选收益。

策略角色生成逐位 one-hot 收缩先验并在后两代保留，alpha、eta 三代都为 1。第三代 11 个查询候选的标准化先验在 -0.4 至 +0.2，历史先验为 +0.8/-0.8，没有触及 ±3 的边界。

独立审阅比较同一历史水位、同一六候选和相同权重下保存的 baseline 与 active 预测：先验改变了三个候选的预测均值，概率分布总变差约 0.0191。按记录的同一随机数，两者都选择 SNDASGSNRFD。因此可以确认先验影响计算，不能声称它改变了本次最终选点；未选候选未做反事实评分。

模型叙述仍需审查：出现过把五处序列差异说成三处、将仅由 explorer 生成的初始化误写为两角色共同生成等错误。生成的策略代码存在注释与实现不完全一致（声明但未使用的 `MAX_PRIOR=1.5`，以及错误声称标准化常量固定为 0/1）；实际数值使用主机提供的 location/scale，本次未触发边界。两条高度相关的序列不足以独立验证各位点贡献。Absolut 在这里只接收 H3 序列，生成几何和机理解释并未得到实际测量验证。

## 验收期间修复

第一次完整试运行 `2e721b96` 暴露出研究回合与实际策略应用的上下文不一致：研究时能看到观测进度和预测反馈，实际应用时没有。旧版审计未覆盖这一点；补充后对旧运行准确报告三个应用都缺少该上下文。

`4694e362` 在同一锁内冻结应用水位、成功观测和按 decision ID 对应的预测，按该快照构造进度及反馈，应用版本升为 2。修复后重新运行上述正式三轮，三个应用均通过新增检查。第一次运行保留在独立目录，不覆盖、不混入正式三轮结果。

正式运行后，`42a9abc9` 消除了 `required_candidates` 容易被理解成必须交满的提示，统一采用与校验器一致的 `candidate_count.minimum/maximum`，并升级研究源和 panel 版本。该提示修正不属于 `4694e362` 的真实轨迹，随后以 H20 角色/恢复定向测试验证，没有再启动额外真实评分。`c3acb756` 修正新测试的有效候选数量，并删除仍依赖同步 `measured_history` 的过时脚本；没有添加兼容层。

## 测试边界

所有测试均在 H20 执行，本机只做静态审阅和 Git 检查。

| 阶段 | 结果 | H20 日志（相对 `/Aidata/shared/ldm/antibody-poc/logs/`） |
| --- | --- | --- |
| 初始 PoC/策略/控制器/实验服务回归 | 241 passed | `full-ldm-pytest-6de83f53.log` |
| GP、Harness、guest 和 MCP 回归 | 49 passed | `full-ldm-harness-tests-6de83f53.log` |
| 新增策略生命周期测试 | 5 passed | `full-ldm-policy-lifecycle.log` |
| 上下文修复后相关回归（与前述有重叠） | 137 passed | `full-ldm-context-tests.log` |
| 数量提示修复后的角色/恢复测试 | 19 passed | `full-ldm-panel-count-tests-final.log` |

受控测试覆盖非空 pending 的真实选择器条件化、按决策匹配的反馈、两处故障注入后的恢复去重，以及研究期间结果返回后应用权重随冻结观测变化。使用的是模拟实验服务和受控策略，不能称为真实设备恢复实验。正式三轮并发为 1，未触发非空 pending 选点、CAS-stale 拒绝或崩溃；审计明确记录 `not_exercised`。

研究 guest 使用 NumPy 1.26.4，构建和实际 VM smoke 均通过；镜像引用 `ldm/antibody-rfd3-research:2a64960feb29`，build ID 为 `b793c47c-8fc1-5eb8-85db-68d35d9f2023`。下载默认 PyPI 较慢，取消了本次构建后改用清华镜像完成，保留两份日志。sidecar 使用既有 `ldm-copper-pi:antibody-async-20261118`，准确镜像 ID 记录于 launch.json。

## 证据与复现

正式运行根目录：

`/Aidata/shared/ldm/antibody-poc/runs/her2-full-ldm-3r-4694e362/`

- `launch.json`：源码提交、配置、模型、镜像 ID、资产清单及 SHA-256。
- `audit-full.json`：独立只读审计结果。
- `campaign/controller.sqlite3`、`experiment-service.sqlite3`：HTTP 两端记录。
- `campaign/harness/`：三个研究角色的会话、技能读取、候选快照及提交。
- `campaign/policy_harness/`、`controller.sqlite3.policy/`：策略回合、不可变代码、应用输入和先验表。
- `campaign/design_jobs/`、`absolut-attempts/`：实际生成和评分证据。

首次试运行根目录为 `her2-full-ldm-3r-2e721b96/`；其中 `audit-context-review.json` 保留增强审计检出缺失上下文的负对照。

运行与审计入口见 [架构和复现说明](poc-full-ldm-harness.md)。评分资产使用独立目录并按既有清单校验；API 凭据仅在 H20 加载，未写入代码或报告。结束时两个运行均无遗留任务进程，GPU 2 已释放；其他工作区和 GPU 任务未被停止。
