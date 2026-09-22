# 原生算法执行与恢复

固定源码为 AlphaBench `31bb94bbb7744177c51c9d011e07c31e3092b93e`。
`core/native_source.py` 校验五个 `searcher/algo` 文件的原始字节哈希，
把带插桩的副本写入运行的私有目录，再加载官方 `create_algo`。
补丁前后文件、加载器和调度代码的哈希写入 `native_source/manifest.json`；
同一运行恢复时不允许这些实现发生变化。源 checkout 不被修改。

当前组件已经执行真实的官方 CoT/ToT/EA Python 算法和原生生成器，并接入
模型请求、动态检查、整批 search 预算及私有 validation。matched 正式 workflow
已接入原生 cold 初始化和终局报告；Qlib searcher source workflow 也已接线。当前验证使用
synthetic oracle，不代表真实市场、W10 或完整 T3 已通过。

## 执行合同

只有一个共享 `CampaignRuntime` 和 Host 写入者；原生路径不创建 LDMEngine。
算法在后台执行，嵌套线程池保持并行。稳定的 worker 身份由父 worker、
线程池创建位置和提交位置组成。模型与 oracle 回调的开始、输入 digest、
返回值、结果交付顺序、UUID、时钟值、锁顺序与状态快照进入现有
`events.jsonl` 的 `native_boundary` 事件，不另建恢复数据库。

完成的叶子回调直接重放原返回值。未完成回调仍须通过原有持久 receipt
执行或对账；日志没有返回值不等于请求未发出。`dispatch_intent` 的未知
结果继续暂停，不自动重发。调用方必须冻结参数，并为所有实际请求预留预算。
会产生嵌套原生事件的操作不能作为可跳过的叶子回调；生成器内部事件接线
需要遵守这一边界。

重放按已提交的全局边界顺序交付，同时重新执行原算法控制流。输入、输出
完整性或调度不一致时暂停。等待没有进展达 60 秒也暂停，避免等待已不存在
的 worker。预算停止、暂停与取消通过 `BaseException` 控制信号穿过上游
`except Exception`，所有后续调度边界停止新增工作；已经发起的请求仍由
receipt 负责核对。

## 最小补丁与行为差异

- 替换算法模块的线程池、锁、UUID 和时钟导入，使它们使用有日志的同等操作。
- 给 ToT 的共享 best 比较与赋值加上其已有互斥锁，消除并行丢失更新；比较规则不变。
- 在 CoT 链、ToT 已评价节点、EA 初始及更新后种群处记录真实状态。EA 的父代选择、
  mutation/crossover、预先发起下一轮生成及更新顺序仍由固定源码执行。

CoT/ToT 依然按 IC 选初始 worker seed；CoT 保留完整链与原接受规则，ToT
保留剪枝和无合格候选时的原 survivor 规则。EA 仍在本轮评价完成前使用
更新前种群发起下一轮生成。上游输出的 elapsed 字段记录其墙钟观察；正式
请求成本必须依据 receipt，不能把跨进程暂停时间作为模型调用耗时。

## 原生生成器

`core/native_generator.py` 验证官方生成器与提示文件的完整源码哈希，只加载
固定提示和四个实际需要的函数；不导入 Qlib、FFO 客户端或上游模型 SDK。
实际请求复用 task 的 `Generator.request`，实际检查复用 `OracleGateway`。
原生最多五次尝试、累积去重、剩余数量提示、达到 N 提前停止，以及耗尽后的
`floor(N/2)` 部分成功和 `N=1` 特例均在原函数中执行。

必要修复包括空 `generated=[]` 的除零，以及单对象模式中非法 name/expression
和非文本 reason 引起的崩溃。前者和非法字段进入原修复循环；非文本 reason
按数组模式的规则丢弃。正常输入的提示、接受、抽样与 quality 字段通过原函数
对照。上游会在正好凑满 N 时也抽样重排，不能省略。

生成器作为一个叶子回调。其内部 UUID、已接受表达式的集合遍历顺序、抽样前后
RNG 状态与下标写入同一 Campaign 的 `native_generation_choice` 事件。恢复未完成
生成时只重放这些选择及原请求 receipt；整个叶子已完成时直接重放输出。RNG
从最后持久化状态继续，已完成选择不倒退全局随机状态。

每份 generation 记录保留原始 occurrence、归一化/访问/检查标记、检查 receipt、
未访问尾部、上游完整返回值及独立计数。基础设施失败与预算停止穿过上游捕获器，
不会变成公式错误而触发更多修复。成功 action 仍通过原有不可变 journal 和成对
IR/SFT 发布；原生不成功但含少量因子的返回值照常交回原算法，不伪造成功或暂停。

请求使用用户指定的 `deepseek-flash`、Responses、`reasoning.effort=max`，并把
上游 JSON 模式映射为 `text.format={"type":"json_object"}`。`temperature` 纳入
冻结协议并传给 provider；[DeepSeek 官方说明](https://api-docs.deepseek.com/zh-cn/guides/responses_api/)
指出思考模式下它不生效。因此 generation 同时记录请求值、不可用的有效值和
`ignored_by_provider_in_thinking_mode`，不能宣称复现了旧模型的温度条件。

cold 初始化同样执行固定 `searcher/pipeline.py` 的 `cold_start_generate`，移除
模块导入副作用并将 30 替换为冻结的 `cold_seed_count`。原提示、五次修复和种子
提取规则保留；即使原生 success 为 false，它返回的部分因素仍按原入口进入
初始化评价。初始化为空时 EA 保留原行为，CoT/ToT 明确暂停为 `paused_no_valid_seeds`。

EA 的下一代预生成可能与本代评价重叠。终局前先核对所有已发起的 model/check
receipt。模型结果未知时暂停；已收到但尚未返回给算法的原始输出记作
`interrupted` generation，`native_result=null`，独立报告中断数量，不能归为普通
生成失败。其全部原始 occurrence 参与独立质量审计，已完成检查可复用。
完成 generation 的 accepted action 由相同发布函数恢复，保留成对 IR/SFT。

## matched workflow

通过 `--protocol-file` 明确冻结 `native_parameters`，正式入口在初始化计费前
检查完整参数集和固定源码。所有方法需 `oracle_workers`、`enable_reason`、
`accept_threshold`；CoT 另需 `workers`，ToT 另需 `workers/N/top_k`，EA 另需
`N/mutation_rate/crossover_rate/pool_size/seeds_top_k`。模型、温度和 rounds 使用
同一个协议字段，拒绝无效或多余参数。原算法固定优化 RankIC，入口拒绝其他
search objective，保留以 IC 选择 worker seed 的原行为。

搜索只打开一个共享 runtime，不启动 LDMEngine。`result.json.execution.kind`
区分 `native_reference` 和 `shared_engine`；原生输出包含原返回值、实际提交的
状态和 `native_final_pool`。预算中断时原返回值为空，池来自停止前已提交的
链/节点/种群。顶层 `final_pool` 仍是 matched 的完整成功观测集合，用于私有
validation 选取和独立多样性评价，两者不会互相替换。

`private/stages` 冻结原生完成结果。finalization 中断后从原结果及 receipt 继续，
不重新生成候选，也不启动新搜索；已完成的运行只读重放。模型与检查结果未知
时不会生成成功终局报告。

## 已验证及边界

`core/native_evaluation.py` 的三个评价回调共享同一个实现。整个 batch 的 E 和
各请求的 worker permits 由共享账本一次写入；任一额度不足不修改任何计数，也
不派发部分 batch。各请求仍使用自己的稳定 usage key，随后网关确认许可和
恢复时不会重扣。相同公式在不同回调或批次位置出现属于不同的新尝试。

CoT/ToT 的种子查询有明确的 seed 角色。在 matched 协议中，它们只读已验证的
共同初始化观测，不消耗新增 E；未知种子暂停。完整初始信息 digest、协议、
并发数和评价适配器 hash 冻结后才能执行。searcher 的 `use_cache=True` 种子查询
复用同合同初始化 receipt；此操作差异写入 source entry 清单。

成功 search 先完成 private validation，再持久化公开 observation；callback 只
返回 search 指标。失败 search 仍计费，记录真实失败，不伪造分数。轨迹按逻辑
回调提交顺序及 batch 原顺序排列，物理完成顺序不改变该横轴。实际 search jobs
在收到结果后即入账，即使之后的 validation 暂停也不会漏掉这些成本。

停止标记阻止新的模型、修复、check 和 search 请求；已完成 receipt 仍可只读
重放。所有 worker 退出后必须调用 `settle()` 核对已发起请求并完成必要的私有
验证。未知在途结果继续暂停，不能用先发生的预算停止掩盖。已预留但未发起的
候选单列 `reserved_but_unstarted`，不生成 observation，不计入实际评价轨迹；
预留计数不会伪装成物理派发数。Host 设置停止标记时不等待原生调度锁，避免
与正在等候 Host 回复的 worker 形成死锁。

在指定服务器运行 `python -m pytest tasks/alphabench/tests/test_native.py -q`。
固定源码对照覆盖三算法的部分生成、重复、阈值与多 worker；完整重放逐项比较
名称、链/树/种群、顺序和时间字段，禁止再次调用已完成的回调。

真正终止子进程的检查分别落在生成提交、评价返回和状态更新之后，覆盖三个
算法。此组检查选择没有未知在途请求的提交边界；恢复使用真实的共享 ledger
和 receipt，所有物理调用计数保持一次。未知结果的暂停由独立 receipt/Host
测试检查，不能把这组恢复结果解释为外部服务的物理 exactly-once 保证。

## source 入口解析与 example 接线

`core/source_profiles.py` 分别解析固定 `searcher/config.yaml`、三个
`searcher/configs/*_config.yaml` 和 `example/search/configs/search_csi300.yaml`。
它校验配置、入口、客户端、算法及种子资源哈希，输出原声明、实际生效值、
参数来源、用户模型覆盖和必要修复差异。此文件是来源解析结果，不能直接作为
`--protocol-file`；执行需另行冻结包含全部预算和数据/环境身份的 `T3Protocol`。

在服务器生成可重复的来源报告，不调用模型或 oracle：

```bash
tasks/alphabench/.venv/bin/python -m tasks.alphabench.core.source_profiles \
  --upstream-root /mnt/data1/Large-Discovery-Models/data/alphabench/AlphaBench \
  --config example/search/configs/search_csi300.yaml --method tot --market sp500 \
  --output /mnt/data1/Large-Discovery-Models/data/alphabench/manifests/source-profiles/benchmark_tot_sp500.json
```

同一输出路径拒绝不同内容覆盖。已在服务器解析七个入口/方法组合在 CSI300、
SP500 上的 14 份报告，索引见 `resources/evidence/source_profiles.json`。

两套入口有不能合并的行为：

- searcher CoT/ToT 按 IC 排序后只选择前 `workers` 个种子；专用配置为 4，
  默认 `searcher/config.yaml` 为 1。其 search/validation/test 源请求为包含端点的
  2016–2021、2021–2022、2022–2025，search 默认 `fast=True`。
- example 先评价全部 42 条 Alpha158，再将 kbar+price 的 13 条分别交给
  CoT/ToT、rolling 的 29 条交给 EA。4 是外层并发上限，全部组内种子都会运行。
  源客户端的实际默认请求是 2023–2024、`fast=False`；原入口没有 validation/test
  阶段。SP500 只传给了 baseline，其余便捷回调仍默认 CSI300。这些源事实不能
  用 searcher 或 matched 的默认值覆盖后声称原样复现。

`core/native_benchmark.py` 执行提取的原 `benchmark_main/run_batch`，修复过期构造
参数及已删除的 `verbose` 参数，给 ToT 注入它实际需要的按名称索引结果。
baseline 与搜索分别绑定现有回调；入口不导入 FFO/Qlib 或模型 SDK。返回原有
局部结果用于报告，不复制搜索循环。参数、源文件和补丁摘要与调度日志共同保存。
真实三算法已通过全组种子、并发、预算停止和完整重放测试；当前仍为 synthetic
oracle 验证，尚未登记 source profile 或真实市场资格。

### Qlib searcher workflow

选择 `profile=upstream_searcher_v1`、`method=alphabench_cot/tot/ea`，
`native_parameters` 必须只有 `source_config`，值为上述对应 searcher YAML 的
仓库相对路径。算法参数从已校验的原始配置与源码解析，不能再覆盖 N、worker、
mutation/crossover 等值。协议中的 rounds、temperature、初始化、过滤、标签、
最终因子数及持仓数须与来源一致；任何不一致在初始化付费前拒绝。
`--dry-run` 同样核对来源并输出完整 `source_entry`，显示实际算法参数和协议差异。

CoT/ToT 使用 30 条原生 cold 种子；EA 自动读取配置中的 125 条固定种子。
不接受其他种子文件、import pool 或 matched 的共享 seed bundle。模型按用户
要求覆盖为 `deepseek-flash`，原温度值保留但在 thinking 模式下无效。

原生轮数及分支不因 E 上限改写。整批请求无法容纳时返回 `paused_budget`，
不发布截短的 source 最终成绩；预算在开始前冻结，不能原地加额度后伪装同一
实验。matched 才允许以新评价预算为停止条件生成明确标注的对照报告。

searcher 对搜索评价失败的因子也执行 private validation，与原 ValEvalTracker
一致。最终集合只取算法实际返回的 `final_pool`，保留名称、顺序和重复成员；
validation 同分时保留源顺序。每个成员都必须有有限 validation 指标，否则
在 test 前返回 `paused_incomplete_validation`。固定源码可回退到 search 排名，
本 task 的完整 T3 合同禁止混排；该差异保存在 `source_entry_contract` 中。
信号多样性按成员位置形成配对，重复公式也参加统计；pair 同时保存公式身份
和左右位置。最终 test 仍为 50 个因子、50 只股票、drop 5、`fast=False`。

执行使用统一入口的 `--protocol-file`、`--upstream-root`、`--data-manifest` 和
`--oracle-url`；mock 仅省略真实数据与服务。数据资格 gate 保留。Assay source
的完整源合同及 example 的 validation/test 扩展仍未完成，入口明确拒绝它们。

## 过滤与质量报告

生成器和原生生成回调共享冻结的过滤合同。`qlib_code_filter_v1` 使用固定
FFO 的 `_check_single_column`，保留原始 NaN 比例大于 1% 时拒绝的行为；
Inf 与 NaN 分别记录。`assay_code_filter_v1` 调用固定 Assay 的无数据 lint，
使用 `lint_checks` 计数，拒绝项标为 `lint_rejected`。
`paper_filter_v1` 要求深度不超过 5、NaN/非有限值比例不超过 1%、动态检查
耗时不超过 30 秒。code filter 接受深度 6 不代表通过 paper filter。

最终质量审计覆盖所有原始 occurrence，复用相同合同的已有检查，缺失检查
使用独立 `quality_checks` 预算。报告分别标明静态、后端检查和 paper 覆盖率；
Assay lint 只产生 `assay_lint_success_rate`，不会伪造动态成功率或行情覆盖。
检查响应若类型、耗时、比例或结果边界不符会暂停，已完成物理作业仍记账，
恢复不重复派发。上述过滤实现不解除真实数据资格的 gate。

## 原生日期与标签边界

后端按冻结 profile 处理请求端点，保存请求区间、实际因子区间和标签读取终点：

| profile / 后端 | 请求终点 | 标签尾部 |
| --- | --- | --- |
| matched / Qlib | 不包含 | 删除最后 n 个信号日；标签查询只读到区间最后一个交易日 |
| matched / Assay | 不包含 | close 删除 n 日，open 删除 n+1 日 |
| source / Qlib | 包含 | 保留全部信号日，按原 FFO 标签读取之后 n 个交易日 |
| source / Assay | 包含 | 不追加未来行情；保留原面板末尾的 n 或 n+1 行未定义 IC |

Qlib 使用真实交易日历确定前向读取范围，日历不足时以 `paused_data_coverage`
暂停，不记成因子质量失败。已启动作业照常记账，恢复读取既有 receipt，不
自动重发。逐日另存每个
forward horizon 的有效样本数。两种后端的完整 portfolio 使用各自保留的信号
区间。Assay source 的尾部 IC 未定义不代表这些日期不参加 portfolio。

端点规则按固定上游逐项核验。searcher 三段日期与 matched 相同，但包含终点，
且没有 matched 的标签 purge。example 的搜索日期是 2023–2024；访问不存在的
validation/test 区间会明确报错，不能自动继承 matched 日期。

该实现已用两个真实后端的数值 fixture 和完整 portfolio 检查。Qlib searcher
已接入这些规则；Assay source 和 example workflow 仍保留独立的未完成 gate。
