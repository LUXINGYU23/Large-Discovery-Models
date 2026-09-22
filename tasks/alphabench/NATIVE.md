# 原生算法执行与恢复

固定源码为 AlphaBench `31bb94bbb7744177c51c9d011e07c31e3092b93e`。
`core/native_source.py` 校验五个 `searcher/algo` 文件的原始字节哈希，
把带插桩的副本写入运行的私有目录，再加载官方 `create_algo`。
补丁前后文件、加载器和调度代码的哈希写入 `native_source/manifest.json`；
同一运行恢复时不允许这些实现发生变化。源 checkout 不被修改。

当前组件已经执行真实的官方 CoT/ToT/EA Python 算法和原生生成器，并接入
模型请求、动态检查、整批 search 预算及私有 validation。matched 正式 workflow
已接入原生 cold 初始化和终局报告；source profiles 仍待完成。当前验证使用
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
并发数和评价适配器 hash 冻结后才能执行。source profile 的初始化规则仍须
独立完成，不能套用 matched 的规则后声称保留了原入口。

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
