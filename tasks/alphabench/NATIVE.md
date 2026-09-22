# 原生算法执行与恢复

固定源码为 AlphaBench `31bb94bbb7744177c51c9d011e07c31e3092b93e`。
`core/native_source.py` 校验五个 `searcher/algo` 文件的原始字节哈希，
把带插桩的副本写入运行的私有目录，再加载官方 `create_algo`。
补丁前后文件、加载器和调度代码的哈希写入 `native_source/manifest.json`；
同一运行恢复时不允许这些实现发生变化。源 checkout 不被修改。

当前组件已经执行真实的官方 CoT/ToT/EA Python 算法，并通过固定回调验证；
尚未接入正式 workflow。原生 generator、完整 batch 预算桥接、source profiles、
私有 validation 和终局报告接线仍须完成。正式入口继续拒绝未完成的方法，
本组件的通过不代表 W10 或完整 T3 已通过。

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

## 已验证及边界

在指定服务器运行 `python -m pytest tasks/alphabench/tests/test_native.py -q`。
固定源码对照覆盖三算法的部分生成、重复、阈值与多 worker；完整重放逐项比较
名称、链/树/种群、顺序和时间字段，禁止再次调用已完成的回调。

真正终止子进程的检查分别落在生成提交、评价返回和状态更新之后，覆盖三个
算法。此组检查选择没有未知在途请求的提交边界；恢复使用真实的共享 ledger
和 receipt，所有物理调用计数保持一次。未知结果的暂停由独立 receipt/Host
测试检查，不能把这组恢复结果解释为外部服务的物理 exactly-once 保证。
