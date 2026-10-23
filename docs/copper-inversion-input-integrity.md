# 输入完整性验收

spec §4 要求只读数组之外，还要核对执行前后输入内容。最终审计发现旧 runner 仅设置只读标志，缺少后置核对；现已补齐。

`runner.input_digest` 对嵌套 problem 的元数据以及数组 shape、dtype、writeable 标志和 C 序内容摘要生成指纹。加载输入后、导入候选前记录指纹，solve 返回后再次核对；变化时拒绝输出。另在保存前显式拒绝 masked ndarray，避免序列化掩码丢失造成假通过。

该检查是正常接口行为的验证，不是对任意恶意 Python 的信任根。正式输入仍由只读挂载保护，控制器验证矿体，独立 scorer 不执行候选代码。

## 远程证据

独立 H20 目录 `/Aidata/shared/ldm/copper-final-audit-20261023`，全部任务回归 **93 passed in 95.53s**，启用真实 Docker 和 `COPPER_REQUIRE_PEAK=1`。新增覆盖：

- 修改只读标志后修改原始 level-set；
- 修改观测 weight；
- 仅修改 mask 的只读标志；
- 返回 masked、Inf、全负、错误 shape、复数或列表；
- 候选在 import 阶段直接退出。

既有 2D/3D 正常案例、独立评分、OOM/超时、断连回收及恢复路径也同时通过。

[机器证据](../tasks/copper_inversion/resources/evidence/runtime/input-integrity.json)记录源码身份、镜像和日志摘要；[完整测试日志](../tasks/copper_inversion/resources/evidence/runtime/input-integrity.log)。

P5 full 使用先前冻结源码继续运行，没有将这次修复注入活动实验。最终算法复评单独记录最终 runner 身份与数值，不能把原模型轨迹重新标记成新实现。
