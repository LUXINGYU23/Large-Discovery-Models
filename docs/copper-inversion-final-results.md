# 铜矿反演 spec 最终工程验收

## 完成范围

完成 P0–P5：科学内核、固定只读接口、封存算法执行、独立评分、单 agent RSI、两种 3D case、恢复与资源计量，以及共享 Coverage 2+2 / GP 四选一。原 Monte Carlo/controller 流程不在新任务运行路径中。

这是工程闭环验收；注册保持 `contract_verified`，实验保持 `draft`。不宣称官方 benchmark 资格、真实矿体恢复精度或 surrogate 优势。逐项依据见 [最终审计](copper-inversion-final-audit.md)。

## 完整 P5 真实运行

H20 `/Aidata/shared/ldm/copper-inversion-20261023/runs/p5-final-20261023/full`，锁定 `ldm_drillholes_3d` profile，官方 DeepSeek `deepseek-flash` / low。基线＋三轮，四分支各提交一份代码，全池通过静态检查和只读接口后，复用共享 RBF GP/LCB 仅选一份正式评分。

| 正式轮次 | loss_total |
| --- | ---: |
| 基线 | 405.0795810831 |
| 第一轮 | 6.5217329551 |
| 第二轮 | 6.5195782652 |
| 第三轮 | **6.5190666600** |

总墙钟 **1552.12 秒（25.87 分钟）**；全部 **182** 条原生请求为官方 Flash/low。

- 3 个主研究 turn＋12 个分支 turn；12 份提案，连同基线 13 个合法候选。
- 26 个接口 job、4 个正式 solver、4 个独立 scorer；未选中候选没有伪造正式分数。
- 三轮每轮四个 acquisition 分数逐一核对 `-mean_loss + std_loss`，实际选中项与最大分数一致，无 fallback。
- 每轮原生源码快照均先于新接口 job 派发；重复源码/语法拒绝及部分屏障恢复另有定向故障测试。
- 第三轮两个旧节点从 history cursor 1 补到 3，收到两条缺失正式测量；内容与权威历史一致。
- 树中已有测量的 credit 与对应 `-loss_total` 一致；末轮留在 checkpoint，未追加预算外总结 turn。
- 完成态恢复的所有计数不变，新增 provider 请求 **0**。

该结果略差于本次单 agent 钻孔的 6.516471；P5 使用更多研究调用，不能据此声称代理模型带来效率或精度优势。

[完整 P5 证据](../tasks/copper_inversion/resources/evidence/research-tree/final-full.json)包含配置、预算、三个 panel、选点预测、正式结果、树归因、快照及归档摘要。完整归档：`/mnt/cephfs/xingyu/ldm/archive/copper-inversion/p5-final-full-20261023.tar.gz`，SHA256 `603e74e06b33aa9dca6784d533784fa9c572faea7a15f3d4ee0799b244af6809`。

## 最终 runner 与验证边界

三轮启动后，逐项审计发现旧 runner 缺少 spec 要求的输入前后核对。修复在独立目录执行，未改变活动 campaign。新增输入指纹及 masked 输出拒绝，输入完整性回归 93 项通过，补齐手写 UBC 和源码入口用例后的最终完整任务回归 **98 项通过**；此前涉及共享模块的远程回归 **228 项通过**。

[源码差异证据](../tasks/copper_inversion/resources/evidence/runtime/final-runtime-diff.json)确认两目录运行代码仅 `core/runner.py` 有上述差异，科学目标、配置、共享引擎及 Harness 不变。最终 runner 随后真实复评：

- 历史 2D 四份；
- 历史 P5 四份；
- 历史两个 3D 单 agent 共八份；
- 最新两个 3D 单 agent 共八份；
- 本次 P5 实际评分的四份。

**28/28** 均重新通过只读接口、solver 和独立 scorer，loss 差值全部为 0；可信工作负载峰值均存在，重复读取结果不增加执行。最终源码身份为 `a71e2cb9f34bb1ce2561d29e23cc3e387a994f3891cfcac047c3957d4cd9ce5d`，与交付工作树静态摘要一致。

原 P5 轨迹保留启动时身份；最终执行器的结果在 [28 份复评证据](../tasks/copper_inversion/resources/evidence/runtime/final-input-replay.json) 和 [日志](../tasks/copper_inversion/resources/evidence/runtime/final-input-replay.log)，没有将旧模型调用改标为新实现。

## 交付与限制

- [单 agent 两个 3D case 对比](copper-inversion-flash-low-rerun.md)
- [输入完整性与 93 项测试](copper-inversion-input-integrity.md)
- [资源监督与隔离验证](copper-inversion-resource-accounting.md)
- [注册、mock 及幂等恢复证据](../tasks/copper_inversion/resources/evidence/runtime/final-registration.json)
- [资格适用性](copper-inversion-qualification-scope.md)

运行仍需要远程 Docker/KVM、具备 memory.peak 的内核、锁定镜像/guest 缓存和受保护 provider key。本机不执行实验。正式地质数据公开，分数衡量合成观测拟合；没有引入隐藏测试集、多 seed 泛化结论或 SFT 训练。

最终回归：[98 项机器证据](../tasks/copper_inversion/resources/evidence/runtime/final-acceptance.json)、[日志](../tasks/copper_inversion/resources/evidence/runtime/final-acceptance.log)。28 份复评归档为 `/mnt/cephfs/xingyu/ldm/archive/copper-inversion/final-input-replay-20261023.tar.gz`，SHA256 `184a75e0da58cba08e0a90894ba635d202707b8980c49012a43dcf7ef7c4b89d`。

最终源码包（含 bootstrap、文档 gate 证据、任务资产及 Pi 源码）：`/mnt/cephfs/xingyu/ldm/archive/copper-inversion/final-source-20261023.tar.gz`。完整身份及镜像依赖见 [交付清单](../tasks/copper_inversion/resources/evidence/runtime/final-delivery.json)。环境、缓存和 provider key 不在源码包中。
