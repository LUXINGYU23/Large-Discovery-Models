# 铜矿任务注册资格适用性审计

## 要求来源

spec §10、§12、§15 要求诚实记录工程 profile、注册 gate 和证据，并明确不把方法有效性或官方 benchmark 认证作为单次工程闭环的结论。仓库 `skills/register-ldm-task/references/qualification.md` 则对 `seed_evaluated` 及后续资格要求主来源 benchmark URL、不可变 commit、task path 和适用的官方预算。

## 当前适用结果

| gate | 结论 | 直接证据或限制 |
| --- | --- | --- |
| registered | 已通过 | task.json、experiment.json、浅层 procedure；共享注册器发现任务 |
| mock_verified | 已通过 | 同一提交 parser、共享 Campaign、假 sidecar；原始 Harness 轨迹明确不转换为 ldm-2.0 IR |
| contract_verified | 已通过 | 科学定义、只读接口、源代码封存、独立评分及恢复测试；当前 H20 回归 228 项 |
| seed_evaluated | 不升级 | 已有真实弱基线及完整算法评估，但铜矿输入来自用户提供 ZIP，不能把所参考的上游 commit 当作这一任务的官方 benchmark 版本 |
| tiny_campaign_verified | 不升级注册资格 | 旧版与当前版本均有真实 P5 tiny。即使工程验证通过，也不绕过前置来源 gate |
| campaign_qualified | 不升级 | 工程 profile 是 task-defined；无铜矿官方 benchmark 预算认证，不运行 require-qualified 来制造资格结论 |

`resources/qualification_evidence.json` 保持 `contract_verified`，`experiment.json` 保持 `draft`。这准确描述注册规范下的资格，不表示已有真实运行是 mock，也不表示要额外索要一个不存在的官方 benchmark 才能完成本 spec 的工程交付。

## 工程完成仍须证明

当前版本的单 agent 两个 3D case 已各完成三轮；每组一个持久会话、8 个接口 job、4 个 solver、4 个 scorer，完成态恢复无新增请求。当前版本已完成 16 份历史源码复评；新协议 P5 tiny 已通过；还需三轮的原生多分支记录，并对 spec 每项边界逐项核对。

相关证据：

- [注册 gate 台账](../tasks/copper_inversion/resources/qualification_evidence.json)
- [来源复核](evidence/copper-inversion/reference-check.json)
- [资源计量及 228 项回归](copper-inversion-resource-accounting.md)
- [当前单 agent 三轮](copper-inversion-flash-low-rerun.md)
- [完整验收台账](copper-inversion-acceptance.md)

不存在多 seed、SFT 训练或 surrogate 效果优于基线的新增完成条件。未来若要提出这些科学结论，需另立独立实验和证据。
