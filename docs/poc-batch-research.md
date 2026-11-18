# HER2 批量研究：32 + 32 → 64 → 32

目标是 20 个研究批次、每批 32 个真实 Absolut 评分，共 640 次。沿用 DeepSeek、low 思考强度、H20 的 PoC 工作区，不向 Claude 工作区同步。

探索和改进角色各提交 32 条符合约束、同批唯一且尚未评估或在途的序列。第二个角色排除第一个角色本批已提交的序列。原始生成次数与最终提案数量分开：配置每批最多 512 条原始序列和 16 个骨架，两名生成角色各分一半，达到目标即可停止，不必用满。预算耗尽却不足时整批报错，不用重复项或短批凑数。

证据审查角色阅读完整的 64 个候选及来源，审查结果保留为意见，不作为第三路重复候选。策略角色研究先验和选择权重。控制器将 64 个唯一候选交给现有 pending-aware UCB 选择器，逐步条件化选出 32 个不同候选，记录为不可变计划，再通过独立 HTTP 实验服务按实际并发分发。

批次大小为 32，评分并发为 4，两者相互独立。上一批 32 个结果全部成功结算后，下一批才读取这些反馈继续研究。下一批使用自己的 64 个提案池；未选择的候选不直接继承占位，未评估序列仍可在后续研究中重新提出。第一批尚无历史，选择记录明确标注未拟合 GP 的探索初始化；从第二批起有 32 条真实观测可用于拟合。

每个评分 attempt 有独立输出目录和锁，按 `assets.json` 精确清单引用本次运行的只读输入库。资产只复制并校验一次，避免为 640 次评分各复制数 GB 缓存。实验身份、取消确认、进程组检查、失败和恢复记录仍按单个候选保存。多个单线程评分进程可以在本次活动限制的 CPU 集合内调度。

配置为 `config/antibody_rfd3/her2_ldm_batch.yaml`。完整运行前先在新目录做一个批次的端到端验收；它是独立的 32 次评分，不计入正式 640 次。所有测试和实际运行都在 H20。

```sh
# 先在 H20 加载现有模型环境；凭据不写入 Git 或命令日志。
PYTHONPATH=. /Aidata/shared/ldm/antibody-poc/env/antibody_campaign/bin/python \
  -m tasks.antibody_rfd3.repro.h20.run_full_ldm \
  --config config/antibody_rfd3/her2_ldm_batch.yaml \
  --rounds 1 --timeout 14400 \
  --output-dir /Aidata/shared/ldm/antibody-poc/runs/<新的单批验收目录> \
  --absolut-source /Aidata/shared/ldm/antibody-poc/runs/her2-full-ldm-3r-4694e362/absolut
```

正式运行使用同一配置，去掉 `--rounds 1`，使用新的输出目录及足够的外部运行时限。启动器记录源码提交、有效配置、镜像、模型和资产散列；只接受干净且已提交的工作树。

只读查看进度：

```sh
PYTHONPATH=. /Aidata/shared/ldm/antibody-poc/env/antibody_campaign/bin/python \
  -m tasks.antibody_rfd3.repro.h20.status_batch_ldm <运行根目录>
```

该规模下 BO 计算并非已知瓶颈：H20 上以合成的 640 条历史和 64 个候选，选出 32 个唯一候选的检查约 0.098 秒。这仅是性能检查，不是实际 HER2 科学结果。生成与评分的真实耗时须由单批验收测量；20 批次运行不能以启动成功代替完成验收。

## 2026-11-18 运行记录

单批真实验收目录：`/Aidata/shared/ldm/antibody-poc/runs/her2-batch-pilot-3f5ca504`，源码提交 `3f5ca504`。两个生成角色通过 6 个真实生成任务合计生成 224 条原始序列，分别提交 32 个候选，去重池为 64。最终 32 次 Absolut 评分全部成功，实际并发峰值 4。`audit-batch.json` 中 `passed=true`、`problems=[]`，完整重放了一个 32 步选择计划。首代没有历史，尚未覆盖拟合后的 GP 与先验结合；受控恢复、取消和竞争由回归测试验证，不能算作这次真实试跑覆盖。

相关 H20 验证包括初版 277 项回归测试、19 项评分与审计测试（其中与前者有重叠），以及新增策略失败保护后的 71 项相关测试。策略轮次不被接受、策略降级、应用缺失或代数不匹配时，会在创建评分计划前暂停；人工恢复不能绕过仍未解决的问题。合法的 disable 和冷启动保持有效。

正式 20 批运行于北京时间 2026-11-18 18:57 启动，源码提交 `8dd9645b`，目录为 `/Aidata/shared/ldm/antibody-poc/runs/her2-batch-20r-8dd9645b`。使用 DeepSeek、low、GPU 1，运行器总时限明确设置为 172800 秒；这个值是上限，不是预计耗时。后台启动进程为 `2484466`，外层日志为 `/Aidata/shared/ldm/antibody-poc/logs/her2-batch-20r-8dd9645b.log`。评分正常结束后自动执行 20/32/64 的批次审计并保存 `audit-batch.json`。

此记录仅确认正式运行已启动，尚未确认完成 640 次评分。以运行目录的 `launch.json`、控制器账本和最终审计为准。未向 Claude 工作区同步任何内容。
