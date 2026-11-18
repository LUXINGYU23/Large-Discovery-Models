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
