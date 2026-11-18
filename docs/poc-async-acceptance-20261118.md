# H20 完整 PoC 异步迁移验收

2026-11-18，代码提交 `c484e176`，分支 `antibody-poc-async`。开发由 6.1 Terra Max 从 agent 完成，主 agent 负责范围、接口审查、H20 执行及验收。测试使用现有环境，未接真实设备，未向 Claude 工作区同步代码或资料。

## 四组原配置冒烟

原四份 `config/antibody_rfd3/smoke-baseline/*.json` 保留。每组仍为两次评估，保留原 seed、生成预算、采样温度和 pool。生成工具为真实 RFD3/ProteinMPNN；评估为原 `mock-v1` 序列哈希。两组 Harness 使用原 Qwen3-8B，本次升级为协议 0.4 的独立侧车镜像。

| 方法 | 成功 / 目标 | ES 执行次数 | 生成记录数 | 验收证据 |
| --- | --- | --- | --- | --- |
| ldm_harness | 2 / 2 | 2 | 3 | 第二次决策为 `direct_max`，使用已入账观测做 GP/EI；3 次历史查询、2 次提交接受 |
| harness | 2 / 2 | 2 | 2 | 2 次历史查询、2 次提交接受 |
| fixed | 2 / 2 | 2 | 3 | 真实生成、顺序选择、HTTP 评估 |
| antbo | 2 / 2 | 2 | 1 | 共享初始化后调用原生 AntBO，优化器及 RNG checkpoint 可重新加载 |

生成记录数包括共享初始化的重放；AntBO 的后续建议由其原生算法产生，不调用结构生成工具。四组首条观测一致。每组控制器与 ES 的 experiment ID、序列评分、预算和终态一致；没有重复候选或执行，审计 `problems=[]`。

部署差异均有记录：输出目录换为新目录，三组初始化指向本次 ldm_harness，新侧车镜像，评估并发为 1 以核对原配置。fixed 首次在 GPU 0 等待空闲时被主动中止，因其他任务正在用卡；随后只将 GPU 改为 2 重跑通过。中止运行完整保留，不计入成功四组，也不作为算法失败。

H20 证据根目录：

- `/Aidata/shared/ldm/antibody-poc/runs/async-smoke-c484e176/`：两组 Harness、原 fixed 中止记录、逐组参数差异及审计。
- `/Aidata/shared/ldm/antibody-poc/runs/async-native-c484e176/`：fixed GPU 2 重跑、AntBO、参数差异和审计。
- 每组 `<method>/` 内有 `config.json`、`summary.json`、`budget.json`、两端 SQLite、`design_jobs/` 和 `history/`。
- 原生两组的补充启动脚本保存在 `/Aidata/shared/ldm/antibody-poc/tmp/native-smoke-c484.py`；可通过对应 `acceptance-summary.json` 的完整命令复跑。

## 软件与故障验收

- 最终实现快照上的 Python 回归：**286 passed**，包含 PoC、ES、Controller、Harness、任务注册、GP 及原有核心接口。
- 新镜像对应的侧车测试：**87 passed**。生产镜像不含测试开发依赖，测试在 Dockerfile 的 `build` 阶段镜像执行。
- 强制终止测试：当已有 1 条观测、另有 1 个 ES 已接单的待完成任务时，对 PoC 控制器发送 SIGKILL；恢复同一目录后复用了原 ES 进程与 URL，最终 **12 条观测 / 12 次执行**，无重复、无审计异常。
- 恢复证据：`/Aidata/shared/ldm/antibody-poc/runs/async-recovery-c484e176/recovery-audit.json`。复现脚本：`tasks/antibody_rfd3/repro/h20/recovery_smoke.py`，仅运行确定性生成与 mock 评估。
- 单实例冲突、首条观测门控、生成调用幂等、历史提交校验、AntBO checkpoint 及真实后端的进程组取消/超时/锁清理均有软件测试。

测试日志位于 `/Aidata/shared/ldm/antibody-poc/logs/async-final-unit-20261118.log` 与 `async-pi-tests-20261118.log`。

## 范围

任务入口已完整迁移到异步框架，原同步入口和整数 fixture 分支已删除。保留原生成与优化算法；GP 对 pending 使用去重排除，不增加幻想观测算法。

本次四组按原配置使用 mock 评分。真实 Absolut 后端已接入执行器，并测试了执行管理，但未实跑真实 Absolut 数据集；这些结果不代表真实实验效果。共享 Harness 已升级到 0.4，此分支其他 Pi 任务也需匹配的新版镜像。

原 H20 PoC 工作树及历史四组结果保留，所有集成改动在 `/Aidata/shared/ldm/antibody-poc/async-code`。
