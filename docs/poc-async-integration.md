# PoC 的完整异步任务入口

`antibody_rfd3` 的四种方法统一使用 `PacedController`、独立 HTTP 实验服务和持久化研究队列。原同步入口与整数 fixture 分支已移除。异步框架、历史查询和 Harness 0.4 来自通用框架提交 `09424112757472c9174c7676f6af0c03a64c206e`，单向引入本 PoC 分支；PoC 内容不回传到 Claude 工作区。

## 运行链路

1. 研究线程调用原生成工具或 Native AntBO，产出实际任务候选。研究代 1 对应原初始化轮 0。
2. Harness 使用 `experiment_history` 查询历史，通过 `submit_designs` 提交已签发的设计 ID。宿主执行历史水位校验、约束检查及去重。
3. 控制器从候选池选择候选；`ldm_harness` 保留原类别 GP/EI，其他方法使用顺序选择。当前 GP 对 pending 的处理是排除重复键，不使用幻想观测。
4. 独立实验服务经 HTTP 接受幂等请求，管理容量、执行状态和结果。控制器回收结果后更新历史与预算。初始化首条结果入账后才启动后续优化研究。

`--oracle mock` 使用原有序列哈希评估，仍可运行真实生成模型和 Pi。`--mock` 启用确定性序列生成样例，两者意义不同。

## H20 复跑

工作树为 `/Aidata/shared/ldm/antibody-poc/async-code`，复用已有环境。常规入口不变：

```sh
cd /Aidata/shared/ldm/antibody-poc/async-code
/Aidata/shared/ldm/antibody-poc/env/antibody_campaign/bin/python \
  scripts/run_ldm_tts.py config/antibody_rfd3/mock.yaml \
  --set args.out-dir=/Aidata/shared/ldm/antibody-poc/runs/new-offline-run
```

恢复时使用相同配置、相同科学参数，并添加 `--set args.resume-run=<原运行目录>`。恢复校验冻结配置，复用数据库和生成记录；不迁移旧同步运行。

四组小规模真实工具验收脚本读取 `config/antibody_rfd3/smoke-baseline/` 的原始配置，保持每组两次评估、原生成预算及采样参数，使用本次新生成的共享初始化。输出目录必须不存在：

```sh
/Aidata/shared/ldm/antibody-poc/env/antibody_campaign/bin/python \
  tasks/antibody_rfd3/repro/h20/async_smoke.py \
  --sidecar-image ldm-copper-pi:antibody-async-20261118 \
  --output-dir /Aidata/shared/ldm/antibody-poc/runs/new-async-smoke
```

脚本启动本地 Qwen3-8B 与 Responses 桥接，先运行两种 Harness，再释放模型服务并运行 fixed 和 AntBO，只清理自身进程。每组并发设为 1 用于原配置配对核验；并发与故障行为由独立软件测试覆盖。原四份配置、历史运行和旧镜像保留。

`acceptance-summary.json` 记录参数变化和执行结果。每组还需执行 `tasks/antibody_rfd3/repro/h20/audit_async_smoke.py <单组运行目录>`，核验序列评分、两端数据库、预算和实际生成产物。

两端分别保存 `controller.sqlite3` 和 `experiment-service.sqlite3`。生成记录位于 `design_jobs/`，历史查询及提交记录位于 `history/`，摘要和预算位于 `summary.json`、`budget.json`。失败或未决的生成记录不自动重放。

## 镜像与验收范围

本次镜像为 `ldm-copper-pi:antibody-async-20261118`，ID 为 `sha256:add36781eef64b4dd708882fd3860de41c91bc53bf2a05dc5127d9ffc6952bee`。构建固定 `SOL_PI_REF=bd005888b9b8a3fcdb511feb91fc27d3dfa8f2b1`。任务工具从 PoC 资源目录挂载。

生产镜像不包含开发依赖，侧车测试在 Dockerfile 的 `build` 阶段运行。此分支其他使用共享 Harness 的任务也需要协议 0.4 镜像；旧 0.3 镜像会被严格拒绝，不提供兼容层。

四组验收使用 mock 评估后端，不连接真实设备。其结果用于检查软件流程，不代表真实实验效果或结合能力；真实 Absolut 后端的执行管理使用独立软件测试核验。
