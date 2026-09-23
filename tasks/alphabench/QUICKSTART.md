# AlphaBench T3 开发验证

当前合同为 `draft`。以下流程验证注册、mock、恢复和数据获取；原生 source profiles、Assay 市场资格、Harness、compiled policy 及完整矩阵仍在实施中，不能把本流程称为完整 T3 验收。

数值执行只在指定服务器进行，仓库位于 `/mnt/data1/Large-Discovery-Models/worktrees/alphabench-t3-implementation`。`uv.lock` 固定依赖，使用清华 PyPI 镜像；AlphaBench/Assay 源码和国际数据按 [DATA.md](DATA.md) 在本机取得并传输。

```bash
cd /mnt/data1/Large-Discovery-Models/worktrees/alphabench-t3-implementation
export UV_CACHE_DIR=/mnt/data1/Large-Discovery-Models/cache/uv
export TMPDIR=/mnt/data1/Large-Discovery-Models/tmp
uv sync --locked --project tasks/alphabench --group dev \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple
T3_PY=tasks/alphabench/.venv/bin/python

"$T3_PY" scripts/validate_tasks.py --task alphabench
"$T3_PY" scripts/check_task_dependencies.py config/alphabench/mock.yaml --no-optional
"$T3_PY" scripts/run_ldm_tts.py config/alphabench/mock.yaml --dry-run
"$T3_PY" -m pytest tasks/alphabench/tests -q
"$T3_PY" scripts/run_ldm_tts.py config/alphabench/mock.yaml
```

mock 使用 `mock_ldm` 合同：30 个 synthetic 初始化种子、一轮 LDM、2 次 search 评价；同一共享 Campaign 执行私有 validation、冻结选择、完整 test 产物、组合和质量审计。每次新运行自动分配 `tasks/alphabench/runs/mock[_N]`，结果中的 `complete_t3` 保持 `false`。

三种原生方法各有完整冻结的两轮资格配置，经同一共享 runner 启动：

```bash
"$T3_PY" scripts/run_ldm_tts.py config/alphabench/native_cot_mock.yaml
"$T3_PY" scripts/run_ldm_tts.py config/alphabench/native_tot_mock.yaml
"$T3_PY" scripts/run_ldm_tts.py config/alphabench/native_ea_mock.yaml
```

配置读取 `resources/protocols/native_*_qualification.json`，使用已离线传入的固定
源码、三个 cold 种子和最多 12 次新 search 尝试。命名配置同时锁定协议文件的
完整内容摘要；运行保存共享 runner 合同，直接恢复也继续验证这一身份。原算法完整执行，搜索使用独立
共享 runtime，私有验证与终局阶段复用同一个 task 网关；结果标记
`execution.kind=native_reference`。正式 matched 比较另用共同的
`--initialization-bundle` 固定初始信息。恢复原生运行时提供相同的
`--upstream-root /mnt/data1/Large-Discovery-Models/data/alphabench/AlphaBench`。

终端返回绝对运行路径。检查该目录中的 `status.json`、`budget.json`、`result.json`、`selection_frozen.json` 和 `trajectory.csv`；初始化有独立的预算和 `seed_manifest.json`。IR/SFT 在 `ldm_data/` 下通过 `current.json` 指向成对完成的 generation，初始化与搜索分别记录。

同一运行恢复时显式提供冻结协议和原目录，例如从仓库根执行：

```bash
T3_RUN=/mnt/data1/Large-Discovery-Models/worktrees/alphabench-t3-implementation/tasks/alphabench/runs/mock
LDM_DATA_COLLECTION_ENABLED=1 "$T3_PY" -m tasks.alphabench.ldm_task.procedure \
  --mock --protocol-file "$T3_RUN/protocol.json" --resume-run "$T3_RUN"
```

完成后的恢复不新增模型请求、评价费用或训练行。中断且物理结果未知的请求保持暂停，通过已有 receipt 对账。

初始化支持 cold（`cold_seed_count=0` 表示空初始集）、Alpha158 分组、TXT/JSON/JSONL 文件，以及外部 `final_pool` JSON/JSONL。后两种通过 `--init-mode file --seed-file ...` 或 `--init-mode import_pool --import-pool ...` 选择；文件中的旧 metrics/validation/test 只保留在 Host 私有来源快照中，表达式重新评价。

来源在首次读取时冻结，逐项记录准入、重复和拒绝原因。恢复可直接使用已冻结内容，不需要原文件继续存在；显式提供与冻结内容不同的源会报错。初始化尚未结束、主搜索尚未创建时也使用同一个 `--resume-run` 入口。每个新运行和初始化拥有独立请求身份，初始化成本不占用新搜索的 E。

固定上游 Alpha158 loader 的分组记录数为 `kbar+price=13`、`rolling=29`、`kbar+rolling=38`、三组全部 `42`；默认是 38 条的 `kbar+rolling`。官方 seed 文件实际读取 125 条，均通过当前静态准入。此来源核验不等于真实行情评价，证据在 `resources/evidence/seed_sources.json`。

matched 方法共用初始信息时，为新运行传入 `--initialization-bundle /absolute/source-run/initialization`。源必须是完整完成的原始初始化目录；导入核对数据、模型、过滤和标签协议，逐条验证 initialization/validation receipt、规范候选、观测和预算。mock bundle 不能用于真实运行。各方法的 `initialization.public_information_digest` 必须相同。

导入只从 Host 读取公共 search 观测和私有 validation，并单独记录创建 bundle 的真实成本。本次初始化预算消耗为零，`including_initialization.generation_cost` 包含原始创建成本；可以把本次 `initialization_evaluations` 上限设为零。新搜索仍有完整 E。源目录的文件清单和哈希冻结后不能替换；导入中断时重新核对同一来源，完成后 `--resume-run` 使用本 run 的固定初始观测。

真实运行必须同时具备冻结的 `protocol-file`、有来源及逐文件哈希的数据 manifest、对应固定源码和受控 oracle。用户允许以同一批不完整数据进行比较时，在协议中设置 `data_policy=partial_comparison` 并使用原审计 manifest 的规范 JSON digest；不编辑其 `qualification` 字段。所有方法仍要固定同一数据身份、市场、切分及缺失规则，报告标记部分数据比较。当前已有 DeepSeek Responses/max 接口预检和 Qlib 真实种子开发探针；它们不满足完整 T3 功能矩阵资格。

CSI300 direct 部分数据 pilot 的冻结协议与 Oracle 配置已保存在 `resources/protocols/partial_csi300_llm_pilot.json` 和 `resources/oracle_configs/partial_csi300_qlib_pilot.json`。在服务器仓库根的两个终端分别执行以下命令，使用新的 run 目录；配置只适用于 `/mnt/data1/` 已安装的指定归档：

```bash
tasks/alphabench/environments/qlib/.venv/bin/python -m tasks.alphabench.core.oracle_service \
  --config tasks/alphabench/resources/oracle_configs/partial_csi300_qlib_pilot.json \
  --root /mnt/data1/Large-Discovery-Models/runs/alphabench-t3/oracle-new --port 19779
```

```bash
LDM_DATA_COLLECTION_ENABLED=1 tasks/alphabench/.venv/bin/python -m tasks.alphabench.ldm_task.procedure \
  --protocol-file tasks/alphabench/resources/protocols/partial_csi300_llm_pilot.json \
  --data-manifest /mnt/data1/Large-Discovery-Models/data/alphabench/manifests/qlib_csi300.json \
  --upstream-root /mnt/data1/Large-Discovery-Models/data/alphabench/AlphaBench \
  --oracle-url http://127.0.0.1:19779 \
  --api-key-file /mnt/data1/Large-Discovery-Models/secrets/alphabench-t3/deepseek.key \
  --out-dir /mnt/data1/Large-Discovery-Models/runs/alphabench-t3/llm-new
```

既有 pilot 的结果哈希、成本与数据缺口见 `resources/evidence/partial_csi300_llm_pilot.json`。这是完整路径的先行测试，不是八方法的正式匹配比较。
