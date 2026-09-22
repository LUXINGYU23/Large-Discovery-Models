# AlphaBench T3 开发验证

当前合同为 `draft`。以下流程验证注册、mock、恢复和数据获取；原生算法、Assay worker、Harness、compiled policy 及完整市场矩阵仍在实施中，不能把本流程称为完整 T3 验收。

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

终端返回绝对运行路径。检查该目录中的 `status.json`、`budget.json`、`result.json`、`selection_frozen.json` 和 `trajectory.csv`；初始化有独立的预算和 `seed_manifest.json`。IR/SFT 在 `ldm_data/` 下通过 `current.json` 指向成对完成的 generation，初始化与搜索分别记录。

同一运行恢复时显式提供冻结协议和原目录，例如从仓库根执行：

```bash
T3_RUN=/mnt/data1/Large-Discovery-Models/worktrees/alphabench-t3-implementation/tasks/alphabench/runs/mock
LDM_DATA_COLLECTION_ENABLED=1 "$T3_PY" -m tasks.alphabench.ldm_task.procedure \
  --mock --protocol-file "$T3_RUN/protocol.json" --resume-run "$T3_RUN"
```

完成后的恢复不新增模型请求、评价费用或训练行。中断且物理结果未知的请求保持暂停，通过已有 receipt 对账。

真实运行必须同时具备冻结的 `protocol-file`、通过资格审核的数据 manifest、对应固定源码和受控 oracle。数据审计仍有缺口时保留 `blocked`，不要编辑 `qualification` 字段把开发探针伪装成正式结果。当前已有 DeepSeek Responses/max 接口预检和 Qlib 真实种子开发探针；它们不满足完整 T3 资格。
