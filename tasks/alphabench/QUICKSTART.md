# AlphaBench T3 快速开始

从仓库根运行。数值计算和运行产物放在服务器的 `/mnt/data1/`；数据获取、固定来源与资格限制见 [DATA.md](DATA.md)。当前任务合同仍是 draft，mock 成功和部分 CSI300 实验都不代表完整 T3 资格。

```bash
export T3_ROOT=/mnt/data1/your-workspace
export ALPHABENCH_SOURCE_ROOT="$T3_ROOT/data/alphabench/AlphaBench"
export ALPHABENCH_RUNTIME_ROOT="$T3_ROOT/runtime/alphabench"
export UV_CACHE_DIR="$T3_ROOT/cache/uv"
export TMPDIR="$T3_ROOT/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$ALPHABENCH_RUNTIME_ROOT"
uv sync --locked --project tasks/alphabench --group dev
T3_PY=tasks/alphabench/.venv/bin/python

"$T3_PY" scripts/validate_tasks.py --task alphabench
"$T3_PY" -m pytest tasks/alphabench/tests -q
"$T3_PY" scripts/run_ldm_tts.py config/alphabench/mock.yaml
```

原生 CoT、ToT、EA 的 mock 配置还需要 `ALPHABENCH_SOURCE_ROOT` 指向已校验的固定 AlphaBench 源码。三个配置使用各自的冻结协议和同一共享 runner：

```bash
for method in cot tot ea; do
  "$T3_PY" scripts/run_ldm_tts.py "config/alphabench/native_${method}_mock.yaml"
done
```

真实实验先启动独立 Oracle 服务，再用新输出目录运行。协议、数据清单、上游源码和服务配置须共同冻结；`resources/evidence/partial_csi300_qlib_pilot_config.json` 只记录旧 pilot 的配置，不是新机器的默认配置。下面的绝对路径由运行者填入，不能直接复用旧研究目录。

```bash
"$T3_PY" -m tasks.alphabench.core.oracle_service \
  --config /absolute/oracle-config.json \
  --root "$T3_ROOT/runs/alphabench/oracle" --port 19779

LDM_DATA_COLLECTION_ENABLED=1 "$T3_PY" -m tasks.alphabench.ldm_task.procedure \
  --protocol-file /absolute/protocol.json \
  --data-manifest /absolute/data-manifest.json \
  --upstream-root "$ALPHABENCH_SOURCE_ROOT" \
  --oracle-url http://127.0.0.1:19779 \
  --out-dir "$T3_ROOT/runs/alphabench/campaign-001"
```

Host 从 `DEEPSEEK_API_KEY` 或 `--api-key-file` 读取凭证。Harness 方法另需 Pi sidecar/KVM；`ALPHABENCH_RUNTIME_ROOT` 可指定可写的 Host 缓存和 socket 根目录，路径须足够短以满足 Unix socket 的长度限制。真实协议应明确 `qualified_only` 或 `partial_comparison`；后者必须标记共享数据缺口，不能宣称原论文数值复现。

恢复只使用原运行目录和原冻结协议，重新提供相同的数据、源码、Oracle 与凭证参数，并将 `--out-dir` 换成 `--resume-run`。已完成运行只核验和重放；未知的模型或 Oracle 物理结果会暂停，必须先核对持久 receipt，不能盲目重发。

已完成的 pilot 的协议、摘要和来源在 [README.md](README.md) 与 `resources/evidence/`。历史 run 保留于各证据记录所指的目录，不属于本发布版的启动接口。
