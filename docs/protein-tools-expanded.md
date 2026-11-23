# 新蛋白工具接入与验证记录

本次以 `codex/validated-protein-tools` 的 `2eff1c71` 为基础，在 `codex/expanded-protein-tools` 开发。主工作树 `/Aidata/shared/ldm/antibody-poc/code` 没有切分支或改代码。所有验证在独立目录运行；提交工具、测试、配置样例和小型证据，不提交权重、运行产物或密钥。

## 目前可交付的范围

| 方法 | 实际完成 | 当前用途 |
|---|---|---|
| Protenix v1 | 完整序列、MSA、三 seed 阳性/错配、严格 PAE 转换、ipSAE | 实验性独立预测；阳性漏检，不能硬筛 |
| Boltz-2 | 六次完整 Fab 预测、原生 PAE/pLDDT、ipSAE | 实验性独立预测；本组对照区分失败，不能硬筛 |
| BoltzGen 0.3.2 | 原生骨架生成＋逆折叠；固定框架 H3；PD-1/HER2 的真实 Qwen3-8B→GP acquisition→Absolut 闭环 | 可选 Pi 生成工具，默认关闭；未验证原生完整排名流程 |
| ESMFold2 | 官方完整模型、三 seed 阳性/错配共六次预测、原生置信度→ipSAE | 实验性独立预测；不设拒绝阈值 |
| PXDesign | PD-1、HER2 的真实 80 aa binder 生成 | 独立蛋白 binder 入口；不属于 11 aa H3 域；未验证 AF2/Protenix 后筛选 |
| Protenix v2 | 支持条件已调研 | 未运行、未接入；官方权重来源/授权仍未解决 |

SolubleMPNN 和原有 ipSAE 工具已在前一个验证分支提交；vanilla ProteinMPNN 继续默认。Absolut 的序列代理结合能不能视为真实亲和力，本记录也不证明这些工具优于基线。

## 服务器目录

共同根目录 `R=/Aidata/shared/ldm/antibody-poc`。

| 内容 | 路径 |
|---|---|
| 本次完整独立验证代码 | `$R/tmp/expanded-protein-tools` |
| 早期独立试验代码 | `$R/tmp/protein-tools-attempt`，不作为最终复现入口 |
| Protenix 环境／缓存 | `$R/env/protenix`、`$R/cache/protenix` |
| Boltz-2 环境／缓存 | `$R/env/boltz2`、`$R/cache/boltz` |
| BoltzGen 环境／缓存 | `$R/env/boltzgen`、`$R/cache/boltzgen-local` |
| ESMFold2 环境／缓存 | `$R/env/esmfold2`、`$R/cache/esmfold2-69869f73` |
| PXDesign 有效环境／源码／缓存 | `$R/env/pxdesign311`、`$R/env/sources/PXDesign`、`$R/cache/pxdesign` |
| ipSAE 上游脚本 | `$R/env/sources/IPSAE/ipsae.py` |
| 原生 BoltzGen 目标／框架配置 | `$R/runs/boltzgen-validation-001/{h3-pd1.yaml,h3-her2.yaml,fab.yaml,1N8Z.cif,5GGR.cif}` |
| 真正的双抗原 LLM 闭环 | `$R/runs/antibody_rfd3/boltzgen-llm-002/{pd1,her2}` |
| Boltz-2 三 seed 对照 | `$R/runs/boltz2-validation-001/calibration-001/` |
| ESMFold2 预测／ipSAE | `$R/runs/esmfold2-validation-001/` |
| PXDesign 双抗原预测 | `$R/runs/pxdesign-validation-001/{wrapper-pd1-001,wrapper-her2-001}` |
| 模型加载／运行日志 | `$R/logs/` 中相应 `boltz2-*`、`esmfold2-*`、`pxdesign-*` 文件 |

代码入口全部位于仓库 `tasks/antibody/tools/complex_screening/`，详细参数可通过各模块 `--help` 查看。下列命令在独立代码目录执行，不需要激活全局环境。

## 实现与运行

### BoltzGen：可选 LLM 生成工具

设置宿主资源后，在既有 `harness` / `ldm_harness` 运行命令中增加 `--enable-boltzgen`：

```bash
R=/Aidata/shared/ldm/antibody-poc
export BOLTZGEN_PYTHON="$R/env/boltzgen/bin/python"
export BOLTZGEN_SCAFFOLD="$R/runs/boltzgen-validation-001/1N8Z.cif"
export BOLTZGEN_CHECKPOINT="$R/cache/boltzgen-local/boltzgen1_adherence.ckpt"
export BOLTZGEN_INVERSE_FOLD_CHECKPOINT="$R/cache/boltzgen-local/boltzgen1_ifold.ckpt"
export BOLTZGEN_MOLDIR="$R/cache/boltzgen-local/mols.zip"
export BOLTZGEN_SPEC_PD1="$R/runs/boltzgen-validation-001/h3-pd1.yaml"
export BOLTZGEN_SPEC_HER2="$R/runs/boltzgen-validation-001/h3-her2.yaml"
```

可独立调用：

```bash
cd "$R/tmp/expanded-protein-tools"
"$BOLTZGEN_PYTHON" -m tasks.antibody.tools.complex_screening.boltzgen_design \
  --python "$BOLTZGEN_PYTHON" --spec "$BOLTZGEN_SPEC_PD1" \
  --scaffold "$BOLTZGEN_SCAFFOLD" --checkpoint "$BOLTZGEN_CHECKPOINT" \
  --inverse-fold-checkpoint "$BOLTZGEN_INVERSE_FOLD_CHECKPOINT" \
  --moldir "$BOLTZGEN_MOLDIR" --out "$R/runs/boltzgen-new-test" \
  --gpu 6 --seed 43 --samples 16
```

请在运行前确认 GPU 空闲，输出目录必须是新目录。调用 `design_h3_boltzgen(count)` 消耗 **1 个骨架＋count 个序列额度**。原生逆折叠生成所有请求样本后，再应用现有芳香族比例、电荷等域约束；不补抽、不放宽约束。生成链按完整框架序列识别，再定位 H3，不能直接截取任意 binder 的 11 位片段。

`h3_from_complex.py` 保留原始输出，只对生成提案做固定框架坐标恢复，并检查 H3 两端肽键。原始框架 CA RMSD 上限 0.25 Å，恢复后 0.05 Å，边界 C–N 1.0–1.8 Å；**不能把该操作应用到预测器的结构/PAE 对上**。

Pi 工具、预算、宿主进程超时与已有设计工具共用运行框架。`design.mjs` 仅在宿主创建启用标志后注册新工具。未启用时默认工具与初始化行为保持原状。完整 native YAML 样例在 `examples/boltzgen/`；需自行准备同目录的 RCSB 原始 mmCIF。5GGR 的 PD-1 是 **label E / author Y**，不能混用；从 PDB 简单重写的 mmCIF 可能缺少 BoltzGen 所需的实体序列元数据。`reset_res_index` 是避免设计插入处编号间隔造成错误肽链几何的必要配置。

真实闭环由公开数据、Qwen3-8B 与实际 Absolut 组成，分别完成三个测量。日志证明 GP acquisition 在后续轮次参与候选选择；LLM 得到了前轮实测历史。PD-1/HER2 各有一个真正 BoltzGen 候选进入评估，能量 -69.01/-84.54。第二次生成均出现 0/16 域内候选，因此仍需要旧候选/其他生成器回退。此实验含显式工具使用指导，不证明模型会自主优先选用它，也不证明生成质量超过 RFD3。

### Boltz-2、ESMFold2：独立辅助预测与 ipSAE

`boltz.py` 使用隔离的原生 CLI；每个样本必须同时产出 CIF、PAE NPZ、pLDDT NPZ 与 confidence JSON。ipSAE runner 自动携带同目录同名 pLDDT，并在隔离目录用相对别名调用上游。不能只复制 PAE 一个文件。Boltz-2 小分子 affinity head 没有被包装成抗体亲和力。

`esmfold2.py` 直接使用官方原生 API：默认 20 loops、200 sampling steps、1 sample、无 MSA、关闭可选 fused kernels、保留原生模型精度；全模型强制 BF16 会导致输出 head dtype 不匹配，已避免该问题。`esmfold2_pae.py` 严格核对 CIF 顺序、token 原子区间与链划分，只转换单位、不估算 PAE。与 Protenix 共用一个完整蛋白复合物 JSON 列表输入格式。

```bash
R=/Aidata/shared/ldm/antibody-poc
cd "$R/tmp/expanded-protein-tools"
"$R/env/esmfold2/bin/python" -m tasks.antibody.tools.complex_screening.esmfold2 \
  --input "$R/runs/esmfold2-validation-001/input-0.json" \
  --out "$R/runs/esmfold2-new-test" --model-dir "$R/cache/esmfold2-69869f73" \
  --gpu 2 --seed 19
"$R/env/boltzgen/bin/python" -m tasks.antibody.tools.complex_screening.esmfold2_pae \
  "$R/runs/esmfold2-new-test/native-confidence.npz" \
  "$R/runs/esmfold2-new-test/model.cif" "$R/runs/esmfold2-new-test/ipsae-confidence.json"
"$R/env/boltzgen/bin/python" -m tasks.antibody.tools.complex_screening \
  --out "$R/runs/esmfold2-new-score" ipsae --python "$R/env/boltzgen/bin/python" \
  --script "$R/env/sources/IPSAE/ipsae.py" \
  --pae "$R/runs/esmfold2-new-test/ipsae-confidence.json" \
  --structure "$R/runs/esmfold2-new-test/model.cif"
```

Boltz-2 和 ESMFold2 的对照都是 PD-1/nivolumab 完整 Fab 阳性与 PD-1/trastuzumab 计算错配；后者并不是实验认证阴性。Boltz-2 使用已有真实 unpaired MSA，ESMFold2 使用无 MSA模式，不能将两者当作严格相同信息预算的模型竞赛。ESMFold2 三 seed（19/43/101）的阳性重链/抗原最小方向性 ipSAE 为 0.446642/0/0.421704，错配为 0/0/0；seed 43 仍漏检阳性。Protenix/Boltz-2 也存在阳性低分，因此三者均不能采用现成 ipSAE 阈值硬筛；全局 ipTM 还会受重轻链配对主导。

### PXDesign：独立完整蛋白 binder

`pxdesign.py` 固定 Python 3.11 / Torch 2.3.1 / `protenix==0.5.0+pxd` / PXDesignBench 0.1.2，显式把固定版本源码加到 Python 路径，解决 wheel 漏装 `configs` namespace。已用官方 2024-06-08 CCD；不同 Protenix 版本的 CCD 不能按文件名替换。

入口只运行上游 raw 生成 API，要求本地真实权重和化学数据预先存在。跳过的是完整 pipeline 自动下载的三个未使用评估模型，不是伪造文件或更换模型。输入快照、资源 SHA256、native config、CIF 和运行记录一并保留；仅支持单目标链，并检查输出 A0 目标与 B0 请求长度 binder。

```bash
R=/Aidata/shared/ldm/antibody-poc
cd "$R/tmp/expanded-protein-tools"
OMP_NUM_THREADS=4 "$R/env/pxdesign311/bin/python" \
  -m tasks.antibody.tools.complex_screening.pxdesign \
  --spec "$R/tmp/pxdesign-her2.yaml" --out "$R/runs/pxdesign-new-test" \
  --source-root "$R/env/sources/PXDesign" \
  --checkpoint-dir "$R/cache/pxdesign/checkpoint" \
  --ccd-dir "$R/cache/pxdesign/ccd_cache" --gpu 7 --seed 19
```

80 aa 完整 binder 不适合当前 11 aa H3 Oracle，因此没有强行注册到该 Pi 设计域。AF2/Protenix 复筛、亲和力与实验有效性没有验证；这些限制在 `run.json` 明确标记。

## 环境与来源

Boltz-2 / BoltzGen 使用单独依赖环境，并经只读 `.pth` 共享现有 Torch 2.7.1 CUDA 库；ESMFold2 则经 `$R/env/shared-torch-2.11-cu128` 只读共享服务器已有 `diffra` 环境的 Torch 2.11.0+cu128。没有升级全局 PyTorch/CUDA/驱动。这些环境依赖共享库原始路径，**不能当作可以整目录迁移的完全独立环境**；迁机时应重新安装相应固定版本。PXDesign 使用自己的 Python3.11/Torch2.3.1 CUDA12.1 依赖。

模型从官方公开来源获取并记录快照/文件哈希。Boltz-2 CCD 使用已验证与官方 tar 全部 45,227 个成员内容一致的官方 BoltzGen zip 解包，容器字节哈希不同，不能声称 tar 本身完全相同。ESMFold2 六个 shard 与 CCD 全部校验官方 LFS SHA256。下载/初始化与 GPU 推理是分开验证的。

官方来源：[Boltz](https://github.com/jwohlwend/boltz)、[BoltzGen](https://github.com/HannesStark/boltzgen)、[ESMFold2](https://huggingface.co/biohub/ESMFold2)、[PXDesign](https://github.com/bytedance/PXDesign)、[Protenix](https://github.com/bytedance/Protenix)。Protenix v2 仍受官方权重可获取性与使用/转交条件限制，未使用无法证明授权的第三方镜像。

## 验证文件

仓库 `tasks/antibody/tools/complex_screening/validation/` 保存：

- `h20-boltzgen-llm.json`：真实双抗原作业、模型哈希、预算、准入、GP acquisition 选择与实测观察。
- `h20-boltz2-calibration.json`：六次原生预测及 ipSAE 方向性分数。
- `h20-esmfold2.json`、`esmfold2-official-model-manifest.json`：实际预测/评分记录、官方模型版本和大文件哈希。
- `h20-pxdesign-raw.json`：双抗原原生生成、资源哈希、输出长度检查。
- `h20-expanded-environments.json`：实际依赖版本与四个隔离环境的依赖一致性检查。
- 原有 `h20-protenix-*`：v1 实际预测、MSA、多 seed 和 ipSAE 的证据。

H20 已通过 complex-screening 的 10 个测试，以及 optional-tools/procedure 的 8 个测试。重点覆盖原子/token 与链的错误映射、置信度尺度、成功退出却缺产物、Boltz pLDDT 附件、venv 解释器符号链接、共享预算及 BoltzGen 关闭行为。真实 GPU 和 LLM/Oracle 运行提供独立的端到端证据。默认不启用新的硬筛选规则。
