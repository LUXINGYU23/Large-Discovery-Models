# HER2 三轮真实计算研究

2026-11-18，在 H20 使用提交 `e484a217` 的异步 PoC 完成三个连续研究回合。用户选择评分对象 HER2（`1N8Z_C`），模型为 `deepseek-flash`、思考强度 `low`；这两个设置已从实际 Harness manifest 核验。

配置：`config/antibody_rfd3/her2_research.yaml`。使用纯 Harness 策略，每轮由模型提交一个设计，实验并发 1、总评估预算 3。真实 RFD3/ProteinMPNN 生成，真实 Absolut 计算；生成器和评分均未使用 mock。未接湿实验，未向 Claude 工作区同步。

## 结果

| 研究回合 | 候选 ID | Absolut energy | 实际行为 |
| --- | --- | ---: | --- |
| 1 | `job-00000/0` | -83.37 | 阅读技能、查询空历史、生成初始化骨架及序列，提交首个合格候选 |
| 2 | `job-00000/5` | -87.75 | 查询首条观测及提案状态；尝试新骨架并在原骨架重采样；选择此前未测的原候选池成员 |
| 3 | `job-00004/6` | -90.71 | 查询历史与 pending；尝试新骨架，并将原骨架重采样温度调为 0.3；提交前再次查询，确认第二轮实际评分后选择新候选 |

三轮均成功，最佳评分较首轮降低 7.34。评分是该格点模型中的计算量，不能据此推断实验亲和力或疗效。

模型确实使用了评估反馈：第三轮开始时第二次评估仍在运行；随后通过历史更新和再次查询获知 -87.75，再完成提交。它不是仅按固定顺序重复生成。

不过，研究解释仍有证据薄弱之处。模型将有利的序列分数称为骨架已被验证，但 Absolut 此处只接收抽取的 11 残基序列，不接收生成坐标；它还把首轮候选的末端残基描述为 V，实际为 Y。因此这些结构机制描述不能当作已证实结论。三个候选也不足以证明一般性的优化能力。

## 审计

- 3 个研究回合、3 条成功观测、3 次 ES 执行；失败 0，无重复执行。
- 5 次显式历史查询、3 次提交接受；第 3 轮收到 1 次历史更新通知。
- 5 个成功生成任务，申请 3 个新骨架、80 条序列；其中包括对已有骨架重采样。
- Harness 记录 13 次 provider 调用、14 次工具调用；3 个回合完整结束。
- 控制器不变量和独立审计均为 `problems=[]`。
- 实验服务和控制器使用独立 SQLite，通过 HTTP 通信。

## 复现与证据

H20 运行目录：

`/Aidata/shared/ldm/antibody-poc/runs/her2-research-3r-e484a217/`

- `launch.json`：源码提交、实际命令、模型设置、独立评分文件的来源与 SHA-256。
- `audit.json`：两端实验标识、预算、执行次数、历史查询与提交校验。
- `campaign/summary.json`、`budget.json`、`events.jsonl`：运行汇总。
- `campaign/controller.sqlite3` 与 `experiment-service.sqlite3`：完整调度及结果记录。
- `campaign/design_jobs/`：生成参数、实际产物与 receipt。
- `campaign/absolut-attempts/`：每次评分请求、进程身份和真实结果。
- `campaign/history/` 与 `campaign/harness/`：可见历史、提交水位、会话和工具记录。

评分目录 `absolut/` 是本次专用副本，来自已验证的 `absolut-her2-check-fUwLve`，同时复制并校验了实际可执行文件，未写入原评分目录。API 凭据仅在 H20 加载，没有写入 Git 或报告。

准备脚本第一次因启动元数据缺失 `guest_cache` 字段退出，当时尚未启动研究或评估。修正启动元数据后使用同一批已准备资产启动，未重复执行任何评估；此准备错误保留在 `launch.json` 中。

复跑时使用相同 YAML、另建输出目录和独立 Absolut 评分目录，并设置 `LLM_BASE_URL`、`LLM_MODEL_NAME`、`LLM_API_KEY`、`LLM_REASONING_EFFORT=low`。在 H20 已有环境中运行：

```sh
python scripts/run_ldm_tts.py config/antibody_rfd3/her2_research.yaml \
  --set args.absolut-path=<独立评分目录> \
  --set args.out-dir=<新运行目录>
```

实际准备与启动脚本留在 H20 的 `tmp/run-her2-research-e484a217.py` 和 `tmp/start-her2-prepared-e484a217.py`；两者均位于 `/Aidata/shared/ldm/antibody-poc/` 下。
