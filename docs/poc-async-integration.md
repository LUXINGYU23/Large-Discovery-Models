# PoC 离线任务的异步调度

`config/antibody_rfd3/mock.yaml` 使用异步调度。标准任务入口把迭代数、并发数和随机种子传给固定版本的异步运行时。离线测试以整数 fixture ID 派发，结果以 `mock_energy` 记录。它覆盖 PoC 任务入口、HTTP 接单、预算、并发与结果落盘。历史冒烟的候选、模型产物和分数不参与这条路径。

运行时固定为 `0942411`，来源与 SHA-256 记录在 `config/async_ldm/runtime.json`。任务启动时校验 H20 的发布包，用独立 Python 进程加载异步框架与实验服务。PoC 的 Python 导入和现有 Harness 使用自己的代码。每次运行都创建独立的控制器和实验服务数据库。

从 H20 的集成工作树运行，使用已有的 PoC Python 环境：

```sh
python scripts/run_ldm_tts.py config/antibody_rfd3/mock.yaml
```

结果包括 `summary.json`、`budget.json`、`events.jsonl`、`controller.sqlite3`、`experiment-service.sqlite3` 和 `async-runtime.json`。任务参数与追溯信息分别写入 `config.json` 和 `ldm_task_spec.json`。异步入口只接收独立的离线固定样例。
