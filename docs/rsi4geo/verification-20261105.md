# RSI4Geo 整理验证：2026-11-05

代码提交：`1e393991`（铜矿 v2、框架预算/并发/计时修复和回归测试）。
此记录验证当前工作区代码，不替代历史科学实验验收。

## 环境

测试在授权 H20 服务器独立目录
`/Aidata/shared/ldm/rsi4geo-review-20261105-PwdKks/` 执行。
未覆盖原实验目录，未启动模型调用或新的科学比较。

Python 使用独立虚拟环境：CPython 3.11.16、pytest 8.4.2、NumPy 1.26.4、
SciPy 1.14.1、scikit-fmm 2025.6.23、discretize 0.11.3、PyYAML 6.0.3。
同步了完整任务资源及测试依赖的 `docs/copper-baseline-audit-20261027/numeric-fixtures/`。

Pi 使用 `ldm-pi-harness:copper-validation-clock-20261025` 镜像的 Node.js 24.7.0。
镜像依赖与当前 `harnesses/pi/package-lock.json` 的 SHA256 相同：
`63afcba0a36246a2f8afe0406946de7051e8cc706a96980fba48f5eae91b6fb4`。
挂载本次源码，在禁止容器外网、2 CPU / 4 GiB 限制下重新编译并测试。

## 结果

| 检查 | 结果 |
| --- | --- |
| Python 回归 | 309 passed，41 skipped，0 failed；31.34 秒 |
| Pi TypeScript 构建及单元测试 | 构建通过；74 passed，0 failed，0 skipped |
| `validate_tasks.py --task copper_inversion` | 注册结构有效；资格证据达到 `registered` |
| 新增代码的 `git diff --check` | 通过 |

Python 的41项跳过包括39项需要显式设置 `COPPER_TEST_IMAGE` 的旧任务集成测试，
以及2项需要指定远程开发数据套件的测试。本次没有启用这些可选条件；
未将跳过项目计为通过。v2 的默认 Docker 求解器/评分器边界测试实际执行并通过。
注册检查同时指出 experiment contract 仍为 `draft`、campaign qualification 未完整；
结构检查通过不代表新增科学验收。

首轮测试因独立环境缺少 discretize、数值夹具和子进程导入路径而失败。
补齐环境并设置 `PYTHONPATH` 后，上述完整 Python 测试选择重新执行通过，
没有为消除失败而修改科学实现或测试断言。

## 执行命令

在远程源码根目录、已激活测试虚拟环境的前提下：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -ra \
  tasks/copper_inversion/tests tests/test_ldm_engine.py \
  tests/test_harness_client.py tests/test_harness_batch_validation.py
python scripts/validate_tasks.py --task copper_inversion
```

Pi 在上述镜像中，将 `/app/node_modules` 连接到测试源码目录后运行 `npm test`。
完整输出：[Python](verification/python-tests.log)、[Pi](verification/pi-tests.log)。

## 整理范围

- 将既有代码改动与报告证据分为独立提交，补齐论文 codebase 导航。
- 纳入 v2 运行必需、此前被通用忽略规则排除的研究提示文件。
- 更新任务入口中已过时的主指标和实验状态说明。
- 保留原报告路径、冻结赢家源码、图像和数据内容；重复 ZIP 导出仅保留本地。
- `.gitattributes` 对冻结报告关闭换行转换并保留生成文件原有空白，避免改变审计摘要。
- 未执行新的训练、验证选型、测试集科学评估或赢家修正。
