# 完整 T3 的数据准备

本机负责国际来源下载和传输。服务器不需要访问国际网络；数值审计、后端转换、调试和实验均在服务器完成，永久资产位于 `/mnt/data1/`。

## 官方实际提供的资产

固定 AlphaBench commit 为 `31bb94bbb7744177c51c9d011e07c31e3092b93e`。
仓库包含题目、因子池和 Alpha158 公式；`benchmark/data/evaluate/raw/download.txt`
链接到 [alphabench_performance_raw.zip](https://drive.google.com/file/d/1DHvASw87V05jr4JQl7EW7PMroxiWgoKj/view)。
[官方输入说明](https://github.com/CityU-MLO/AlphaBench/blob/31bb94bbb7744177c51c9d011e07c31e3092b93e/example/evaluation/01_build_datasets/README.md)
列出 CSI300/SP500 的逐日 IC、RankIC 表，供 T2/T4 构造标签。预计算表不能评价 T3 新生成的任意因子。

已核对该压缩包的 ZIP 目录：仅有上述四张 CSV，总压缩包大小为 202,921,647 字节。
核对使用文件末尾的目录，不代表已完整下载或校验这个包；证据在
[official_data_catalog.json](resources/evidence/official_data_catalog.json)。当前只接入 T3，故不将这个包列为行情准备依赖。

[官方 Prerequisites](https://github.com/CityU-MLO/AlphaBench/tree/31bb94bbb7744177c51c9d011e07c31e3092b93e#prerequisites)
要求另外下载 Qlib 行情。该版本未提供覆盖完整 T3 所有市场的固定行情包，也没有原实验行情的全量文件哈希。第三方公开行情可用于独立接入验证，不能冒充原论文数据或数值复现。

截至 2026-09-23，[Qlib 的数据准备说明](https://github.com/microsoft/qlib#data-preparation)
明确表示其官方数据集暂时停用，转而推荐社区维护的
`chenditc/investment_data`。下文固定的 CN 归档就是该推荐来源的指定版本，
不是 AlphaBench 或 Qlib 官方发布的数据。不能把执行 Qlib 旧下载命令视为
已经获得更完整的官方 T3 行情；任何新来源仍须独立冻结哈希并通过同一覆盖审计。

旧版 Qlib 下载器使用的 Azure 地址在本机返回 HTTP 409（禁止公共访问）。
[当前下载器源码](https://github.com/microsoft/qlib/blob/main/qlib/tests/data.py)
转而指向 [SunsetWolf/qlib_dataset 的 GitHub release](https://github.com/SunsetWolf/qlib_dataset/releases/tag/v3)。
`v3` 的 CN/US 日线归档可以从本机访问，但都发布于 2024-05-22，
不可能包含完整的 2024 年下半年 T3 行情；这只是 Qlib 示例数据镜像，
也不是 AlphaBench 原实验数据。没有把它传到服务器或作为缺口补证。
对应的 URL、响应与资产大小记录在
[official_data_catalog.json](resources/evidence/official_data_catalog.json)。

## 固定来源

版本、地址和已知 SHA-256 在 `resources/data_sources.json`，不用 `latest`。

- CN：investment_data 的 `2026-09-22` Qlib 归档，含 CSI300/500/1000 历史成员区间、行情、复权 factor 和指数基准。归档和 provenance 分别校验。
- CN 停牌：Dolt 快照 `d9digdvvk5spruculluv4ujhen2350ut`。仅精确证券/日期的 `tradestatus=0` 可佐证停牌；缺行不是停牌证据。该公开表只到 `2023-06-09`。
- CN 剩余缺口：BaoStock `0.9.4` 客户端获取日线交易状态及上市/退市日期；沪深300和中证500成分股接口还接受历史查询日期。服务没有不可变版本的历史快照；查询计划、原始响应、获取时间和逐文件哈希组成冻结采集批次。重新在线获取不保证相同内容，离线复现使用保存的批次。该客户端没有中证1000的对应成分股接口。
- US 成员：固定 Assay commit `06179ef75140ce5d9b94405e4af243faddc70b9e` 中的 SP500 基线/变更与 NASDAQ100 年度成员文件，保留生效日期、实现及资源哈希。
- US 行情：Dolt `post-no-preference/stocks` 快照 `s156r5d03kp2mlt6a8pbai703vtrldc4` 的未复权 OHLCV、split、dividend、symbol。SQL 必须含 `AS OF`，保存完整原始响应和 SHA-256。

两个 US 市场按全部历史成员并集下载。代码更名/重用、退市结算、公司行动完整性、实际指数基准仍须审计；不把 ETF 或等权组合当作官方指数。曾核查的 Yahoo 访问失败仅是来源调查，不是行情缺失证据。

候选补充来源也要核对证券身份，不能只按 ticker 拼接。2026-09-23 查询
[HistoricalData.net 的 LLL 公开目录](https://historicaldata.net/api/symbol/LLL)，其日线文件只列
`JX Luxventure Limited`；但 [L3 Technologies 的 SEC 文件](https://www.sec.gov/Archives/edgar/data/1039101/000114036118040588/form425.htm)
明确记录该公司 2018 年的交易代码同为 `LLL`。目录自身也声明首末日期不保证逐日完整。
因此该来源尚不能填补 L3 的历史行情缺口或证明 US 证券身份完整；未购买、导入或提升为合格来源。

## 本机下载

从实际实现仓库根运行 PowerShell。缓存放到仓库外，不提交数据或凭证。

```powershell
uv sync --locked --project tasks/alphabench --default-index https://pypi.tuna.tsinghua.edu.cn/simple
$T3_PY = '.\tasks\alphabench\.venv\Scripts\python.exe'
$T3_STAGE = 'D:\Data\alphabench-transfer'
# 需要代理时使用本机实际监听地址；可直连时省略。
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'

& $T3_PY -m tasks.alphabench.core.data acquire-sources --root $T3_STAGE
& $T3_PY -m tasks.alphabench.core.data sources --root $T3_STAGE
& $T3_PY -m tasks.alphabench.core.data acquire-cn --root $T3_STAGE --connections 8
& $T3_PY -m tasks.alphabench.core.data plan-us --root $T3_STAGE --assay-root "$T3_STAGE\Assay"
& $T3_PY -m tasks.alphabench.core.data acquire-plan --root $T3_STAGE `
  --plan "$T3_STAGE\manifests\us-download-plan.json" --connections 4
```

`acquire-*` 是唯一联网阶段。CN 大文件按 4 MiB Range 分块续传，拼接后校验完整归档哈希。SQL 错误响应单独保留，不能把超时后的部分 rows 当作完整数据；重新执行同一计划可复用有效缓存。瞬时读取失败最多重试三次，401/403/429 不循环重试；连续八次获取失败暂停余下请求，网络恢复后重新执行同一命令。进度在 `manifests/*-progress.json`，未完成请求导致非零退出码。

US 固定历史成员并集为 780 只。价格按日查询全部成员，公司行动和证券元数据每组 5 只，共 3,100 页。Dolt 此接口最多返回 1,000 行；必须同时满足 `Success` 和未达到查询上限，`RowLimit` 一律视为不完整响应。

## 本机到服务器

`pack` 只打包 `raw/` 的完整文件，排除下载块、临时文件、环境、凭证与可变审计报告。包内 manifest 校验逐文件哈希，外部 SHA-256 校验传输包。

```powershell
& $T3_PY -m tasks.alphabench.core.data pack --root $T3_STAGE --bundle "$T3_STAGE\t3-data.tar.gz"
$T3_HASH = (Get-FileHash "$T3_STAGE\t3-data.tar.gz" -Algorithm SHA256).Hash.ToLowerInvariant()
scp -P 5320 "$T3_STAGE\t3-data.tar.gz" zsgpu@111.2.199.31:/mnt/data1/Large-Discovery-Models/tmp/t3-data.tar.gz
$T3_REMOTE = '/mnt/data1/Large-Discovery-Models/worktrees/alphabench-t3-implementation'
ssh -p 5320 zsgpu@111.2.199.31 "cd $T3_REMOTE && tasks/alphabench/.venv/bin/python -m tasks.alphabench.core.data install --root /mnt/data1/Large-Discovery-Models/data/alphabench --bundle /mnt/data1/Large-Discovery-Models/tmp/t3-data.tar.gz --sha256 $T3_HASH"
```

`install` 写入前验证来源合同、目录清单和全部哈希，拒绝链接、越界路径、重复成员及不同内容覆盖。重传相同文件幂等。第三方镜像不能改变固定内容哈希。

## 服务器离线准备

服务器环境按同一个 `uv.lock` 从国内 PyPI 镜像安装。以下操作均不联网：

```bash
cd /mnt/data1/Large-Discovery-Models/worktrees/alphabench-t3-implementation
T3_PY=tasks/alphabench/.venv/bin/python
T3_DATA=/mnt/data1/Large-Discovery-Models/data/alphabench

# 新目录解包已传入的固定源码；已有独立源码归档可继续使用。
"$T3_PY" -m tasks.alphabench.core.data sources --root "$T3_DATA"
"$T3_PY" -m tasks.alphabench.core.data cn --root "$T3_DATA"
"$T3_PY" -m tasks.alphabench.core.data plan-us --root "$T3_DATA" --assay-root "$T3_DATA/Assay"
"$T3_PY" -m tasks.alphabench.core.data plan-cn-status --root "$T3_DATA"
```

CN 初审生成缺口查询计划。取回本机下载，随后重复打包和传输：

```powershell
scp -P 5320 zsgpu@111.2.199.31:/mnt/data1/Large-Discovery-Models/data/alphabench/manifests/cn-status-plan.json "$T3_STAGE\manifests\cn-status-plan.json"
& $T3_PY -m tasks.alphabench.core.data acquire-plan --root $T3_STAGE `
  --plan "$T3_STAGE\manifests\cn-status-plan.json" --connections 4
```

最后在服务器运行 `cn-status --root "$T3_DATA"`，离线核对所有固定快照响应并更新审计。缺失响应不会从服务器补发。

对剩余 CN 缺口，在服务器生成 BaoStock 补证计划：

```bash
"$T3_PY" -m tasks.alphabench.core.data plan-cn-baostock --root "$T3_DATA"
```

取回本机并运行显式的数据获取命令。请求逐个执行，不需要账号；按证券/年度划分，避免触发客户端的分页边界。成功的缓存不重新获取，错误响应保留证据并终止采集，同一命令可恢复。

```powershell
uv sync --locked --project tasks/alphabench --group acquisition --default-index https://pypi.tuna.tsinghua.edu.cn/simple
$T3_BAO = "$T3_STAGE\raw\cn-baostock-2026-09-23"
New-Item -ItemType Directory -Force $T3_BAO | Out-Null
scp -P 5320 zsgpu@111.2.199.31:/mnt/data1/Large-Discovery-Models/data/alphabench/raw/cn-baostock-2026-09-23/plan.json "$T3_BAO\plan.json"
& $T3_PY -m tasks.alphabench.core.data acquire-cn-baostock --root $T3_STAGE
```

再次 `pack → scp → install` 后，在服务器运行 `cn-baostock --root "$T3_DATA"`。该审计重新读取全部固定 Dolt 响应与 BaoStock 响应，只有明确的停牌状态可消除对应行情缺口；返回为空、报告仍交易但归档无价、历史成员区间跨退市日期都会保留为未解决问题。不会填充假行情或修改历史成员区间。结果及逐响应哈希在 `manifests/cn_baostock_status.json`。

沪深300/中证500 的归档成员区间与 `all.txt` 冲突时，可再采集这些区间首末交易日的独立快照。`plan-cn-membership` 从服务器既有的 `qlib_csi300.json` 和 `qlib_csi500.json` 生成去重日期计划，并将两份审计哈希固定；本机按固定 BaoStock 客户端联网，服务器只做离线复核：

```powershell
$T3_MEM_STAGE = 'D:\Data\alphabench-membership-2026-09-23'
$T3_PY = '.\tasks\alphabench\.venv\Scripts\python.exe'
$T3_DATA = '/mnt/data1/Large-Discovery-Models/data/alphabench'
$T3_REMOTE = '/mnt/data1/Large-Discovery-Models/worktrees/alphabench-t3-implementation'
New-Item -ItemType Directory -Force "$T3_MEM_STAGE\manifests" | Out-Null
scp -P 5320 `
  zsgpu@111.2.199.31:/mnt/data1/Large-Discovery-Models/data/alphabench/manifests/qlib_csi300.json `
  zsgpu@111.2.199.31:/mnt/data1/Large-Discovery-Models/data/alphabench/manifests/qlib_csi500.json `
  "$T3_MEM_STAGE\manifests\"
& $T3_PY -m tasks.alphabench.core.data plan-cn-membership --root $T3_MEM_STAGE
& $T3_PY -m tasks.alphabench.core.data acquire-cn-membership --root $T3_MEM_STAGE
& $T3_PY -m tasks.alphabench.core.data cn-membership --root $T3_MEM_STAGE
& $T3_PY -m tasks.alphabench.core.data pack --root $T3_MEM_STAGE --bundle "$T3_MEM_STAGE\cn-membership.tar.gz"
$T3_MEM_HASH = (Get-FileHash "$T3_MEM_STAGE\cn-membership.tar.gz" -Algorithm SHA256).Hash.ToLowerInvariant()
scp -P 5320 "$T3_MEM_STAGE\cn-membership.tar.gz" zsgpu@111.2.199.31:/mnt/data1/Large-Discovery-Models/tmp/cn-membership.tar.gz
ssh -p 5320 zsgpu@111.2.199.31 "cd $T3_REMOTE && tasks/alphabench/.venv/bin/python -m tasks.alphabench.core.data install --root $T3_DATA --bundle /mnt/data1/Large-Discovery-Models/tmp/cn-membership.tar.gz --sha256 $T3_MEM_HASH"
ssh -p 5320 zsgpu@111.2.199.31 "cd $T3_REMOTE && tasks/alphabench/.venv/bin/python -m tasks.alphabench.core.data cn-membership --root $T3_DATA"
```

每个原始快照保存请求日期、服务返回日期、`updateDate`、全量行和获取时间。少于指数应有的 300/500 行仅记为不完整证据；即使满额也不自动改写 Qlib 成员或提升数据资格。报告在 `manifests/cn_membership_probe.json`，中证1000冲突仍需另一独立来源。

US 全部响应传入后，用固定 Assay 环境中的交易所日历审计：

```bash
UV_CACHE_DIR=/mnt/data1/Large-Discovery-Models/cache/uv \
  uv sync --locked --project tasks/alphabench/environments/assay \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple
tasks/alphabench/environments/assay/.venv/bin/python \
  -m tasks.alphabench.core.data us --root "$T3_DATA"
```

Assay 环境锁定的本地源码路径是 `/mnt/data1/Large-Discovery-Models/data/alphabench/Assay`，由上面的离线 `sources` 创建。审计逐日使用生效成员，检查 OHLCV、重复证券/日期、公司行动范围、缺页及交易所非交易日；输出 `us_sp500.json`、`us_nasdaq100.json` 和响应哈希清单。此来源没有 VWAP、真实指数基准或历史行业分类，公司行动知悉日及证券代码沿革也未验证，因此即使下载全部完成仍不能自动获得完整 T3 数据资格。

## 产物及资格

```text
raw/cn-2026-09-22/             CN 归档及 provenance
raw/cn-status-<commit>/        固定停牌快照的响应
raw/cn-membership-<capture>/   BaoStock 历史成分股快照、查询计划与获取时间
raw/us-dolt-<commit>/          US 价格、公司行动和证券元数据响应
raw/<market>-membership.json   历史成员及来源哈希
qlib/cn-2026-09-22/            服务器的 Qlib 数据
manifests/*-plan.json          确定的查询计划
manifests/*-progress.json      本机获取进度、失败身份
manifests/qlib_<market>.json    服务器字段、日历、成员、缺口审计
manifests/*.files.json         服务器实际行情文件 SHA-256
manifests/cn_suspensions.json   停牌来源、响应哈希及核对结果
```

CN 审计检查成员区间、日历、表达式字段、复权 factor、基准和有限行情覆盖。
2015 年是最低 warmup 期，保留到 2025 年 1 月检查前向标签边界；长窗口表达式仍须动态检查。
`coverage_verified` 只表示结构化覆盖检查通过。默认 `qualified_only` 仍要求独立复核后的 `qualified` 数据清单。按用户允许的部分数据比较，冻结协议显式选择 `partial_comparison` 时可使用有明确 `issues` 的原始 `blocked` 审计清单；各方法必须绑定同一规范 JSON digest 和物理文件清单，结果保留缺口、成员冲突与 `partial_comparison` 标记，不手改数据资格，也不声称原论文数值复现。

Assay 需要 `price_raw`、`adj_events`、`universe_snapshots`；CN 另需交易限制，分组算子需要历史分类。Qlib 归一化 adjusted close 不能当作 raw close，累计 factor 不能可靠拆成分红和拆股。资产未齐前，Assay 转换与完整 T3 矩阵保持未通过。

真实 run 每次启动或恢复时，Host 按冻结 digest 检查 manifest，并重新计算物理资产哈希。
Qlib 的 `qlib_<market>.files.json` 须与 manifest 的 `files_sha256` 一致；逐文件行情、
交易日历、市场成分和 `all.txt` 都重新核对。Assay 的七个离线资产使用 manifest
内的相对路径与哈希，并核对清单总摘要。缺失或改动在模型请求前拒绝运行；
该入口检查不代替 Oracle worker 执行期间的数据不变性验收。
Oracle 服务配置还必须提供同一 `data_manifest`；Qlib 的 `data_root` 和
`benchmark` 必须与该清单一致。每个物理 Qlib worker 在读取行情前重复校验
完整文件清单，Assay worker 读取资产时校验七个文件。配置或数据漂移使请求
返回 `paused_data_integrity`，不能把失败回测当作有效的零收益样本。
服务启动时还冻结整份 Oracle 配置摘要；之后即使只改上游源码路径，
物理 worker 也拒绝执行。Host 在同一 run 内固定服务报告的配置摘要，
配置确需变化时须创建新的服务和新的实验身份。

## 本次实际准备结果

2026-09-23 的本机采集和服务器离线导入已完成：US 固定快照 3,100 页、1,609,180 条行情；CN 固定停牌快照 3,745 页及 BaoStock 补证 232 个响应。传输包共 10,069 个原始文件，SHA-256 为 `b20140d6954440fb2f9951d99ef78f14e0203781934b5b5b3397fdaddd766962`。

BaoStock 额外佐证 774 个停牌日期。CN 剩余缺口如下，单位为证券/交易日组合：

| 市场 | 仍有缺口的证券数 | 缺口日期数 |
| --- | ---: | ---: |
| CSI300 | 4 | 107 |
| CSI500 | 6 | 1,028 |
| CSI1000 | 6 | 259 |

归档 `all.txt` 的证券有效区间与历史成分区间也独立核对；三个市场分别有
5、8、11 条冲突区间。这是归档内部冲突，不能单凭 `all.txt` 推断真实退市日。
未解决证券日中，329 项跨越来源报告的退市日期，493 项把尚未启用的
`SZ302132` 放进 2022–2024 年成分股，570 项是 `SH689009` 有 BaoStock
正常交易记录而社区 Qlib 归档缺价，另有 2 项没有状态证据。
[深交所披露的实施公告](https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-02-15/cedb693a-f5ee-4463-9682-ea33d406b569.PDF)
明确 `300114` 到 `302132` 的代码变更于 2025-02-17 生效；归档自身的
`instruments/all.txt` 也从该日才列出 `SZ302132`，但 `csi500.txt` 和
`csi1000.txt` 在此前同时列出旧、新代码。BaoStock 将旧价挂在现行代码下，
不能据此把 `SZ302132` 旧日期直接补入 Qlib。`SH689009` 的原始 OHLCV
已捕获，但仍须核对复权/成交额口径并在独立衍生数据版本中补齐；不能原地
修改已冻结归档。退市后的成员关系也须以指数变更记录核对，不能简单截断。

2026-09-23 另取得沪深300/中证500 冲突区间边界的 22 个历史成分股快照；13 条冲突区间的两端对应快照均达到 300/500 行。CSI300 中 4 条在区间末端缺席、1 条仍在；CSI500 中 4 条首末均缺席、4 条仅在起点出现。`updateDate` 可早于查询日期，满额快照也不能独自证明每日生效成员。原始 23 文件传输包 SHA-256 为 `be00b8939a3921125d6ed19299007222bbb2b3925e7d4f6fdfdbb272cd189ef5`；离线报告 SHA-256 为 `09faf44f57e3820996950cd80286c56a83f93f63fa6ac9ad990a6f23e2fe90fb`。这批证据仍标记 `evidence_only`，没有变更三个 CN 市场的 `blocked` 资格。

US 审计发现 SP500 59,717 个缺口日期、NASDAQ100 13,605 个缺口日期；源数据还有 825 条异常 OHLCV、10 条非交易日记录和 34 只证券缺失元数据。VWAP、真实指数基准、历史行业分类及公司行动/代码沿革资格仍未解决。下载完整不能等同于数据完整，当前全部市场保持 `blocked`。

紧凑证据在 [data_preparation.json](resources/evidence/data_preparation.json)，完整响应、逐文件哈希、证券区间冲突与缺口清单在服务器数据根目录。更换来源需另行冻结和审计，不能手动修改上述资格字段。
