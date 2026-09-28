# Assay execution contract

The task calls the pinned Assay Python engine and portfolio entry point inside
the same bounded, durable worker used by Qlib. It never calls an external data
service. Link the verified Assay source into the environment project, then install its locked dependencies on the server:

```bash
export T3_ROOT=/mnt/data1/your-workspace
export ALPHABENCH_ASSAY_SOURCE_ROOT="$T3_ROOT/data/alphabench/Assay"
export UV_CACHE_DIR="$T3_ROOT/cache/uv"
export TMPDIR="$T3_ROOT/tmp"
ln -s "$ALPHABENCH_ASSAY_SOURCE_ROOT" tasks/alphabench/environments/assay/source
uv sync --locked --project tasks/alphabench/environments/assay --group test \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple
tasks/alphabench/environments/assay/.venv/bin/python -m pytest \
  tasks/alphabench/tests/test_assay_execution.py -q
```

The service config must contain `backend=assay`, `market`, `data_digest`,
`environment_digest` and the absolute `data_manifest` path. Start it with the
Assay environment's Python. Health and dispatch both bind the market; each
worker verifies the data manifest and the actual SHA-256 of every input asset.
The qualification in current acquisition audits remains **blocked**. Fixture
execution is not a market-data qualification or a completed W06 gate.

## Offline input contract

The manifest retains the common data qualification fields and adds `assets`.
Each entry is `{ "path": "relative/path", "sha256": "..." }`. Paths resolve
inside the manifest directory. All seven entries are mandatory:

| Asset | Required contents |
| --- | --- |
| `calendar` | UTF-8 file, one distinct ISO session date per line, ascending; includes the frozen history origin and brackets the requested interval. |
| `prices` | Parquet: `date`, `symbol`, raw `open/high/low/close/volume/vwap`, `as_of_date`. VWAP is an actual source field, not an OHLC average. |
| `events` | Parquet: native Assay action fields `event_id`, `symbol`, `ex_date`, `as_of_date`, `split_ratio`, `dividend_cash`. An empty but verified event table is permitted only when the qualified source establishes there were no actions. |
| `membership` | Parquet: one `date/symbol/as_of_date` row per effective index constituent per session, including warmup. Retain full snapshots effective on non-trading dates too: source requests resolve the snapshot at the requested end date. |
| `groups` | Parquet: `symbol`, `effective_date`, `as_of_date` and named string classification columns such as `sector`. Only documented historical labels may populate this asset. |
| `execution` | Parquet: `date`, `symbol`, `as_of_date`, Boolean `tradable` for the full historical union, including names outside the index that may still be held. CN also requires actual raw `close/up_limit/down_limit`. |
| `benchmark` | Parquet: `date`, `close` for the actual named index in `manifest.benchmark`. Every loaded session must be covered. |

Date columns are Parquet `Date`; prices are numeric. Duplicate/null identity
keys, absent knowledge dates and records known after their effective date are
rejected. These are verified execution inputs, not a new data provider. The
acquisition process in [DATA.md](DATA.md) must supply and qualify them before a
real campaign can use this adapter. The presently downloaded assets cannot yet
produce a qualified snapshot; no command fabricates the missing inputs.

## Factor and label semantics

The matched profile loads the union of historical constituents with all prices from the
frozen history origin. Time-series operators retain pre-entry price history.
Each cross-sectional operation sees only that session's actual membership.
Group kernels run with each session's latest effective, already-known labels;
missing active labels are errors. The final factor is masked again before IC
and score export. Assay's official parser handles the canonical Qlib and native
DSL forms, including expanded `advN` and positionalized `safe_div(fill=...)`.
The registry adds the guide's exact `Tanh(x)` and `Mask(condition,value)` kernels,
matching the task's Qlib extensions. It does not rewrite other operators.

Matched adjustment uses native `forward_adjust`: split for US and total for CN, anchored
to the last loaded session. Actual VWAP receives the same price adjustment as
OHLC. The frozen origin also determines recursive EMA initialization. Insufficient
finite-window history fails explicitly; available history is never silently
shortened to make a formula run.

The source profile follows the pinned `AssayService` / `FactorEngine.from_store`:
choose the latest historical universe snapshot at the requested end date and use
that fixed cross-section throughout the inclusive requested interval. A snapshot
effective on a non-trading end date still applies. Load no feature history before
the requested start; native rolling warmup NaNs remain in the panel. Both CN and
US use the source default `adj=split`. The FFO bridge does not forward `label`,
so the effective source label is `next_open`, not its client's `close_return`.
Source resolution freezes these defaults and verifies both repositories' files.

Source evaluation runs the original `FactorEngine.diagnose`. Diagnostic errors
and any `failure_mode`, including the `CONSTANT` warning, fail exactly as the FFO
bridge does. Diagnostics remain in the private receipt. Missing correlations
remain unavailable; the task does not adopt FFO's conversion of null metrics to
zero. A factor with no finite daily IC is an unsuccessful evaluation.

The unchanged FFO payload loads OHLCV and supplies no group data. Full guide
support therefore explicitly extends that payload through the Python engine:
actual adjusted VWAP, the guide's Tanh/Mask operators and supplied group labels.
Source groups are the last known effective labels at the requested end, passed
to the original kernels as one fixed vector, matching the service's explicit
`group_data` argument. Matched groups remain point-in-time daily vectors. These
extensions are recorded in `source_entry_contract`; they are not claims that
the unchanged FFO transport supports those inputs.

`label=close_return` means Assay `next_close`: close[t+n]/close[t]-1.
`label=open_return` means `next_open`: open[t+1+n]/open[t+1]-1. Both request
`horizons=[forward_n]`. The matched profile excludes the requested end and
removes the last n or n+1 signal sessions so no label crosses the split.
Source profiles include the requested end and retain the native panel tail:
the pinned Assay service builds returns from the in-window panel, so its last
n or n+1 IC rows have unavailable labels. Those rows remain NaN, with zero
valid pairs; they are not converted to zero IC or filled by reading future bars.
The portfolio still covers the full source interval. Events after that interval
cannot change either the factor or label adjustment basis.

The response records the requested interval and inclusion rule, actual dates,
panel policy, adjustment, feature history origin, label read end, purged dates
and unavailable tail dates. The raw Assay IC report,
daily IC/RankIC, finite sample counts and scores remain Host-private.

With `paper_filter_v1`, the check operation evaluates only factors and exports
no returns, IC or scores. With `assay_code_filter_v1`, it instead invokes the
pinned Assay `lint` implementation without opening any market asset. Its full
diagnostics are retained, including bare-field and constant rejection. A lint
result has no measured NaN/non-finite ratio or paper dynamic success rate;
it is metered as `lint_checks`, separately from `dynamic_checks`.

## Independent portfolio

`T3Protocol.assay_portfolio` freezes every field of the pinned
`PortfolioBacktestConfig.to_dict()`. The evaluator sets its period/as-of fields
to the requested inclusive source interval or the actual retained matched split,
and returns the complete effective config. Its
market and universe must match the protocol. Qlib `stock_topk/stock_n_drop` are
not translated into Assay controls; Assay uses its own weight construction,
schedule, execution and accounting semantics.

The original FFO Assay bridge ignores `fast=False` and does not run a portfolio.
The independent portfolio is an explicit full-T3 extension. Source workflow
preflight loads only the verified, dependency-free native config module and
validates the complete configuration before initialization. The worker uses the
same task constraints with the installed native config class; there is no second
schema or implicit portfolio default. The frozen implementation plan requires
Host to call `/v1/portfolio/backtest`. The bounded worker mounts the pinned
Assay portfolio router on a one-request loopback HTTP server. Its process-local
service binds the already verified offline panel, actual custom index series and
trading mask to that request; the POST sends the frozen signal identity, complete
native config and `as_of` through the original route before the pinned
`PortfolioBacktester.run` pipeline executes. A random per-worker API key protects
the route. There is no HTTP retry; the shared receipt contract handles unknown
outcomes. The configured worker timeout is a task limit, not an original Assay
compute limit. W06 still requires qualified market assets and the real matrix.

Source factor IC and scores use the service's split adjustment in both markets.
For an A-share source portfolio the official backtester independently rebuilds
its factor and prices on a total-return basis; the task does the same before
calling the pinned portfolio pipeline. US portfolios retain split adjustment.
The raw report's `lineage.adj_version` and the signal contract record the
portfolio basis separately from the factor report's adjustment. A deterministic
source CN fixture with a split and cash dividend matches the native backtester's
NAV, benchmark NAV and trade log when both receive the same actual controls.

The actual index series is supplied as `benchmark=custom`, with an exact
`benchmark_symbol` match to the data manifest. Trade and position logs and daily
NAV are mandatory. Input controls without an implemented data feed are rejected:
sector-neutral portfolio construction, bid/ask, flow/connect, anticipation,
IPO/ST/reconstitution filters and nonzero impact models. The current explicit
contract uses `slippage_model=zero`; costs still come from the fully frozen
native commission/tax configuration. Open or close execution is allowed;
the source's VWAP-to-close proxy is rejected.

Multi-factor analysis first forms the frozen daily cross-sectional z-score,
equal-weight signal (mean over finite factors for each security/session) and
then uses the same independent portfolio pipeline. Its explicit signal identity
includes the expression list, direction and missing-value rule; it does not use
a sum expression whose NaN propagation would describe a different signal.
Raw native reports are retained with canonical daily/holdings/actions projections.
All return and drawdown units remain fractions, annualization is the source's
252-session convention. No claim of Qlib/Assay numerical equivalence is made.

Remaining gates include qualified source assets, CN limit/cost qualification,
full operator numerical audits, physical-process interruption/reconciliation
and the real backend/market matrix. Passing the deterministic offline fixtures
does not close these gates.
