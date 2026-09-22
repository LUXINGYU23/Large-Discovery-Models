# Assay execution contract

The task calls the pinned Assay Python engine and portfolio entry point inside
the same bounded, durable worker used by Qlib. It never calls an external data
service. Install the locked Assay environment on the server using the mirror:

```bash
export UV_CACHE_DIR=/mnt/data1/Large-Discovery-Models/cache/uv
export TMPDIR=/mnt/data1/Large-Discovery-Models/tmp
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
| `membership` | Parquet: one `date/symbol/as_of_date` row per effective index constituent per session, including warmup. |
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

The adapter loads the union of historical constituents with all prices from the
frozen history origin. Time-series operators retain pre-entry price history.
Each cross-sectional operation sees only that session's actual membership.
Group kernels run with each session's latest effective, already-known labels;
missing active labels are errors. The final factor is masked again before IC
and score export. Assay's official parser handles the canonical Qlib and native
DSL forms, including expanded `advN` and positionalized `safe_div(fill=...)`.
The registry adds the guide's exact `Tanh(x)` and `Mask(condition,value)` kernels,
matching the task's Qlib extensions. It does not rewrite other operators.

Adjustment uses native `forward_adjust`: split for US and total for CN, anchored
to the last loaded session. Actual VWAP receives the same price adjustment as
OHLC. The frozen origin also determines recursive EMA initialization. Insufficient
finite-window history fails explicitly; available history is never silently
shortened to make a formula run.

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
label read end, purged dates and unavailable tail dates. The raw Assay IC report,
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
to the actual retained split, and returns the complete effective config. Its
market and universe must match the protocol. Qlib `stock_topk/stock_n_drop` are
not translated into Assay controls; Assay uses its own weight construction,
schedule, execution and accounting semantics.

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
