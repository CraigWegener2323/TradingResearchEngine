# Trading Research Engine

A first, deliberately small Python backtesting engine for NQ and MNQ futures.
It is for research only: it does not connect to a broker or submit live orders,
and the included EMA crossover is a test strategy, not a claim of profitability.

## Project layout

```text
config/                       Saved experiment configurations
data/raw/                     User-supplied source data (not committed)
data/processed/               Derived data (not committed)
logs/                         Local logs (not committed)
runs/                         Local backtest exports (not committed)
src/trading_research_engine/
  data.py                     CSV/Parquet import and OHLCV validation
  engine.py                   Chronological simulator, stop modes, accounting
  export.py                   Non-overwriting result export
  strategies.py               Strategy contract and EMA crossover example
  strategy_families/          Four independent, rule-based strategy families
  experiments.py              Parameter sweeps, split validation, walk-forward
  forward.py                  Provider interface, replay and persisted forward tests
  dashboard.py                Streamlit monitor for forward-only sessions
  __main__.py                 Command-line backtest entry point
tests/                        Deterministic pytest coverage
```

## Set up on Windows

Run these PowerShell commands from the project root. Create `.venv` only if it
does not already exist:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of
`python`; changing the system execution policy is not necessary. Run the tests:

```powershell
python -m pytest
```

### Open the dashboard from the Windows desktop

The desktop shortcut **Trading Research Engine Dashboard** starts the
hypothetical forward-test monitor. Its launcher is
[launch_forward_dashboard.bat](./launch_forward_dashboard.bat). This project
uses Python 3.14 for the optional Streamlit environment because some dashboard
dependencies do not yet provide wheels for Python 3.15. The dashboard
environment is separate from `.venv`:

```powershell
py -3.14 -m venv .venv-dashboard
.\.venv-dashboard\Scripts\python.exe -m pip install -e ".[dashboard]"
```

The dashboard opens even before any forward sessions exist. To display a
session, point the dashboard sidebar to the forward-test output directory.

### Deploy from GitHub to Streamlit Community Cloud

The root-level [streamlit_app.py](./streamlit_app.py) is the Community Cloud
entry point. Push this repository to GitHub, then in Streamlit Community Cloud
create an app from that repository and select `streamlit_app.py` as the main
file. The root [requirements.txt](./requirements.txt) installs the app and its
dashboard dependencies. Use a Python version supported by Streamlit Cloud and
compatible with the listed dependencies.

Community Cloud runs on its own Linux machine; it cannot see your PC's files.
The local `runs/` folder is intentionally ignored by Git, so a deployed app
will show the empty-session message unless forward session artifacts are made
available to the deployment. The app does not connect to a feed or persist
forward-test sessions to external storage.

## Strategy experimentation

The experiment runner compares independent strategies using the same
`BacktestEngine` and market execution model:

1. **Multi-timeframe trend pullback:** long when the 14 EMA is above the 50 EMA,
   the latest fully completed higher-timeframe close is above its 50 EMA, and
   the completed bar crosses back above the 14 EMA. Short rules are mirrored.
2. **FVG/order-block pullback:** an FVG is a three-bar gap
   (`low[t] > high[t-2]` bullish, inverse bearish), with a minimum ATR-scaled
   gap; displacement body must exceed an ATR threshold. An order block is the
   nearest opposite-colour candle within the configured lookback before the
   displacement. Its zone is `[low, open]` bullish or `[open, high]` bearish.
   Only the intersection of those zones is used; a later retest and close back
   out of the intersection triggers. No subjective annotations are used.
3. **Range mean reversion/failed breakout:** use the prior `lookback` bars'
   high/low, excluding the current bar. Trade a break that closes back inside
   the range, or a reversal beyond the configured ATR edge, only if the range
   width is below its ATR cap.
4. **Volatility regime:** classify ATR/close as low, normal, or high using
   rolling quantiles calculated only from prior bars. Combine this with the
   14/50 EMA direction. High-volatility states trade confirmed range breakouts;
   low-volatility states trade reversals at the prior range boundaries; normal
   volatility produces no signal.

All family strategies default to a 09:30-16:00 America/New_York weekday entry
filter. Configure a `session` object or pass `null` to disable it. The filter
uses the completed bar's end time and controls signal eligibility, not forced
liquidation; stops and strategy exits remain active outside the entry window. Higher-timeframe bars are aggregated
from completed input bars. Input timestamps are treated as bar-start times for
aggregation, and signals still fill no earlier than the next bar open.

`BacktestConfig` supports `stop_mode="fixed"|"atr"|"structure"`, and
`take_profit_mode="fixed"|"risk_reward"|"none"`. Structural stops use the
lowest low/highest high over `structure_lookback` completed bars before the
signal. ATR stops use Wilder-style exponentially smoothed true range at the
signal close. Risk-reward targets use the actual entry-to-stop distance. All
entries, stops, targets, commissions, and slippage use the same simulator.
Risk-sized quantity is determined from equity and stop risk at signal time;
an entry gap can change realized stop distance.

### Batch sweep and validation

Edit the split dates in
[experiment example](./config/strategy_experiment_example.json)
to match your data **before** running. Use non-overlapping chronological
development, validation, and final out-of-sample intervals. The example expands
four families across NQ/MNQ, three stop modes, and fixed/risk-reward/uncapped
targets; that creates many comparisons, so consider a smaller initial grid.

```powershell
python -m trading_research_engine.experiments sweep `
  data\raw\nq_bars.csv `
  --spec config\strategy_experiment_example.json `
  --output runs\experiment-2026-01
```

The sweep tests every candidate on development, then validates up to the
configured top-N **per strategy family** so one family cannot eliminate all
other families before the validation comparison. It never runs final OOS
automatically.
Every attempted candidate/stage, including errors, is appended to
`experiments.jsonl`; completed runs also save trades, orders, equity, and run
metadata. `comparison_report.md` includes candidate ranks, sample size,
Bonferroni-adjusted bootstrap expectancy intervals, commission/slippage
estimates, and breakdowns by direction, session date, year, and objective
market-regime label. Adjusted intervals are percentile bootstraps over trades;
serial dependence can make them optimistic.

After reviewing validation, explicitly choose candidate IDs from the report or
`experiments.jsonl`, then run only those on the untouched final period:

```powershell
python -m trading_research_engine.experiments out-of-sample `
  data\raw\nq_bars.csv `
  --spec config\strategy_experiment_example.json `
  --output runs\experiment-2026-01 `
  --select "VALIDATION_SELECTED_ID"
```

The default screening criteria are recorded with every attempt: at least 100
validation trades (30 OOS), positive Bonferroni-adjusted 95% expectancy-CI
lower bound, profit factor at least 1.1, maximum drawdown at most 20%, and at
least 60% positive active calendar months. These are configurable research
screens, not universal financial standards. The system reports `screen_passed`
or `screen_not_passed`; passing is **not** a profitability declaration or
guarantee.

### Walk-forward and selection bias

The Python API `run_walk_forward` accepts explicit expanding/rolling training
and subsequent test folds. It selects parameters using only each fold's
training interval, then records the next interval's result and selection
frequency. Parameter stability is summarized across folds. Walk-forward folds
are still historical validation and do not replace the single final untouched
OOS period.

The sweep corrects bootstrap interval confidence levels for the number of
candidate comparisons using Bonferroni. This does not eliminate data snooping,
correlated strategies, regime dependence, non-stationarity, or the limitations
of trade-level bootstrap resampling. Pre-register the candidate grid and
criteria, preserve unsuccessful runs, report long/short results separately,
and treat repeated grid redesign after seeing validation as additional
multiple testing. Never alter a selected strategy after inspecting final OOS.

## Hypothetical forward testing

Forward testing consumes completed bars or timestamped trades and reuses the
same strategy interface, sizing, stops, targets, accounting, and simulator as
backtesting. `MarketDataProvider` is the vendor-independent adapter contract;
`HistoricalReplayProvider` is implemented now. A future CME feed adapter can
implement `events(after=...)` to yield `BarEvent` or `TradeEvent` values, catch
up from the supplied cursor after reconnect, then continue the stream. Trade
events require a configured UTC bar interval; the aggregator never emits the
current incomplete bar. Bar events must explicitly set `is_complete=True` and
should provide `completed_at` (when the full bar became available); this is the
clock from which configured order latency is measured. `timestamp` identifies
the bar for chronological storage and execution.

Duplicate IDs/timestamps and out-of-order events are rejected and logged.
Expected missing bars are recorded; the engine does not invent replacement
bars. The JSON state is atomically checkpointed, and restart reconstructs the
execution state by replaying persisted completed bars. Resume with the exact
same strategy version, parameters, risk controls, and execution settings.
Each session is written under its own forward-only directory (for example
`runs\forward_tests\session-1`) with `forward_state.json`, `forward_events.jsonl`,
`forward_signals.csv`, and forward-only orders/trades/equity CSVs; it does not
write to historical backtest run directories.

Execution costs are shared `BacktestConfig` settings: commissions per
contract, estimated bid/ask spread in ticks, adverse slippage in ticks, and
latency in milliseconds. Market orders are accepted after configured latency
and fill at the first later bar open; protective stop/target behavior follows
the existing conservative OHLC assumptions. Limit touches use a deterministic
hash of the execution seed, order ID, and bar timestamp to decide whether a
touch fills, at `limit_touch_fill_probability`; set it below 1 to model
queue/fill uncertainty (forward sessions reject 1.0). Optional `max_contracts_per_fill` caps a market entry
fill per bar and cancels the unfilled remainder (an explicit, simplified
partial-fill assumption). The forward CLI defaults limit-touch probability to
0.5; use the same complete `BacktestConfig` for replay comparisons.

Replay a CSV through the forward pipeline and compare trades, orders, equity,
and key metrics with the batch backtest:

```powershell
python -m trading_research_engine.forward data\raw\nq_bars.csv `
  --output runs\forward_tests\nq-replay `
  --strategy-family ema_crossover `
  --parameters '{"fast_period":9,"slow_period":21}' `
  --expected-interval 1min
```

Run the Streamlit monitor after installing its optional dependencies:

```powershell
python -m pip install -e ".[dashboard]"
python -m streamlit run src\trading_research_engine\dashboard.py
```

The dashboard reads persisted hypothetical sessions and displays strategies,
signals, positions, marked-to-market equity/drawdown, and execution logs.
It does not connect to a data vendor or a broker.

### Requirements for a future real-time CME feed adapter

Before selecting a vendor, confirm CME market-data entitlement and any
redistribution/storage terms for NQ/MNQ, the needed top-of-book versus depth
level, professional/non-professional classification, real-time versus delayed
access, contract symbol/expiry conventions, and feed/API rate limits. The
adapter must document timestamp semantics and timezone, exchange sequence
numbers or stable event IDs, heartbeat/connection health, reconnect and
historical catch-up behavior, duplicate/out-of-order handling, correction and
cancellation messages, session calendars/holidays, and how quotes/trades map
to the engine's OHLCV contract and tick size. Validate the adapter against
vendor-provided samples and recorded feed data before unattended use. Obtain
legal approval for retention and storage of exchange data.

No vendor is selected or connected by this implementation. No live order
execution code or broker adapter is present; forward orders are always
hypothetical. Any later live-trading work requires a separate explicit
authorization and independent safety review, including exchange/broker
permissions, hard risk limits, kill switch, reconciliation, duplicate-order
protection, and operational monitoring.

## Input data contract

The importer accepts `.csv`, `.parquet`, and `.pq` files with these columns:

```text
timestamp,open,high,low,close,volume
```

Column names are case-insensitive. Timestamps must be parseable; timezone-aware
timestamps are converted to UTC and naive timestamps are interpreted as UTC.
Rows are sorted chronologically. Duplicate timestamps, null/non-finite values,
negative volume, and invalid OHLC relationships are rejected. Supply
`--expected-interval` (for example `1min` or `5min`) to report timestamp gaps.
Non-contract input columns are discarded so arbitrary precomputed or
future-derived fields cannot silently enter strategy history; indicators
should be computed causally from the completed history.
Gap detection does not apply a CME session calendar; closures and session
breaks can therefore be reported as gaps unless the input is segmented or
validated with suitable expectations.
Parquet files require the optional PyArrow dependency, installable with
`python -m pip install -e ".[parquet]"`.

## Run a backtest

Put your own historical bars in `data/raw/`, then run, for example:

```powershell
python -m trading_research_engine data\raw\nq_bars.csv `
  --instrument NQ `
  --fast-ema 9 `
  --slow-ema 21 `
  --stop-loss-points 20 `
  --take-profit-points 40 `
  --expected-interval 1min
```

Use `MNQ` for Micro E-mini Nasdaq-100 futures. By default the engine uses a
fixed one-contract position, no protective stop/target, a $2.50 commission per
contract per side, and one tick of slippage (a 0.25-point tick). Configure costs
to match the data source, broker, and assumptions you intend to study. To use
equity-risk sizing, provide both `--risk-fraction` and `--stop-loss-points`;
size is floored to whole contracts and capped at 10 by default.
Use `--periods-per-year` to set the annualization factor to suit the bar
frequency and session convention in your dataset.

The command creates a new directory under `runs/` unless `--output` specifies
another path. Exports include `trades.csv`, `equity.csv`, `orders.csv`, and
`run.json` (configuration, data-quality summary, performance metrics, and run
metadata including a configuration hash). Existing output directories are
never overwritten.

## Execution and accounting assumptions

- A strategy is called once per completed bar and receives a copy of history
  ending at that bar. It must return `Signal.HOLD`, `Signal.LONG`,
  `Signal.SHORT`, or `Signal.FLAT`. Signals are submitted and accepted at that
  close, then market orders fill at the next available bar's open. A flat
  signal exits an open position without reversing. A signal on the final bar
  is canceled because no later open exists.
- The simulator holds at most one position. An opposite signal closes and
  reverses at the next bar open. Orders are accepted immediately; queue
  position, exchange latency, partial fills, and rejected orders are not
  modelled.
- Market and stop orders fill with configured adverse slippage. A stop gapped
  through fills from the worse of the stop price and bar open, then applies
  slippage. A target is treated as a limit: it fills at its target price only,
  with no price improvement and no slippage. If one OHLC bar touches both stop
  and target and order cannot be established, the stop is assumed to fill
  first. Finer-grained data is required to resolve the true sequence.
- Stops and targets are distances in index points from the actual entry fill.
  Stop/target checks include the entry bar after its open fill. End-of-data
  liquidation uses the final close with market slippage; this is an explicit
  research convention rather than a guarantee that a real order could fill
  there.
- NQ is accounted at $20 per index point per contract; MNQ at $2. Commissions
  are charged on entry and exit. Realized P&L is net of both commissions;
  unrealized P&L includes the entry commission and is marked to each bar close.
  Equity is initial equity plus realized and unrealized P&L.
- Risk sizing uses `equity * risk_fraction / (stop_points * point_value)`,
  floored to whole contracts and capped by `max_position_size`. It does not
  reserve additional risk for slippage or commissions.
- Sharpe and Sortino ratios use bar-close equity percentage returns and assume
  zero risk-free return. `periods_per_year` defaults to 98,280 (252 sessions
  times 390 regular-session one-minute bars); override this in code for the
  actual bar frequency and annualization convention, or pass
  `--periods-per-year` on the CLI. Sortino downside deviation uses negative
  returns against a zero-return target. Ratios are `null` when the relevant
  return deviation is zero or unavailable; profit factor is `null` when there
  are no losing trades.
- `EMACrossover` uses only closes in the supplied completed history. Results
  are sensitive to the chosen data, configuration, and execution assumptions;
  this sample strategy is not an investment recommendation.

## Current scope and limitations

This version provides CSV/Parquet import, validation, the strategy contract,
four rule-based strategy families, one-position market execution, fixed/ATR/
structure stops, target exit modes, metrics, batch sweeps, chronological
validation, explicit OOS evaluation, walk-forward fold selection, a persisted
provider-independent forward/replay pipeline, and a Streamlit monitor. It does
not yet include a live CME provider, DuckDB experiment store,
CME holiday/session-aware gap validation, limit/stop entry orders, partial
position exits, or daily risk limits. Forward limit-touch and partial-fill
assumptions are simplified and must be calibrated against actual market data.
Walk-forward scoring is not a full multiple-hypothesis correction such as
Deflated Sharpe or White's Reality Check.
No real historical data was present in the project when this version was
implemented, so only synthetic tests can be run until you supply your data.
PyArrow is optional and the Parquet round-trip test is skipped when its engine
is not installed; this environment's Python 3.15 did not have a compatible
PyArrow wheel, so Parquet could not be exercised here.
