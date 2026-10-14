"""Provider-independent hypothetical forward testing and historical replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterator, Protocol, TypeAlias
from uuid import uuid4

import pandas as pd

from .data import validate_ohlcv
from .engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    IncrementalBacktestEngine,
)
from .strategies import Strategy


LIVE_ORDER_EXECUTION_ENABLED = False


@dataclass(frozen=True)
class BarEvent:
    event_id: str
    timestamp: str | pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_complete: bool = True
    completed_at: str | pd.Timestamp | None = None

    def as_bar(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(frozen=True)
class TradeEvent:
    event_id: str
    timestamp: str | pd.Timestamp
    price: float
    quantity: float


MarketEvent: TypeAlias = BarEvent | TradeEvent


class MarketDataProvider(Protocol):
    """A replay or real-time adapter yields uniquely identified, ordered events.

    A future real-time adapter should resume from ``after`` by first requesting
    any available historical catch-up, then continuing its live subscription.
    Provider callbacks must not submit broker orders.
    """

    def events(
        self, *, after: pd.Timestamp | None = None
    ) -> Iterator[MarketEvent]: ...


class HistoricalReplayProvider:
    """Yield validated complete bars as provider-shaped events."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars, _ = validate_ohlcv(bars)

    def events(
        self, *, after: pd.Timestamp | None = None
    ) -> Iterator[BarEvent]:
        for row in self.bars.itertuples(index=False):
            timestamp = pd.Timestamp(row.timestamp)
            if after is not None and timestamp <= _utc(after):
                continue
            yield BarEvent(
                event_id=f"bar:{timestamp.isoformat()}",
                timestamp=timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )


class TradeBarAggregator:
    """Aggregate trades into UTC-aligned bars; never emits the live partial bar."""

    def __init__(self, interval: str | pd.Timedelta) -> None:
        self.interval = pd.Timedelta(interval)
        if self.interval <= pd.Timedelta(0):
            raise ValueError("Trade aggregation interval must be positive.")
        self._bucket_start: pd.Timestamp | None = None
        self._ohlcv: dict[str, float] | None = None
        self._last_timestamp: pd.Timestamp | None = None

    def push(self, trade: TradeEvent) -> BarEvent | None:
        timestamp = _utc(trade.timestamp)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Trades must arrive in chronological order.")
        if not math.isfinite(trade.price) or trade.price <= 0:
            raise ValueError("Trade price must be finite and positive.")
        if not math.isfinite(trade.quantity) or trade.quantity < 0:
            raise ValueError("Trade quantity must be finite and non-negative.")
        self._last_timestamp = timestamp
        bucket_start = timestamp.floor(self.interval)
        if self._bucket_start is None:
            self._start_bucket(bucket_start, trade)
            return None
        if bucket_start < self._bucket_start:
            raise ValueError("Trades must not move to an earlier aggregation bucket.")
        if bucket_start == self._bucket_start:
            assert self._ohlcv is not None
            self._ohlcv["high"] = max(self._ohlcv["high"], trade.price)
            self._ohlcv["low"] = min(self._ohlcv["low"], trade.price)
            self._ohlcv["close"] = trade.price
            self._ohlcv["volume"] += trade.quantity
            return None

        completed = BarEvent(
            event_id=f"tradebar:{self._bucket_start.isoformat()}",
            timestamp=self._bucket_start,
            **self._ohlcv,
            completed_at=self._bucket_start + self.interval,
        )
        self._start_bucket(bucket_start, trade)
        return completed

    def state(self) -> dict[str, object]:
        return {
            "bucket_start": (
                self._bucket_start.isoformat()
                if self._bucket_start is not None
                else None
            ),
            "ohlcv": self._ohlcv,
            "last_timestamp": (
                self._last_timestamp.isoformat()
                if self._last_timestamp is not None
                else None
            ),
        }

    def restore(self, state: dict[str, object]) -> None:
        start = state.get("bucket_start")
        last = state.get("last_timestamp")
        self._bucket_start = _utc(start) if start else None
        self._last_timestamp = _utc(last) if last else None
        values = state.get("ohlcv")
        self._ohlcv = dict(values) if isinstance(values, dict) else None

    def _start_bucket(self, bucket_start: pd.Timestamp, trade: TradeEvent) -> None:
        self._bucket_start = bucket_start
        self._ohlcv = {
            "open": float(trade.price),
            "high": float(trade.price),
            "low": float(trade.price),
            "close": float(trade.price),
            "volume": float(trade.quantity),
        }


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


class ForwardTestSession:
    """Persisted forward session using the shared incremental execution engine.

    This class can create hypothetical orders only. It has no broker interface.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_directory: str | Path,
        strategy: Strategy,
        config: BacktestConfig | None = None,
        *,
        expected_interval: str | pd.Timedelta | None = None,
        trade_interval: str | pd.Timedelta | None = None,
        resume: bool = False,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.strategy = strategy
        self.config = config or BacktestConfig(limit_touch_fill_probability=0.5)
        if self.config.limit_touch_fill_probability >= 1:
            raise ValueError(
                "Forward testing requires limit_touch_fill_probability below 1 "
                "to model fill uncertainty."
            )
        self.expected_interval = (
            pd.Timedelta(expected_interval) if expected_interval is not None else None
        )
        if self.expected_interval is not None and self.expected_interval <= pd.Timedelta(0):
            raise ValueError("expected_interval must be positive.")
        self._trade_aggregator = (
            TradeBarAggregator(trade_interval) if trade_interval is not None else None
        )
        self._seen_event_ids: set[str] = set()
        self._completed_bar_times: set[str] = set()
        self._last_event_timestamp: pd.Timestamp | None = None
        self._bars: list[dict[str, object]] = []
        self._events: list[dict[str, object]] = []
        self._session_event_counter = 0
        self._finalized = False
        self._poisoned = False

        if resume:
            self._restore()
        else:
            self.output_directory.mkdir(parents=True, exist_ok=False)
            self._engine = IncrementalBacktestEngine(self.strategy, self.config)
            self._persist()

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        strategy: Strategy,
        config: BacktestConfig | None = None,
        **options: object,
    ) -> "ForwardTestSession":
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = root / f"forward-{stamp}-{uuid4().hex[:8]}"
        return cls(directory, strategy, config, **options)

    @property
    def last_event_timestamp(self) -> pd.Timestamp | None:
        return self._last_event_timestamp

    @property
    def active_strategy(self) -> dict[str, object]:
        return {
            "name": self.strategy.name,
            "version": self.strategy.version,
            "parameters": _json_safe(self.strategy.parameters),
        }

    @property
    def snapshot_events(self) -> list[dict[str, object]]:
        return [dict(event) for event in self._events]

    def process_event(self, event: MarketEvent) -> str:
        if self._poisoned:
            raise RuntimeError(
                "Session state may be inconsistent after a processing/persistence error; resume it."
            )
        if self._finalized:
            raise RuntimeError("Cannot process events after finalization.")
        if not event.event_id:
            raise ValueError("Every market event must have a non-empty event_id.")

        timestamp = _utc(event.timestamp)
        if event.event_id in self._seen_event_ids:
            self._record("duplicate_event_rejected", timestamp, source_event_id=event.event_id)
            self._persist()
            return "duplicate"
        if isinstance(event, TradeEvent) and self._trade_aggregator is None:
            raise ValueError(
                "TradeEvent input requires trade_interval when creating the session."
            )
        if (
            self._last_event_timestamp is not None
            and timestamp < self._last_event_timestamp
        ):
            self._seen_event_ids.add(event.event_id)
            self._record("out_of_order_event_rejected", timestamp, source_event_id=event.event_id)
            self._persist()
            return "out_of_order"

        self._seen_event_ids.add(event.event_id)
        self._last_event_timestamp = timestamp
        if isinstance(event, TradeEvent):
            assert self._trade_aggregator is not None
            try:
                completed = self._trade_aggregator.push(event)
            except ValueError as exc:
                self._record(
                    "market_data_rejected",
                    timestamp,
                    source_event_id=event.event_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._persist()
                return "rejected"
            if completed is None:
                self._record(
                    "incomplete_trade_bar_buffered",
                    timestamp,
                    source_event_id=event.event_id,
                )
                self._persist()
                return "buffered"
            status = self._process_complete_bar(completed)
            self._persist()
            return status

        if not event.is_complete:
            self._record(
                "incomplete_bar_ignored",
                timestamp,
                source_event_id=event.event_id,
            )
            self._persist()
            return "incomplete"
        status = self._process_complete_bar(event)
        self._persist()
        return status

    def run(
        self,
        provider: MarketDataProvider,
        *,
        finalize: bool = False,
    ) -> BacktestResult:
        for event in provider.events(after=self._last_event_timestamp):
            self.process_event(event)
        if finalize:
            self.finalize()
        return self.result()

    def finalize(self) -> None:
        if self._poisoned:
            raise RuntimeError("Cannot finalize a poisoned session.")
        if not self._finalized:
            self._engine.finalize()
            self._finalized = True
            timestamp = self._engine.last_timestamp or pd.Timestamp.now(tz="UTC")
            self._record("forward_session_finalized", timestamp)
            self._persist()

    def result(self) -> BacktestResult:
        return self._engine.result()

    def snapshot(self) -> dict[str, object]:
        result = self.result()
        latest_equity = result.equity.iloc[-1] if not result.equity.empty else None
        return {
            "session_type": "forward_test",
            "schema_version": self.SCHEMA_VERSION,
            "output_directory": str(self.output_directory),
            "active_strategy": self.active_strategy,
            "finalized": self._finalized,
            "poisoned": self._poisoned,
            "last_event_timestamp": (
                self._last_event_timestamp.isoformat()
                if self._last_event_timestamp is not None
                else None
            ),
            "active_position": self._engine.active_position,
            "realized_pnl": (
                float(latest_equity["realized_pnl"])
                if latest_equity is not None
                else 0.0
            ),
            "unrealized_pnl": (
                float(latest_equity["unrealized_pnl"])
                if latest_equity is not None
                else 0.0
            ),
            "metrics": result.metrics,
            "event_count": len(self._events),
            "configuration": {
                "backtest": asdict(self.config),
                "expected_interval": (
                    str(self.expected_interval) if self.expected_interval else None
                ),
                "trade_interval": (
                    str(self._trade_aggregator.interval)
                    if self._trade_aggregator is not None
                    else None
                ),
            },
        }

    def _process_complete_bar(self, event: BarEvent) -> str:
        timestamp = _utc(event.timestamp)
        bar_key = timestamp.isoformat()
        if bar_key in self._completed_bar_times:
            self._record(
                "duplicate_bar_rejected",
                timestamp,
                source_event_id=event.event_id,
            )
            return "duplicate_bar"

        try:
            normalized, _ = validate_ohlcv(pd.DataFrame([event.as_bar()]))
        except ValueError as exc:
            self._record(
                "market_data_rejected",
                timestamp,
                source_event_id=event.event_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return "rejected"

        previous = self._engine.last_timestamp
        if previous is not None and timestamp <= previous:
            self._record(
                "out_of_order_bar_rejected",
                timestamp,
                source_event_id=event.event_id,
            )
            return "out_of_order"
        if previous is not None and self.expected_interval is not None:
            delta = timestamp - previous
            if delta > self.expected_interval:
                missing_count = max(
                    int(delta // self.expected_interval) - 1, 0
                )
                self._record(
                    "missing_bars_detected",
                    timestamp,
                    after=previous.isoformat(),
                    missing_bars=missing_count,
                    interval_aligned=(delta % self.expected_interval == pd.Timedelta(0)),
                )

        before = len(self._engine.event_log)
        decision_time = (
            _utc(event.completed_at)
            if event.completed_at is not None
            else timestamp
        )
        try:
            self._engine.process_bar(
                normalized,
                signal_timestamp=decision_time,
            )
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
            self._engine = IncrementalBacktestEngine(self.strategy, self.config)
            for saved_bar in self._bars:
                self._engine.process_bar(saved_bar)
            self._record(
                "execution_error",
                timestamp,
                source_event_id=event.event_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._poisoned = True
            self._persist()
            raise
        self._completed_bar_times.add(bar_key)
        self._bars.append(
            {
                "timestamp": timestamp.isoformat(),
                "signal_timestamp": decision_time.isoformat(),
                "open": float(normalized.iloc[0]["open"]),
                "high": float(normalized.iloc[0]["high"]),
                "low": float(normalized.iloc[0]["low"]),
                "close": float(normalized.iloc[0]["close"]),
                "volume": float(normalized.iloc[0]["volume"]),
            }
        )
        for item in self._engine.event_log.iloc[before:].to_dict("records"):
            engine_event_id = str(item.pop("event_id"))
            self._record(
                str(item.pop("event_type")),
                _utc(str(item.pop("timestamp"))),
                engine_event_id=engine_event_id,
                **item,
            )
        self._record(
            "complete_bar_processed",
            timestamp,
            source_event_id=event.event_id,
        )
        return "accepted"

    def _record(
        self,
        event_type: str,
        timestamp: pd.Timestamp,
        **details: object,
    ) -> None:
        self._session_event_counter += 1
        self._events.append(
            {
                "event_id": f"F{self._session_event_counter:08d}",
                "event_type": event_type,
                "timestamp": timestamp.isoformat(),
                **details,
            }
        )

    def _persist(self) -> None:
        snapshot = {
            **self.snapshot(),
            "strategy_identity": self.active_strategy,
            "bars": self._bars,
            "seen_event_ids": sorted(self._seen_event_ids),
            "completed_bar_timestamps": sorted(self._completed_bar_times),
            "last_event_timestamp": (
                self._last_event_timestamp.isoformat()
                if self._last_event_timestamp is not None
                else None
            ),
            "trade_aggregator": (
                self._trade_aggregator.state()
                if self._trade_aggregator is not None
                else None
            ),
            "events": self._events,
        }
        self._atomic_json(self.output_directory / "forward_state.json", snapshot)
        self._atomic_jsonl(self.output_directory / "forward_events.jsonl", self._events)
        signals = pd.DataFrame(
            [event for event in self._events if event["event_type"] == "signal"]
        )
        self._atomic_csv(self.output_directory / "forward_signals.csv", signals)
        result = self.result()
        self._atomic_csv(self.output_directory / "forward_trades.csv", result.trades)
        self._atomic_csv(self.output_directory / "forward_equity.csv", result.equity)
        self._atomic_csv(self.output_directory / "forward_orders.csv", result.orders)

    def _restore(self) -> None:
        state_path = self.output_directory / "forward_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"No forward-test state found at {state_path}.")
        with state_path.open("r", encoding="utf-8") as file:
            state = json.load(file)
        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported forward-test state schema version.")
        expected_strategy = _canonical(self.active_strategy)
        if _canonical(state.get("strategy_identity")) != expected_strategy:
            raise ValueError("Strategy identity differs from the persisted session.")
        expected_config = _canonical(
            {
                "backtest": asdict(self.config),
                "expected_interval": (
                    str(self.expected_interval) if self.expected_interval else None
                ),
                "trade_interval": (
                    str(self._trade_aggregator.interval)
                    if self._trade_aggregator is not None
                    else None
                ),
            }
        )
        if _canonical(state.get("configuration")) != expected_config:
            raise ValueError("Execution/data configuration differs from persisted session.")

        self._bars = list(state.get("bars", []))
        self._seen_event_ids = set(state.get("seen_event_ids", []))
        self._completed_bar_times = set(state.get("completed_bar_timestamps", []))
        last = state.get("last_event_timestamp")
        self._last_event_timestamp = _utc(last) if last else None
        self._events = list(state.get("events", []))
        self._session_event_counter = len(self._events)
        self._finalized = bool(state.get("finalized", False))
        self._poisoned = bool(state.get("poisoned", False))
        aggregator_state = state.get("trade_aggregator")
        if aggregator_state is not None:
            if self._trade_aggregator is None:
                raise ValueError("Persisted session requires a trade aggregation interval.")
            self._trade_aggregator.restore(aggregator_state)
        self._engine = IncrementalBacktestEngine(self.strategy, self.config)
        for bar in self._bars:
            self._engine.process_bar(
                bar,
                signal_timestamp=bar.get("signal_timestamp"),
            )
        if self._finalized:
            self._engine.finalize()

    @staticmethod
    def _atomic_json(path: Path, payload: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(_json_safe(payload), file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)

    @staticmethod
    def _atomic_jsonl(path: Path, records: list[dict[str, object]]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(_json_safe(record), sort_keys=True, allow_nan=False))
                file.write("\n")
        os.replace(temporary, path)

    @staticmethod
    def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)


def _canonical(value: object) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def main() -> None:
    import argparse

    from .data import load_market_data
    from .experiments import STRATEGY_FACTORIES, _build_strategy

    parser = argparse.ArgumentParser(
        description="Replay historical data through the hypothetical forward-test pipeline."
    )
    parser.add_argument("data", type=Path, help="CSV or Parquet OHLCV input.")
    parser.add_argument("--output", required=True, type=Path, help="New forward-session directory.")
    parser.add_argument(
        "--strategy-family",
        choices=tuple(STRATEGY_FACTORIES),
        default="ema_crossover",
    )
    parser.add_argument(
        "--parameters",
        default="{}",
        help="JSON object of strategy parameters.",
    )
    parser.add_argument(
        "--execution-config",
        type=Path,
        help="Optional JSON object of BacktestConfig fields.",
    )
    parser.add_argument(
        "--expected-interval",
        help="Expected bar interval, e.g. 1min; gaps are recorded, not synthesized.",
    )
    args = parser.parse_args()

    parameters = json.loads(args.parameters)
    if not isinstance(parameters, dict):
        parser.error("--parameters must be a JSON object.")
    strategy = _build_strategy(args.strategy_family, parameters)
    config_values: dict[str, object] = {}
    if args.execution_config:
        with args.execution_config.open("r", encoding="utf-8") as file:
            config_values = json.load(file)
        if not isinstance(config_values, dict):
            parser.error("--execution-config must contain a JSON object.")
    config_values.setdefault("limit_touch_fill_probability", 0.5)
    config = BacktestConfig(**config_values)
    bars, _ = load_market_data(args.data, expected_interval=args.expected_interval)
    comparison = run_historical_replay(
        bars,
        strategy,
        args.output,
        config,
        expected_interval=args.expected_interval,
    )
    print(
        json.dumps(
            {
                "session_directory": str(comparison.session_directory.resolve()),
                "matches_backtest": comparison.matches_backtest,
                "differences": comparison.differences,
                "forward_metrics": comparison.forward_result.metrics,
                "backtest_metrics": comparison.backtest_result.metrics,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    if not comparison.matches_backtest:
        raise SystemExit(1)


@dataclass(frozen=True)
class ReplayComparison:
    session_directory: Path
    matches_backtest: bool
    differences: tuple[str, ...]
    forward_result: BacktestResult
    backtest_result: BacktestResult


def run_historical_replay(
    bars: pd.DataFrame,
    strategy: Strategy,
    output_directory: str | Path,
    config: BacktestConfig | None = None,
    *,
    expected_interval: str | pd.Timedelta | None = None,
) -> ReplayComparison:
    """Replay historical complete bars through the forward path and compare."""
    provider = HistoricalReplayProvider(bars)
    session = ForwardTestSession(
        output_directory,
        strategy,
        config,
        expected_interval=expected_interval,
    )
    forward_result = session.run(provider, finalize=True)
    historical_bars, quality = validate_ohlcv(bars, expected_interval)
    backtest_result = BacktestEngine(config or session.config).run(
        historical_bars, strategy, data_quality=quality
    )
    differences: list[str] = []
    for field in ("trades", "equity", "orders"):
        try:
            pd.testing.assert_frame_equal(
                getattr(forward_result, field),
                getattr(backtest_result, field),
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-9,
            )
        except AssertionError as exc:
            differences.append(f"{field}: {exc}")
    for metric in (
        "trade_count",
        "ending_equity",
        "max_drawdown_dollars",
        "max_drawdown_percent",
    ):
        if forward_result.metrics.get(metric) != backtest_result.metrics.get(metric):
            differences.append(
                f"metrics.{metric}: forward={forward_result.metrics.get(metric)} "
                f"backtest={backtest_result.metrics.get(metric)}"
            )
    return ReplayComparison(
        session_directory=Path(output_directory),
        matches_backtest=not differences,
        differences=tuple(differences),
        forward_result=forward_result,
        backtest_result=backtest_result,
    )


if __name__ == "__main__":
    main()
