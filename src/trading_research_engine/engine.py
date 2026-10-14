"""Chronological OHLCV backtest engine with conservative bar-based fills."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from typing import Any

import numpy as np
import pandas as pd

from .data import DataQualityReport, validate_ohlcv
from .strategies import Signal, Strategy


POINT_VALUES = {"NQ": 20.0, "MNQ": 2.0}


@dataclass(frozen=True)
class BacktestConfig:
    instrument: str = "NQ"
    initial_equity: float = 100_000.0
    quantity: int = 1
    risk_fraction: float | None = None
    max_position_size: int = 10
    stop_mode: str = "fixed"
    stop_loss_points: float | None = None
    stop_atr_period: int = 14
    stop_atr_multiple: float = 2.0
    structure_lookback: int = 20
    take_profit_mode: str = "fixed"
    take_profit_points: float | None = None
    risk_reward_multiple: float = 2.0
    commission_per_contract: float = 2.50
    slippage_ticks: float = 1.0
    spread_ticks: float = 0.0
    latency_ms: int = 0
    limit_touch_fill_probability: float = 1.0
    max_contracts_per_fill: int | None = None
    execution_seed: int = 0
    tick_size: float = 0.25
    periods_per_year: int = 98_280

    def __post_init__(self) -> None:
        if self.instrument not in POINT_VALUES:
            raise ValueError("instrument must be 'NQ' or 'MNQ'.")
        numeric_values = (
            self.initial_equity,
            self.commission_per_contract,
            self.slippage_ticks,
            self.spread_ticks,
            self.tick_size,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Configuration amounts must be finite.")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive.")
        if (
            not isinstance(self.quantity, int)
            or not isinstance(self.max_position_size, int)
            or self.quantity < 1
            or self.max_position_size < 1
        ):
            raise ValueError("quantity and max_position_size must be positive integers.")
        if self.risk_fraction is not None and not 0 < self.risk_fraction <= 1:
            raise ValueError("risk_fraction must be greater than 0 and at most 1.")
        if self.stop_mode not in ("fixed", "atr", "structure"):
            raise ValueError("stop_mode must be 'fixed', 'atr', or 'structure'.")
        if self.take_profit_mode not in ("fixed", "risk_reward", "none"):
            raise ValueError("take_profit_mode must be 'fixed', 'risk_reward', or 'none'.")
        if self.risk_fraction is not None and self.stop_mode == "fixed" and self.stop_loss_points is None:
            raise ValueError("Fixed-stop risk sizing requires stop_loss_points.")
        if self.stop_atr_period < 2 or self.structure_lookback < 1:
            raise ValueError("ATR period and structure lookback must be positive.")
        if not math.isfinite(self.stop_atr_multiple) or self.stop_atr_multiple <= 0:
            raise ValueError("stop_atr_multiple must be finite and positive.")
        if not math.isfinite(self.risk_reward_multiple) or self.risk_reward_multiple <= 0:
            raise ValueError("risk_reward_multiple must be finite and positive.")
        for name in ("stop_loss_points", "take_profit_points"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive when configured.")
        if self.take_profit_mode == "risk_reward" and self.stop_mode == "fixed" and self.stop_loss_points is None:
            raise ValueError("Risk-reward target requires a configured stop.")
        if (
            self.commission_per_contract < 0
            or self.slippage_ticks < 0
            or self.spread_ticks < 0
        ):
            raise ValueError("Commissions, spread, and slippage cannot be negative.")
        if not isinstance(self.latency_ms, int) or self.latency_ms < 0:
            raise ValueError("latency_ms must be a non-negative integer.")
        if not isinstance(self.execution_seed, int):
            raise ValueError("execution_seed must be an integer.")
        if not 0 <= self.limit_touch_fill_probability <= 1:
            raise ValueError("limit_touch_fill_probability must be between 0 and 1.")
        if (
            self.max_contracts_per_fill is not None
            and (
                not isinstance(self.max_contracts_per_fill, int)
                or self.max_contracts_per_fill < 1
            )
        ):
            raise ValueError("max_contracts_per_fill must be a positive integer.")
        if self.tick_size <= 0 or self.periods_per_year < 1:
            raise ValueError("tick_size and periods_per_year must be positive.")
        if not isinstance(self.periods_per_year, int):
            raise ValueError("periods_per_year must be a positive integer.")


def calculate_position_size(
    equity: float,
    config: BacktestConfig,
    stop_distance: float | None = None,
) -> int:
    """Size contracts by fixed quantity or equity risk divided by stop risk."""
    if equity <= 0:
        return 0
    if config.risk_fraction is None:
        return min(config.quantity, config.max_position_size)
    distance = stop_distance if stop_distance is not None else config.stop_loss_points
    if distance is None or distance <= 0:
        return 0
    risk_per_contract = distance * POINT_VALUES[config.instrument]
    quantity = math.floor(equity * config.risk_fraction / risk_per_contract)
    return min(quantity, config.max_position_size)


@dataclass
class _Order:
    order_id: str
    order_type: str
    side: str
    quantity: int
    submitted_at: pd.Timestamp
    accepted_at: pd.Timestamp
    reason: str
    status: str = "accepted"
    filled_at: pd.Timestamp | None = None
    reference_price: float | None = None
    fill_price: float | None = None
    commission: float = 0.0
    cancel_reason: str | None = None
    filled_quantity: int = 0

    def as_record(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "order_type": self.order_type,
            "side": self.side,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "submitted_at": self.submitted_at.isoformat(),
            "accepted_at": self.accepted_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at is not None else None,
            "reference_price": self.reference_price,
            "fill_price": self.fill_price,
            "commission": self.commission,
            "status": self.status,
            "reason": self.reason,
            "cancel_reason": self.cancel_reason,
        }


@dataclass
class _Position:
    side: int
    quantity: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    entry_commission: float
    entry_slippage: float
    stop_price: float | None
    target_price: float | None
    stop_order: _Order | None
    target_order: _Order | None
    entry_order_id: str


def _equity_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    periods_per_year: int,
) -> dict[str, float | int | None]:
    if equity.empty:
        maximum_drawdown = maximum_drawdown_pct = 0.0
        returns = pd.Series(dtype=float)
    else:
        values = equity["equity"].astype(float)
        peaks = values.cummax()
        drawdowns = values - peaks
        drawdowns_pct = drawdowns.div(peaks.where(peaks != 0))
        maximum_drawdown = float(-drawdowns.min())
        maximum_drawdown_pct = float(-drawdowns_pct.min() * 100)
        returns = values.pct_change().dropna()

    if trades.empty:
        win_rate = expectancy = average_winner = average_loser = 0.0
        profit_factor: float | None = None
    else:
        net = trades["net_pnl"].astype(float)
        winners, losers = net[net > 0], net[net < 0]
        win_rate = float((net > 0).mean() * 100)
        expectancy = float(net.mean())
        average_winner = float(winners.mean()) if not winners.empty else 0.0
        average_loser = float(losers.mean()) if not losers.empty else 0.0
        loss_sum = float(-losers.sum())
        profit_factor = float(winners.sum() / loss_sum) if loss_sum > 0 else None

    standard_deviation = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside_deviation = (
        float(np.sqrt(np.square(returns.clip(upper=0)).mean()))
        if not returns.empty
        else 0.0
    )
    sharpe = (
        float(returns.mean() / standard_deviation * math.sqrt(periods_per_year))
        if standard_deviation > 0
        else None
    )
    sortino = (
        float(returns.mean() / downside_deviation * math.sqrt(periods_per_year))
        if downside_deviation > 0
        else None
    )

    return {
        "trade_count": int(len(trades)),
        "win_rate_percent": win_rate,
        "expectancy_per_trade": expectancy,
        "profit_factor": profit_factor,
        "average_winner": average_winner,
        "average_loser": average_loser,
        "max_drawdown_dollars": maximum_drawdown,
        "max_drawdown_percent": maximum_drawdown_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "ending_equity": float(equity["equity"].iloc[-1]) if not equity.empty else None,
    }


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict[str, float | int | None]
    configuration: dict[str, Any]
    metadata: dict[str, Any]


class BacktestEngine:
    """Runs one strategy against validated, chronologically processed OHLCV bars."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        market_data: pd.DataFrame,
        strategy: Strategy,
        data_quality: DataQualityReport | None = None,
        start_time: str | pd.Timestamp | None = None,
        end_time: str | pd.Timestamp | None = None,
    ) -> BacktestResult:
        bars, validated_quality = validate_ohlcv(market_data)
        quality = data_quality or validated_quality
        start = self._as_utc(start_time) if start_time is not None else bars["timestamp"].iloc[0]
        end = self._as_utc(end_time) if end_time is not None else bars["timestamp"].iloc[-1] + pd.Timedelta(1, "ns")
        if end <= start:
            raise ValueError("end_time must be later than start_time.")
        self._orders: list[_Order] = []
        self._trades: list[dict[str, object]] = []
        self._position: _Position | None = None
        self._realized_pnl = 0.0
        self._order_counter = 0
        self._events: list[dict[str, object]] = []
        self._event_counter = 0
        self._pending: dict[str, Any] | None = None
        equity_records: list[dict[str, object]] = []
        last_bar: pd.Series | None = None

        for index, bar in bars.iterrows():
            timestamp = bar["timestamp"]
            if timestamp < start:
                continue
            if timestamp >= end:
                break
            last_bar = bar
            self._current_timestamp = timestamp
            self._process_open(bar)
            self._process_intrabar(bar)

            unrealized = self._unrealized(float(bar["close"]))
            equity_records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "realized_pnl": self._realized_pnl,
                    "unrealized_pnl": unrealized,
                    "equity": self.config.initial_equity + self._realized_pnl + unrealized,
                }
            )

            # A strategy receives a copy of the completed prefix only. It cannot
            # inspect future bars or mutate the engine's source data.
            history = bars.iloc[: index + 1].copy(deep=True)
            signal = strategy.on_bar(history)
            if not isinstance(signal, Signal):
                raise TypeError("Strategy.on_bar must return a Signal enum value.")
            self._record_event(
                "signal",
                timestamp,
                signal=signal.value,
                close=float(bar["close"]),
            )
            if signal != Signal.HOLD and not self._has_signal_position(signal):
                if self._pending is not None:
                    self._record_event(
                        "order_rejected",
                        timestamp,
                        reason="another_order_is_pending",
                        signal=signal.value,
                    )
                else:
                    self._submit_signal(
                        signal,
                        timestamp,
                        float(equity_records[-1]["equity"]),
                        history,
                    )

        if last_bar is None:
            raise ValueError("Evaluation period contains no market bars.")
        if self._pending is not None:
            self._cancel(self._pending["entry_order"], "end_of_data_before_fill")
            if self._pending["exit_order"] is not None:
                self._cancel(self._pending["exit_order"], "end_of_data_before_fill")
            self._pending = None
        if self._position is not None:
            timestamp = last_bar["timestamp"]
            self._current_timestamp = timestamp
            order = self._new_order(
                "market",
                "sell" if self._position.side == 1 else "buy",
                self._position.quantity,
                timestamp,
                "end_of_data",
            )
            fill = self._fill(order, float(last_bar["close"]), timestamp)
            self._close_position(fill, timestamp, "end_of_data", order)
            equity_records[-1]["realized_pnl"] = self._realized_pnl
            equity_records[-1]["unrealized_pnl"] = 0.0
            equity_records[-1]["equity"] = self.config.initial_equity + self._realized_pnl

        equity = pd.DataFrame(equity_records)
        if not equity.empty:
            peaks = equity["equity"].cummax()
            equity["drawdown_dollars"] = equity["equity"] - peaks
            equity["drawdown_percent"] = equity["drawdown_dollars"].div(peaks) * 100
        trades = pd.DataFrame(self._trades)
        orders = pd.DataFrame([order.as_record() for order in self._orders])

        configuration: dict[str, Any] = {
            "backtest": asdict(self.config),
            "strategy": {
                "name": strategy.name,
                "version": strategy.version,
                "parameters": strategy.parameters,
            },
            "data_quality": {
                "row_count": quality.row_count,
                "duplicate_count": quality.duplicate_count,
                "missing_bar_count": quality.missing_bar_count,
                "gaps": list(quality.gaps),
            },
            "evaluation_period": {
                "start": start.isoformat(),
                "end_exclusive": end.isoformat(),
            },
        }
        canonical = json.dumps(configuration, sort_keys=True, default=str, separators=(",", ":"))
        data_digest = hashlib.sha256()
        data_digest.update(b"timestamp:int64-utc-ns;ohlcv:float64;")
        data_digest.update(
            bars["timestamp"].array.asi8.astype("<i8", copy=False).tobytes()
        )
        for column in ("open", "high", "low", "close", "volume"):
            data_digest.update(np.asarray(bars[column], dtype="<f8").tobytes())
        metadata = {
            "engine_version": "0.1.0",
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "data_sha256": data_digest.hexdigest(),
        }
        return BacktestResult(
            trades=trades,
            equity=equity,
            orders=orders,
            metrics=_equity_metrics(equity, trades, self.config.periods_per_year),
            configuration=configuration,
            metadata=metadata,
        )

    @staticmethod
    def _as_utc(value: str | pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _record_event(
        self,
        event_type: str,
        timestamp: pd.Timestamp,
        **details: object,
    ) -> None:
        self._event_counter += 1
        self._events.append(
            {
                "event_id": f"E{self._event_counter:08d}",
                "event_type": event_type,
                "timestamp": timestamp.isoformat(),
                **details,
            }
        )

    def _new_order(
        self,
        order_type: str,
        side: str,
        quantity: int,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> _Order:
        self._order_counter += 1
        accepted_at = timestamp + (
            pd.Timedelta(milliseconds=self.config.latency_ms)
            if order_type == "market" and reason != "end_of_data"
            else pd.Timedelta(0)
        )
        order = _Order(
            order_id=f"O{self._order_counter:06d}",
            order_type=order_type,
            side=side,
            quantity=quantity,
            submitted_at=timestamp,
            accepted_at=accepted_at,
            reason=reason,
        )
        self._orders.append(order)
        self._record_event(
            "order_submitted",
            timestamp,
            order_id=order.order_id,
            order_type=order.order_type,
            side=order.side,
            quantity=order.quantity,
            reason=reason,
        )
        self._record_event(
            "order_accepted",
            accepted_at,
            order_id=order.order_id,
            order_type=order.order_type,
        )
        return order

    def _submit_signal(
        self,
        signal: Signal,
        timestamp: pd.Timestamp,
        equity: float,
        history: pd.DataFrame,
    ) -> None:
        if signal == Signal.FLAT:
            if self._position is None:
                return
            exit_order = self._new_order(
                "market",
                "sell" if self._position.side == 1 else "buy",
                self._position.quantity,
                timestamp,
                "signal_flat",
            )
            self._pending = {
                "side": 0,
                "quantity": 0,
                "entry_order": None,
                "exit_order": exit_order,
            }
            return

        side = 1 if signal == Signal.LONG else -1
        stop_distance, stop_anchor = self._planned_stop(history, side)
        if self.config.stop_mode != "fixed" and stop_distance is None:
            self._record_event(
                "order_rejected",
                timestamp,
                reason="stop_distance_unavailable",
                signal=signal.value,
            )
            return
        if self.config.risk_fraction is not None and stop_distance is None:
            self._record_event(
                "order_rejected",
                timestamp,
                reason="risk_sizing_requires_stop",
                signal=signal.value,
            )
            return
        quantity = calculate_position_size(equity, self.config, stop_distance)
        if quantity == 0:
            self._record_event(
                "order_rejected",
                timestamp,
                reason="position_size_is_zero",
                signal=signal.value,
            )
            return

        exit_order = None
        if self._position is not None:
            exit_order = self._new_order(
                "market",
                "sell" if self._position.side == 1 else "buy",
                self._position.quantity,
                timestamp,
                "signal_reversal",
            )
        entry_order = self._new_order(
            "market",
            "buy" if side == 1 else "sell",
            quantity,
            timestamp,
            "strategy_signal",
        )
        self._pending = {
            "side": side,
            "quantity": quantity,
            "stop_distance": stop_distance,
            "stop_anchor": stop_anchor,
            "entry_order": entry_order,
            "exit_order": exit_order,
        }

    def _process_open(self, bar: pd.Series) -> None:
        timestamp, opening = bar["timestamp"], float(bar["open"])
        self._process_gap_protection(bar)
        if self._pending is None:
            return

        pending = self._pending
        exit_order = pending["exit_order"]
        entry_order = pending["entry_order"]
        due_orders = [
            order
            for order in (exit_order, entry_order)
            if order is not None
        ]
        if due_orders and timestamp < max(order.accepted_at for order in due_orders):
            return
        if exit_order is not None:
            if self._position is not None:
                fill = self._fill(exit_order, opening, timestamp)
                self._close_position(fill, timestamp, exit_order.reason, exit_order)
            else:
                self._cancel(exit_order, "position_already_closed")

        if self._position is None and entry_order is not None:
            order = entry_order
            estimated_fill = opening + (
                (
                    self.config.slippage_ticks
                    + self.config.spread_ticks / 2
                )
                * self.config.tick_size
                if order.side == "buy"
                else -(
                    self.config.slippage_ticks
                    + self.config.spread_ticks / 2
                )
                * self.config.tick_size
            )
            stop_anchor = pending["stop_anchor"]
            if stop_anchor is not None and (
                (pending["side"] == 1 and stop_anchor >= estimated_fill)
                or (pending["side"] == -1 and stop_anchor <= estimated_fill)
            ):
                self._cancel(order, "entry_gap_invalidates_structure_stop")
                self._pending = None
                return
            fill_quantity = order.quantity
            if self.config.max_contracts_per_fill is not None:
                fill_quantity = min(fill_quantity, self.config.max_contracts_per_fill)
            fill = self._fill(order, opening, timestamp, quantity=fill_quantity)
            if fill_quantity < order.quantity:
                order.cancel_reason = "partial_fill_remainder_cancelled"
                self._record_event(
                    "order_remainder_canceled",
                    timestamp,
                    order_id=order.order_id,
                    unfilled_quantity=order.quantity - fill_quantity,
                    reason=order.cancel_reason,
                )
            self._open_position(
                pending["side"],
                fill_quantity,
                timestamp,
                fill,
                order,
                pending["stop_distance"],
                pending["stop_anchor"],
            )
        elif pending["entry_order"] is not None:
            self._cancel(pending["entry_order"], "position_already_open")
        self._pending = None

    def _process_gap_protection(self, bar: pd.Series) -> None:
        position = self._position
        if position is None:
            return
        opening, timestamp = float(bar["open"]), bar["timestamp"]
        stop_hit = position.stop_price is not None and (
            (position.side == 1 and opening <= position.stop_price)
            or (position.side == -1 and opening >= position.stop_price)
        )
        target_hit = position.target_price is not None and (
            (position.side == 1 and opening >= position.target_price)
            or (position.side == -1 and opening <= position.target_price)
        )
        if target_hit and position.target_order is not None:
            target_hit = self._limit_fill_available(
                position.target_order, timestamp
            )
        if stop_hit:
            order = position.stop_order
            assert order is not None
            fill = self._fill(order, opening, timestamp)
            self._close_position(fill, timestamp, "stop_loss_gap", order)
        elif target_hit:
            order = position.target_order
            assert order is not None
            fill = self._fill(order, position.target_price, timestamp, apply_slippage=False)
            self._close_position(fill, timestamp, "take_profit_gap", order)

    def _process_intrabar(self, bar: pd.Series) -> None:
        position = self._position
        if position is None:
            return
        opening, high, low = (float(bar[column]) for column in ("open", "high", "low"))
        timestamp = bar["timestamp"]
        stop_hit = position.stop_price is not None and (
            (position.side == 1 and low <= position.stop_price)
            or (position.side == -1 and high >= position.stop_price)
        )
        target_hit = position.target_price is not None and (
            (position.side == 1 and high >= position.target_price)
            or (position.side == -1 and low <= position.target_price)
        )
        target_touched = target_hit
        if target_hit and position.target_order is not None:
            target_hit = self._limit_fill_available(
                position.target_order, timestamp
            )

        # OHLC bars cannot establish the order of intrabar touches. Stop-first
        # is deliberately conservative when both protective levels are touched.
        if stop_hit:
            order = position.stop_order
            assert order is not None and position.stop_price is not None
            reference = (
                min(opening, position.stop_price)
                if position.side == 1
                else max(opening, position.stop_price)
            )
            fill = self._fill(order, reference, timestamp)
            reason = "stop_loss_ambiguous" if target_touched else "stop_loss"
            self._close_position(fill, timestamp, reason, order)
        elif target_hit:
            order = position.target_order
            assert order is not None and position.target_price is not None
            fill = self._fill(
                order, position.target_price, timestamp, apply_slippage=False
            )
            self._close_position(fill, timestamp, "take_profit", order)

    def _open_position(
        self,
        side: int,
        quantity: int,
        timestamp: pd.Timestamp,
        fill: float,
        entry_order: _Order,
        stop_distance: float | None,
        stop_anchor: float | None,
    ) -> None:
        commission = self.config.commission_per_contract * quantity
        stop_price = stop_anchor
        if stop_price is None and stop_distance is not None:
            stop_price = fill - side * stop_distance
        if self.config.take_profit_mode == "none":
            target_distance = None
        elif self.config.take_profit_mode == "risk_reward":
            actual_risk = abs(fill - stop_price) if stop_price is not None else None
            target_distance = (
                actual_risk * self.config.risk_reward_multiple
                if actual_risk is not None
                else None
            )
        else:
            target_distance = self.config.take_profit_points
        target_price = (
            fill + side * target_distance
            if target_distance is not None
            else None
        )
        stop_order = (
            self._new_order(
                "stop",
                "sell" if side == 1 else "buy",
                quantity,
                timestamp,
                "protective_stop",
            )
            if stop_price is not None
            else None
        )
        target_order = (
            self._new_order(
                "limit",
                "sell" if side == 1 else "buy",
                quantity,
                timestamp,
                "protective_target",
            )
            if target_price is not None
            else None
        )
        self._position = _Position(
            side=side,
            quantity=quantity,
            signal_time=entry_order.submitted_at,
            entry_time=timestamp,
            entry_price=fill,
            entry_commission=commission,
            entry_slippage=abs(float(entry_order.reference_price) - fill),
            stop_price=stop_price,
            target_price=target_price,
            stop_order=stop_order,
            target_order=target_order,
            entry_order_id=entry_order.order_id,
        )
        self._record_event(
            "position_opened",
            timestamp,
            side="long" if side == 1 else "short",
            quantity=quantity,
            entry_price=fill,
            stop_price=stop_price,
            target_price=target_price,
            order_id=entry_order.order_id,
        )

    def _planned_stop(
        self, history: pd.DataFrame, side: int
    ) -> tuple[float | None, float | None]:
        if self.config.stop_mode == "fixed":
            return self.config.stop_loss_points, None
        if self.config.stop_mode == "structure":
            recent = history.tail(self.config.structure_lookback)
            anchor = (
                float(recent["low"].min())
                if side == 1
                else float(recent["high"].max())
            )
            close = float(history["close"].iloc[-1])
            distance = close - anchor if side == 1 else anchor - close
            return (distance, anchor) if distance > 0 else (None, anchor)

        previous_close = history["close"].shift(1)
        true_range = pd.concat(
            [
                history["high"] - history["low"],
                (history["high"] - previous_close).abs(),
                (history["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        value = true_range.ewm(
            alpha=1 / self.config.stop_atr_period,
            adjust=False,
            min_periods=self.config.stop_atr_period,
        ).mean().iloc[-1]
        if pd.isna(value):
            return None, None
        return float(value) * self.config.stop_atr_multiple, None

    def _close_position(
        self,
        fill: float,
        timestamp: pd.Timestamp,
        reason: str,
        exit_order: _Order,
    ) -> None:
        position = self._position
        if position is None:
            return
        gross = (
            (fill - position.entry_price)
            * position.side
            * position.quantity
            * POINT_VALUES[self.config.instrument]
        )
        commission = position.entry_commission + exit_order.commission
        net = gross - commission
        self._realized_pnl += net
        exit_slippage = abs(float(exit_order.reference_price) - fill)
        self._trades.append(
            {
                "signal_time": position.signal_time.isoformat(),
                "entry_time": position.entry_time.isoformat(),
                "exit_time": timestamp.isoformat(),
                "side": "long" if position.side == 1 else "short",
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "exit_price": fill,
                "gross_pnl": gross,
                "commission": commission,
                "slippage_points": position.entry_slippage + exit_slippage,
                "net_pnl": net,
                "exit_reason": reason,
                "entry_order_id": position.entry_order_id,
                "exit_order_id": exit_order.order_id,
            }
        )
        self._record_event(
            "position_closed",
            timestamp,
            side="long" if position.side == 1 else "short",
            quantity=position.quantity,
            exit_price=fill,
            net_pnl=net,
            reason=reason,
            order_id=exit_order.order_id,
        )
        self._record_event(
            "exit",
            timestamp,
            exit_reason=reason,
            net_pnl=net,
            order_id=exit_order.order_id,
        )
        self._cancel(position.stop_order, "position_closed")
        self._cancel(position.target_order, "position_closed")
        self._position = None

    def _fill(
        self,
        order: _Order,
        reference_price: float,
        timestamp: pd.Timestamp,
        apply_slippage: bool = True,
        quantity: int | None = None,
    ) -> float:
        filled_quantity = quantity if quantity is not None else order.quantity
        impact = (
            (self.config.slippage_ticks + self.config.spread_ticks / 2)
            * self.config.tick_size
            if apply_slippage
            else 0.0
        )
        fill = reference_price + impact if order.side == "buy" else reference_price - impact
        order.filled_quantity += filled_quantity
        order.status = (
            "filled"
            if order.filled_quantity >= order.quantity
            else "partially_filled"
        )
        order.filled_at = timestamp
        order.reference_price = reference_price
        order.fill_price = fill
        order.commission = self.config.commission_per_contract * filled_quantity
        self._record_event(
            "fill",
            timestamp,
            order_id=order.order_id,
            order_type=order.order_type,
            side=order.side,
            quantity=filled_quantity,
            reference_price=reference_price,
            fill_price=fill,
            commission=order.commission,
        )
        return fill

    def _cancel(self, order: _Order | None, reason: str) -> None:
        if order is not None and order.status in ("accepted", "partially_filled"):
            order.status = "canceled"
            order.cancel_reason = reason
            self._record_event(
                "order_canceled",
                getattr(self, "_current_timestamp", order.accepted_at),
                order_id=order.order_id,
                reason=reason,
            )

    def _limit_fill_available(
        self, order: _Order, timestamp: pd.Timestamp
    ) -> bool:
        probability = self.config.limit_touch_fill_probability
        if probability >= 1:
            return True
        if probability <= 0:
            self._record_event(
                "limit_touch_unfilled",
                timestamp,
                order_id=order.order_id,
                fill_probability=probability,
            )
            return False
        key = (
            f"{self.config.execution_seed}:{order.order_id}:"
            f"{timestamp.value}"
        ).encode("utf-8")
        sample = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        filled = sample < probability
        if not filled:
            self._record_event(
                "limit_touch_unfilled",
                timestamp,
                order_id=order.order_id,
                fill_probability=probability,
            )
        return filled

    def _unrealized(self, mark: float) -> float:
        position = self._position
        if position is None:
            return 0.0
        gross = (
            (mark - position.entry_price)
            * position.side
            * position.quantity
            * POINT_VALUES[self.config.instrument]
        )
        return gross - position.entry_commission

    def _has_signal_position(self, signal: Signal) -> bool:
        if signal == Signal.FLAT:
            return self._position is None
        if self._position is None:
            return False
        return (signal == Signal.LONG and self._position.side == 1) or (
            signal == Signal.SHORT and self._position.side == -1
        )


class IncrementalBacktestEngine(BacktestEngine):
    """Stateful bar processor sharing the batch engine's execution implementation."""

    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.strategy = strategy
        self._orders = []
        self._trades = []
        self._position = None
        self._realized_pnl = 0.0
        self._order_counter = 0
        self._events = []
        self._event_counter = 0
        self._pending = None
        self._history: list[pd.Series] = []
        self._equity_records: list[dict[str, object]] = []
        self._finalized = False
        self._current_timestamp: pd.Timestamp | None = None

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        return self._history[-1]["timestamp"] if self._history else None

    @property
    def event_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._events).copy()

    @property
    def active_position(self) -> dict[str, object] | None:
        position = self._position
        if position is None:
            return None
        return {
            "side": "long" if position.side == 1 else "short",
            "quantity": position.quantity,
            "entry_time": position.entry_time.isoformat(),
            "entry_price": position.entry_price,
            "stop_price": position.stop_price,
            "target_price": position.target_price,
            "unrealized_pnl": self._unrealized(
                float(self._history[-1]["close"])
            ),
        }

    def process_bar(
        self,
        bar: pd.DataFrame | pd.Series | dict[str, object],
        *,
        signal_timestamp: str | pd.Timestamp | None = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot process bars after this session is finalized.")
        frame = bar.to_frame().T if isinstance(bar, pd.Series) else pd.DataFrame([bar]) if isinstance(bar, dict) else bar
        normalized, _ = validate_ohlcv(frame)
        if len(normalized) != 1:
            raise ValueError("Incremental processing accepts exactly one bar at a time.")
        completed = normalized.iloc[0]
        if self.last_timestamp is not None and completed["timestamp"] <= self.last_timestamp:
            raise ValueError("Bars must be processed in strictly chronological order.")

        decision_time = (
            self._as_utc(signal_timestamp)
            if signal_timestamp is not None
            else completed["timestamp"]
        )
        self._history.append(completed)
        self._current_timestamp = completed["timestamp"]
        self._process_open(completed)
        self._process_intrabar(completed)
        history = pd.DataFrame(self._history).reset_index(drop=True)
        unrealized = self._unrealized(float(completed["close"]))
        self._equity_records.append(
            {
                "timestamp": completed["timestamp"].isoformat(),
                "realized_pnl": self._realized_pnl,
                "unrealized_pnl": unrealized,
                "equity": self.config.initial_equity + self._realized_pnl + unrealized,
            }
        )

        signal = self.strategy.on_bar(history.copy(deep=True))
        if not isinstance(signal, Signal):
            raise TypeError("Strategy.on_bar must return a Signal enum value.")
        self._record_event(
            "signal",
            decision_time,
            signal=signal.value,
            close=float(completed["close"]),
        )
        if signal != Signal.HOLD and not self._has_signal_position(signal):
            if self._pending is not None:
                self._record_event(
                    "order_rejected",
                    completed["timestamp"],
                    reason="another_order_is_pending",
                    signal=signal.value,
                )
            else:
                self._submit_signal(
                    signal,
                    decision_time,
                    float(self._equity_records[-1]["equity"]),
                    history,
                )

    def finalize(self) -> None:
        if self._finalized:
            return
        if not self._history:
            raise ValueError("Cannot finalize a forward session without completed bars.")
        timestamp = self._history[-1]["timestamp"]
        self._current_timestamp = timestamp
        if self._pending is not None:
            self._cancel(self._pending["entry_order"], "end_of_data_before_fill")
            self._cancel(self._pending["exit_order"], "end_of_data_before_fill")
            self._pending = None
        if self._position is not None:
            order = self._new_order(
                "market",
                "sell" if self._position.side == 1 else "buy",
                self._position.quantity,
                timestamp,
                "end_of_data",
            )
            fill = self._fill(
                order, float(self._history[-1]["close"]), timestamp
            )
            self._close_position(fill, timestamp, "end_of_data", order)
            self._equity_records[-1]["realized_pnl"] = self._realized_pnl
            self._equity_records[-1]["unrealized_pnl"] = 0.0
            self._equity_records[-1]["equity"] = (
                self.config.initial_equity + self._realized_pnl
            )
        self._finalized = True

    def result(self) -> BacktestResult:
        history = pd.DataFrame(self._history)
        if history.empty:
            empty = pd.DataFrame()
            return BacktestResult(
                trades=empty,
                equity=empty,
                orders=empty,
                metrics=_equity_metrics(empty, empty, self.config.periods_per_year),
                configuration={
                    "backtest": asdict(self.config),
                    "strategy": {
                        "name": self.strategy.name,
                        "version": self.strategy.version,
                        "parameters": self.strategy.parameters,
                    },
                },
                metadata={"engine_version": "0.1.0"},
            )
        quality = validate_ohlcv(history)[1]
        equity = pd.DataFrame(self._equity_records)
        peaks = equity["equity"].cummax()
        equity["drawdown_dollars"] = equity["equity"] - peaks
        equity["drawdown_percent"] = equity["drawdown_dollars"].div(peaks) * 100
        trades = pd.DataFrame(self._trades)
        orders = pd.DataFrame([order.as_record() for order in self._orders])
        configuration: dict[str, Any] = {
            "backtest": asdict(self.config),
            "strategy": {
                "name": self.strategy.name,
                "version": self.strategy.version,
                "parameters": self.strategy.parameters,
            },
            "data_quality": {
                "row_count": quality.row_count,
                "duplicate_count": quality.duplicate_count,
                "missing_bar_count": quality.missing_bar_count,
                "gaps": list(quality.gaps),
            },
            "evaluation_period": {
                "start": history["timestamp"].iloc[0].isoformat(),
                "end_exclusive": (
                    history["timestamp"].iloc[-1] + pd.Timedelta(1, "ns")
                ).isoformat(),
            },
        }
        canonical = json.dumps(
            configuration, sort_keys=True, default=str, separators=(",", ":")
        )
        data_digest = hashlib.sha256()
        data_digest.update(b"timestamp:int64-utc-ns;ohlcv:float64;")
        data_digest.update(
            history["timestamp"].array.asi8.astype("<i8", copy=False).tobytes()
        )
        for column in ("open", "high", "low", "close", "volume"):
            data_digest.update(np.asarray(history[column], dtype="<f8").tobytes())
        return BacktestResult(
            trades=trades,
            equity=equity,
            orders=orders,
            metrics=_equity_metrics(equity, trades, self.config.periods_per_year),
            configuration=configuration,
            metadata={
                "engine_version": "0.1.0",
                "python_version": platform.python_version(),
                "pandas_version": pd.__version__,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "configuration_sha256": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
                "data_sha256": data_digest.hexdigest(),
            },
        )
