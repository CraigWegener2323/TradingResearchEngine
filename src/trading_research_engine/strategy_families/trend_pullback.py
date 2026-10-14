"""Causal multi-timeframe EMA trend and pullback strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..strategies import Signal
from .common import SessionFilter, ema, finite, session_allows_completed_bar, strategy_parameters


@dataclass(frozen=True)
class MultiTimeframeTrendPullback:
    fast_period: int = 14
    slow_period: int = 50
    higher_timeframe: str = "15min"
    higher_timeframe_ema: int = 50
    session: SessionFilter | None = field(default_factory=SessionFilter)

    def __post_init__(self) -> None:
        if self.fast_period < 2 or self.slow_period <= self.fast_period:
            raise ValueError("Require 2 <= fast_period < slow_period.")
        if self.higher_timeframe_ema < 2 or pd.Timedelta(self.higher_timeframe) <= pd.Timedelta(0):
            raise ValueError("Higher-timeframe duration and EMA period must be positive.")

    @property
    def name(self) -> str:
        return "multi_timeframe_trend_pullback"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameters(self) -> dict[str, object]:
        return strategy_parameters(self)

    def on_bar(self, history: pd.DataFrame) -> Signal:
        if len(history) < self.slow_period + 2:
            return Signal.HOLD
        if not session_allows_completed_bar(self.session, history):
            return Signal.HOLD

        close = history["close"]
        fast, slow = ema(close, self.fast_period), ema(close, self.slow_period)
        current, previous = len(history) - 1, len(history) - 2
        if not all(finite(series.iloc[current]) and finite(series.iloc[previous]) for series in (fast, slow)):
            return Signal.HOLD

        timestamps = pd.DatetimeIndex(history["timestamp"])
        differences = timestamps.to_series().diff().dropna()
        base_interval = differences.median() if not differences.empty else pd.Timedelta(minutes=1)
        bar_ends = timestamps + base_interval
        completed = history.set_index(bar_ends)
        higher = completed.resample(
            self.higher_timeframe, label="right", closed="right", origin="start_day"
        ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        higher = higher.dropna(subset=["close"])
        higher_ema = ema(higher["close"], self.higher_timeframe_ema)
        higher = higher.loc[higher.index <= bar_ends[-1]]
        if len(higher) < 2 or not finite(higher_ema.loc[higher.index[-1]]):
            return Signal.HOLD

        bullish_trend = (
            fast.iloc[current] > slow.iloc[current]
            and higher["close"].iloc[-1] > higher_ema.loc[higher.index[-1]]
        )
        bearish_trend = (
            fast.iloc[current] < slow.iloc[current]
            and higher["close"].iloc[-1] < higher_ema.loc[higher.index[-1]]
        )
        if bullish_trend and close.iloc[previous] <= fast.iloc[previous] and close.iloc[current] > fast.iloc[current]:
            return Signal.LONG
        if bearish_trend and close.iloc[previous] >= fast.iloc[previous] and close.iloc[current] < fast.iloc[current]:
            return Signal.SHORT
        return Signal.HOLD
