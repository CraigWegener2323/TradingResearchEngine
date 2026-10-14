"""Volatility state classification and state-conditional trading rules."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..strategies import Signal
from .common import (
    SessionFilter,
    atr,
    ema,
    finite,
    session_allows_completed_bar,
    strategy_parameters,
)


@dataclass(frozen=True)
class VolatilityRegimeStrategy:
    atr_period: int = 14
    regime_lookback: int = 250
    lower_percentile: float = 0.25
    upper_percentile: float = 0.75
    range_lookback: int = 30
    trend_fast: int = 14
    trend_slow: int = 50
    session: SessionFilter | None = field(default_factory=SessionFilter)

    def __post_init__(self) -> None:
        if self.atr_period < 2 or self.regime_lookback < 20 or self.range_lookback < 3:
            raise ValueError("ATR, regime, and range lookbacks are too short.")
        if not 0 < self.lower_percentile < self.upper_percentile < 1:
            raise ValueError("Require 0 < lower_percentile < upper_percentile < 1.")
        if self.trend_fast < 2 or self.trend_slow <= self.trend_fast:
            raise ValueError("Require 2 <= trend_fast < trend_slow.")

    @property
    def name(self) -> str:
        return "volatility_regime"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameters(self) -> dict[str, object]:
        return strategy_parameters(self)

    def regime(self, history: pd.DataFrame) -> str:
        current = len(history) - 1
        if current < max(self.atr_period + self.regime_lookback, self.trend_slow):
            return "unclassified"
        normalized_atr = atr(history, self.atr_period) / history["close"]
        prior = normalized_atr.iloc[current - self.regime_lookback : current]
        current_value = normalized_atr.iloc[current]
        if not finite(current_value) or prior.dropna().empty:
            return "unclassified"
        lower = prior.quantile(self.lower_percentile)
        upper = prior.quantile(self.upper_percentile)
        if current_value <= lower:
            return "low_volatility"
        if current_value >= upper:
            return "high_volatility"
        return "normal_volatility"

    def on_bar(self, history: pd.DataFrame) -> Signal:
        current = len(history) - 1
        if current < max(self.atr_period + self.regime_lookback, self.trend_slow):
            return Signal.HOLD
        if not session_allows_completed_bar(self.session, history):
            return Signal.HOLD
        state = self.regime(history)
        previous = history.iloc[current - self.range_lookback : current]
        high, low = float(previous["high"].max()), float(previous["low"].min())
        bar = history.iloc[current]

        if state == "high_volatility":
            fast, slow = ema(history["close"], self.trend_fast), ema(
                history["close"], self.trend_slow
            )
            if bar["close"] > high and fast.iloc[current] > slow.iloc[current]:
                return Signal.LONG
            if bar["close"] < low and fast.iloc[current] < slow.iloc[current]:
                return Signal.SHORT
        elif state == "low_volatility":
            if bar["low"] <= low and bar["close"] > low:
                return Signal.LONG
            if bar["high"] >= high and bar["close"] < high:
                return Signal.SHORT
        return Signal.HOLD
