"""Range mean-reversion and failed-breakout strategy with explicit thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..strategies import Signal
from .common import SessionFilter, atr, finite, session_allows_completed_bar, strategy_parameters


@dataclass(frozen=True)
class RangeMeanReversion:
    lookback: int = 30
    atr_period: int = 14
    maximum_range_atr: float = 6.0
    edge_atr: float = 0.5
    session: SessionFilter | None = field(default_factory=SessionFilter)

    def __post_init__(self) -> None:
        if self.lookback < 3 or self.atr_period < 2:
            raise ValueError("Range lookback and ATR period are too short.")
        if self.maximum_range_atr <= 0 or self.edge_atr <= 0:
            raise ValueError("Range and edge ATR thresholds must be positive.")

    @property
    def name(self) -> str:
        return "range_mean_reversion_failed_breakout"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameters(self) -> dict[str, object]:
        return strategy_parameters(self)

    def on_bar(self, history: pd.DataFrame) -> Signal:
        current = len(history) - 1
        if current < max(self.lookback, self.atr_period + 1):
            return Signal.HOLD
        if not session_allows_completed_bar(self.session, history):
            return Signal.HOLD

        previous = history.iloc[current - self.lookback : current]
        range_high, range_low = float(previous["high"].max()), float(previous["low"].min())
        volatility = atr(history, self.atr_period).iloc[current]
        if not finite(volatility) or range_high - range_low > self.maximum_range_atr * float(volatility):
            return Signal.HOLD
        bar = history.iloc[current]

        if float(bar["high"]) > range_high and float(bar["close"]) <= range_high:
            return Signal.SHORT
        if float(bar["low"]) < range_low and float(bar["close"]) >= range_low:
            return Signal.LONG
        if float(bar["low"]) <= range_low + self.edge_atr * float(volatility):
            if float(bar["close"]) > range_low + self.edge_atr * float(volatility):
                return Signal.LONG
        if float(bar["high"]) >= range_high - self.edge_atr * float(volatility):
            if float(bar["close"]) < range_high - self.edge_atr * float(volatility):
                return Signal.SHORT
        return Signal.HOLD
