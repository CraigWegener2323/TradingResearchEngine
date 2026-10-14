"""Strategy interface and a deliberately simple EMA crossover example."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pandas as pd


class Signal(str, Enum):
    HOLD = "hold"
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Strategy(Protocol):
    """A strategy sees only completed bars through the current decision bar."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def parameters(self) -> dict[str, object]: ...

    def on_bar(self, history: pd.DataFrame) -> Signal: ...


@dataclass(frozen=True)
class EMACrossover:
    fast_period: int = 9
    slow_period: int = 21

    def __post_init__(self) -> None:
        if self.fast_period < 1 or self.slow_period < 2:
            raise ValueError("EMA periods must be positive; slow_period must be at least 2.")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period.")

    @property
    def name(self) -> str:
        return "ema_crossover"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameters(self) -> dict[str, object]:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}

    def on_bar(self, history: pd.DataFrame) -> Signal:
        if len(history) < self.slow_period + 1:
            return Signal.HOLD

        close = history["close"]
        fast = close.ewm(span=self.fast_period, adjust=False, min_periods=self.fast_period).mean()
        slow = close.ewm(span=self.slow_period, adjust=False, min_periods=self.slow_period).mean()
        previous_fast, current_fast = fast.iloc[-2], fast.iloc[-1]
        previous_slow, current_slow = slow.iloc[-2], slow.iloc[-1]
        if pd.isna(previous_fast) or pd.isna(previous_slow):
            return Signal.HOLD
        if previous_fast <= previous_slow and current_fast > current_slow:
            return Signal.LONG
        if previous_fast >= previous_slow and current_fast < current_slow:
            return Signal.SHORT
        return Signal.HOLD
