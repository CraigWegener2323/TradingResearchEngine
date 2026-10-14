"""Algorithmic FVG and last-opposite-candle order-block pullbacks."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..strategies import Signal
from .common import SessionFilter, atr, finite, session_allows_completed_bar, strategy_parameters


@dataclass(frozen=True)
class FVGOrderBlockPullback:
    atr_period: int = 14
    order_block_lookback: int = 12
    maximum_zone_age: int = 20
    displacement_atr: float = 1.0
    minimum_gap_atr: float = 0.1
    session: SessionFilter | None = field(default_factory=SessionFilter)

    def __post_init__(self) -> None:
        if self.atr_period < 2 or self.order_block_lookback < 1 or self.maximum_zone_age < 1:
            raise ValueError("ATR period and order-block lookback must be positive.")
        if self.displacement_atr <= 0 or self.minimum_gap_atr < 0:
            raise ValueError("Displacement threshold must be positive and gap threshold non-negative.")

    @property
    def name(self) -> str:
        return "fvg_order_block_pullback"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def parameters(self) -> dict[str, object]:
        return strategy_parameters(self)

    def on_bar(self, history: pd.DataFrame) -> Signal:
        current = len(history) - 1
        if current < max(self.atr_period + 1, 3):
            return Signal.HOLD
        timestamp = history["timestamp"].iloc[current]
        if not session_allows_completed_bar(self.session, history):
            return Signal.HOLD
        atr_values = atr(history, self.atr_period)
        prior = history.iloc[:current]
        active_zone = self._latest_overlapping_zone(
            prior, atr_values.iloc[:current], self.maximum_zone_age
        )
        if active_zone is None:
            return Signal.HOLD
        side, lower, upper = active_zone
        bar = history.iloc[current]
        touches_zone = float(bar["low"]) <= upper and float(bar["high"]) >= lower
        if not touches_zone:
            return Signal.HOLD
        if side == 1 and float(bar["close"]) > upper and float(bar["close"]) > float(bar["open"]):
            return Signal.LONG
        if side == -1 and float(bar["close"]) < lower and float(bar["close"]) < float(bar["open"]):
            return Signal.SHORT
        return Signal.HOLD

    def _latest_overlapping_zone(
        self,
        history: pd.DataFrame,
        atr_values: pd.Series,
        maximum_zone_age: int,
    ) -> tuple[int, float, float] | None:
        latest = len(history) - 1
        earliest = max(2, latest - maximum_zone_age + 1)
        for index in range(latest, earliest - 1, -1):
            volatility = atr_values.iloc[index]
            if not finite(volatility):
                continue
            first, displacement = (
                history.iloc[index - 2],
                history.iloc[index],
            )
            body = float(displacement["close"]) - float(displacement["open"])
            if abs(body) < self.displacement_atr * float(volatility):
                continue

            if body > 0:
                gap_low, gap_high = float(first["high"]), float(displacement["low"])
                if gap_high - gap_low < self.minimum_gap_atr * float(volatility):
                    continue
                candidates = range(index - 1, max(-1, index - self.order_block_lookback - 1), -1)
                block = next(
                    (
                        history.iloc[prior]
                        for prior in candidates
                        if float(history.iloc[prior]["close"]) < float(history.iloc[prior]["open"])
                    ),
                    None,
                )
                if block is None:
                    continue
                block_low, block_high = float(block["low"]), float(block["open"])
                lower, upper = max(gap_low, block_low), min(gap_high, block_high)
                if lower < upper:
                    return 1, lower, upper
            elif body < 0:
                gap_low, gap_high = float(displacement["high"]), float(first["low"])
                if gap_high - gap_low < self.minimum_gap_atr * float(volatility):
                    continue
                candidates = range(index - 1, max(-1, index - self.order_block_lookback - 1), -1)
                block = next(
                    (
                        history.iloc[prior]
                        for prior in candidates
                        if float(history.iloc[prior]["close"]) > float(history.iloc[prior]["open"])
                    ),
                    None,
                )
                if block is None:
                    continue
                block_low, block_high = float(block["open"]), float(block["high"])
                lower, upper = max(gap_low, block_low), min(gap_high, block_high)
                if lower < upper:
                    return -1, lower, upper
        return None
