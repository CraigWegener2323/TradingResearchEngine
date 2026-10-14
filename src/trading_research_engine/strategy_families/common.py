"""Shared causal calculations and explicit session filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionFilter:
    start: str = "09:30"
    end: str = "16:00"
    timezone: str = "America/New_York"
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)

    def __post_init__(self) -> None:
        try:
            start, end = time.fromisoformat(self.start), time.fromisoformat(self.end)
            ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("Session filter requires valid HH:MM times and IANA timezone.") from exc
        if start == end:
            raise ValueError("Session start and end must differ.")
        if not self.weekdays or any(
            not isinstance(day, int) or day < 0 or day > 6 for day in self.weekdays
        ):
            raise ValueError("Session weekdays must contain values from 0 (Monday) to 6 (Sunday).")

    def allows(self, timestamp: pd.Timestamp) -> bool:
        local = timestamp.tz_convert(ZoneInfo(self.timezone))
        current, start, end = local.time().replace(tzinfo=None), time.fromisoformat(
            self.start
        ), time.fromisoformat(self.end)
        in_session = start <= current < end if start < end else current >= start or current < end
        weekday = local.weekday()
        if start > end and current < end:
            weekday = (weekday - 1) % 7
        return weekday in self.weekdays and in_session


def strategy_parameters(instance: object) -> dict[str, object]:
    values = asdict(instance)
    session = values.get("session")
    if isinstance(session, dict):
        session["weekdays"] = list(session["weekdays"])
    return values


def atr(history: pd.DataFrame, period: int) -> pd.Series:
    previous_close = history["close"].shift(1)
    true_range = pd.concat(
        [
            history["high"] - history["low"],
            (history["high"] - previous_close).abs(),
            (history["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def session_allows_completed_bar(
    session: SessionFilter | None, history: pd.DataFrame
) -> bool:
    if session is None:
        return True
    timestamps = pd.DatetimeIndex(history["timestamp"])
    differences = timestamps.to_series().diff().dropna()
    interval = differences.median() if not differences.empty else pd.Timedelta(minutes=1)
    return session.allows(timestamps[-1] + interval)
