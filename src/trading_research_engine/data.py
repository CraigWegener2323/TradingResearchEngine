"""Historical OHLCV import and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    duplicate_count: int
    missing_bar_count: int
    gaps: tuple[dict[str, object], ...]


def validate_ohlcv(
    frame: pd.DataFrame,
    expected_interval: str | pd.Timedelta | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Normalize and validate OHLCV bars; naive timestamps are interpreted as UTC."""
    if frame.empty:
        raise ValueError("Market data contains no rows.")

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    if len(set(normalized.columns)) != len(normalized.columns):
        raise ValueError("Column names are duplicated after case normalization.")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing_columns)}")
    normalized = normalized.loc[:, REQUIRED_COLUMNS].copy()

    try:
        normalized["timestamp"] = pd.to_datetime(
            normalized["timestamp"], errors="raise", utc=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Timestamps must be parseable datetimes.") from exc
    if normalized["timestamp"].isna().any():
        raise ValueError("Timestamps cannot be null.")
    duplicate_count = int(normalized["timestamp"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate timestamp(s).")

    for column in ("open", "high", "low", "close", "volume"):
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{column}' must contain numeric values.") from exc

    numeric = normalized[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite and non-null.")
    if (normalized["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    invalid_ohlc = (
        (normalized["high"] < normalized["low"])
        | (normalized["high"] < normalized[["open", "close"]].max(axis=1))
        | (normalized["low"] > normalized[["open", "close"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise ValueError(f"Found {int(invalid_ohlc.sum())} invalid OHLC bar(s).")

    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    gaps: list[dict[str, object]] = []
    missing_bar_count = 0
    if expected_interval is not None:
        interval = pd.Timedelta(expected_interval)
        if interval <= pd.Timedelta(0):
            raise ValueError("expected_interval must be a positive duration.")
        differences = normalized["timestamp"].diff()
        for index in differences[differences > interval].index:
            delta = differences.iloc[index]
            missing = max(int(delta // interval) - 1, 0)
            missing_bar_count += missing
            gaps.append(
                {
                    "after": normalized.at[index - 1, "timestamp"].isoformat(),
                    "before": normalized.at[index, "timestamp"].isoformat(),
                    "missing_bars": missing,
                    "interval_aligned": delta % interval == pd.Timedelta(0),
                }
            )

    report = DataQualityReport(
        row_count=len(normalized),
        duplicate_count=duplicate_count,
        missing_bar_count=missing_bar_count,
        gaps=tuple(gaps),
    )
    return normalized, report


def load_market_data(
    path: str | Path,
    expected_interval: str | pd.Timedelta | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load CSV or Parquet OHLCV data, then validate and sort chronologically."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Market data file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in (".parquet", ".pq"):
        try:
            frame = pd.read_parquet(source)
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires an engine; install it with "
                "`python -m pip install -e \".[parquet]\"`."
            ) from exc
    else:
        raise ValueError("Market data file must have a .csv, .parquet, or .pq extension.")
    return validate_ohlcv(frame, expected_interval=expected_interval)
