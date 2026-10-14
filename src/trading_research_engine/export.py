"""Non-overwriting export of a backtest run and its reproducibility metadata."""

from __future__ import annotations

import json
from pathlib import Path

from .engine import BacktestResult


def export_result(result: BacktestResult, output_directory: str | Path) -> Path:
    """Write a complete run to a new directory; existing paths are never replaced."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    result.trades.to_csv(destination / "trades.csv", index=False)
    result.equity.to_csv(destination / "equity.csv", index=False)
    result.orders.to_csv(destination / "orders.csv", index=False)
    payload = {
        "configuration": result.configuration,
        "metadata": result.metadata,
        "performance_metrics": result.metrics,
    }
    with (destination / "run.json").open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    return destination
