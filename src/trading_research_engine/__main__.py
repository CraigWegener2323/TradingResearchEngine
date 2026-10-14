"""Command-line entry point for a reproducible EMA crossover backtest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .data import load_market_data
from .engine import BacktestConfig, BacktestEngine
from .export import export_result
from .strategies import EMACrossover


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a historical OHLCV backtest.")
    parser.add_argument("data", type=Path, help="Input CSV or Parquet OHLCV file.")
    parser.add_argument("--output", type=Path, help="New output directory (must not exist).")
    parser.add_argument("--instrument", choices=("NQ", "MNQ"), default="NQ")
    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--fast-ema", type=int, default=9)
    parser.add_argument("--slow-ema", type=int, default=21)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--max-position-size", type=int, default=10)
    parser.add_argument("--risk-fraction", type=float)
    parser.add_argument("--stop-loss-points", type=float)
    parser.add_argument("--take-profit-points", type=float)
    parser.add_argument("--commission-per-contract", type=float, default=2.50)
    parser.add_argument("--slippage-ticks", type=float, default=1.0)
    parser.add_argument("--tick-size", type=float, default=0.25)
    parser.add_argument("--periods-per-year", type=int, default=98_280)
    parser.add_argument("--expected-interval", help="Expected bar interval, e.g. 1min or 5min.")
    args = parser.parse_args()

    bars, report = load_market_data(args.data, expected_interval=args.expected_interval)
    config = BacktestConfig(
        instrument=args.instrument,
        initial_equity=args.initial_equity,
        quantity=args.quantity,
        max_position_size=args.max_position_size,
        risk_fraction=args.risk_fraction,
        stop_loss_points=args.stop_loss_points,
        take_profit_points=args.take_profit_points,
        commission_per_contract=args.commission_per_contract,
        slippage_ticks=args.slippage_ticks,
        tick_size=args.tick_size,
        periods_per_year=args.periods_per_year,
    )
    result = BacktestEngine(config).run(
        bars, EMACrossover(args.fast_ema, args.slow_ema), data_quality=report
    )
    output = args.output or (
        Path("runs")
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    location = export_result(result, output)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    print(f"Results saved to: {location.resolve()}")


if __name__ == "__main__":
    main()
