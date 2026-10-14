import pandas as pd
import pytest

from trading_research_engine.engine import (
    BacktestConfig,
    BacktestEngine,
    calculate_position_size,
)
from trading_research_engine.export import export_result
from trading_research_engine.strategies import Signal


class ScheduledStrategy:
    name = "scheduled_test_strategy"
    version = "test"
    parameters = {}

    def __init__(self, signals):
        self.signals = signals
        self.history_lengths = []
        self.last_timestamps = []

    def on_bar(self, history):
        self.history_lengths.append(len(history))
        self.last_timestamps.append(history["timestamp"].iloc[-1])
        return self.signals.get(len(history) - 1, Signal.HOLD)


def make_bars(rows):
    timestamps = pd.date_range("2025-01-02T14:30:00Z", periods=len(rows), freq="min")
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "open": row[0],
                "high": row[1],
                "low": row[2],
                "close": row[3],
                "volume": 100,
            }
            for timestamp, row in zip(timestamps, rows)
        ]
    )


def run_trade(rows, config=None, direction=Signal.LONG):
    strategy = ScheduledStrategy({0: direction})
    result = BacktestEngine(config or BacktestConfig()).run(make_bars(rows), strategy)
    return result, strategy


def test_long_trade_fills_next_bar_and_closes_at_end_of_data():
    result, _ = run_trade(
        [(100, 101, 99, 100), (100, 102, 99.5, 101), (102, 104, 101, 103)],
        BacktestConfig(commission_per_contract=0, slippage_ticks=0),
    )

    trade = result.trades.iloc[0]
    assert trade["side"] == "long"
    assert trade["entry_price"] == 100
    assert trade["exit_price"] == 103
    assert trade["gross_pnl"] == 60
    assert trade["net_pnl"] == 60
    assert trade["exit_reason"] == "end_of_data"
    assert trade["signal_time"] == "2025-01-02T14:30:00+00:00"
    entry = result.orders[result.orders["reason"] == "strategy_signal"].iloc[0]
    assert entry["status"] == "filled"
    assert entry["submitted_at"] == "2025-01-02T14:30:00+00:00"
    assert entry["filled_at"] == "2025-01-02T14:31:00+00:00"


def test_short_trade_accounts_for_direction():
    result, _ = run_trade(
        [(100, 101, 99, 100), (100, 101, 98, 99), (98, 99, 96, 97)],
        BacktestConfig(commission_per_contract=0, slippage_ticks=0),
        direction=Signal.SHORT,
    )

    trade = result.trades.iloc[0]
    assert trade["side"] == "short"
    assert trade["entry_price"] == 100
    assert trade["exit_price"] == 97
    assert trade["net_pnl"] == 60


def test_opposite_signal_closes_and_reverses_at_next_open():
    bars = make_bars(
        [
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (101, 102, 100, 101),
            (99, 100, 98, 99),
        ]
    )
    strategy = ScheduledStrategy({0: Signal.LONG, 1: Signal.SHORT})
    result = BacktestEngine(
        BacktestConfig(commission_per_contract=0, slippage_ticks=0)
    ).run(bars, strategy)

    assert list(result.trades["side"]) == ["long", "short"]
    assert list(result.trades["entry_price"]) == [100, 101]
    assert list(result.trades["exit_reason"]) == ["signal_reversal", "end_of_data"]


def test_flat_signal_closes_without_opening_a_reverse_position():
    bars = make_bars(
        [
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (102, 103, 101, 102),
            (103, 104, 102, 103),
        ]
    )
    strategy = ScheduledStrategy({0: Signal.LONG, 1: Signal.FLAT})
    result = BacktestEngine(
        BacktestConfig(commission_per_contract=0, slippage_ticks=0)
    ).run(bars, strategy)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "signal_flat"
    assert result.trades.iloc[0]["exit_time"] == bars.iloc[2]["timestamp"].isoformat()
    assert not (result.trades["side"] == "short").any()


def test_stop_loss_exit_and_stop_market_slippage():
    result, _ = run_trade(
        [(100, 101, 99.5, 100), (100, 101, 98, 100), (100, 101, 99, 100)],
        BacktestConfig(
            stop_loss_points=1,
            commission_per_contract=0,
            slippage_ticks=1,
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(99)
    assert trade["net_pnl"] == pytest.approx(-25)


def test_take_profit_fills_at_limit_price_without_price_improvement():
    result, _ = run_trade(
        [(100, 100.5, 99.5, 100), (100, 103, 99.5, 102), (102, 103, 101, 102)],
        BacktestConfig(
            take_profit_points=1,
            commission_per_contract=0,
            slippage_ticks=2,
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == pytest.approx(101.5)
    assert trade["net_pnl"] == pytest.approx(20)


def test_commissions_and_slippage_are_included_in_net_pnl():
    result, _ = run_trade(
        [(100, 100.5, 99.5, 100), (100, 101, 99.5, 100.5), (101, 102, 100, 101)],
        BacktestConfig(
            commission_per_contract=2.5,
            slippage_ticks=1,
            tick_size=0.25,
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(100.25)
    assert trade["exit_price"] == pytest.approx(100.75)
    assert trade["commission"] == pytest.approx(5)
    assert trade["slippage_points"] == pytest.approx(0.5)
    assert trade["gross_pnl"] == pytest.approx(10)
    assert trade["net_pnl"] == pytest.approx(5)


def test_risk_based_position_size_uses_contract_multiplier_and_cap():
    config = BacktestConfig(
        instrument="NQ",
        risk_fraction=0.01,
        stop_loss_points=10,
        max_position_size=3,
    )
    micro = BacktestConfig(
        instrument="MNQ",
        risk_fraction=0.01,
        stop_loss_points=10,
        max_position_size=100,
    )

    assert calculate_position_size(100_000, config) == 3
    assert calculate_position_size(100_000, micro) == 50
    assert calculate_position_size(-1, config) == 0


def test_risk_sizing_sets_the_filled_position_quantity():
    config = BacktestConfig(
        risk_fraction=0.01,
        stop_loss_points=10,
        max_position_size=3,
        commission_per_contract=0,
        slippage_ticks=0,
    )
    result, _ = run_trade(
        [(100, 101, 99, 100), (100, 101, 99.5, 100), (100, 101, 99, 100)],
        config,
    )

    assert result.trades.iloc[0]["quantity"] == 3


def test_stop_gap_fills_at_worse_open_plus_slippage():
    result, _ = run_trade(
        [
            (100, 101, 99.5, 100),
            (100, 100.5, 99.5, 100),
            (95, 97, 94, 96),
            (96, 97, 95, 96),
        ],
        BacktestConfig(
            stop_loss_points=1,
            commission_per_contract=0,
            slippage_ticks=1,
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss_gap"
    assert trade["exit_price"] == pytest.approx(94.75)
    assert trade["net_pnl"] == pytest.approx(-110)


def test_end_of_data_closes_open_position_and_cancels_last_bar_signal():
    strategy = ScheduledStrategy({0: Signal.LONG, 2: Signal.SHORT})
    bars = make_bars(
        [(100, 101, 99, 100), (100, 101, 99, 100), (101, 102, 100, 101)]
    )
    result = BacktestEngine(
        BacktestConfig(commission_per_contract=0, slippage_ticks=0)
    ).run(bars, strategy)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "end_of_data"
    assert result.trades.iloc[0]["exit_time"] == bars.iloc[-1]["timestamp"].isoformat()
    unfilled = result.orders[result.orders["reason"] == "strategy_signal"].iloc[-1]
    assert unfilled["status"] == "canceled"
    assert unfilled["cancel_reason"] == "end_of_data_before_fill"
    assert result.equity.iloc[-1]["unrealized_pnl"] == 0


def test_ambiguous_intrabar_stop_and_target_uses_stop_first():
    result, _ = run_trade(
        [(100, 100.5, 99.5, 100), (100, 102, 98, 100), (100, 101, 99, 100)],
        BacktestConfig(
            stop_loss_points=1,
            take_profit_points=1,
            commission_per_contract=0,
            slippage_ticks=0,
        ),
    )

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop_loss_ambiguous"
    assert trade["exit_price"] == 99
    assert trade["net_pnl"] == -20


def test_strategy_only_receives_completed_history_without_future_bars():
    bars = make_bars(
        [(100, 101, 99, 100), (100, 101, 99, 100), (500, 501, 499, 500)]
    )
    strategy = ScheduledStrategy({0: Signal.LONG})
    result = BacktestEngine(
        BacktestConfig(commission_per_contract=0, slippage_ticks=0)
    ).run(bars, strategy)

    assert strategy.history_lengths == [1, 2, 3]
    assert strategy.last_timestamps == list(bars["timestamp"])
    assert result.trades.iloc[0]["entry_price"] == 100
    assert result.trades.iloc[0]["entry_time"] == bars.iloc[1]["timestamp"].isoformat()


def test_equity_tracks_realized_unrealized_and_drawdown():
    result, _ = run_trade(
        [(100, 101, 99, 100), (100, 102, 99, 101), (101, 102, 97, 98)],
        BacktestConfig(commission_per_contract=0, slippage_ticks=0),
    )

    assert result.equity.iloc[1]["unrealized_pnl"] == 20
    assert result.equity.iloc[-1]["realized_pnl"] == -40
    assert result.equity.iloc[-1]["equity"] == 99_960
    assert result.metrics["max_drawdown_dollars"] == 60


def test_export_writes_run_files_and_refuses_existing_directory(tmp_path):
    result, _ = run_trade(
        [(100, 101, 99, 100), (100, 101, 99, 100), (101, 102, 100, 101)],
    )
    output = tmp_path / "run-1"

    export_result(result, output)

    assert {path.name for path in output.iterdir()} == {
        "trades.csv",
        "equity.csv",
        "orders.csv",
        "run.json",
    }
    assert result.metadata["data_sha256"]
    assert result.metadata["configuration_sha256"]
    with pytest.raises(FileExistsError):
        export_result(result, output)
