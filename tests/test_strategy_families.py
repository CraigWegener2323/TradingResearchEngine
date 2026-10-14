import pandas as pd
import pytest

from trading_research_engine.engine import BacktestConfig, BacktestEngine
from trading_research_engine.strategies import Signal
from trading_research_engine.strategy_families import (
    FVGOrderBlockPullback,
    MultiTimeframeTrendPullback,
    RangeMeanReversion,
    VolatilityRegimeStrategy,
)
from trading_research_engine.strategy_families.common import SessionFilter


def bars_from_prices(rows):
    timestamps = pd.date_range("2025-01-06T14:30:00Z", periods=len(rows), freq="min")
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "open": opening,
                "high": high,
                "low": low,
                "close": close,
                "volume": 100,
            }
            for timestamp, (opening, high, low, close) in zip(timestamps, rows)
        ]
    )


def test_multitimeframe_trend_pullback_long_and_short_rules():
    long_closes = [10, 11, 12, 13, 12, 8, 20]
    short_closes = [20, 19, 18, 17, 18, 22, 10]

    def make(closes):
        return bars_from_prices(
            [(value, value + 0.2, value - 0.2, value) for value in closes]
        )

    strategy = MultiTimeframeTrendPullback(
        fast_period=2,
        slow_period=3,
        higher_timeframe="1min",
        higher_timeframe_ema=2,
        session=None,
    )
    assert strategy.on_bar(make(long_closes)) == Signal.LONG
    assert strategy.on_bar(make(short_closes)) == Signal.SHORT


@pytest.mark.parametrize(
    "setup, expected",
    [
        (
            [
                (12, 12, 10, 11),
                (13.5, 14, 12.5, 13),
                (13, 16, 13, 15),
                (13, 14, 12.7, 13.8),
            ],
            Signal.LONG,
        ),
        (
            [
                (12.5, 14, 12, 13),
                (11.6, 12.8, 11.5, 12.4),
                (11.6, 11.9, 9, 9.5),
                (12, 12.1, 11, 11.7),
            ],
            Signal.SHORT,
        ),
    ],
)
def test_fvg_and_order_block_pullbacks_are_algorithmic(setup, expected):
    strategy = FVGOrderBlockPullback(
        atr_period=2,
        order_block_lookback=3,
        displacement_atr=0.1,
        minimum_gap_atr=0,
        session=None,
    )

    assert strategy.on_bar(bars_from_prices(setup)) == expected


@pytest.mark.parametrize(
    "last_bar, expected",
    [
        ((100, 100.5, 98, 100), Signal.LONG),
        ((100, 102, 99.5, 100), Signal.SHORT),
    ],
)
def test_range_failed_breakout_signals_both_directions(last_bar, expected):
    base = [(100, 101, 99, 100)] * 8
    strategy = RangeMeanReversion(
        lookback=5,
        atr_period=2,
        maximum_range_atr=20,
        edge_atr=0.5,
        session=None,
    )

    assert strategy.on_bar(bars_from_prices(base + [last_bar])) == expected


def test_volatility_regime_is_causal_and_has_explicit_state_names():
    bars = bars_from_prices([(100, 100.2, 99.8, 100)] * 80)
    first = VolatilityRegimeStrategy(
        atr_period=2,
        regime_lookback=20,
        range_lookback=5,
        trend_fast=2,
        trend_slow=3,
        session=None,
    )
    first_labels = first.regime(bars.iloc[:70])
    changed_future = bars.copy()
    changed_future.loc[70:, ["open", "high", "low", "close"]] += 50
    second = VolatilityRegimeStrategy(
        atr_period=2,
        regime_lookback=20,
        range_lookback=5,
        trend_fast=2,
        trend_slow=3,
        session=None,
    )

    assert first_labels == "low_volatility"
    assert second.regime(changed_future.iloc[:70]) == first_labels


def test_atr_stop_mode_is_applied_by_shared_execution_engine():
    bars = bars_from_prices(
        [
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99.5, 100),
            (100, 101, 97, 98),
            (98, 99, 97, 98),
        ]
    )

    class LongOnce:
        name = "long_once"
        version = "test"
        parameters = {}

        def on_bar(self, history):
            return Signal.LONG if len(history) == 2 else Signal.HOLD

    result = BacktestEngine(
        BacktestConfig(
            stop_mode="atr",
            stop_atr_period=2,
            stop_atr_multiple=1,
            commission_per_contract=0,
            slippage_ticks=1,
        )
    ).run(bars, LongOnce())

    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(98)


def test_structure_stop_and_risk_reward_target_use_shared_execution():
    bars = bars_from_prices(
        [
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99.5, 100),
            (100, 103, 99.5, 102),
            (102, 103, 101, 102),
        ]
    )

    class LongOnce:
        name = "long_once"
        version = "test"
        parameters = {}

        def on_bar(self, history):
            return Signal.LONG if len(history) == 2 else Signal.HOLD

    result = BacktestEngine(
        BacktestConfig(
            stop_mode="structure",
            structure_lookback=2,
            take_profit_mode="risk_reward",
            risk_reward_multiple=2,
            commission_per_contract=0,
            slippage_ticks=0,
        )
    ).run(bars, LongOnce())

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "take_profit"
    assert trade["entry_price"] == 100
    assert trade["exit_price"] == 102


def test_session_filter_uses_new_york_time_and_weekday():
    regular = SessionFilter(start="09:30", end="16:00")

    assert regular.allows(pd.Timestamp("2025-01-06T14:30:00Z"))
    assert not regular.allows(pd.Timestamp("2025-01-06T14:29:00Z"))
    assert not regular.allows(pd.Timestamp("2025-01-04T15:00:00Z"))
