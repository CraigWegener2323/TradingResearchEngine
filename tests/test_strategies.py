import pandas as pd

from trading_research_engine.strategies import EMACrossover, Signal


def test_ema_crossover_waits_for_history_then_signals_on_cross():
    close = [3.0, 2.0, 1.0, 2.0, 3.0, 4.0]
    bars = pd.DataFrame({"close": close})
    strategy = EMACrossover(fast_period=2, slow_period=3)

    signals = [strategy.on_bar(bars.iloc[:index + 1]) for index in range(len(bars))]

    assert signals[:4] == [Signal.HOLD] * 4
    assert signals[4] == Signal.LONG
    assert signals[5] == Signal.HOLD


def test_ema_strategy_rejects_invalid_periods():
    try:
        EMACrossover(fast_period=3, slow_period=3)
    except ValueError as error:
        assert "fast_period" in str(error)
    else:
        raise AssertionError("Expected equal EMA periods to be rejected.")


def test_ema_strategy_has_explicit_version_metadata():
    strategy = EMACrossover()

    assert strategy.name == "ema_crossover"
    assert strategy.version == "1.0.0"
    assert strategy.parameters == {"fast_period": 9, "slow_period": 21}
