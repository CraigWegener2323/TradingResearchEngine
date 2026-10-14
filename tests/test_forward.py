import json

import pandas as pd
import pytest

from trading_research_engine.engine import BacktestConfig
from trading_research_engine.forward import (
    BarEvent,
    ForwardTestSession,
    HistoricalReplayProvider,
    LIVE_ORDER_EXECUTION_ENABLED,
    TradeEvent,
    run_historical_replay,
)
from trading_research_engine.strategies import Signal


class OneShotLong:
    name = "one_shot_long"
    version = "1.0"

    @property
    def parameters(self):
        return {}

    def on_bar(self, history):
        return Signal.LONG if len(history) == 1 else Signal.HOLD


class FailingStrategy(OneShotLong):
    name = "failing_strategy"

    def on_bar(self, history):
        raise RuntimeError("intentional strategy failure")


def bars(timestamps=None):
    timestamps = timestamps or pd.date_range(
        "2025-01-06T14:30:00Z", periods=5, freq="min"
    )
    prices = [100, 100, 102, 101, 103]
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 10,
            }
            for timestamp, price in zip(timestamps, prices)
        ]
    )


def config(**overrides):
    values = {
        "instrument": "MNQ",
        "commission_per_contract": 0,
        "slippage_ticks": 0,
        "limit_touch_fill_probability": 0.5,
    }
    values.update(overrides)
    return BacktestConfig(**values)


def bar_event(row, event_id=None, is_complete=True):
    return BarEvent(
        event_id=event_id or f"bar:{pd.Timestamp(row.timestamp).isoformat()}",
        timestamp=row.timestamp,
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
        is_complete=is_complete,
    )


def test_replay_pipeline_matches_batch_backtest_and_keeps_data_separate(tmp_path):
    comparison = run_historical_replay(
        bars(),
        OneShotLong(),
        tmp_path / "forward-session",
        config(),
        expected_interval="1min",
    )

    assert comparison.matches_backtest
    assert comparison.forward_result.trades.equals(comparison.backtest_result.trades)
    assert (comparison.session_directory / "forward_state.json").is_file()
    assert (comparison.session_directory / "forward_trades.csv").is_file()
    assert not (comparison.session_directory / "trades.csv").exists()
    state = json.loads(
        (comparison.session_directory / "forward_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["session_type"] == "forward_test"
    assert state["finalized"] is True


def test_restart_recovers_position_and_deduplicates_replayed_event(tmp_path):
    frame = bars()
    directory = tmp_path / "persistent"
    session = ForwardTestSession(directory, OneShotLong(), config())
    first, second = list(HistoricalReplayProvider(frame).events())[:2]
    first = BarEvent(
        **{
            **first.__dict__,
            "completed_at": pd.Timestamp(first.timestamp) + pd.Timedelta(minutes=1),
        }
    )
    assert session.process_event(first) == "accepted"
    assert session.process_event(second) == "accepted"
    assert session.snapshot()["active_position"]["quantity"] == 1

    recovered = ForwardTestSession(
        directory, OneShotLong(), config(), resume=True
    )
    assert recovered.snapshot()["active_position"] == session.snapshot()["active_position"]
    assert recovered.result().orders.iloc[0]["accepted_at"] == (
        pd.Timestamp(second.timestamp).isoformat()
    )
    assert recovered.process_event(second) == "duplicate"
    recovered.run(HistoricalReplayProvider(frame), finalize=True)

    assert len(recovered.result().trades) == 1
    assert any(
        item["event_type"] == "duplicate_event_rejected"
        for item in recovered.snapshot_events
    )


def test_out_of_order_and_incomplete_events_never_create_signals(tmp_path):
    frame = bars().iloc[:2]
    session = ForwardTestSession(tmp_path / "events", OneShotLong(), config())
    first = frame.iloc[0]
    incomplete = bar_event(first, "partial:first", is_complete=False)
    assert session.process_event(incomplete) == "incomplete"
    assert session.process_event(bar_event(first, "complete:first")) == "accepted"

    older = first.copy()
    older["timestamp"] = pd.Timestamp(first["timestamp"]) - pd.Timedelta(minutes=1)
    assert session.process_event(bar_event(older, "older")) == "out_of_order"
    assert sum(
        event["event_type"] == "signal"
        for event in session.snapshot_events
    ) == 1
    assert session.snapshot()["active_position"] is None


def test_missing_bars_are_reported_without_synthetic_bars(tmp_path):
    frame = bars()
    session = ForwardTestSession(
        tmp_path / "gap",
        OneShotLong(),
        config(),
        expected_interval="1min",
    )
    events = list(HistoricalReplayProvider(frame).events())
    session.process_event(events[0])
    session.process_event(events[2])

    missing = [
        event
        for event in session.snapshot_events
        if event["event_type"] == "missing_bars_detected"
    ]
    assert missing[0]["missing_bars"] == 1
    assert len(session.result().equity) == 2


def test_partial_market_fill_records_filled_and_unfilled_quantity(tmp_path):
    session = ForwardTestSession(
        tmp_path / "partial",
        OneShotLong(),
        config(quantity=3, max_contracts_per_fill=1),
    )
    events = list(HistoricalReplayProvider(bars()).events())
    session.process_event(events[0])
    session.process_event(events[1])

    assert session.snapshot()["active_position"]["quantity"] == 1
    entry = session.result().orders.query("reason == 'strategy_signal'").iloc[0]
    assert entry["status"] == "partially_filled"
    assert entry["filled_quantity"] == 1
    assert entry["cancel_reason"] == "partial_fill_remainder_cancelled"
    assert any(
        event["event_type"] == "order_remainder_canceled"
        for event in session.snapshot_events
    )


def test_configured_latency_defers_market_fill_until_order_is_accepted(tmp_path):
    session = ForwardTestSession(
        tmp_path / "latency",
        OneShotLong(),
        config(latency_ms=120_000),
    )
    events = list(HistoricalReplayProvider(bars()).events())
    session.process_event(events[0])
    session.process_event(events[1])
    assert session.snapshot()["active_position"] is None

    session.process_event(events[2])
    assert session.snapshot()["active_position"]["entry_time"] == pd.Timestamp(
        events[2].timestamp
    ).isoformat()
    order = session.result().orders.query("reason == 'strategy_signal'").iloc[0]
    assert pd.Timestamp(order["accepted_at"]) == (
        pd.Timestamp(events[0].timestamp) + pd.Timedelta(seconds=120)
    )


def test_latency_starts_when_the_bar_is_complete_not_at_its_start(tmp_path):
    session = ForwardTestSession(
        tmp_path / "completed-at",
        OneShotLong(),
        config(latency_ms=1_000),
    )
    events = list(HistoricalReplayProvider(bars()).events())
    first = BarEvent(
        **{
            **events[0].__dict__,
            "completed_at": pd.Timestamp(events[0].timestamp) + pd.Timedelta(minutes=1),
        }
    )
    session.process_event(first)
    session.process_event(events[1])
    assert session.snapshot()["active_position"] is None
    session.process_event(events[2])

    order = session.result().orders.query("reason == 'strategy_signal'").iloc[0]
    assert pd.Timestamp(order["accepted_at"]) == (
        pd.Timestamp(events[1].timestamp) + pd.Timedelta(seconds=1)
    )
    assert session.snapshot()["active_position"]["entry_time"] == pd.Timestamp(
        events[2].timestamp
    ).isoformat()


def test_configured_spread_is_applied_adversely_on_market_fills(tmp_path):
    session = ForwardTestSession(
        tmp_path / "spread",
        OneShotLong(),
        config(spread_ticks=2, tick_size=0.25),
    )
    session.run(HistoricalReplayProvider(bars()), finalize=True)
    trade = session.result().trades.iloc[0]

    assert trade["entry_price"] == pytest.approx(100.25)
    assert trade["exit_price"] == pytest.approx(102.75)
    assert trade["slippage_points"] == pytest.approx(0.5)


def test_zero_limit_fill_probability_does_not_fill_a_touch(tmp_path):
    session = ForwardTestSession(
        tmp_path / "limit-uncertainty",
        OneShotLong(),
        config(
            stop_loss_points=5,
            take_profit_points=1,
            limit_touch_fill_probability=0,
        ),
    )
    frame = bars().iloc[:4].copy()
    frame.loc[2, ["high", "close"]] = [103, 102]
    session.run(HistoricalReplayProvider(frame), finalize=True)

    assert session.result().trades.iloc[0]["exit_reason"] == "end_of_data"
    assert not any(
        event["event_type"] == "fill" and event.get("order_type") == "limit"
        for event in session.snapshot_events
    )
    assert any(
        event["event_type"] == "limit_touch_unfilled"
        for event in session.snapshot_events
    )


def test_execution_error_is_logged_persisted_and_blocks_continuation(tmp_path):
    directory = tmp_path / "execution-error"
    session = ForwardTestSession(directory, FailingStrategy(), config())
    event = next(HistoricalReplayProvider(bars()).events())
    with pytest.raises(RuntimeError, match="intentional strategy failure"):
        session.process_event(event)

    state = json.loads(
        (directory / "forward_state.json").read_text(encoding="utf-8")
    )
    assert state["poisoned"] is True
    assert state["events"][-1]["event_type"] == "execution_error"
    recovered = ForwardTestSession(
        directory, FailingStrategy(), config(), resume=True
    )
    with pytest.raises(RuntimeError, match="state may be inconsistent"):
        recovered.process_event(event)


def test_trade_input_aggregates_only_completed_bars(tmp_path):
    session = ForwardTestSession(
        tmp_path / "trades",
        OneShotLong(),
        config(),
        trade_interval="1min",
    )
    first = pd.Timestamp("2025-01-06T14:30:10Z")
    assert session.process_event(
        TradeEvent("t1", first, 100.0, 1)
    ) == "buffered"
    assert session.process_event(
        TradeEvent("t2", first + pd.Timedelta(seconds=20), 101.0, 2)
    ) == "buffered"
    assert len(session.result().equity) == 0
    assert session.process_event(
        TradeEvent("t3", first + pd.Timedelta(minutes=1), 102.0, 1)
    ) == "accepted"
    assert len(session.result().equity) == 1
    assert session.result().equity.iloc[0]["timestamp"] == "2025-01-06T14:30:00+00:00"


def test_session_refuses_mismatched_restart_configuration(tmp_path):
    directory = tmp_path / "mismatch"
    ForwardTestSession(directory, OneShotLong(), config())

    with pytest.raises(ValueError, match="configuration differs"):
        ForwardTestSession(
            directory,
            OneShotLong(),
            config(spread_ticks=2),
            resume=True,
        )


def test_forward_mode_requires_limit_fill_uncertainty_and_never_enables_live_orders(tmp_path):
    assert LIVE_ORDER_EXECUTION_ENABLED is False
    with pytest.raises(ValueError, match="fill uncertainty"):
        ForwardTestSession(
            tmp_path / "certain-limit",
            OneShotLong(),
            BacktestConfig(),
        )
