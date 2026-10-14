import json

import pandas as pd
import pytest

from trading_research_engine.experiments import (
    CandidateSpec,
    ExperimentPeriods,
    ValidationCriteria,
    WalkForwardFold,
    expand_parameter_grid,
    run_parameter_sweep,
    run_selected_out_of_sample,
    run_walk_forward,
)


def sample_bars():
    timestamps = pd.date_range("2025-01-06T14:30:00Z", periods=18, freq="min")
    prices = [100, 101, 99, 102, 98, 103, 97, 104, 96] * 2
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + (0.5 if index % 2 else -0.5),
                "volume": 100,
            }
            for index, (timestamp, price) in enumerate(zip(timestamps, prices))
        ]
    )


def sample_periods():
    return ExperimentPeriods(
        development_start="2025-01-06T14:30:00Z",
        development_end="2025-01-06T14:36:00Z",
        validation_start="2025-01-06T14:36:00Z",
        validation_end="2025-01-06T14:42:00Z",
        out_of_sample_start="2025-01-06T14:42:00Z",
        out_of_sample_end="2025-01-06T14:48:00Z",
    )


def candidate(edge_atr=0.5):
    return CandidateSpec(
        strategy_family="range_mean_reversion_failed_breakout",
        strategy_parameters={
            "lookback": 3,
            "atr_period": 2,
            "maximum_range_atr": 50,
            "edge_atr": edge_atr,
            "session": None,
        },
        backtest_parameters={
            "instrument": "MNQ",
            "commission_per_contract": 0,
            "slippage_ticks": 0,
        },
    )


def test_grid_expansion_is_deterministic_and_unique():
    candidates = expand_parameter_grid(
        "ema_crossover",
        {"fast_period": [2, 3], "slow_period": [5]},
        {"instrument": ["NQ", "MNQ"]},
    )

    assert len(candidates) == 4
    assert len({item.experiment_id for item in candidates}) == 4
    assert candidates == expand_parameter_grid(
        "ema_crossover",
        {"slow_period": [5], "fast_period": [2, 3]},
        {"instrument": ["NQ", "MNQ"]},
    )


def test_periods_reject_overlap_and_walk_forward_requires_train_before_test():
    with pytest.raises(ValueError, match="chronological"):
        ExperimentPeriods(
            "2025-01-01", "2025-02-01", "2025-01-15", "2025-03-01",
            "2025-03-01", "2025-04-01",
        )
    with pytest.raises(ValueError, match="disjoint"):
        WalkForwardFold("bad", "2025-01-01", "2025-03-01", "2025-02-01", "2025-04-01")


def test_sweep_records_configs_and_does_not_use_oos_prices(tmp_path):
    data = sample_bars()
    variants = [candidate(0.5), candidate(1.0)]
    criteria = ValidationCriteria(
        minimum_validation_trades=1,
        minimum_oos_trades=1,
        bootstrap_resamples=100,
    )
    first = run_parameter_sweep(
        data, variants, sample_periods(), tmp_path / "sweep-a",
        validation_top_n=1, criteria=criteria,
    )
    changed = data.copy()
    changed.loc[changed["timestamp"] >= "2025-01-06T14:42:00Z", ["open", "high", "low", "close"]] += 500
    second = run_parameter_sweep(
        changed, variants, sample_periods(), tmp_path / "sweep-b",
        validation_top_n=1, criteria=criteria,
    )

    assert len(first.development) == 2
    assert len(first.validation) == 1
    assert all(record["stage"] == "development" for record in first.development)
    first_digest = {row["experiment_id"]: row["metadata"]["data_sha256"] for row in first.development}
    second_digest = {row["experiment_id"]: row["metadata"]["data_sha256"] for row in second.development}
    assert first_digest == second_digest
    recorded = [
        json.loads(line)
        for line in (first.output_directory / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(recorded) == 3
    assert {record["experiment_id"] for record in recorded} == {
        item.experiment_id for item in variants
    }
    report = (first.output_directory / "comparison_report.md").read_text(encoding="utf-8")
    assert "Bonferroni" in report
    assert "not a profitability guarantee" in report
    assert "Net P&L by market regime" in report


def test_invalid_candidate_is_recorded_as_an_unsuccessful_attempt(tmp_path):
    invalid = CandidateSpec(strategy_family="unknown_strategy")
    valid = candidate()
    batch = run_parameter_sweep(
        sample_bars(),
        [valid, invalid],
        sample_periods(),
        tmp_path / "failed-run",
        validation_top_n=1,
        criteria=ValidationCriteria(bootstrap_resamples=100),
    )

    invalid_record = next(
        record for record in batch.development
        if record["experiment_id"] == invalid.experiment_id
    )
    assert invalid_record["status"] == "error"
    assert "Unknown strategy family" in invalid_record["error"]
    assert "configuration_error" in invalid_record["configuration"]


def test_oos_requires_explicit_selection_and_records_costs(tmp_path):
    data = sample_bars()
    item = candidate()
    criteria = ValidationCriteria(bootstrap_resamples=100)
    batch = run_parameter_sweep(
        data, [item], sample_periods(), tmp_path / "sweep",
        criteria=criteria,
    )
    with pytest.raises(ValueError, match="OOS cannot be auto-selected"):
        run_selected_out_of_sample(
            data, [item], [], sample_periods(), batch.output_directory, criteria
        )

    results = run_selected_out_of_sample(
        data, [item], [item.experiment_id], sample_periods(),
        batch.output_directory, criteria,
    )

    assert results[0]["stage"] == "out_of_sample"
    assert "total_trading_cost" in results[0]["metrics"]
    assert results[0]["metrics"]["sample_size_trades"] >= 0
    assert results[0]["criteria"]["interpretation"].startswith("A statistical screening")
    assert "Final out-of-sample results" in (
        batch.output_directory / "comparison_report.md"
    ).read_text(encoding="utf-8")


def test_candidate_and_record_preserve_predeclared_configuration():
    item = candidate()
    serialized = item.to_dict()
    rebuilt = CandidateSpec(
        strategy_family=serialized["strategy_family"],
        strategy_parameters=serialized["strategy_parameters"],
        backtest_parameters=serialized["backtest_parameters"],
    )

    assert rebuilt.experiment_id == item.experiment_id
    assert serialized["strategy_parameters"]["session"] is None


def test_walk_forward_selects_on_train_and_records_next_fold(tmp_path):
    item = candidate()
    fold = WalkForwardFold(
        name="fold-1",
        train_start="2025-01-06T14:30:00Z",
        train_end="2025-01-06T14:36:00Z",
        test_start="2025-01-06T14:36:00Z",
        test_end="2025-01-06T14:42:00Z",
    )
    result = run_walk_forward(
        sample_bars(),
        [item],
        [fold],
        tmp_path / "walk-forward",
        ValidationCriteria(bootstrap_resamples=100),
    )

    assert result["selection_frequency"] == {item.experiment_id: 1}
    assert result["folds"][0]["test"]["stage"] == "fold-1_forward_test"
    assert result["candidate_stability"][item.experiment_id]["selected_folds"] == 1
    assert "not independent final OOS proof" in result["interpretation"]
