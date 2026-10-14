"""Reproducible sweeps, chronological validation, walk-forward tests, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import validate_ohlcv
from .engine import BacktestConfig, BacktestEngine, BacktestResult, POINT_VALUES
from .export import export_result
from .strategies import EMACrossover, Strategy
from .strategy_families import (
    FVGOrderBlockPullback,
    MultiTimeframeTrendPullback,
    RangeMeanReversion,
    VolatilityRegimeStrategy,
)
from .strategy_families.common import SessionFilter


STRATEGY_FACTORIES = {
    "multi_timeframe_trend_pullback": MultiTimeframeTrendPullback,
    "fvg_order_block_pullback": FVGOrderBlockPullback,
    "range_mean_reversion_failed_breakout": RangeMeanReversion,
    "volatility_regime": VolatilityRegimeStrategy,
    "ema_crossover": EMACrossover,
}


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class ExperimentPeriods:
    development_start: str
    development_end: str
    validation_start: str
    validation_end: str
    out_of_sample_start: str
    out_of_sample_end: str

    def __post_init__(self) -> None:
        bounds = [_utc(value) for value in asdict(self).values()]
        if not (
            bounds[0] < bounds[1] <= bounds[2] < bounds[3] <= bounds[4] < bounds[5]
        ):
            raise ValueError(
                "Periods must be strictly chronological, non-overlapping, and use start < end."
            )

    def range(self, stage: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        return {
            "development": (_utc(self.development_start), _utc(self.development_end)),
            "validation": (_utc(self.validation_start), _utc(self.validation_end)),
            "out_of_sample": (_utc(self.out_of_sample_start), _utc(self.out_of_sample_end)),
        }[stage]


@dataclass(frozen=True)
class CandidateSpec:
    strategy_family: str
    strategy_parameters: dict[str, Any] = field(default_factory=dict)
    backtest_parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_id(self) -> str:
        serialized = json.dumps(
            self.to_dict(include_id=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, include_id: bool = True) -> dict[str, Any]:
        try:
            strategy = _build_strategy(self.strategy_family, self.strategy_parameters)
            record = {
                "strategy_family": self.strategy_family,
                "strategy_version": strategy.version,
                "strategy_parameters": strategy.parameters,
                "backtest_parameters": asdict(BacktestConfig(**self.backtest_parameters)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            record = {
                "strategy_family": self.strategy_family,
                "strategy_parameters": self.strategy_parameters,
                "backtest_parameters": self.backtest_parameters,
                "configuration_error": f"{type(exc).__name__}: {exc}",
            }
        record = json.loads(
            json.dumps(
                record,
                default=lambda value: asdict(value)
                if hasattr(value, "__dataclass_fields__")
                else str(value),
            )
        )
        if include_id:
            record["experiment_id"] = self.experiment_id
        return record


def _build_strategy(family: str, parameters: dict[str, Any]) -> Strategy:
    if family not in STRATEGY_FACTORIES:
        raise ValueError(f"Unknown strategy family: {family}")
    supplied = dict(parameters)
    if isinstance(supplied.get("session"), dict):
        supplied["session"] = SessionFilter(
            **{
                **supplied["session"],
                "weekdays": tuple(supplied["session"].get("weekdays", (0, 1, 2, 3, 4))),
            }
        )
    return STRATEGY_FACTORIES[family](**supplied)


def expand_parameter_grid(
    strategy_family: str,
    strategy_grid: dict[str, list[Any]],
    execution_grid: dict[str, list[Any]],
) -> list[CandidateSpec]:
    """Expand independent dimensions deterministically into immutable candidates."""
    def mappings(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        keys = sorted(grid)
        if any(not values for values in grid.values()):
            raise ValueError("Parameter grid dimensions cannot be empty.")
        return [
            dict(zip(keys, values))
            for values in itertools.product(*(grid[key] for key in keys))
        ] or [{}]

    return [
        CandidateSpec(strategy_family, strategy_params, execution_params)
        for strategy_params, execution_params in itertools.product(
            mappings(strategy_grid), mappings(execution_grid)
        )
    ]


@dataclass(frozen=True)
class ValidationCriteria:
    """Predeclared screening bar. Passing is not a profitability guarantee."""

    minimum_validation_trades: int = 100
    minimum_oos_trades: int = 30
    minimum_expectancy_ci_lower: float = 0.0
    minimum_profit_factor: float = 1.1
    maximum_drawdown_percent: float = 20.0
    minimum_positive_period_fraction: float = 0.6
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000
    random_seed: int = 1729

    def __post_init__(self) -> None:
        if self.minimum_validation_trades < 1 or self.minimum_oos_trades < 1:
            raise ValueError("Minimum trade counts must be positive.")
        if self.minimum_profit_factor <= 0 or self.maximum_drawdown_percent <= 0:
            raise ValueError("Profit-factor and drawdown criteria must be positive.")
        if not 0 < self.minimum_positive_period_fraction <= 1:
            raise ValueError("minimum_positive_period_fraction must be in (0, 1].")
        if not 0 < self.confidence_level < 1 or self.bootstrap_resamples < 100:
            raise ValueError("Confidence level must be in (0,1), with at least 100 bootstrap draws.")


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def __post_init__(self) -> None:
        bounds = [_utc(self.train_start), _utc(self.train_end), _utc(self.test_start), _utc(self.test_end)]
        if not (bounds[0] < bounds[1] <= bounds[2] < bounds[3]):
            raise ValueError("Each walk-forward fold must have train before a disjoint test period.")


def _new_output(path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True, default=str, allow_nan=False) + "\n")


def _trade_statistics(
    result: BacktestResult,
    bars: pd.DataFrame,
    comparisons: int,
    criteria: ValidationCriteria,
) -> dict[str, Any]:
    trades = result.trades
    net = trades["net_pnl"].astype(float).to_numpy() if not trades.empty else np.array([])
    seed = criteria.random_seed + int(result.metadata["configuration_sha256"][:8], 16)
    if len(net):
        generator = np.random.default_rng(seed)
        sampled = generator.choice(net, size=(criteria.bootstrap_resamples, len(net)), replace=True)
        means = sampled.mean(axis=1)
        tail = (1 - criteria.confidence_level) / (2 * max(comparisons, 1))
        lower, upper = np.quantile(means, [tail, 1 - tail])
        ci_method = "percentile bootstrap of trade net P&L mean; Bonferroni-adjusted across candidates"
    else:
        lower = upper = None
        ci_method = "unavailable: no completed trades"

    timestamp = pd.to_datetime(trades["entry_time"], utc=True) if len(trades) else pd.Series(dtype="datetime64[ns, UTC]")
    period_values = pd.DataFrame(
        {"period": timestamp.dt.tz_localize(None).dt.to_period("M"), "net_pnl": net}
    )
    monthly = period_values.groupby("period")["net_pnl"].sum() if len(period_values) else pd.Series(dtype=float)
    positive_period_fraction = float((monthly > 0).mean()) if len(monthly) else 0.0
    total_cost = float(trades["commission"].sum()) if len(trades) else 0.0
    point_value = POINT_VALUES[result.configuration["backtest"]["instrument"]]
    slippage_cost = (
        float(
            (
                trades["slippage_points"].astype(float)
                * trades["quantity"].astype(float)
                * point_value
            ).sum()
        )
        if len(trades)
        else 0.0
    )
    metrics = dict(result.metrics)
    metrics.update(
        {
            "sample_size_trades": int(len(trades)),
            "months_with_trades": int(len(monthly)),
            "positive_month_fraction": positive_period_fraction,
            "expectancy_ci_lower": float(lower) if lower is not None else None,
            "expectancy_ci_upper": float(upper) if upper is not None else None,
            "expectancy_ci_method": ci_method,
            "multiple_testing_comparisons": int(comparisons),
            "commission_cost": total_cost,
            "estimated_slippage_cost": slippage_cost,
            "total_trading_cost": total_cost + slippage_cost,
            "by_year": _grouped_pnl(trades, timestamp, "year"),
            "by_session_date": _grouped_pnl(trades, timestamp, "session_date"),
            "by_market_regime": _regime_statistics(trades, bars),
            "by_direction": _direction_statistics(trades),
        }
    )
    regime_groups = metrics["by_market_regime"]
    metrics["positive_regime_fraction"] = (
        float(sum(group["net_pnl"] > 0 for group in regime_groups.values()) / len(regime_groups))
        if regime_groups
        else 0.0
    )
    metrics["robustness_score"] = metrics["positive_regime_fraction"]
    return metrics


def _grouped_pnl(
    trades: pd.DataFrame, timestamps: pd.Series, grouping: str
) -> dict[str, dict[str, float | int]]:
    if trades.empty:
        return {}
    if grouping == "year":
        keys = timestamps.dt.year.astype(str)
    else:
        keys = timestamps.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    result: dict[str, dict[str, float | int]] = {}
    for key, indices in keys.groupby(keys).groups.items():
        values = trades.loc[list(indices), "net_pnl"].astype(float)
        result[str(key)] = {
            "trade_count": int(len(values)),
            "net_pnl": float(values.sum()),
            "expectancy": float(values.mean()),
            "positive": int(values.sum() > 0),
        }
    return result


def classify_market_regimes(bars: pd.DataFrame, lookback: int = 250) -> pd.Series:
    """Causal volatility quantiles and 14/50 EMA direction define nine states."""
    if lookback < 20:
        raise ValueError("Regime lookback must be at least 20 bars.")
    close = bars["close"].astype(float)
    prior_close = close.shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prior_close).abs(),
            (bars["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    normalized_atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / close
    prior = normalized_atr.shift(1).rolling(lookback, min_periods=max(20, lookback // 4))
    low_threshold, high_threshold = prior.quantile(0.25), prior.quantile(0.75)
    volatility = np.select(
        [normalized_atr <= low_threshold, normalized_atr >= high_threshold],
        ["low_volatility", "high_volatility"],
        default="normal_volatility",
    )
    fast = close.ewm(span=14, adjust=False, min_periods=14).mean()
    slow = close.ewm(span=50, adjust=False, min_periods=50).mean()
    trend = np.select(
        [fast > slow, fast < slow], ["uptrend", "downtrend"], default="neutral"
    )
    labels = pd.Series(
        [f"{vol}_{direction}" for vol, direction in zip(volatility, trend)],
        index=bars.index,
        dtype="object",
    )
    labels[(low_threshold.isna()) | (high_threshold.isna())] = "unclassified"
    return labels


def _regime_statistics(trades: pd.DataFrame, bars: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {}
    regimes = classify_market_regimes(bars)
    lookup = pd.Series(regimes.to_numpy(), index=bars["timestamp"])
    signal_times = pd.to_datetime(trades["signal_time"], utc=True)
    assigned = lookup.reindex(signal_times).fillna("unclassified").reset_index(drop=True)
    grouped = trades.reset_index(drop=True).assign(regime=assigned)
    stats: dict[str, Any] = {}
    for regime, subset in grouped.groupby("regime"):
        values = subset["net_pnl"].astype(float)
        stats[str(regime)] = {
            "trade_count": int(len(values)),
            "net_pnl": float(values.sum()),
            "expectancy": float(values.mean()),
            "positive": int(values.sum() > 0),
        }
    return stats


def _direction_statistics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {}
    stats: dict[str, Any] = {}
    for side, subset in trades.groupby("side"):
        values = subset["net_pnl"].astype(float)
        stats[str(side)] = {
            "trade_count": int(len(values)),
            "net_pnl": float(values.sum()),
            "expectancy": float(values.mean()),
            "win_rate_percent": float((values > 0).mean() * 100),
        }
    return stats


def _data_before(
    market_data: pd.DataFrame, end: pd.Timestamp
) -> pd.DataFrame:
    columns = {str(column).strip().lower(): column for column in market_data.columns}
    if "timestamp" not in columns:
        raise ValueError("Market data must contain a timestamp column.")
    timestamps = pd.to_datetime(market_data[columns["timestamp"]], errors="raise", utc=True)
    return market_data.loc[timestamps < end].copy()


def _criteria_result(
    metrics: dict[str, Any], criteria: ValidationCriteria, stage: str
) -> dict[str, Any]:
    minimum_trades = (
        criteria.minimum_oos_trades if stage == "out_of_sample" else criteria.minimum_validation_trades
    )
    profit_factor = metrics.get("profit_factor")
    checks = {
        "minimum_trade_count": metrics["sample_size_trades"] >= minimum_trades,
        "positive_adjusted_expectancy_ci": (
            metrics["expectancy_ci_lower"] is not None
            and metrics["expectancy_ci_lower"] > criteria.minimum_expectancy_ci_lower
        ),
        "minimum_profit_factor": (
            profit_factor is not None and profit_factor >= criteria.minimum_profit_factor
        ),
        "maximum_drawdown": (
            metrics["max_drawdown_percent"] <= criteria.maximum_drawdown_percent
        ),
        "period_consistency": (
            metrics["positive_month_fraction"] >= criteria.minimum_positive_period_fraction
        ),
    }
    return {
        "status": "screen_passed" if all(checks.values()) else "screen_not_passed",
        "checks": checks,
        "interpretation": "A statistical screening result, not a profitability guarantee.",
    }


def _rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [record for record in records if record.get("status") == "completed"]
    if not successful:
        return records
    frame = pd.DataFrame(
        [
            {
                "id": record["experiment_id"],
                "expectancy": record["metrics"].get("expectancy_per_trade"),
                "drawdown": record["metrics"].get("max_drawdown_percent"),
                "profit_factor": record["metrics"].get("profit_factor"),
                "consistency": record["metrics"].get("positive_month_fraction"),
                "robustness": record["metrics"].get("positive_regime_fraction"),
            }
            for record in successful
        ]
    ).set_index("id")
    ranks = pd.DataFrame(index=frame.index)
    for column in ("expectancy", "profit_factor", "consistency", "robustness"):
        ranks[column] = frame[column].replace([np.inf, -np.inf], np.nan).fillna(
            frame[column].replace([np.inf, -np.inf], np.nan).min()
        ).rank(pct=True)
    ranks["drawdown"] = -frame["drawdown"].rank(pct=True)
    ranks["composite"] = ranks.mean(axis=1)
    sorted_ids = ranks["composite"].sort_values(ascending=False).index.tolist()
    rank_by_id = {experiment_id: position + 1 for position, experiment_id in enumerate(sorted_ids)}
    for record in records:
        if record.get("status") == "completed":
            record["rank"] = rank_by_id[record["experiment_id"]]
            record["rank_score"] = float(ranks.loc[record["experiment_id"], "composite"])
    return records


def _execute(
    bars: pd.DataFrame,
    candidate: CandidateSpec,
    stage: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    criteria: ValidationCriteria,
    comparisons: int,
    output: Path,
) -> dict[str, Any]:
    base = {
        "experiment_id": candidate.experiment_id,
        "stage": stage,
        "configuration": candidate.to_dict(),
        "predeclared_criteria": asdict(criteria),
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "status": "error",
    }
    try:
        config = BacktestConfig(**candidate.backtest_parameters)
        strategy = _build_strategy(candidate.strategy_family, candidate.strategy_parameters)
        available_bars = bars.loc[bars["timestamp"] < end].copy()
        result = BacktestEngine(config).run(
            available_bars, strategy, start_time=start, end_time=end
        )
        metrics = _trade_statistics(result, available_bars, comparisons, criteria)
        base.update(
            {
                "status": "completed",
                "metrics": metrics,
                "criteria": (
                    _criteria_result(metrics, criteria, stage)
                    if stage == "validation"
                    or stage == "out_of_sample"
                    or stage.endswith("_forward_test")
                    else {
                        "status": "not_evaluated",
                        "checks": {},
                        "interpretation": "Development results are for candidate ranking, not validation.",
                    }
                ),
                "metadata": result.metadata,
            }
        )
        artifact_path = output / "artifacts" / candidate.experiment_id / stage
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        export_result(result, artifact_path)
        base["artifacts"] = str(artifact_path)
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


@dataclass
class ExperimentBatch:
    output_directory: Path
    development: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    shortlisted_ids: list[str]


def run_parameter_sweep(
    market_data: pd.DataFrame,
    candidates: list[CandidateSpec],
    periods: ExperimentPeriods,
    output_directory: str | Path,
    validation_top_n: int = 10,
    criteria: ValidationCriteria | None = None,
) -> ExperimentBatch:
    """Sweep only development; validate top development candidates; never read OOS."""
    if not candidates or validation_top_n < 1:
        raise ValueError("Supply candidates and a positive validation_top_n.")
    ids = [candidate.experiment_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate configurations must be unique.")
    criteria = criteria or ValidationCriteria()
    output = _new_output(output_directory)
    records_path = output / "experiments.jsonl"
    dev_start, dev_end = periods.range("development")
    _, validation_end = periods.range("validation")
    bars = validate_ohlcv(_data_before(market_data, validation_end))[0]
    development = [
        _execute(bars, candidate, "development", dev_start, dev_end, criteria, len(candidates), output)
        for candidate in candidates
    ]
    development = _rank_records(development)
    for record in development:
        _append_jsonl(records_path, record)
    eligible = [
        record for record in development if record["status"] == "completed"
    ]
    lookup = {candidate.experiment_id: candidate for candidate in candidates}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in sorted(eligible, key=lambda row: row["rank"]):
        family = record["configuration"]["strategy_family"]
        family_records = by_family.setdefault(family, [])
        if len(family_records) < validation_top_n:
            family_records.append(record)
    selected_ids = [
        record["experiment_id"]
        for family_records in by_family.values()
        for record in family_records
    ]
    val_start, val_end = periods.range("validation")
    validation = [
        _execute(
            bars,
            lookup[experiment_id],
            "validation",
            val_start,
            val_end,
            criteria,
            len(candidates),
            output,
        )
        for experiment_id in selected_ids
    ]
    validation = _rank_records(validation)
    for record in validation:
        _append_jsonl(records_path, record)
    batch = ExperimentBatch(output, development, validation, selected_ids)
    write_comparison_report(batch, criteria)
    return batch


def run_selected_out_of_sample(
    market_data: pd.DataFrame,
    candidates: list[CandidateSpec],
    selected_experiment_ids: list[str],
    periods: ExperimentPeriods,
    output_directory: str | Path,
    criteria: ValidationCriteria | None = None,
) -> list[dict[str, Any]]:
    """Run explicitly selected candidates once on untouched OOS data."""
    if not selected_experiment_ids:
        raise ValueError("Select candidate IDs after reviewing validation; OOS cannot be auto-selected.")
    lookup = {candidate.experiment_id: candidate for candidate in candidates}
    unknown = set(selected_experiment_ids) - set(lookup)
    if unknown:
        raise ValueError(f"Selected IDs are not in the candidate set: {sorted(unknown)}")
    if len(selected_experiment_ids) != len(set(selected_experiment_ids)):
        raise ValueError("Selected candidate IDs must be unique.")
    prior_records = output_directory / "experiments.jsonl"
    if not prior_records.is_file():
        raise FileNotFoundError("Run and review a development/validation sweep before selecting OOS candidates.")
    validated_ids: set[str] = set()
    for line in prior_records.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("stage") == "validation":
                validated_ids.add(record["experiment_id"])
    if set(selected_experiment_ids) - validated_ids:
        raise ValueError("Every OOS selection must have a recorded validation attempt.")
    bars, _ = validate_ohlcv(
        _data_before(market_data, periods.range("out_of_sample")[1])
    )
    criteria = criteria or ValidationCriteria()
    output = Path(output_directory)
    if not output.is_dir():
        raise FileNotFoundError(f"Experiment output directory not found: {output}")
    start, end = periods.range("out_of_sample")
    records = [
        _execute(
            bars,
            lookup[experiment_id],
            "out_of_sample",
            start,
            end,
            criteria,
            len(selected_experiment_ids),
            output,
        )
        for experiment_id in selected_experiment_ids
    ]
    _rank_records(records)
    records_path = output / "experiments.jsonl"
    for record in records:
        _append_jsonl(records_path, record)
    _append_report(output / "comparison_report.md", records, criteria, "Final out-of-sample results")
    return records


def run_walk_forward(
    market_data: pd.DataFrame,
    candidates: list[CandidateSpec],
    folds: list[WalkForwardFold],
    output_directory: str | Path,
    criteria: ValidationCriteria | None = None,
) -> dict[str, Any]:
    """Select on each training window, evaluate only on its following test window."""
    if not candidates or not folds:
        raise ValueError("Walk-forward requires candidates and folds.")
    if any(
        _utc(previous.test_end) > _utc(current.test_start)
        for previous, current in zip(folds, folds[1:])
    ):
        raise ValueError("Walk-forward test folds must be chronological and non-overlapping.")
    maximum_test_end = max(_utc(fold.test_end) for fold in folds)
    bars, _ = validate_ohlcv(_data_before(market_data, maximum_test_end))
    criteria = criteria or ValidationCriteria()
    output = _new_output(output_directory)
    records_path = output / "experiments.jsonl"
    selections: list[dict[str, Any]] = []
    for fold in folds:
        train_start, train_end = _utc(fold.train_start), _utc(fold.train_end)
        test_start, test_end = _utc(fold.test_start), _utc(fold.test_end)
        training = [
            _execute(
                bars, candidate, f"{fold.name}_development", train_start, train_end,
                criteria, len(candidates), output,
            )
            for candidate in candidates
        ]
        training = _rank_records(training)
        for record in training:
            _append_jsonl(records_path, record)
        valid = [record for record in training if record["status"] == "completed"]
        if not valid:
            selections.append({"fold": fold.name, "status": "no_successful_training_runs"})
            continue
        selected_id = min(valid, key=lambda record: record["rank"])["experiment_id"]
        chosen = next(candidate for candidate in candidates if candidate.experiment_id == selected_id)
        test = _execute(
            bars, chosen, f"{fold.name}_forward_test", test_start, test_end,
            criteria, len(candidates), output,
        )
        _append_jsonl(records_path, test)
        selections.append(
            {
                "fold": fold.name,
                "selected_experiment_id": selected_id,
                "status": test["status"],
                "training_rank": min(valid, key=lambda row: row["rank"])["rank"],
                "test": test,
            }
        )

    frequencies: dict[str, int] = {}
    for selection in selections:
        if "selected_experiment_id" in selection:
            key = selection["selected_experiment_id"]
            frequencies[key] = frequencies.get(key, 0) + 1
    completed_folds = [
        selection for selection in selections if selection.get("status") == "completed"
    ]
    candidate_stability: dict[str, Any] = {}
    for experiment_id, count in frequencies.items():
        selected_records = [
            selection["test"]
            for selection in completed_folds
            if selection.get("selected_experiment_id") == experiment_id
        ]
        candidate_stability[experiment_id] = {
            "selected_folds": count,
            "positive_test_folds": sum(
                row["metrics"]["expectancy_per_trade"] > 0 for row in selected_records
            ),
            "mean_test_expectancy": float(
                np.mean([row["metrics"]["expectancy_per_trade"] for row in selected_records])
            ),
            "worst_test_drawdown_percent": max(
                row["metrics"]["max_drawdown_percent"] for row in selected_records
            ),
        }
    summary = {
        "selection_frequency": frequencies,
        "candidate_stability": candidate_stability,
        "folds": selections,
        "positive_test_fold_fraction": (
            sum(
                selection["test"]["metrics"]["expectancy_per_trade"] > 0
                for selection in completed_folds
            )
            / max(len(completed_folds), 1)
        ),
        "interpretation": "Walk-forward is historical validation, not independent final OOS proof.",
    }
    (output / "walk_forward_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _append_report(output / "comparison_report.md", [
        selection["test"] for selection in selections if "test" in selection
    ], criteria, "Walk-forward forward-test folds")
    return summary


def write_comparison_report(batch: ExperimentBatch, criteria: ValidationCriteria) -> Path:
    destination = batch.output_directory / "comparison_report.md"
    text = [
        "# Strategy Experiment Comparison",
        "",
        "> Research evidence only. A screening pass is not a profitability guarantee.",
        "> Final out-of-sample data was not accessed during the parameter sweep.",
        "",
        "## Predeclared screening criteria",
        "",
        f"```json\n{json.dumps(asdict(criteria), indent=2, sort_keys=True)}\n```",
        "",
        "## Development ranking",
        "",
    ]
    text.extend(_markdown_table(batch.development))
    text.extend(["", "## Validation of development-ranked shortlist", ""])
    text.extend(_markdown_table(batch.validation))
    text.extend(["", "## Validation break-downs", ""])
    text.extend(_breakdown_lines(batch.validation))
    text.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- Rankings combine percentile ranks of net expectancy per trade, lower drawdown, profit factor, positive-month consistency, and positive-regime consistency. They are comparative, not predictive.",
            "- Expectancy intervals are percentile bootstraps over trades with a Bonferroni adjustment across tested candidates. Serial dependence can make these intervals optimistic.",
            "- Trading costs include recorded commissions and slippage estimate. Session/regime groupings are descriptive and are not independent statistical samples.",
            "- The report records failed configurations in `experiments.jsonl`; inspect each status and error before interpreting rankings.",
            "- OOS results can only be produced by explicitly selecting IDs with `run_selected_out_of_sample`; do not change parameters after viewing OOS.",
        ]
    )
    destination.write_text("\n".join(text) + "\n", encoding="utf-8")
    return destination


def _markdown_table(records: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| Rank | Score | Experiment | Family | Status | Trades | Net expectancy | PF | Max DD % | Positive months | Positive regimes | Adjusted expectancy CI | Commission | Slippage est. | Screen |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for record in records:
        config = record["configuration"]
        metrics = record.get("metrics", {})
        ci = (
            f"[{metrics['expectancy_ci_lower']:.4g}, {metrics['expectancy_ci_upper']:.4g}]"
            if metrics.get("expectancy_ci_lower") is not None
            else "n/a"
        )
        pf = metrics.get("profit_factor")
        rows.append(
            f"| {record.get('rank', '-')} | {_format(record.get('rank_score'))} | "
            f"{record['experiment_id']} | {config['strategy_family']} | {record['status']} | "
            f"{metrics.get('sample_size_trades', 0)} | "
            f"{_format(metrics.get('expectancy_per_trade'))} | {_format(pf)} | "
            f"{_format(metrics.get('max_drawdown_percent'))} | "
            f"{_format(metrics.get('positive_month_fraction'))} | "
            f"{_format(metrics.get('positive_regime_fraction'))} | {ci} | "
            f"{_format(metrics.get('commission_cost'))} | "
            f"{_format(metrics.get('estimated_slippage_cost'))} | "
            f"{record.get('criteria', {}).get('status', 'not evaluated')} |"
        )
        if record.get("error"):
            rows.append(f"\n  Error: `{record['error']}`\n")
    return rows


def _breakdown_lines(records: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for record in records:
        if record.get("status") != "completed":
            continue
        metrics = record["metrics"]
        lines.extend(
            [
                f"### {record['experiment_id']}",
                "",
                f"- Trades by direction: `{json.dumps(metrics.get('by_direction', {}), sort_keys=True)}`",
                f"- Net P&L by year: `{json.dumps(metrics.get('by_year', {}), sort_keys=True)}`",
                f"- Net P&L by market regime: `{json.dumps(metrics.get('by_market_regime', {}), sort_keys=True)}`",
                f"- Trades by session date: `{json.dumps(metrics.get('by_session_date', {}), sort_keys=True)}`",
                "",
            ]
        )
    return lines or ["No successful validation records to break down."]


def _format(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)


def _append_report(path: Path, records: list[dict[str, Any]], criteria: ValidationCriteria, heading: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(f"\n## {heading}\n\n")
        file.write(f"Screening criteria: `{json.dumps(asdict(criteria), sort_keys=True)}`\n\n")
        file.write("\n".join(_markdown_table(records)))
        file.write("\n\n")
        file.write("\n".join(_breakdown_lines(records)))
        file.write("\n")


def main() -> None:
    import argparse

    from .data import load_market_data

    parser = argparse.ArgumentParser(
        description="Run reproducible strategy sweeps, walk-forward tests, or selected OOS evaluations."
    )
    parser.add_argument("mode", choices=("sweep", "out-of-sample", "walk-forward"))
    parser.add_argument("data", type=Path, help="Historical CSV or Parquet OHLCV data.")
    parser.add_argument("--spec", type=Path, required=True, help="JSON experiment specification.")
    parser.add_argument("--output", type=Path, required=True, help="New output path for sweep/walk-forward, existing path for OOS.")
    parser.add_argument(
        "--select",
        nargs="+",
        help="Explicit experiment IDs to evaluate on untouched OOS data.",
    )
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    bars, _ = load_market_data(args.data)
    candidates = [
        CandidateSpec(
            strategy_family=row["strategy_family"],
            strategy_parameters=row.get("strategy_parameters", {}),
            backtest_parameters=row.get("backtest_parameters", {}),
        )
        for row in spec.get("candidates", [])
    ]
    for grid in spec.get("grids", []):
        candidates.extend(
            expand_parameter_grid(
                grid["strategy_family"],
                grid.get("strategy_grid", {}),
                grid.get("execution_grid", {}),
            )
        )
    if not candidates:
        parser.error("Experiment spec must define at least one candidate or parameter grid.")
    criteria = ValidationCriteria(**spec.get("criteria", {}))
    if args.mode == "sweep":
        periods = ExperimentPeriods(**spec["periods"])
        batch = run_parameter_sweep(
            bars,
            candidates,
            periods,
            args.output,
            validation_top_n=spec.get("validation_top_n", 10),
            criteria=criteria,
        )
        print(f"Sweep complete. Selected for validation: {len(batch.shortlisted_ids)}")
        print(f"Review report: {batch.output_directory / 'comparison_report.md'}")
    elif args.mode == "out-of-sample":
        if not args.select:
            parser.error("out-of-sample mode requires explicit --select experiment IDs.")
        periods = ExperimentPeriods(**spec["periods"])
        records = run_selected_out_of_sample(
            bars, candidates, args.select, periods, args.output, criteria
        )
        print(f"Out-of-sample evaluations completed: {sum(record['status'] == 'completed' for record in records)}")
        print(f"Review report: {args.output / 'comparison_report.md'}")
    else:
        folds = [WalkForwardFold(**row) for row in spec["walk_forward_folds"]]
        summary = run_walk_forward(bars, candidates, folds, args.output, criteria)
        print(f"Walk-forward folds completed: {len(summary['folds'])}")
        print(f"Review report: {args.output / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
