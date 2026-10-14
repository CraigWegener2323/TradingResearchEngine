import pandas as pd
import pytest

from trading_research_engine.data import load_market_data, validate_ohlcv


def sample_frame():
    return pd.DataFrame(
        {
            "timestamp": [
                "2025-01-02T14:31:00Z",
                "2025-01-02T14:30:00Z",
                "2025-01-02T14:33:00Z",
            ],
            "open": [101, 100, 102],
            "high": [102, 101, 103],
            "low": [100, 99, 101],
            "close": [101.5, 100.5, 102.5],
            "volume": [10, 20, 30],
        }
    )


def test_validation_sorts_bars_and_reports_missing_intervals():
    bars, report = validate_ohlcv(sample_frame(), expected_interval="1min")

    assert bars["timestamp"].is_monotonic_increasing
    assert str(bars["timestamp"].dt.tz) == "UTC"
    assert report.row_count == 3
    assert report.duplicate_count == 0
    assert report.missing_bar_count == 1
    assert report.gaps[0]["missing_bars"] == 1


def test_validation_drops_non_contract_columns():
    frame = sample_frame()
    frame["future_return"] = [999, 999, 999]

    bars, _ = validate_ohlcv(frame)

    assert list(bars.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_validation_rejects_duplicate_timestamps():
    frame = pd.concat([sample_frame(), sample_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(frame)


def test_validation_rejects_duplicate_instants_with_different_timezones():
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-02T14:30:00Z", "2025-01-02T09:30:00-05:00"],
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
            "volume": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(frame)


def test_validation_rejects_invalid_ohlc_and_null_fields():
    invalid = sample_frame()
    invalid.loc[0, "high"] = 98
    with pytest.raises(ValueError, match="invalid OHLC"):
        validate_ohlcv(invalid)

    missing = sample_frame()
    missing.loc[0, "volume"] = None
    with pytest.raises(ValueError, match="finite and non-null"):
        validate_ohlcv(missing)


def test_csv_import_normalizes_columns_and_sorts(tmp_path):
    source = tmp_path / "bars.csv"
    frame = sample_frame().rename(columns=str.upper)
    frame.to_csv(source, index=False)

    bars, report = load_market_data(source, expected_interval="1min")

    assert list(bars.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert bars["timestamp"].is_monotonic_increasing
    assert report.row_count == 3


def test_parquet_import(tmp_path):
    pytest.importorskip("pyarrow")
    source = tmp_path / "bars.parquet"
    sample_frame().to_parquet(source, index=False)

    bars, _ = load_market_data(source)

    assert len(bars) == 3
    assert bars["timestamp"].is_monotonic_increasing


def test_import_rejects_unsupported_file_extension(tmp_path):
    source = tmp_path / "bars.txt"
    source.write_text("not market data", encoding="utf-8")

    with pytest.raises(ValueError, match="extension"):
        load_market_data(source)


def test_parquet_import_explains_missing_optional_engine(tmp_path, monkeypatch):
    source = tmp_path / "bars.parquet"
    source.touch()

    def missing_engine(*args, **kwargs):
        raise ImportError("no parquet engine")

    monkeypatch.setattr(pd, "read_parquet", missing_engine)
    with pytest.raises(ImportError, match="Parquet support"):
        load_market_data(source)
