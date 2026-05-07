from __future__ import annotations

import pandas as pd

from app.services.trader_health.forward_backtest_service import (
    _normalize_pnl_frame,
    _normalize_trades_frame,
    _to_naive_datetime_series,
    _to_naive_timestamp,
)


def test_normalize_pnl_frame_handles_duplicate_date_columns() -> None:
    pnl_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "BALANCE": [10000.0, 10050.0],
            "EQUITY": [10000.0, 10040.0],
        }
    ).set_index("date")
    pnl_df["date"] = ["2024-01-01", "2024-01-02"]

    out = _normalize_pnl_frame(pnl_df)

    assert list(out.columns) == ["date", "balance", "equity"]
    assert len(out) == 2
    assert out["date"].isna().sum() == 0


def test_normalize_trades_frame_handles_duplicate_columns() -> None:
    trades_df = pd.DataFrame(
        [
            ["2024-01-01", "2024-01-05", 25.0, "buy"],
            ["2024-01-06", "2024-01-08", -10.0, "sell"],
        ],
        columns=["entry_time", "exit_time", "profit", "side"],
    )
    trades_df.columns = ["entry_time", "exit_time", "profit", "profit"]

    out = _normalize_trades_frame(trades_df)

    assert list(out.columns) == ["entry_time", "exit_time", "profit", "side"]
    assert len(out) == 2
    assert out["profit"].tolist() == [25.0, -10.0]


def test_timestamp_helpers_normalize_tz_aware_values_to_naive() -> None:
    aware = "2026-04-24T18:00:00+00:00"
    ts = _to_naive_timestamp(aware)
    series = _to_naive_datetime_series(pd.Series([aware, "2026-04-25T18:00:00+00:00"]))

    assert ts.tzinfo is None
    assert series.dt.tz is None
