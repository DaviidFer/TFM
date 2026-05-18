import pandas as pd

from app.runtime.development_operational_supervisor import DevelopmentOperationalSupervisor


def test_normalize_pnl_frame_handles_duplicate_date_columns() -> None:
    base_dates = pd.date_range("2026-01-01", periods=3, freq="D", name="date")
    pnl_df = pd.DataFrame(
        {
            "date": base_dates.astype(str),
            "BALANCE": [10000.0, 10010.0, 10020.0],
            "EQUITY": [10000.0, 10012.0, 10025.0],
        },
        index=base_dates,
    )

    out = DevelopmentOperationalSupervisor._normalize_pnl_frame(pnl_df)

    assert list(out.columns) == ["date", "balance", "equity"]
    assert len(out) == 3
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert float(out["equity"].iloc[-1]) == 10025.0


def test_normalize_pnl_frame_preserves_existing_date_column_from_csv_reload() -> None:
    pnl_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "balance": [10000.0, 10010.0, 10020.0],
            "equity": [10000.0, 10015.0, 10030.0],
        }
    )

    out = DevelopmentOperationalSupervisor._normalize_pnl_frame(pnl_df)

    assert list(out.columns) == ["date", "balance", "equity"]
    assert out["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert float(out["equity"].iloc[-1]) == 10030.0


def test_normalize_trades_frame_handles_duplicate_columns() -> None:
    trades_df = pd.DataFrame(
        [
            ["2026-01-02", "2026-01-03", 12.5, "buy"],
            ["2026-01-04", "2026-01-05", -4.0, "sell"],
        ],
        columns=["entry_time", "exit_time", "gross_profit", "type"],
    )
    trades_df = pd.concat([trades_df, trades_df[["entry_time"]]], axis=1)

    out = DevelopmentOperationalSupervisor._normalize_trades_frame(trades_df)

    assert list(out.columns) == ["entry_time", "exit_time", "profit", "side"]
    assert len(out) == 2
    assert pd.api.types.is_datetime64_any_dtype(out["entry_time"])
    assert pd.api.types.is_datetime64_any_dtype(out["exit_time"])
    assert out["profit"].tolist() == [12.5, -4.0]


def test_compute_trade_streaks_tracks_consecutive_wins_and_losses() -> None:
    max_win, max_loss = DevelopmentOperationalSupervisor._compute_trade_streaks(
        [1.0, 2.0, -1.0, -2.0, -3.0, 5.0, 6.0, 7.0]
    )

    assert max_win == 3
    assert max_loss == 3


def test_compute_trade_streaks_ignores_breakeven_without_resetting() -> None:
    """Ceros intermedios no deben dejar racha máxima en 1 (caso PyEventBT)."""
    max_win, max_loss = DevelopmentOperationalSupervisor._compute_trade_streaks(
        [1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0]
    )
    assert max_win == 3
    assert max_loss == 3  # -1, luego -1, -1 con ceros que no cortan
