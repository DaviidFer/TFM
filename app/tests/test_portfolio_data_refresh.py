from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.execution.local_data_provider import LocalMarketDataProvider
from app.services.portfolio_support.data_refresh import PortfolioOHLCRefreshService
from app.toolbox.data_download import download as download_module


def test_discover_universe_from_local_data_prefers_existing_csvs(monkeypatch, tmp_path: Path) -> None:
    stocks_dir = tmp_path / "Stocks"
    etfs_dir = tmp_path / "ETFs"
    stocks_dir.mkdir(parents=True, exist_ok=True)
    etfs_dir.mkdir(parents=True, exist_ok=True)
    (stocks_dir / "AAPL.csv").write_text("Date,Close\n2024-01-01,100\n", encoding="utf-8")
    (stocks_dir / "MSFT.csv").write_text("Date,Close\n2024-01-01,200\n", encoding="utf-8")
    (etfs_dir / "XLV.csv").write_text("Date,Close\n2024-01-01,50\n", encoding="utf-8")

    monkeypatch.setattr(download_module, "STOCKS_DIR", stocks_dir)
    monkeypatch.setattr(download_module, "ETFS_DIR", etfs_dir)

    stocks, etfs, df = download_module.discover_universe_from_local_data(pd)

    assert stocks == ["AAPL", "MSFT"]
    assert etfs == ["XLV"]
    assert set(df["symbol"].tolist()) == {"AAPL", "MSFT", "XLV"}


def test_portfolio_refresh_uses_only_requested_symbols(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "AAPL.csv"
    pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Close": [100.0, 101.0]}).to_csv(csv_path, index=False)
    provider = LocalMarketDataProvider({"AAPL": str(csv_path)})
    monkeypatch.setattr(provider, "refresh_symbol_registry", lambda: None)

    captured: dict[str, object] = {}

    def _fake_run_data_download(symbols=None):
        captured["symbols"] = set(symbols or [])
        universe_df = pd.DataFrame({"symbol": ["AAPL"]})
        all_results = pd.DataFrame(
            [
                {"symbol": "AAPL", "status": "updated"},
                {"symbol": "MSFT", "status": "updated"},
            ]
        )
        failed = pd.DataFrame(columns=["symbol", "status"])
        return universe_df, all_results, failed

    monkeypatch.setattr("app.services.portfolio_support.data_refresh.run_data_download", _fake_run_data_download)

    service = PortfolioOHLCRefreshService(provider)
    result = service.refresh(["AAPL"])

    assert captured["symbols"] == {"AAPL"}
    assert result.refreshed_symbols == ["AAPL"]
    assert result.n_requested_symbols == 1
