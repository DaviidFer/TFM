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


def test_load_or_build_universe_prefers_mt5_over_local_csvs(monkeypatch, tmp_path: Path) -> None:
    stocks_txt = tmp_path / "dz_stocks.txt"
    etfs_txt = tmp_path / "dz_etfs.txt"
    universe_csv = tmp_path / "dz_universe_full.csv"
    mt5_raw_csv = tmp_path / "mt5_all_symbols_raw.csv"

    monkeypatch.setattr(download_module, "REBUILD_UNIVERSE_FROM_MT5", True)
    monkeypatch.setattr(
        download_module,
        "discover_universe_from_local_data",
        lambda _pd: (["LOCAL"], [], pd.DataFrame([{"asset_type": "Stock", "symbol": "LOCAL", "description": "", "path": "LOCAL"}])),
    )
    monkeypatch.setattr(download_module, "_import_dependencies", lambda: (pd, None, object()))

    def _fake_build_universe_from_mt5(**kwargs):
        df = pd.DataFrame(
            [
                {"asset_type": "Stock", "symbol": "AAPL", "description": "", "path": "MT5"},
                {"asset_type": "ETF", "symbol": "XLV", "description": "", "path": "MT5"},
            ]
        )
        return ["AAPL"], ["XLV"], df

    monkeypatch.setattr(download_module, "build_universe_from_mt5", _fake_build_universe_from_mt5)

    stocks, etfs, df = download_module.load_or_build_universe(
        pd=pd,
        stocks_txt=stocks_txt,
        etfs_txt=etfs_txt,
        universe_csv=universe_csv,
        mt5_raw_csv=mt5_raw_csv,
    )

    assert stocks == ["AAPL"]
    assert etfs == ["XLV"]
    assert set(df["symbol"].tolist()) == {"AAPL", "XLV"}


def test_run_data_download_full_rebuild_removes_stale_and_failed_csvs(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "datos"
    stocks_dir = data_root / "Stocks"
    etfs_dir = data_root / "ETFs"
    universe_dir = data_root / "_universe"
    stocks_dir.mkdir(parents=True, exist_ok=True)
    etfs_dir.mkdir(parents=True, exist_ok=True)
    universe_dir.mkdir(parents=True, exist_ok=True)

    for folder, symbol in (
        (stocks_dir, "AAPL"),
        (stocks_dir, "BAD"),
        (stocks_dir, "OLD"),
        (etfs_dir, "XLV"),
    ):
        pd.DataFrame({"Date": ["2024-01-01"], "Close": [100.0]}).to_csv(folder / f"{symbol}.csv", index=False)

    monkeypatch.setattr(download_module, "DATA_ROOT", data_root)
    monkeypatch.setattr(download_module, "STOCKS_DIR", stocks_dir)
    monkeypatch.setattr(download_module, "ETFS_DIR", etfs_dir)
    monkeypatch.setattr(download_module, "UNIVERSE_DIR", universe_dir)
    monkeypatch.setattr(download_module, "_prepare_directories", lambda: None)
    monkeypatch.setattr(download_module, "_import_dependencies", lambda: (pd, object(), object()))

    universe_df = pd.DataFrame(
        [
            {"asset_type": "Stock", "symbol": "AAPL", "description": "", "path": "MT5"},
            {"asset_type": "Stock", "symbol": "BAD", "description": "", "path": "MT5"},
            {"asset_type": "ETF", "symbol": "XLV", "description": "", "path": "MT5"},
        ]
    )
    monkeypatch.setattr(
        download_module,
        "load_or_build_universe",
        lambda **kwargs: (["AAPL", "BAD"], ["XLV"], universe_df.copy()),
    )

    def _fake_update_group(_pd, _yf, symbols, asset_type, _folder):
        rows = []
        for symbol in symbols:
            status = "failed" if symbol == "BAD" else "updated"
            rows.append(
                {
                    "asset_type": asset_type,
                    "symbol": symbol,
                    "status": status,
                    "rows_added": 1 if status == "updated" else 0,
                    "yahoo_symbol": symbol,
                    "error": None if status == "updated" else "not_found",
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(download_module, "update_group", _fake_update_group)

    rebuilt_universe, all_results, failed = download_module.run_data_download()

    assert set(rebuilt_universe["symbol"].tolist()) == {"AAPL", "XLV"}
    assert failed["symbol"].tolist() == ["BAD"]
    assert (stocks_dir / "AAPL.csv").exists()
    assert (etfs_dir / "XLV.csv").exists()
    assert not (stocks_dir / "BAD.csv").exists()
    assert not (stocks_dir / "OLD.csv").exists()


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
