from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class LocalMarketDataProvider:
    """
    Sustituye la petición de datos MT5/forex por datos locales de Stocks/ETFs.
    """

    def __init__(self, asset_csv_by_symbol: Dict[str, str] | None = None) -> None:
        if asset_csv_by_symbol:
            self.asset_csv_by_symbol = {k.upper(): v for k, v in asset_csv_by_symbol.items()}
        else:
            self.asset_csv_by_symbol = self._discover_default_csvs()
        self._cache: Dict[str, pd.DataFrame] = {}

    def _discover_default_csvs(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for root in [Path("datos/Stocks"), Path("datos/ETFs")]:
            if not root.exists():
                continue
            for p in root.glob("*.csv"):
                out[p.stem.upper()] = str(p)
        return out

    def _load_symbol_df(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        if sym in self._cache:
            return self._cache[sym]
        path = self.asset_csv_by_symbol.get(sym)
        if not path:
            raise FileNotFoundError(f"No hay CSV local para símbolo {sym}")
        df = pd.read_csv(path)
        rename = {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        df = df.rename(columns=rename)
        if "date" not in df.columns:
            raise ValueError(f"CSV inválido para {sym}: falta columna date/Date")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        self._cache[sym] = df
        return df

    def get_latest_bar(self, symbol: str) -> Dict[str, object]:
        df = self._load_symbol_df(symbol)
        if df.empty:
            raise ValueError(f"Sin datos para {symbol}")
        row = df.iloc[-1]
        return {
            "symbol": symbol.upper(),
            "date": row["date"].isoformat(),
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "volume": float(row.get("volume", 0.0)) if "volume" in df.columns else None,
            "rows": int(len(df)),
        }

    def get_range_info(self, symbol: str) -> Dict[str, object]:
        df = self._load_symbol_df(symbol)
        if df.empty:
            return {"symbol": symbol.upper(), "rows": 0, "start_date": None, "end_date": None}
        return {
            "symbol": symbol.upper(),
            "rows": int(len(df)),
            "start_date": df["date"].min().date().isoformat(),
            "end_date": df["date"].max().date().isoformat(),
        }

    def has_symbol(self, symbol: str) -> bool:
        return symbol.upper() in self.asset_csv_by_symbol

    def get_csv_path(self, symbol: str) -> Optional[str]:
        return self.asset_csv_by_symbol.get(symbol.upper())

    def invalidate_cache(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._cache.clear()
            return
        self._cache.pop(symbol.upper(), None)

    def refresh_symbol_registry(self) -> None:
        self.asset_csv_by_symbol = self._discover_default_csvs()
        self.invalidate_cache()

