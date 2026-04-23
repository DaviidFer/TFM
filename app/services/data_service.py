from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_asset_ohlc(asset_csv_path: str | Path) -> pd.DataFrame:
    """
    Carga un CSV estilo Yahoo y devuelve OHLC normalizado:
    index: DatetimeIndex
    columns: open, high, low, close
    """
    p = Path(asset_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe CSV de activo: {p}")

    df = pd.read_csv(p)
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    }
    df = df.rename(columns=rename_map)
    needed = ["date", "open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sin columnas requeridas {missing}: {p}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if out.empty:
        raise ValueError(f"OHLC vacío tras limpieza: {p}")
    return out

