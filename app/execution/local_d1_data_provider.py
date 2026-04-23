from __future__ import annotations

from datetime import datetime
from queue import Queue
from typing import Dict

import pandas as pd

from app.execution.local_data_provider import LocalMarketDataProvider
from app.execution.mt5_events import DataEvent


class LocalD1DataProvider:
    """
    Data provider compatible con el runtime live, pero usando CSV locales.
    Emite un DataEvent cuando detecta una barra D1 más reciente.
    """

    def __init__(self, *, events_queue: Queue, market_data: LocalMarketDataProvider, symbol_list: list[str]) -> None:
        self.events_queue = events_queue
        self.market_data = market_data
        self.symbols = list(symbol_list)
        self.timeframe = "1d"
        self.last_bar_datetime: Dict[str, datetime] = {s: datetime.min for s in self.symbols}

    def get_latest_closed_bars(self, symbol: str, timeframe: str, num_bars: int = 260) -> pd.DataFrame:
        path = self.market_data.get_csv_path(symbol)
        if not path:
            return pd.DataFrame()
        df = pd.read_csv(path)
        rename = {"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "vol"}
        df = df.rename(columns=rename)
        if "time" not in df.columns:
            return pd.DataFrame()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        df = df.tail(max(1, int(num_bars))).copy()
        df = df.set_index("time")
        if "tickvol" not in df.columns:
            df["tickvol"] = 0.0
        if "spread" not in df.columns:
            df["spread"] = 0.0
        if "vol" not in df.columns:
            df["vol"] = 0.0
        return df[["open", "high", "low", "close", "tickvol", "vol", "spread"]]

    def check_for_new_data(self) -> None:
        for symbol in self.symbols:
            bars = self.get_latest_closed_bars(symbol, timeframe="1d", num_bars=1)
            if bars.empty:
                continue
            latest_bar = bars.iloc[-1]
            if latest_bar.name > self.last_bar_datetime[symbol]:
                self.last_bar_datetime[symbol] = latest_bar.name
                self.events_queue.put(DataEvent(symbol=symbol, data=latest_bar))

