from __future__ import annotations

from datetime import datetime
from queue import Queue
from typing import Dict

import pandas as pd

from app.core.structured_logging import emit_log
from app.execution.mt5_events import DataEvent


class MT5DataProvider:
    """
    Integración directa del concepto de data_provider de mt5-framework.
    Publica DataEvent cuando detecta nueva vela cerrada.
    """

    def __init__(self, *, events_queue: Queue, symbol_list: list[str], timeframe: str = "1d") -> None:
        self.events_queue = events_queue
        self.symbols = list(symbol_list)
        self.timeframe = timeframe
        self.last_bar_datetime: Dict[str, datetime] = {symbol: datetime.min for symbol in self.symbols}
        self._mt5 = None
        self._no_bar_warned_symbols: set[str] = set()

    def _get_mt5(self):
        if self._mt5 is not None:
            return self._mt5
        try:
            import MetaTrader5 as mt5
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Falta dependencia MetaTrader5. "
                "Instala con `python -m pip install MetaTrader5` o usa modo paper/local."
            ) from exc

        self._mt5 = mt5
        return self._mt5

    def _map_timeframes(self, timeframe: str) -> int:
        mt5 = self._get_mt5()
        timeframe_mapping = {
            "1min": mt5.TIMEFRAME_M1,
            "5min": mt5.TIMEFRAME_M5,
            "15min": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "1w": mt5.TIMEFRAME_W1,
            "1M": mt5.TIMEFRAME_MN1,
        }
        tf = timeframe_mapping.get(timeframe)
        if tf is None:
            raise ValueError(f"Timeframe '{timeframe}' no es válido.")
        return tf

    def _dateprint(self) -> str:
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]

    def get_latest_closed_bar(self, symbol: str, timeframe: str) -> pd.Series:
        mt5 = self._get_mt5()
        tf = self._map_timeframes(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 1, 1)  # 1 = última barra cerrada
        if rates is None or len(rates) == 0:
            return pd.Series(dtype="float64")
        bars = pd.DataFrame(rates)
        bars["time"] = pd.to_datetime(bars["time"], unit="s")
        bars.set_index("time", inplace=True)
        bars.rename(columns={"tick_volume": "tickvol", "real_volume": "vol"}, inplace=True)
        bars = bars[["open", "high", "low", "close", "tickvol", "vol", "spread"]]
        return bars.iloc[-1]

    def get_latest_closed_bars(self, symbol: str, timeframe: str, num_bars: int = 200) -> pd.DataFrame:
        mt5 = self._get_mt5()
        tf = self._map_timeframes(timeframe)
        bars_np = mt5.copy_rates_from_pos(symbol, tf, 1, max(1, int(num_bars)))
        if bars_np is None or len(bars_np) == 0:
            return pd.DataFrame()
        bars = pd.DataFrame(bars_np)
        bars["time"] = pd.to_datetime(bars["time"], unit="s")
        bars.set_index("time", inplace=True)
        bars.rename(columns={"tick_volume": "tickvol", "real_volume": "vol"}, inplace=True)
        bars = bars[["open", "high", "low", "close", "tickvol", "vol", "spread"]]
        return bars

    def check_for_new_data(self, *, force_emit_snapshot: bool = False) -> None:
        for symbol in self.symbols:
            latest_bar = self.get_latest_closed_bar(symbol, self.timeframe)
            if latest_bar.empty:
                if symbol not in self._no_bar_warned_symbols:
                    print(f"{self._dateprint()} - ERROR: No se pudo obtener datos de {symbol}.")
                    self._no_bar_warned_symbols.add(symbol)
                emit_log("mt5_data_provider", "no_bar_data", console=False, symbol=symbol, timeframe=self.timeframe)
                continue
            if force_emit_snapshot or latest_bar.name > self.last_bar_datetime[symbol]:
                self.last_bar_datetime[symbol] = latest_bar.name
                emit_log(
                    "mt5_data_provider",
                    "new_bar_detected",
                    console=False,
                    symbol=symbol,
                    timeframe=self.timeframe,
                    bar_time=str(latest_bar.name),
                    close=float(latest_bar.get("close", 0.0)),
                )
                self.events_queue.put(DataEvent(symbol=symbol, data=latest_bar))

