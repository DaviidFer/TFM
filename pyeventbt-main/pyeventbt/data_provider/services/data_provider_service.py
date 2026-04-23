"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from ..core.interfaces.data_provider_interface import IDataProvider
from ..connectors.csv_data_connector import CSVDataProvider
from ..connectors.mt5_live_data_connector import Mt5LiveDataProvider
from ..core.configurations.data_provider_configurations import (
    BaseDataConfig,
    MT5LiveDataConfig,
    CSVBacktestDataConfig)
import pyeventbt.trading_context.trading_context as trading_context

from pyeventbt.events.events import BarEvent
from queue import Queue
import pandas as pd
from decimal import Decimal

class DataProvider(IDataProvider):
    
    def __init__(self, events_queue: Queue, data_config: BaseDataConfig, trading_context: trading_context.TypeContext = trading_context.TypeContext.BACKTEST) -> None:
        
        self.trading_context = trading_context
        self.events_queue = events_queue
        self.DATA_PROVIDER = self._get_data_provider(data_config)
        self.continue_backtest = True
        self.close_positions_end_of_data = False

    def _get_data_provider(self, data_config: BaseDataConfig) -> IDataProvider:
        
        if isinstance(data_config, MT5LiveDataConfig):
            return Mt5LiveDataProvider(configs=data_config)
        elif isinstance(data_config, CSVBacktestDataConfig):
            return CSVDataProvider(configs=data_config)
        else:
            raise Exception(f"Unknown Data Provider: {data_config}")

    def get_latest_bar(self, symbol: str, timeframe: str) -> pd.Series:
        return self.DATA_PROVIDER.get_latest_bar(symbol, timeframe)

    def get_latest_bars(self, symbol: str, timeframe: str = None, N: int = 2) -> pd.DataFrame:
        return self.DATA_PROVIDER.get_latest_bars(symbol, timeframe, N)
    
    def get_latest_tick(self, symbol: str) -> dict:
        return self.DATA_PROVIDER.get_latest_tick(symbol)
    
    def get_latest_bid(self, symbol: str) -> Decimal:
        return self.DATA_PROVIDER.get_latest_bid(symbol)

    def get_latest_ask(self, symbol: str) -> Decimal:
        return self.DATA_PROVIDER.get_latest_ask(symbol)
    
    def get_latest_datetime(self, symbol: str, timeframe: str = None) -> pd.Timestamp:
        return self.DATA_PROVIDER.get_latest_datetime(symbol, timeframe)
    
    def update_bars(self) -> None:
        for bar_event in self.DATA_PROVIDER.update_bars():
            self._put_bar_event(bar_event)
        
        if self.trading_context == "BACKTEST":
            self.close_positions_end_of_data = self.DATA_PROVIDER.close_positions_end_of_data
            self.continue_backtest = self.DATA_PROVIDER.continue_backtest
    
    def _put_bar_event(self, bar_event: BarEvent) -> None:
        self.events_queue.put(bar_event)
