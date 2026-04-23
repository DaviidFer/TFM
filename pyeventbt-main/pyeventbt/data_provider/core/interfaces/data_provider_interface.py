"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Protocol
from pyeventbt.events.events import BarEvent
#from ..entities.bar import Bar
from decimal import Decimal

class IDataProvider:
    """
    IDataProvider is an interface for all subsequent (inherited) data providers (both live and historic).

    The goal of a (derived) DataProvider object is to output a generated set of bars for each symbol requested.

    This will replicate how a live strategy would function as current market data would be sent "down the pipe".
    Thus a historic and live system will be treated identically by the rest of the backtesting suite.
    """

    def get_latest_bar(self, symbol: str, timeframe: str):
        """
        Returns the latest bar available in the data handler.
        """
        raise NotImplementedError()

    def get_latest_bars(self, symbol: str, timeframe: str, N:int):
        """
        Returns the latest bars from the data source.

        :return: A list of Bar objects representing the latest bars.
        """
        raise NotImplementedError()
    
    def get_latest_tick(self, symbol: str) -> dict:
        """Returns a dict with the MT5 Tick object structure"""
        raise NotImplementedError()

    def get_latest_bid(self, symbol: str) -> Decimal:
        """
        Returns the latest bid price for the symbol.
        """
        raise NotImplementedError()

    def get_latest_ask(self, symbol: str) -> Decimal:
        """
        Returns the latest ask price for the symbol.
        """
        raise NotImplementedError()

    def get_latest_datetime(self, symbol: str, timeframe: str):
        """
        Returns a Timestamp object for the last tick/bar.
        """
        raise NotImplementedError()

    def update_bars(self) -> list[BarEvent] | None:
        """
        In connectors: returns a list with BarEvents (to the service)
        In service: returns None
        """
        raise NotImplementedError()


    # Implementar como servicio
    # def get_latest_bar_datetime(self) -> pd.Timestamp:
    #     return self.get_latest_bar().datetime
