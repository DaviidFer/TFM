"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from decimal import Decimal
from typing import Protocol

import pandas as pd
from pyeventbt.events.events import BarEvent
from ..entities.open_position import OpenPosition
from ..entities.closed_position import ClosedPosition
from ..entities.pending_order import PendingOrder

class IPortfolio:
    """
    PortfolioBase is an abstract base class providing an interface for all subsequent (inherited) portfolio handling classes (both live and historic).

    The goal of a (derived) Portfolio object is to generate SignalEvents. ?? It is market event trigers strategy and if signal the strategy trigers signal event
    These are then sent to an (inherited) ExecutionHandler object.
    The Portfolio object maintains a copy of a DataHandler object which it uses to obtain bars as necessary.
    It also maintains a copy of a Strategy object which it uses to generate the SignalEvents.
    """

    def _update_portfolio(self, bar_event: BarEvent) -> None:
        """
        Updates the portfolio current state.
        """
        raise NotImplementedError("Should implement update_portfolio()")
    
    def get_positions(self, symbol: str = '', ticket: int = None) -> tuple[OpenPosition]:
        """
        Returns a tuple of OpenPosition or OpenPosition objects as defined in our Domain entities
        """
        raise NotImplementedError("Should implement get_open_positions()")
    
    def get_pending_orders(self, symbol: str = '', ticket: int = None) -> tuple[PendingOrder]:
        """
        Returns a tuple of PendingOrder objects as defined in our Domain entities
        """
        raise NotImplementedError("Should implement get_pending_orders()")
    
    def get_number_of_strategy_open_positions_by_symbol(self, symbol: str) -> dict[str, int]:
        """
        Returns a dictionary with the number of open positions per symbol
        """
        raise NotImplementedError("Should implement get_number_of_strategy_open_positions_by_symbol()")
    
    def get_number_of_strategy_pending_orders_by_symbol(self, symbol: str) -> dict[str, int]:
        """
        Returns a dictionary with the number of pending orders per symbol
        """
        raise NotImplementedError("Should implement get_number_of_strategy_pending_orders_by_symbol()")
    
    # def get_closed_positions(self) -> list[ClosedPosition]:
    #     """
    #     Returns a list of ClosedPosition objects
    #     """
    #     raise NotImplementedError("Should implement get_closed_positions()")
    
    def get_account_balance(self) -> Decimal:
        """
        Returns the current balance of the account.
        """
        raise NotImplementedError("Should implement get_account_balance()")
    
    def get_account_equity(self) -> Decimal:
        """
        Returns the current equity of the account.
        """
        raise NotImplementedError("Should implement get_account_equity()")
    
    def get_account_unrealised_pnl(self) -> Decimal:
        """
        Returns the current unrealised profit and loss of the account.
        """
        raise NotImplementedError("Should implement get_account_unrealised_pnl()")
    
    def get_account_realised_pnl(self) -> Decimal:
        """
        Returns the realised profit and loss of the account.
        """
        raise NotImplementedError("Should implement get_account_realised_pnl()")
    
    def _export_historical_pnl_dataframe() -> pd.DataFrame:
        raise NotImplementedError()
    
    def _export_historical_pnl_json(self) -> str:
        """
        Returns a dictionary with the historical PnL data.
        """
        raise NotImplementedError()

    def _export_csv_historical_pnl(self, file_path: str) -> None:
        raise NotImplementedError()
    
    def _update_portfolio_end_of_backtest(self) -> None:
        raise NotImplementedError()