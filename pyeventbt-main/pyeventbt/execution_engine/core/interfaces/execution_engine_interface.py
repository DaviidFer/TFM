"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Protocol
from pyeventbt.events.events import BarEvent, OrderEvent
from pyeventbt.broker.mt5_broker.core.entities.order_send_result import OrderSendResult
from pyeventbt.portfolio.core.entities.open_position import OpenPosition
from decimal import Decimal

class IExecutionEngine:
    """
    The IExecutionEngine interface handles the interaction between a set of order objects created by a portfolio and the
    market data used by the strategy. It receives Order events from the event queue and processes them, converting them
    into Fill events that are placed back onto the event queue.
    """
    
    # def check_if_pending_orders_filled(self, bar_event: BarEvent) -> None:
    #     """
    #     Checks if any pending order have been filled and generates a fill event, for every order, if they have.
    #     """
    #     raise NotImplementedError()
    
    def _process_order_event(self, order_event: OrderEvent) -> None:
        """
        Processes an order event
        """
        raise NotImplementedError()
    
    def _update_values_and_check_executions_and_fills(self, bar_event: BarEvent) -> None:
        """
        Updates the account values, checks if any pending order has been filled and if any SL/TP has been hit.
        """
        raise NotImplementedError()
    
    def _send_market_order(self, order_event: OrderEvent) -> OrderSendResult:
        """
        Executes a Market Order and returns an OrderSendResult
        """
        raise NotImplementedError()
    
    def _send_pending_order(self, order_event: OrderEvent) -> OrderSendResult:
        """
        Sends a pending order to the broker
        """
        raise NotImplementedError()
    
    def close_position(self, position_ticket: int) -> OrderSendResult:
        """
        Close a currently opened position
        """
        raise NotImplementedError()
    
    def cancel_pending_order(self, order_ticket: int) -> OrderSendResult:
        """
        Cancels a pending order
        """
        raise NotImplementedError()
    
    def cancel_all_strategy_pending_orders(self) -> None:
        """
        Cancels all pending orders from the strategy, in all symbols.
        """
        raise NotImplementedError()
    
    def cancel_all_strategy_pending_orders_by_type_and_symbol(self, order_type:str, symbol: str) -> None:
        """
        Cancels all specific type of pending orders from the strategy, in a specific symbol.
        Example: cancels all BUY_LIMIT orders in EURUSD.
        """
        raise NotImplementedError()
    
    def close_all_strategy_positions(self) -> None:
        """
        Close all positions from the strategy
        """
        raise NotImplementedError()
    
    def close_strategy_long_positions_by_symbol(self, symbol: str) -> None:
        raise NotImplementedError()
    
    def close_strategy_short_positions_by_symbol(self, symbol: str) -> None:
        raise NotImplementedError()
    
    def update_position_sl_tp(self, position_ticket: int, new_sl: float = 0.0, new_tp: float = 0.0) -> None:
        raise NotImplementedError()
    
    # Here we'll define helper methods (THAT WILL BE ACCESSED FROM THE PORTFOLIO CLASS) for getting important data from the MT5 account like account currency, balance, equity, etc.
    def _get_account_currency(self) -> str:
        """Get account currency"""
        raise NotImplementedError()
    
    def _get_account_balance(self) -> Decimal:
        """Get account balance in account currency"""
        raise NotImplementedError()
    
    def _get_account_equity(self) -> Decimal:
        """Get account equity in account currency"""
        raise NotImplementedError()
    
    def _get_account_floating_profit(self) -> Decimal:
        """Get account floating profit in account currency"""
        raise NotImplementedError()
    
    def _get_account_used_margin(self) -> Decimal:
        """Get account used margin in account currency"""
        raise NotImplementedError()
    
    def _get_account_free_margin(self) -> Decimal:
        """Get account free margin in account currency"""
        raise NotImplementedError()
    
    def _get_total_number_of_pending_orders(self) -> int:
        """Get total number of active pending orders"""
        raise NotImplementedError()
    
    def _get_strategy_pending_orders(self) -> tuple:
        """Get current pending orders"""
        raise NotImplementedError()
    
    def _get_total_number_of_positions(self) -> int:
        """Get total number of active pending orders"""
        raise NotImplementedError()
    
    def _get_strategy_positions(self) -> tuple[OpenPosition]:
        """Get current positions"""
        raise NotImplementedError()
    
    def _get_symbol_min_volume(self, symbol: str) -> Decimal:
        """Get symbol min volume"""
        raise NotImplementedError()
    
    def enable_trading(self) -> None:
        pass
        
    def disable_trading(self) -> None:
        pass

    
    #etc