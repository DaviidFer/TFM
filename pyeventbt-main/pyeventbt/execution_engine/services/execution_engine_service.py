"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.utils.utils import TerminalColors, colorize
from ..core.interfaces.execution_engine_interface import IExecutionEngine
from ..core.configurations.execution_engine_configurations import BaseExecutionConfig, MT5SimulatedExecutionConfig, MT5LiveExecutionConfig
from ..connectors.mt5_live_execution_engine_connector import Mt5LiveExecutionEngineConnector
from ..connectors.mt5_simulator_execution_engine_connector import Mt5SimulatorExecutionEngineConnector

from pyeventbt.broker.mt5_broker.core.entities.order_send_result import OrderSendResult

from pyeventbt.data_provider.core.interfaces.data_provider_interface import IDataProvider
from pyeventbt.events.events import BarEvent, OrderEvent, FillEvent
from queue import Queue
import logging

logger = logging.getLogger("pyeventbt")

class ExecutionEngine(IExecutionEngine):
    
    def __init__(self, events_queue: Queue, data_provider: IDataProvider, execution_config: BaseExecutionConfig) -> None:
        self.events_queue = events_queue
        self.DATA_PROVIDER = data_provider
        self.EXECUTION_ENGINE = self._get_execution_engine(execution_config)
        self.__enable_trading = True

    def _get_execution_engine(self, execution_config: BaseExecutionConfig) -> IExecutionEngine:
        
        # The dependency for the events queue passed to the connectors is because their methods already return another value and can't
        # return the event to put to the queue in a generic method here in the service.
        if isinstance(execution_config, MT5LiveExecutionConfig):
            return Mt5LiveExecutionEngineConnector(configs=execution_config, events_queue=self.events_queue, data_provider=self.DATA_PROVIDER)
        
        elif isinstance(execution_config, MT5SimulatedExecutionConfig):
            return Mt5SimulatorExecutionEngineConnector(configs=execution_config, events_queue=self.events_queue, data_provider=self.DATA_PROVIDER)
        
        else:
            raise Exception(f"Unknown Execution Engine: {execution_config}")
    
    def _put_fill_event(self, fill_event: FillEvent)-> None:
        self.events_queue.put(fill_event)

    def enable_trading(self) -> None:
        self.__enable_trading = True
        
    def disable_trading(self) -> None:
        self.__enable_trading = False

    # Wrapping of the IExecutionEngine methods using the appropiate Exec Engine
    def _process_order_event(self, order_event: OrderEvent) -> None:
        """
        Processes an order event
        """
        if self.__enable_trading:
            self.EXECUTION_ENGINE._process_order_event(order_event)
        else:
            logger.warning(f"ExecutionEngine Trading is disbled. Order {order_event} won't be processed\nUse ExecutionEngine.enable_trading() to allow execution")
    
    def _update_values_and_check_executions_and_fills(self, bar_event: BarEvent) -> None:
        """
        Updates the account values, checks if any pending order has been filled and if any SL/TP has been hit.
        """
        self.EXECUTION_ENGINE._update_values_and_check_executions_and_fills(bar_event)
    
    def _send_market_order(self, order_event: OrderEvent) -> OrderSendResult:
        """
        Executes a Market Order and returns an OrderSendResult
        """
        return self.EXECUTION_ENGINE._send_market_order(order_event)
    
    def _send_pending_order(self, order_event: OrderEvent) -> OrderSendResult:
        """
        Sends a pending order to the broker
        """
        return self.EXECUTION_ENGINE._send_pending_order(order_event)
    
    def close_position(self, position_ticket: int) -> OrderSendResult:
        """
        Close a currently opened position
        """
        return self.EXECUTION_ENGINE.close_position(position_ticket)
    
    def close_all_strategy_positions(self) -> None:
        """
        Close all positions from the strategy
        """
        self.EXECUTION_ENGINE.close_all_strategy_positions()
    
    def close_strategy_long_positions_by_symbol(self, symbol: str) -> None:
        """
        Close all long positions from the strategy by symbol
        """
        self.EXECUTION_ENGINE.close_strategy_long_positions_by_symbol(symbol)
    
    def close_strategy_short_positions_by_symbol(self, symbol: str) -> None:
        """
        Close all short positions from the strategy by symbol
        """
        self.EXECUTION_ENGINE.close_strategy_short_positions_by_symbol(symbol)
    
    def cancel_pending_order(self, order_ticket: int) -> OrderSendResult:
        """
        Cancels a pending order
        """
        return self.EXECUTION_ENGINE.cancel_pending_order(order_ticket)
    
    def cancel_all_strategy_pending_orders(self) -> None:
        """
        Cancels all pending orders from the strategy, in all symbols.
        """
        self.EXECUTION_ENGINE.cancel_all_strategy_pending_orders()

    def cancel_all_strategy_pending_orders_by_type_and_symbol(self, order_type:str, symbol: str) -> None:
        """
        Cancels all specific type of pending orders from the strategy, in a specific symbol.
        Example: cancels all BUY_LIMIT orders in EURUSD.
        """
        self.EXECUTION_ENGINE.cancel_all_strategy_pending_orders_by_type_and_symbol(order_type, symbol)
    
    def update_position_sl_tp(self, position_ticket: int, new_sl: float = 0.0, new_tp: float = 0.0) -> None:
        """
        Update the SL and TP of a position
        """
        self.EXECUTION_ENGINE.update_position_sl_tp(position_ticket, new_sl, new_tp)
    
    # Here we'll define helper methods for getting important data from the MT5 account like account currency, balance, equity, etc.
    def _get_account_currency(self) -> str:
        """Get account currency"""
        return self.EXECUTION_ENGINE._get_account_currency()
    
    def _get_account_balance(self) -> float:
        """Get account balance in account currency"""
        return self.EXECUTION_ENGINE._get_account_balance()
    
    def _get_account_equity(self) -> float:
        """Get account equity in account currency"""
        return self.EXECUTION_ENGINE._get_account_equity()
    
    def _get_account_floating_profit(self) -> float:
        """Get account floating profit in account currency"""
        return self.EXECUTION_ENGINE._get_account_floating_profit()
    
    def _get_account_used_margin(self) -> float:
        """Get account used margin in account currency"""
        return self.EXECUTION_ENGINE._get_account_used_margin()
    
    def _get_account_free_margin(self) -> float:
        """Get account free margin in account currency"""
        return self.EXECUTION_ENGINE._get_account_free_margin()
    
    def _get_total_number_of_pending_orders(self) -> int:
        """Get total number of active pending orders"""
        return self.EXECUTION_ENGINE._get_total_number_of_pending_orders()
    
    def _get_strategy_pending_orders(self) -> tuple:
        """Get current pending orders"""
        return self.EXECUTION_ENGINE._get_strategy_pending_orders()
    
    def _get_total_number_of_positions(self) -> int:
        """Get total number of active pending orders"""
        return self.EXECUTION_ENGINE._get_total_number_of_positions()
    
    def _get_strategy_positions(self) -> tuple:
        """Get current positions"""
        return self.EXECUTION_ENGINE._get_strategy_positions()
    
    def _get_symbol_min_volume(self, symbol: str) -> float:
        """Get symbol min volume"""
        return self.EXECUTION_ENGINE._get_symbol_min_volume(symbol)