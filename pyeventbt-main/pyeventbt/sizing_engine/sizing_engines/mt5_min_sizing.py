"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.utils.utils import check_platform_compatibility
from ..core.interfaces.sizing_engine_interface import ISizingEngine
from pyeventbt.events.events import SignalEvent
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder
import pyeventbt.trading_context.trading_context as trading_context 
from decimal import Decimal


class MT5MinSizing(ISizingEngine):
    """MT5 implementation of minimum position sizing strategy.
    
    This sizing engine uses the minimum allowed volume for the symbol
    as specified by the broker. It automatically adapts to different
    symbols and their respective minimum volume requirements.
    """
    
    def __init__(self, trading_context: trading_context.TypeContext = trading_context.TypeContext.BACKTEST) -> None:
        """Initialize the minimum sizing engine.
        
        Args:
            trading_context: Trading context (BACKTEST or LIVE), defaults to BACKTEST
        """
        if trading_context == "BACKTEST":
            from pyeventbt.broker.mt5_broker.mt5_simulator_wrapper import Mt5SimulatorWrapper as mt5
        else:
            check_platform_compatibility()
            try:
                import MetaTrader5 as mt5
            except ImportError:
                mt5 = None
        self.mt5 = mt5
    
    def get_suggested_order(self, signal_event: SignalEvent, *args, **kwargs) -> SuggestedOrder:
        """Generate a suggested order with minimum allowed volume for the symbol.
        
        Args:
            signal_event: The trading signal event containing symbol information
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)
            
        Returns:
            SuggestedOrder: Order suggestion with the minimum volume for the symbol
        """
        return SuggestedOrder(signal_event=signal_event,
                            volume=Decimal(str(self.mt5.symbol_info(signal_event.symbol).volume_min)))