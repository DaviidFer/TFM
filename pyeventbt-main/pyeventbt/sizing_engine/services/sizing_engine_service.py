"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Callable
from pyeventbt.strategy.core.modules import Modules
from ..core.interfaces.sizing_engine_interface import ISizingEngine
from ..sizing_engines.mt5_min_sizing import MT5MinSizing
from ..sizing_engines.mt5_fixed_sizing import MT5FixedSizing
from ..sizing_engines.mt5_risk_pct_sizing import MT5RiskPctSizing
from ..core.configurations.sizing_engine_configurations import (
    BaseSizingConfig,
    MinSizingConfig,
    FixedSizingConfig,
    RiskPctSizingConfig)
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder
from pyeventbt.data_provider.core.interfaces.data_provider_interface import IDataProvider
from pyeventbt.events.events import SignalEvent
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder
from queue import Queue

class SizingEngineService:
    """Service for managing position sizing strategies based on different configurations."""
    
    def __init__(self, events_queue: Queue, modules: Modules, sizing_config: BaseSizingConfig = BaseSizingConfig()) -> None:
        """Initialize the sizing engine service.
        
        Args:
            events_queue: Queue for handling events
            modules: Trading modules containing context and services
            sizing_config: Configuration for the sizing engine (defaults to BaseSizingConfig)
        """
        self.modules = modules
        self.events_queue = events_queue
        self.sizing_engine = self._get_position_sizing_method(sizing_config)

    def _get_position_sizing_method(self, sizing_config: BaseSizingConfig) -> ISizingEngine:
        """Get the appropriate position sizing engine based on configuration.
        
        Args:
            sizing_config: Configuration object specifying the sizing strategy
            
        Returns:
            ISizingEngine: The initialized sizing engine instance
        """
        if isinstance(sizing_config, MinSizingConfig):
            return MT5MinSizing(self.modules.TRADING_CONTEXT)
        
        elif isinstance(sizing_config, FixedSizingConfig):
            return MT5FixedSizing(configs=sizing_config)
        
        elif isinstance(sizing_config, RiskPctSizingConfig):
            return MT5RiskPctSizing(configs=sizing_config, trading_context=self.modules.TRADING_CONTEXT)
        
        else:
            return MT5MinSizing(trading_context=self.modules.TRADING_CONTEXT)

    def get_suggested_order(self, signal_event: SignalEvent) -> SuggestedOrder:
        """Get a suggested order based on a signal event.
        
        Args:
            signal_event: The signal event to process
            
        Returns:
            SuggestedOrder: The suggested order to be sent to the risk manager via portfolio handler
        """
        return self.sizing_engine.get_suggested_order(signal_event, self.modules)

    def set_suggested_order_function(self, fn: Callable[[SignalEvent, Modules], SuggestedOrder]):
        """Override the default suggested order function with a custom implementation.
        
        Args:
            fn: Custom function that takes a SignalEvent and Modules, returns a SuggestedOrder
        """
        self.sizing_engine.get_suggested_order = fn