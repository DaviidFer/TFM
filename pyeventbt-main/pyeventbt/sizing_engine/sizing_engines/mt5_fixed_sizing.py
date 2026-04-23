"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from ..core.interfaces.sizing_engine_interface import ISizingEngine
from ..core.configurations.sizing_engine_configurations import FixedSizingConfig
from pyeventbt.events.events import SignalEvent
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder
from decimal import Decimal

class MT5FixedSizing(ISizingEngine):
    """MT5 implementation of fixed position sizing strategy.
    
    This sizing engine uses a fixed volume/quantity for all positions,
    regardless of market conditions or account balance.
    """
    
    def __init__(self, configs: FixedSizingConfig) -> None:
        """Initialize the fixed sizing engine.
        
        Args:
            configs: Configuration containing the fixed volume size
        """
        self.fixed_volume_size = Decimal(configs.volume)
    
    def get_suggested_order(self, signal_event: SignalEvent, *args, **kwargs) -> SuggestedOrder:
        """Generate a suggested order with fixed volume size.
        
        Args:
            signal_event: The trading signal event
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)
            
        Returns:
            SuggestedOrder: Order suggestion with the fixed volume size
        """
        return SuggestedOrder(signal_event=signal_event,
                            volume=self.fixed_volume_size)
        