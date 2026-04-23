"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Protocol
from pyeventbt.strategy.core.modules import Modules
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder
from pyeventbt.events.events import SignalEvent

class ISizingEngine(Protocol):
    """Interface for position sizing engines that determine order sizes.
    
    This protocol defines the contract that all position sizing engines must implement.
    Different implementations can use various strategies like fixed sizing, risk percentage,
    or minimum sizing to calculate appropriate position sizes.
    """
    
    def get_suggested_order(self, signal_event: SignalEvent, modules: Modules) -> SuggestedOrder:
        """Generate a suggested order with calculated position size.
        
        Args:
            signal_event: The trading signal event containing trade information
            modules: Trading modules containing context and services
            
        Returns:
            SuggestedOrder: Order suggestion with calculated size and position details
            
        Raises:
            NotImplementedError: Must be implemented by concrete sizing engines
        """
        raise NotImplementedError()