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

class IRiskEngine(Protocol):
    """
    The IRiskEngine interface handles the risk management at a portfolio level (strategy level)
    """
    
    def assess_order(self, suggested_order: SuggestedOrder, modules: Modules) -> float:
        """
        Assess the given signal event and generate an order event based on the specified method.
        """
        raise NotImplementedError()