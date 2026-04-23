"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.strategy.core.modules import Modules
from ..core.interfaces.risk_engine_interface import IRiskEngine
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder

class PassthroughRiskEngine(IRiskEngine):

    def assess_order(self, suggested_order: SuggestedOrder, modules: Modules) -> float:
        # This risk manager lets the proposed volume pass through
        return suggested_order.volume