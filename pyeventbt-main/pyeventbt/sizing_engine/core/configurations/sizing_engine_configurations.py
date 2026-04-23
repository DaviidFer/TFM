"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel
from decimal import Decimal

class BaseSizingConfig(BaseModel):
    """Base configuration class for all position sizing strategies."""
    pass

class MinSizingConfig(BaseSizingConfig):
    """Configuration for minimum position sizing strategy."""
    pass

class FixedSizingConfig(BaseSizingConfig):
    """Configuration for fixed position sizing strategy.
    
    Attributes:
        volume: Fixed volume/quantity for all positions
    """
    volume: Decimal

class RiskPctSizingConfig(BaseSizingConfig):
    """Position sizing configuration for the risk percentage method.
    
    This method calculates the position size based on the risk percentage
    and the account balance and the stop loss level of the trade.

    Attributes:
        risk_pct: Risk percentage to be used for the position sizing, 
                 1 means 1% of the account balance
    """
    
    risk_pct: float
