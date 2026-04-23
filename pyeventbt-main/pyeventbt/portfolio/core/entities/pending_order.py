"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel
from typing import Optional
from decimal import Decimal


# This means we can have more than one position per symbol which is non-standard (possible in the MT world with hedging accounts)
class PendingOrder(BaseModel):
    price: Decimal
    type: str               # BUY(0), SELL(1), BUY_LIMIT(2), SELL_LIMIT(3), BUY_STOP(4), SELL_STOP(5)
    symbol: str             # ticker
    ticket: int             # position identifier in the trading platform
    volume: Decimal
    strategy_id: str
    sl: Optional[Decimal]
    tp: Optional[Decimal]
    comment: Optional[str]
