"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from decimal import Decimal


# This means we can have more than one position per symbol which is non-standard (possible in the MT world with hedging accounts)
class OpenPosition(BaseModel):
    time_entry: datetime    # in milliseconds
    price_entry: Decimal
    type: str               # BUY or SELL
    symbol: str             # ticker
    ticket: int             # position identifier in the trading platform
    volume: Decimal
    unrealized_profit: Decimal
    strategy_id: str
    sl: Optional[Decimal]
    tp: Optional[Decimal]
    swap: Optional[Decimal]
    comment: Optional[str]
