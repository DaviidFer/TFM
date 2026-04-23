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
from decimal import Decimal

class ClosedPosition(BaseModel):
    time_entry: int
    price_entry: Decimal
    magic: int   # Strategy ID
    ticket: int
    symbol: str
    direction: str
    volume: Decimal
    sl: Decimal
    tp: Decimal
    commission: Decimal  # The total roundtrip commission
    swap: Decimal
    time_exit: int
    price_exit: Decimal
    comment: str
    profit: Decimal