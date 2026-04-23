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

class ClosedPosition(BaseModel):
    time_entry: datetime
    price_entry: Decimal
    time_exit: datetime
    price_exit: Decimal
    strategy_id: str
    ticket: int
    symbol: str
    direction: str
    volume: Decimal
    commission: Decimal
    pnl: Decimal
    sl: Optional[Decimal]
    tp: Optional[Decimal]
    swap: Optional[Decimal]
    comment: Optional[str]