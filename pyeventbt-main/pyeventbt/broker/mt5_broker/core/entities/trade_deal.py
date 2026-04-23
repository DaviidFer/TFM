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

# https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties
class TradeDeal(BaseModel):
    ticket: int         # deal ticket (unique identifier of the deal)
    order: int          
    time: int
    time_msc: int
    type: int           # deal type: 0 buy, 1 sell, 2 balance, 3 credit
    entry: int          # deal entry: 0 is in, 1 is out
    magic: int
    position_id: int    # position id that originated the deal
    reason: int
    volume: Decimal
    price: Decimal
    commission: Decimal
    swap: Decimal
    profit: Decimal       # profit in deposit currency. Is 0 for entry: 0
    fee: Decimal
    symbol: str
    comment: str
    external_id: str