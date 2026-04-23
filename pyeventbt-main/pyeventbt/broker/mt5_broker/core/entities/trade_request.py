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

class TradeRequest(BaseModel):
    action: int
    magic: int
    order: int
    symbol: str
    volume: Decimal
    price: Decimal
    stoplimit: Decimal
    sl: Decimal
    tp: Decimal
    deviation: int
    type: int
    type_filling: int
    type_time: int
    expiration: int
    comment: str
    position: int
    position_by: int