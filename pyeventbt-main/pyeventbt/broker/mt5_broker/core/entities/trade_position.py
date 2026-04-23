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

class TradePosition(BaseModel):
    ticket: int
    time: int               # open time
    time_msc: int           # open time in milliseconds
    time_update: int
    time_update_msc: int
    type: int               # direction: 0 buy, 1 sell
    magic: int              # Strategy ID
    identifier: int
    reason: int             # https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties#enum_deal_reason   Expert, Mobile, etc
    volume: Decimal
    price_open: Decimal
    sl: Decimal
    tp: Decimal
    price_current: Decimal
    swap: Decimal
    profit: Decimal           # Decimal pnl
    symbol: str
    comment: str
    external_id: str
    # Add an optional field to store used margin
    used_margin_acc_ccy: Optional[Decimal]