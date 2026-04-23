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

class TradeOrder(BaseModel):
    ticket: int
    time_setup: int
    time_setup_msc: int
    time_done: int
    time_done_msc: int
    time_expiration: int
    type: int
    type_time: int
    type_filling: int               # https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling
    state: int                      # https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_state
    magic: int
    position_id: int
    position_by_id: int
    reason: int                     # https://www.mql5.com/en/docs/constants/tradingconstants/orderproperties#enum_order_reason
    volume_initial: Decimal
    volume_current: Decimal
    price_open: Decimal
    sl: Decimal
    tp: Decimal
    price_current: Decimal
    price_stoplimit: Decimal
    symbol: str
    comment: str
    external_id: str
