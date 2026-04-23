"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel
from ..entities.trade_request import TradeRequest
from decimal import Decimal

class OrderSendResult(BaseModel):
    retcode: int
    deal: int               # deal ticket (unique identifier of the deal)
    order: int              # order ticket (unique identifier of the order) - aparece como Ticket en la posición de MT5
    volume: Decimal
    price: Decimal
    bid: Decimal
    ask: Decimal
    comment: str
    request_id: int
    retcode_external: int
    request: TradeRequest