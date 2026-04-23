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

# https://www.mql5.com/en/docs/constants/structures/mqltick
class Tick(BaseModel):
    time: int
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: int
    time_msc: int
    flags: int
    volume_real: Decimal
