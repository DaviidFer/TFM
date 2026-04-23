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
from pyeventbt.events.events import SignalEvent
from decimal import Decimal

class SuggestedOrder(BaseModel):
    signal_event:   SignalEvent
    volume:         Decimal
    buffer_data:    Optional[dict] = None