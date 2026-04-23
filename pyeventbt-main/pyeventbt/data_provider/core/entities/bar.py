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

class Bar(BaseModel):
    """
    A class representing a financial bar, containing information about the opening, closing, high, low, and volume of a financial instrument.

    Attributes:
    -----------
    datetime : datetime
        The datetime of the bar.
    open : float
        The opening price of the bar.
    high : float
        The highest price of the bar.
    low : float
        The lowest price of the bar.
    close : float
        The closing price of the bar.
    adj_close : Optional[float]
        The adjusted closing price of the bar.
    volume : Optional[int]
        The volume of the bar.
    spread : Optional[int]
        The spread of the bar.
    open_interest : Optional[int]
        The open interest of the bar.
    """
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float]
    volume: Optional[int]
    spread: Optional[int]
    open_interest: Optional[int]
