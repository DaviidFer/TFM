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

class BaseExecutionConfig(BaseModel):
    pass

class MT5LiveExecutionConfig(BaseExecutionConfig):
    magic_number: int

class MT5SimulatedExecutionConfig(BaseExecutionConfig):
    initial_balance: Decimal
    account_currency: str
    account_leverage: int
    magic_number: int