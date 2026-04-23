"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from enum import Enum

class TypeContext(str, Enum):
    LIVE = "LIVE"
    BACKTEST = "BACKTEST"

trading_context = TypeContext.BACKTEST

def get_trading_context():
    return trading_context

def set_trading_context(context: TypeContext):
    global trading_context
    trading_context = context
