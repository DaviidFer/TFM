"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel
import pandas as pd
from datetime import datetime

class BaseDataConfig(BaseModel):
    pass

class MT5LiveDataConfig(BaseDataConfig):
    tradeable_symbol_list: list
    timeframes_list: list

class CSVBacktestDataConfig(BaseDataConfig):
    csv_path: str
    account_currency: str
    tradeable_symbol_list: list
    base_timeframe: str
    timeframes_list: list
    backtest_start_timestamp: datetime | None = None
    backtest_end_timestamp: datetime = datetime.now() 