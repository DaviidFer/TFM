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
from pyeventbt.config.configs import Mt5PlatformConfig

class BaseTradingSessionConfig(BaseModel):
    pass

class MT5BacktestSessionConfig(BaseTradingSessionConfig):
    initial_capital: float
    start_date: datetime
    backtest_name: str

class MT5LiveSessionConfig(BaseTradingSessionConfig):
    symbol_list: list[str]
    heartbeat: float
    platform_config: Mt5PlatformConfig

