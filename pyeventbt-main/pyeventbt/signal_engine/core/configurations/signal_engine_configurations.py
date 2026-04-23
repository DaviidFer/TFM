"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel
from enum import Enum

from pyeventbt.core.entities.hyper_parameter import HyperParameter

class BaseSignalEngineConfig(BaseModel):
    strategy_id: str
    signal_timeframe: str

class MAType(str, Enum):
    SIMPLE = "SIMPLE"
    EXPONENTIAL = "EXPONENTIAL"

class MACrossoverConfig(BaseSignalEngineConfig):
    strategy_id: str
    ma_type: MAType = MAType.SIMPLE  # Let's set default to simple
    signal_timeframe: str
    fast_period: int | HyperParameter
    slow_period: int | HyperParameter