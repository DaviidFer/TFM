"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""


from enum import Enum
from datetime import timedelta



class StrategyTimeframes(str, Enum):
    ONE_MIN = '1min' # KEY = pandas timeframe
    TWO_MIN = '2min'
    THREE_MIN = '3min'
    FOUR_MIN = '4min'
    FIVE_MIN = '5min'
    SIX_MIN = '6min'
    TEN_MIN = '10min'
    TWELVE_MIN = '12min'
    FIFTEEN_MIN = '15min'
    TWENTY_MIN = '20min'
    THIRTY_MIN = '30min'
    ONE_HOUR = '1h'
    TWO_HOUR = '2h'
    THREE_HOUR ='3h'
    FOUR_HOUR = '4h'
    SIX_HOUR = '6h'
    EIGHT_HOUR = '8h'
    TWELVE_HOUR = '12h'
    ONE_DAY = '1D'
    ONE_WEEK = '1W'
    ONE_MONTH = '1M'
    SIX_MONTH = '6M'
    ONE_YEAR = '12M'

    def to_timedelta(self):
        timeframes_to_intervals = {
            StrategyTimeframes.ONE_MIN.value: timedelta(minutes=1),
            StrategyTimeframes.TWO_MIN.value: timedelta(minutes=2),
            StrategyTimeframes.THREE_MIN.value: timedelta(minutes=3),
            StrategyTimeframes.FOUR_MIN.value: timedelta(minutes=4),
            StrategyTimeframes.FIVE_MIN.value: timedelta(minutes=5),
            StrategyTimeframes.SIX_MIN.value: timedelta(minutes=6),
            StrategyTimeframes.TEN_MIN.value: timedelta(minutes=10),
            StrategyTimeframes.TWELVE_MIN.value: timedelta(minutes=12),
            StrategyTimeframes.FIFTEEN_MIN.value: timedelta(minutes=15),
            StrategyTimeframes.TWENTY_MIN.value: timedelta(minutes=20),
            StrategyTimeframes.THIRTY_MIN.value: timedelta(minutes=30),
            StrategyTimeframes.ONE_HOUR.value: timedelta(hours=1),
            StrategyTimeframes.TWO_HOUR.value: timedelta(hours=2),
            StrategyTimeframes.THREE_HOUR.value: timedelta(hours=3),
            StrategyTimeframes.FOUR_HOUR.value: timedelta(hours=4),
            StrategyTimeframes.SIX_HOUR.value: timedelta(hours=6),
            StrategyTimeframes.EIGHT_HOUR.value: timedelta(hours=8),
            StrategyTimeframes.TWELVE_HOUR.value: timedelta(hours=12),
            StrategyTimeframes.ONE_DAY.value: timedelta(days=1),
            StrategyTimeframes.ONE_WEEK.value: timedelta(weeks=1),
            StrategyTimeframes.ONE_MONTH.value: timedelta(days=30),
            StrategyTimeframes.SIX_MONTH.value: timedelta(days=30*6),
            StrategyTimeframes.ONE_YEAR.value: timedelta(days=365), # Note: timedelta does not directly support months due to variable days in months, so we approximate with 30 days.
        }
        
        return timeframes_to_intervals[self.value]


    def __eq__(self, value: object) -> bool:
        
        if isinstance(value, str):
            return self.value == value
        elif isinstance(value, StrategyTimeframes):        
            return self.to_timedelta() == value.to_timedelta()
        else:
            raise ValueError(f"Cannot compare StrategyTimeframes with {type(value)}")
    
    def __gt__(self, value: str) -> bool:
        return self.to_timedelta() > value.to_timedelta()

    def __lt__(self, value: str) -> bool:
        return self.to_timedelta() < value.to_timedelta()

    def __hash__(self) -> int:
        return hash(self.value)