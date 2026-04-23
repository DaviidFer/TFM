"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from decimal import Decimal
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field

from pyeventbt.strategy.core.strategy_timeframes import StrategyTimeframes

class EventType(str, Enum):
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    SCHEDULED_EVENT = "SCHEDULED_EVENT"

class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    CONT = "CONT"

class DealType(str, Enum):
    IN = "IN"
    OUT = "OUT"


class EventBase(BaseModel):
    type: EventType

    # This will allow us to use arbitrary types in the event like pd.Series
    class Config:
        arbitrary_types_allowed = True


# This is the payload, with only fixed-width numerics to be comppact (arount 56B)
@dataclass(slots=True)
class Bar:
    open:    int
    high:    int
    low:     int
    close:   int
    tickvol: int
    volume:  int
    spread:  int
    digits:  int        # Number of decimals to reconstruct the decimal prices
    
    __price_factor: float = field(init=False, repr=False, default=None)
    
    @property
    def price_factor(self) -> float:
        if self.__price_factor is None:
            self.__price_factor = 10 ** self.digits
        return self.__price_factor
    
    @property
    def open_f(self) -> float:
        """
        Returns the open price as a float.
        """
        return self.open / self.price_factor
    
    @property
    def high_f(self) -> float:
        """
        Returns the high price as a float.
        """
        return self.high / self.price_factor
    
    @property
    def low_f(self) -> float:
        """
        Returns the low price as a float.
        """
        return self.low / self.price_factor
    
    @property
    def close_f(self) -> float:
        """
        Returns the close price as a float.
        """
        return self.close / self.price_factor
    
    @property
    def spread_f(self) -> float:
        """
        Returns the spread as a float.
        """
        return self.spread / self.price_factor


# BarEvent is the envelope: carries metadata (symbol, timeframe, timestamp) around the payload (the Bar)
class BarEvent(EventBase):
    type: EventType = EventType.BAR
    symbol: str
    datetime: datetime
    data: Bar               # Holds the lightweight payload (the Bar --> a row of data)
    timeframe: str


class SignalEvent(EventBase):
    type:           EventType = EventType.SIGNAL
    symbol:         str
    time_generated: datetime                    # moment of generating the signal
    strategy_id:    str                         # magic number
    forecast:       Optional[float] = 0.0       # value from -20 to +20
    signal_type:    SignalType                  # BUY or SELL
    order_type:     OrderType                   # MARKET, LIMIT, STOP
    order_price:    Optional[Decimal] = Decimal('0.0')
    sl:             Optional[Decimal] = Decimal('0.0')
    tp:             Optional[Decimal] = Decimal('0.0')
    rollover:       Optional[tuple] = (False, "", "")       # Structure is True, "original_contract", "new_contract". If first element is False, it means no rollover is needed


class OrderEvent(EventBase):
    """
    Handles the event of sending an Order to an execution system.
    """
    type:           EventType = EventType.ORDER
    symbol:         str
    time_generated: datetime
    strategy_id:    str
    volume:         Decimal                   # The forecast is here transformed into a volume
    signal_type:    SignalType              # BUY or SELL
    order_type:     OrderType               # MARKET, LIMIT, STOP
    order_price:    Optional[Decimal] = Decimal('0.0')
    sl:             Optional[Decimal] = Decimal('0.0')
    tp:             Optional[Decimal] = Decimal('0.0')
    rollover:       Optional[tuple] = (False, "", "")
    buffer_data:    Optional[dict] = None     # Extra data that will be needed down the line in the execution engine


class FillEvent(EventBase):
    """
    Encapsulates the notion of a Filled Order, as returned from a brokerage.
    """
    type:           EventType = EventType.FILL
    deal:           DealType
    symbol:         str
    time_generated: datetime
    position_id:    int
    strategy_id:    str
    exchange:       str
    volume:         Decimal
    price:          Decimal
    signal_type:    SignalType
    commission:     Decimal
    swap:           Decimal
    fee:            Decimal
    gross_profit:   Decimal
    ccy:            str         # Currency in which costs and profits are specified

    
class ScheduledEvent(EventBase):
    
    type: EventType = Field(default=EventType.SCHEDULED_EVENT, init_var=False)
    schedule_timeframe: StrategyTimeframes
    symbol: str
    timestamp: pd.Timestamp
    former_execution_timestamp: pd.Timestamp | None = None
    