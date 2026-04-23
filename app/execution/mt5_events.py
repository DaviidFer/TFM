from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MT5EventType(str, Enum):
    DATA = "DATA"
    SIGNAL = "SIGNAL"
    SIZING = "SIZING"
    ORDER = "ORDER"


@dataclass(frozen=True)
class DataEvent:
    symbol: str
    data: pd.Series
    event_type: MT5EventType = MT5EventType.DATA
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class SignalEvent:
    symbol: str
    signal: str
    target_order: str
    target_price: float
    magic_number: int
    sl: float
    tp: float
    extra: Dict[str, Any] = field(default_factory=dict)
    event_type: MT5EventType = MT5EventType.SIGNAL
    created_at: str = field(default_factory=utc_now_iso)

