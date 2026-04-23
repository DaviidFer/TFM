from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionMode(str, Enum):
    PAPER = "paper"
    LIVE_MT5 = "live_mt5"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class OrderIntent:
    trader_id: str
    symbol: str
    side: OrderSide
    volume: float
    order_type: str = "market"
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: str = ""
    requested_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderResult:
    accepted: bool
    mode: ExecutionMode
    ticket: str
    reason: str = ""
    broker_payload: Dict[str, Any] = field(default_factory=dict)
    processed_at: str = field(default_factory=utc_now_iso)

