from __future__ import annotations

from typing import List
from uuid import uuid4

from app.contracts import PromotedTraderSpec


def build_promoted_spec(
    *,
    asset: str,
    timeframe: str,
    experiment_id: str,
    winners_long_stable: List[str],
    winners_short_stable: List[str],
) -> PromotedTraderSpec:
    trader_id = f"tr_{asset.lower()}_{timeframe.lower()}_{uuid4().hex[:8]}"
    return PromotedTraderSpec(
        trader_id=trader_id,
        asset=asset,
        timeframe=timeframe,
        long_rules=list(winners_long_stable),
        short_rules=list(winners_short_stable),
        origin_experiment_id=experiment_id,
        metadata={"source": "validation_agent"},
    )

