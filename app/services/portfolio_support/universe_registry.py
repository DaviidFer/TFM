from __future__ import annotations

from typing import Dict, Mapping

from app.contracts import PromotedTraderSpec, TraderLifecycleState
from app.storage import StateStore


class UniverseRegistry:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def sync_promoted_specs(self, promoted_specs: Mapping[str, PromotedTraderSpec]) -> Dict[str, Dict[str, object]]:
        synced: Dict[str, Dict[str, object]] = {}
        for trader_id, spec in promoted_specs.items():
            self.store.upsert_portfolio_universe_member(
                trader_id=trader_id,
                asset=spec.asset,
                timeframe=spec.timeframe,
                promotion_date=str(spec.promoted_at),
                lifecycle_state=str(spec.lifecycle_state.value),
                metadata=dict(spec.metadata or {}),
            )
            synced[trader_id] = {
                "trader_id": trader_id,
                "asset": spec.asset,
                "timeframe": spec.timeframe,
                "promotion_date": str(spec.promoted_at),
                "lifecycle_state": str(spec.lifecycle_state.value),
                "metadata": dict(spec.metadata or {}),
            }
        return synced

    def update_lifecycle(self, trader_id: str, lifecycle_state: TraderLifecycleState) -> None:
        members = {row["trader_id"]: row for row in self.store.list_portfolio_universe_members()}
        row = members.get(str(trader_id))
        if row is None:
            return
        self.store.upsert_portfolio_universe_member(
            trader_id=str(trader_id),
            asset=str(row["asset"]),
            timeframe=str(row["timeframe"]),
            promotion_date=str(row["promotion_date"]),
            lifecycle_state=lifecycle_state.value,
            metadata=dict(row.get("metadata") or {}),
        )

    def list_members(self) -> Dict[str, Dict[str, object]]:
        rows = self.store.list_portfolio_universe_members()
        return {str(row["trader_id"]): dict(row) for row in rows}
