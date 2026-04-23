from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.contracts import DesignRiskProfile
from app.storage import StateStore


def _tmp_db_path() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"risk_profile_{uuid4().hex[:8]}.sqlite"


def test_design_risk_profile_serialization_roundtrip() -> None:
    store = StateStore(db_path=_tmp_db_path())
    profile = DesignRiskProfile(
        trader_id="tr_A",
        asset="AAPL",
        timeframe="D1",
        promoted_at="2026-01-01T00:00:00+00:00",
        design_start="2016-01-01",
        design_end="2025-12-31",
        sharpe_design=1.2,
        profit_factor_design=1.8,
        max_drawdown_design=0.12,
        avg_loss_design=-15.0,
        avg_win_design=22.0,
        winrate_design=0.56,
        expectancy_design=4.2,
        max_losing_streak_design=4,
        trades_design=120,
        metadata={"source": "unit_test"},
    )
    store.upsert_trader_design_profile(
        trader_id=profile.trader_id,
        asset=profile.asset,
        timeframe=profile.timeframe,
        promoted_at=profile.promoted_at,
        profile=profile.to_dict(),
    )
    loaded = store.get_trader_design_profile("tr_A")
    assert loaded is not None
    restored = DesignRiskProfile(**dict(loaded["profile"]))
    assert restored.trader_id == profile.trader_id
    assert restored.asset == profile.asset
    assert restored.sharpe_design == profile.sharpe_design
    assert restored.metadata["source"] == "unit_test"
