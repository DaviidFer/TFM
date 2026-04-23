from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.contracts import EventType, PromotedTraderSpec, TraderLifecycleState
from app.storage import StateStore


def _tmp_db_path() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"state_{uuid4().hex[:8]}.sqlite"


def test_promoted_spec_serialization_contains_lifecycle_value() -> None:
    spec = PromotedTraderSpec(
        trader_id="tr_test",
        asset="AAPL",
        timeframe="D1",
        long_rules=["r1"],
        short_rules=["r2"],
        origin_experiment_id="exp_test",
    )
    data = spec.to_dict()
    assert data["lifecycle_state"] == TraderLifecycleState.PROMOTED.value
    assert data["asset"] == "AAPL"
    assert isinstance(data["long_rules"], list)


def test_state_store_roundtrip_for_state_metrics_events() -> None:
    db_path = _tmp_db_path()
    if db_path.exists():
        db_path.unlink()
    store = StateStore(db_path=db_path)

    trader_id = f"tr_{uuid4().hex[:6]}"
    store.upsert_trader_state(
        trader_id=trader_id,
        asset="AAPL",
        timeframe="D1",
        state=TraderLifecycleState.PROMOTED,
        notes="unit test",
    )
    store.upsert_trader_metrics(
        trader_id=trader_id,
        as_of="2026-01-01T00:00:00+00:00",
        pnl=123.0,
        sharpe_rolling=1.1,
        drawdown_rolling=0.05,
        trade_count=10,
        extra={"corr_penalty": 0.1},
    )
    store.append_event(
        event_id=f"evt_{uuid4().hex[:10]}",
        event_type=EventType.TRADER_METRICS_UPDATED,
        producer="test_agent",
        payload={"trader_id": trader_id},
    )

    row = store.get_trader_state(trader_id)
    assert row is not None
    assert row.state == TraderLifecycleState.PROMOTED
    metrics = store.get_trader_metrics(trader_id)
    assert metrics is not None
    assert metrics["trade_count"] == 10
    events = store.list_events(limit=20)
    assert any(e["event_type"] == EventType.TRADER_METRICS_UPDATED.value for e in events)

