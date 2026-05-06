from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.storage import StateStore


def _tmp_db_path() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"risk_store_{uuid4().hex[:8]}.sqlite"


def test_state_store_risk_tables_roundtrip() -> None:
    store = StateStore(db_path=_tmp_db_path())
    store.create_risk_evaluation_run(run_id="risk_1", run_type="manual")
    store.save_risk_evaluation_detail(
        evaluation_run_id="risk_1",
        trader_id="tr_A",
        asset="AAPL",
        timeframe="D1",
        previous_state="live",
        new_state="retraining",
        action="retraining",
        health_score=55.0,
        reasons=["pf deterioration"],
        flags={"insufficient_evidence": False},
        retrain_request={},
    )
    store.create_retrain_request(
        request_id="rr_1",
        trader_id="tr_A",
        asset="AAPL",
        timeframe="D1",
        reason="health low",
        priority="high",
        payload={"source": "unit_test"},
    )
    pending = store.list_pending_retrain_requests()
    assert len(pending) == 1
    assert pending[0]["request_id"] == "rr_1"
    store.mark_retrain_request_running("rr_1")
    store.mark_retrain_request_completed("rr_1", payload={"new_trader_id": "tr_A_v2"})
    all_requests = store.list_retrain_requests()
    assert all_requests[0]["status"] == "completed"
    assert all_requests[0]["payload"]["new_trader_id"] == "tr_A_v2"
    store.complete_risk_evaluation_run(
        run_id="risk_1",
        status="completed",
        evaluated_traders=1,
        retraining_count=1,
        retrain_requests_count=1,
    )
    runs = store.list_risk_evaluation_runs()
    details = store.list_risk_evaluation_details()
    assert runs[0]["evaluated_traders"] == 1
    assert details[0]["trader_id"] == "tr_A"
