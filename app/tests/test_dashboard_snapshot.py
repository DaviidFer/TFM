from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.phase5_check import main as run_phase5_check
from app.ui.dashboard_data import load_dashboard_snapshot


def _tmp_phase5_db() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"phase5_{uuid4().hex[:8]}.sqlite"


def test_dashboard_snapshot_after_phase5_cycle() -> None:
    db_path = _tmp_phase5_db()
    rc = int(run_phase5_check(db_path=db_path))
    assert rc == 0

    snap = load_dashboard_snapshot(db_path=db_path, event_limit=300)

    assert snap.summary["n_agents"] >= 4
    assert snap.summary["n_traders"] >= 1
    assert snap.summary["n_events"] >= 10
    assert "traders_by_state" in snap.summary
    assert "events_by_type" in snap.summary

