from __future__ import annotations

from pathlib import Path

from app.phase5_check import main as run_phase5_check
from app.ui.dashboard_data import load_dashboard_snapshot


def main() -> int:
    print("=== Phase 6 Check ===")
    db_path = Path("app/.tmp/phase5/phase5.sqlite")
    if not db_path.exists():
        print("No phase5 sqlite found. Running phase5_check first...")
        rc = int(run_phase5_check())
        if rc != 0:
            raise RuntimeError("phase5_check failed; cannot validate dashboard.")

    snap = load_dashboard_snapshot(db_path=db_path, event_limit=300)
    print(f"db_path: {snap.db_path}")
    print(f"n_agents: {snap.summary['n_agents']}")
    print(f"n_traders: {snap.summary['n_traders']}")
    print(f"n_metrics: {snap.summary['n_metrics']}")
    print(f"n_events: {snap.summary['n_events']}")

    if snap.summary["n_traders"] <= 0:
        raise RuntimeError("Expected at least one trader in dashboard snapshot.")
    if snap.summary["n_events"] <= 0:
        raise RuntimeError("Expected at least one event in dashboard snapshot.")

    print("Dashboard snapshot load: OK")
    print("Phase 6 check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

