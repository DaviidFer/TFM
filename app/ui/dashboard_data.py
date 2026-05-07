from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from app.storage import StateStore


@dataclass
class DashboardSnapshot:
    db_path: Path
    agent_status: List[Dict[str, str]]
    trader_states: List[Dict[str, str]]
    trader_metrics: List[Dict[str, object]]
    events: List[Dict[str, object]]
    summary: Dict[str, object]
    human_resources_summary: Dict[str, object]


def load_dashboard_snapshot(db_path: str | Path, *, event_limit: int = 200) -> DashboardSnapshot:
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe base de datos: {p}")

    store = StateStore(db_path=p)
    agent_status = store.list_agent_status()
    trader_states_rows = store.list_trader_states()
    trader_metrics = store.list_trader_metrics()
    events = store.list_events(limit=event_limit)

    trader_states: List[Dict[str, str]] = []
    for r in trader_states_rows:
        trader_states.append(
            {
                "trader_id": r.trader_id,
                "asset": r.asset,
                "timeframe": r.timeframe,
                "state": r.state.value,
                "updated_at": r.updated_at,
                "notes": r.notes,
            }
        )

    by_state: Dict[str, int] = {}
    for r in trader_states:
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1

    by_event_type: Dict[str, int] = {}
    for e in events:
        et = str(e.get("event_type"))
        by_event_type[et] = by_event_type.get(et, 0) + 1

    summary = {
        "n_agents": len(agent_status),
        "n_traders": len(trader_states),
        "n_metrics": len(trader_metrics),
        "n_events": len(events),
        "traders_by_state": by_state,
        "events_by_type": by_event_type,
    }
    latest_review_runs = store.list_trader_review_runs(limit=1)
    latest_review = latest_review_runs[0] if latest_review_runs else {}
    human_resources_summary = {
        "latest_run": latest_review,
        "pending_retrain_requests": len(store.list_pending_retrain_requests(limit=1000)),
    }

    return DashboardSnapshot(
        db_path=p,
        agent_status=agent_status,
        trader_states=trader_states,
        trader_metrics=trader_metrics,
        events=events,
        summary=summary,
        human_resources_summary=human_resources_summary,
    )
