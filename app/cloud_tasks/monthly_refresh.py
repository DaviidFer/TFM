from __future__ import annotations

import json
from datetime import datetime, timezone

from app.cloud.heartbeat import build_heartbeat_payload, upload_heartbeat_if_enabled, write_heartbeat
from app.cloud_tasks import build_supervisor, summarize_result


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    result: dict[str, object]
    try:
        supervisor = build_supervisor()
        refresh_out = supervisor.run_portfolio_monthly_refresh(force=True, as_of=_now_iso())
        trader_health_out = supervisor.force_trader_health_evaluation(force_backtest=True)
        retrain_out = supervisor.process_pending_retrain_requests()
        result = {
            "status": "completed",
            "refresh": refresh_out,
            "trader_health": trader_health_out,
            "retrain_requests": retrain_out,
        }
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}

    heartbeat = build_heartbeat_payload()
    heartbeat["monthly_refresh"] = result
    write_heartbeat(heartbeat)
    upload_heartbeat_if_enabled(heartbeat)
    print(json.dumps(summarize_result("monthly_refresh", result), ensure_ascii=True))
    return 0 if str(result.get("status")) != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())

