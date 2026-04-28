from __future__ import annotations

import json

from app.cloud.heartbeat import build_heartbeat_payload, upload_heartbeat_if_enabled, write_heartbeat
from app.cloud_tasks import build_supervisor, summarize_result


def main() -> int:
    result: dict[str, object]
    try:
        supervisor = build_supervisor()
        runtime_state = supervisor.start_operational_runtime()
        runtime = getattr(supervisor, "_runtime", None)
        if runtime is not None and hasattr(runtime, "force_rebalance_now"):
            rebalance_out = runtime.force_rebalance_now(reason="cloud_weekly_rebalance")
            result = {
                "status": "completed",
                "runtime": runtime_state,
                "rebalance": rebalance_out,
            }
        else:
            result = {
                "status": "warning",
                "warning": "runtime_not_available",
                "runtime": runtime_state,
            }
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}

    heartbeat = build_heartbeat_payload()
    heartbeat["weekly_rebalance"] = result
    write_heartbeat(heartbeat)
    upload_heartbeat_if_enabled(heartbeat)
    print(json.dumps(summarize_result("weekly_rebalance", result), ensure_ascii=True))
    return 0 if str(result.get("status")) != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())

