from __future__ import annotations

import json
import time

from app.cloud.heartbeat import build_heartbeat_payload, upload_heartbeat_if_enabled, write_heartbeat
from app.cloud_tasks import build_supervisor, summarize_result


def main() -> int:
    supervisor = build_supervisor()
    supervisor.start()
    runtime_state = supervisor.start_operational_runtime()
    heartbeat = build_heartbeat_payload()
    heartbeat["run_runtime"] = runtime_state
    write_heartbeat(heartbeat)
    upload_heartbeat_if_enabled(heartbeat)
    print(json.dumps(summarize_result("run_runtime", runtime_state), ensure_ascii=True))
    try:
        while True:
            time.sleep(60)
            print(json.dumps({"task": "run_runtime", "status": supervisor.get_status()}, ensure_ascii=True))
    except KeyboardInterrupt:
        shutdown = getattr(supervisor, "_shutdown", None)
        if shutdown is not None:
            shutdown.set()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

