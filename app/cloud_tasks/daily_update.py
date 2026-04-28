from __future__ import annotations

import importlib
import json

from app.cloud.heartbeat import build_heartbeat_payload, upload_heartbeat_if_enabled, write_heartbeat
from app.cloud_tasks import summarize_result


def main() -> int:
    result: dict[str, object] = {"status": "warning", "warning": "data_download_not_available"}
    try:
        module = importlib.import_module("data_download.download")
        runner = getattr(module, "run_data_download", None)
        if callable(runner):
            universe_df, all_results, failed = runner()
            result = {
                "status": "completed",
                "n_universe_rows": int(len(universe_df)),
                "n_results": int(len(all_results)),
                "n_failed": int(len(failed)),
            }
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}

    heartbeat = build_heartbeat_payload()
    heartbeat["daily_update"] = result
    write_heartbeat(heartbeat)
    upload_heartbeat_if_enabled(heartbeat)
    print(json.dumps(summarize_result("daily_update", result), ensure_ascii=True))
    return 0 if str(result.get("status")) != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())

