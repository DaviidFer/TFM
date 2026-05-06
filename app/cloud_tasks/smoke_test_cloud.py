from __future__ import annotations

import importlib
import json

from app.cloud import CLOUD_PATHS, S3Storage, load_cloud_config
from app.cloud.heartbeat import build_heartbeat_payload, upload_heartbeat_if_enabled, write_heartbeat
from app.cloud_tasks import summarize_result


def main() -> int:
    config = load_cloud_config()
    result: dict[str, object] = {
        "status": "completed",
        "imports": {},
        "paths": {
            "project_dir": str(config.project_dir),
            "artifacts_root": str(config.artifacts_root),
            "db_path": str(config.db_path),
        },
        "path_exists": {
            "project_dir": config.project_dir.exists(),
            "artifacts_root_parent": config.artifacts_root.parent.exists(),
            "db_parent": config.db_path.parent.exists(),
        },
    }
    required_modules = (
        "app",
        "app.cloud",
        "app.runtime",
        "pydantic",
        "polars",
        "numba",
        "yaml",
    )
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            result["imports"][module_name] = "ok"
        except Exception as exc:
            result["imports"][module_name] = f"error: {exc}"
            result["status"] = "error"

    mt5_expected = bool(config.mt5_path or config.mt5_login or config.mt5_server)
    if mt5_expected:
        try:
            importlib.import_module("MetaTrader5")
            result["imports"]["MetaTrader5"] = "ok"
        except Exception as exc:
            result["imports"]["MetaTrader5"] = f"error: {exc}"
            result["status"] = "error"

    try:
        importlib.import_module("quantdle")
        result["imports"]["quantdle"] = "ok"
    except Exception as exc:
        result["imports"]["quantdle"] = f"error: {exc}"

    if config.enable_s3 and config.has_s3_bucket:
        try:
            storage = S3Storage(config)
            keys = storage.list_prefix(CLOUD_PATHS.deployment)
            result["s3"] = {"status": "ok", "keys_found": len(keys)}
        except Exception as exc:
            result["s3"] = {"status": "error", "error": str(exc)}
            result["status"] = "error"
    else:
        result["s3"] = {"status": "skipped", "reason": "s3_disabled"}

    heartbeat = build_heartbeat_payload()
    heartbeat["smoke_test_cloud"] = result
    write_heartbeat(heartbeat)
    upload_heartbeat_if_enabled(heartbeat)
    print(json.dumps(summarize_result("smoke_test_cloud", result), ensure_ascii=True))
    return 0 if str(result.get("status")) != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())

