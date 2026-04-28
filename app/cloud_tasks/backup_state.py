from __future__ import annotations

import json
from datetime import datetime, timezone

from app.cloud import CLOUD_PATHS, S3Storage, load_cloud_config
from app.cloud.heartbeat import build_heartbeat_payload, upload_heartbeat_if_enabled, write_heartbeat
from app.cloud_tasks import summarize_result


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    config = load_cloud_config()
    storage = S3Storage(config)
    if not storage.enabled():
        result = {"status": "warning", "warning": "s3_disabled"}
    else:
        try:
            uploads: dict[str, object] = {}
            if config.db_path.exists():
                uploads["sqlite_backup"] = storage.upload_file(
                    config.db_path,
                    CLOUD_PATHS.join(CLOUD_PATHS.sqlite_backups, f"state_{_timestamp_token()}.sqlite"),
                )
            uploads["artifacts"] = storage.upload_directory(config.artifacts_root, CLOUD_PATHS.artifacts)
            data_dir = config.project_dir / "datos"
            uploads["datos"] = storage.upload_directory(data_dir, CLOUD_PATHS.datos)
            uploads["logs"] = storage.upload_directory(config.logs_dir, CLOUD_PATHS.logs)
            result = {"status": "completed", "uploads": uploads}
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}

    heartbeat = build_heartbeat_payload()
    heartbeat["backup_state"] = result
    write_heartbeat(heartbeat)
    upload_heartbeat_if_enabled(heartbeat)
    print(json.dumps(summarize_result("backup_state", result), ensure_ascii=True))
    return 0 if str(result.get("status")) != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())

