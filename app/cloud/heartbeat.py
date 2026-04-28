from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.cloud.cloud_config import load_cloud_config
from app.cloud.cloud_paths import CLOUD_PATHS
from app.cloud.s3_storage import S3Storage


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_heartbeat_payload() -> dict[str, Any]:
    config = load_cloud_config()
    return {
        "ts_utc": _utc_now_iso(),
        "tfm_env": config.tfm_env,
        "aws_region": config.aws_region,
        "project_dir": str(config.project_dir),
        "artifacts_root": str(config.artifacts_root),
        "db_path": str(config.db_path),
        "paths": {
            "project_dir_exists": config.project_dir.exists(),
            "artifacts_root_exists": config.artifacts_root.exists(),
            "db_path_exists": config.db_path.exists(),
        },
        "cloud": {
            "enable_s3": config.enable_s3,
            "enable_cloudwatch": config.enable_cloudwatch,
            "s3_bucket": config.s3_bucket,
            "s3_prefix": config.s3_prefix,
        },
    }


def write_heartbeat(payload: dict[str, Any] | None = None) -> Path:
    config = load_cloud_config()
    heartbeat_payload = payload or build_heartbeat_payload()
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = config.logs_dir / "heartbeat.json"
    heartbeat_path.write_text(json.dumps(heartbeat_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return heartbeat_path


def upload_heartbeat_if_enabled(payload: dict[str, Any]) -> str | None:
    config = load_cloud_config()
    storage = S3Storage(config)
    if not storage.enabled():
        return None
    return storage.upload_json(payload, CLOUD_PATHS.join(CLOUD_PATHS.deployment, "heartbeat.json"))


def main() -> int:
    payload = build_heartbeat_payload()
    local_path = write_heartbeat(payload)
    remote_key = upload_heartbeat_if_enabled(payload)
    print(json.dumps({"heartbeat_path": str(local_path), "s3_key": remote_key}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

