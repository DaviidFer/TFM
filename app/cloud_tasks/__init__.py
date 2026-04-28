from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.cloud import load_cloud_config
from app.runtime import DevelopmentOperationalSupervisor


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_runtime_environment() -> None:
    config = load_cloud_config()
    if "EXECUTION_MODE" not in os.environ:
        trading_mode = str(config.trading_mode).strip().lower()
        os.environ["EXECUTION_MODE"] = "live_mt5" if trading_mode in {"live", "live_mt5"} else "paper"


def build_supervisor() -> DevelopmentOperationalSupervisor:
    configure_runtime_environment()
    config = load_cloud_config()
    db_path = Path(config.db_path)
    return DevelopmentOperationalSupervisor(db_path=db_path)


def summarize_result(name: str, result: Any) -> dict[str, Any]:
    return {"task": name, "ts_utc": utc_now_iso(), "result": result}

