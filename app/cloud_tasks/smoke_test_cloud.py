from __future__ import annotations

import importlib
import json

from app.cloud import CLOUD_PATHS, S3Storage, load_cloud_config
from app.cloud.heartbeat import build_heartbeat_payload, upload_heartbeat_if_enabled, write_heartbeat
from app.cloud_tasks import summarize_result

# Importaciones que PyEventBT / event-backtest resuelven en runtime (no solo al cargar `runner`).
EVENT_BACKTEST_PROBE_MODULES: tuple[str, ...] = (
    "numba",
    "yaml",
    "pyeventbt.indicators.indicators",
    "app.toolbox.backtest_eventos.runner",
)


def _numpy_compatible_with_numba_stack(result: dict[str, object]) -> None:
    """Numba en PyEventBT no admite NumPy 2.4+ en la matriz probada; requirements.txt pinna numpy<2.4."""
    try:
        import numpy as np

        parts = np.__version__.split(".")
        maj, minor = int(parts[0]), int(parts[1])
        if maj > 2 or (maj == 2 and minor >= 4):
            result["imports"]["numpy_numba_abi"] = (
                f"error: numpy {np.__version__} — usar numpy<2.4 con numba/pyeventbt"
            )
            result["status"] = "error"
        else:
            result["imports"]["numpy_numba_abi"] = f"ok ({np.__version__})"
    except Exception as exc:
        result["imports"]["numpy_numba_abi"] = f"error: {exc}"
        result["status"] = "error"


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
        "pandas",
        "polars",
        "numpy",
        "numba",
        "boto3",
    )
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            result["imports"][module_name] = "ok"
        except Exception as exc:
            result["imports"][module_name] = f"error: {exc}"
            result["status"] = "error"

    _numpy_compatible_with_numba_stack(result)

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

    for module_name in EVENT_BACKTEST_PROBE_MODULES:
        key = f"bt_probe:{module_name}"
        try:
            importlib.import_module(module_name)
            result["imports"][key] = "ok"
        except Exception as exc:
            result["imports"][key] = f"error: {exc}"
            result["status"] = "error"

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

