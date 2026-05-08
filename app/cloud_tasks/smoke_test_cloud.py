from __future__ import annotations

import importlib
import json
from pathlib import Path

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
    """Numba 0.62.x no admite NumPy 2.4+; requirements.txt y scripts cloud fijan numpy<=2.3.5."""
    try:
        import numpy as np

        parts = np.__version__.split(".")
        maj, minor = int(parts[0]), int(parts[1])
        if maj > 2 or (maj == 2 and minor >= 4):
            result["imports"]["numpy_numba_abi"] = (
                f"error: numpy {np.__version__} — usar numpy<=2.3.5 con numba/pyeventbt (pip install \"numpy>=1.24,<=2.3.5\")"
            )
            result["status"] = "error"
        else:
            result["imports"]["numpy_numba_abi"] = f"ok ({np.__version__})"
    except Exception as exc:
        result["imports"]["numpy_numba_abi"] = f"error: {exc}"
        result["status"] = "error"


def _numba_version_compatible_with_pyeventbt(result: dict[str, object]) -> None:
    """PyEventBT 0.0.9 requiere numba>=0.62.1,<0.63.0 en la instalacion cloud soportada."""
    try:
        import sys
        import numba

        if tuple(sys.version_info[:2]) != (3, 11):
            result["imports"]["numba_pyeventbt_range"] = f"skipped (python {sys.version_info.major}.{sys.version_info.minor})"
            return

        parts = numba.__version__.split(".")
        maj, minor = int(parts[0]), int(parts[1])
        if maj != 0 or minor < 62 or minor >= 63:
            result["imports"]["numba_pyeventbt_range"] = (
                f"error: numba {numba.__version__} — usar numba>=0.62.1,<0.63.0 con pyeventbt 0.0.9"
            )
            result["status"] = "error"
        else:
            result["imports"]["numba_pyeventbt_range"] = f"ok ({numba.__version__})"
    except Exception as exc:
        result["imports"]["numba_pyeventbt_range"] = f"error: {exc}"
        result["status"] = "error"


def _paths_align_with_project(result: dict[str, object], *, project_dir: Path, artifacts_root: Path, db_path: Path) -> None:
    try:
        project_dir = project_dir.resolve()
        artifacts_root = artifacts_root.resolve()
        db_path = db_path.resolve()
    except Exception:
        pass

    artifacts_under_project = True
    try:
        artifacts_root.relative_to(project_dir)
    except Exception:
        artifacts_under_project = False
    db_under_project = True
    try:
        db_path.relative_to(project_dir)
    except Exception:
        db_under_project = False

    result["paths"]["artifacts_under_project"] = artifacts_under_project
    result["paths"]["db_under_project"] = db_under_project
    if not (artifacts_under_project and db_under_project):
        result["imports"]["cloud_path_alignment"] = (
            f"error: rutas mezcladas; project_dir={project_dir}, artifacts_root={artifacts_root}, db_path={db_path}"
        )
        result["status"] = "error"
    else:
        result["imports"]["cloud_path_alignment"] = "ok"


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
    _numba_version_compatible_with_pyeventbt(result)
    _paths_align_with_project(
        result,
        project_dir=config.project_dir,
        artifacts_root=config.artifacts_root,
        db_path=config.db_path,
    )

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

