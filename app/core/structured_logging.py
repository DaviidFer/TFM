from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.cloud.cloud_paths import LOCAL_PATHS


LOG_FILE_PATH: Path = LOCAL_PATHS.runtime_log


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_log(component: str, event: str, *, console: bool = True, **fields: Any) -> None:
    """
    Emite log estructurado en formato JSONL:
    - stdout (visible en ejecución)
    - fichero persistente para auditoría
    """
    payload: Dict[str, Any] = {
        "ts_utc": utc_now_iso(),
        "component": component,
        "event": event,
    }
    payload.update(fields)

    line = json.dumps(payload, ensure_ascii=True)
    if console:
        # Salida humana legible en consola.
        print("")
        print("=" * 100)
        print(f"[{payload['ts_utc']}] {component} -> {event}")
        for key, value in fields.items():
            txt = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            if len(txt) > 280:
                txt = txt[:280] + "..."
            print(f" - {key}: {txt}")
        print("=" * 100)
        # Salida estructurada JSON para persistencia/auditoría.
        print(line)

    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

