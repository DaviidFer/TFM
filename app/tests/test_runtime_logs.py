from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from app.core.structured_logging import LOG_FILE_PATH
from app.phase5_check import main as run_phase5_check


def _tmp_phase5_db() -> Path:
    base = Path("app/.tmp/tests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"phase5_{uuid4().hex[:8]}.sqlite"


def test_runtime_structured_logs_include_key_events() -> None:
    rc = int(run_phase5_check(db_path=_tmp_phase5_db()))
    assert rc == 0
    assert LOG_FILE_PATH.exists()

    lines = LOG_FILE_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0

    events = []
    for line in lines:
        obj = json.loads(line)
        assert "ts_utc" in obj
        assert "component" in obj
        assert "event" in obj
        events.append(obj["event"])

    assert "dataset_ready" in events
    assert "rule_generation_selected_models" in events
    assert "validation_pipeline_started" in events
    assert "trader_promoted_with_rules" in events

