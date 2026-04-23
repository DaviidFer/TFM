from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.storage import StateStore


@dataclass
class AgentContext:
    store: StateStore
    artifacts_root: Path
    execution_router: Any | None = None

