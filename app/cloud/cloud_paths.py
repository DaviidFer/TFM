from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CloudPathSet:
    datos: str = "datos"
    artifacts: str = "artifacts"
    ppo_models: str = "ppo_models"
    backtests: str = "backtests"
    sqlite_backups: str = "sqlite_backups"
    logs: str = "logs"
    risk_reports: str = "risk_reports"
    deployment: str = "deployment"

    def join(self, *parts: str) -> str:
        clean_parts = [str(part).strip("/\\") for part in parts if str(part).strip("/\\")]
        return "/".join(clean_parts)


CLOUD_PATHS = CloudPathSet()

