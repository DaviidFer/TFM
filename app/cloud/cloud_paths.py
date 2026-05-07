"""
Configuración centralizada de paths del proyecto.

Reúne en un único módulo:

1. **Prefijos S3** (`CLOUD_PATHS`): los prefijos lógicos del bucket S3 donde el
   sistema persiste artefactos en la nube.
2. **Paths locales** (`LOCAL_PATHS`): la raíz `app/.tmp` y todos sus
   subdirectorios estables (logs, backtests, supervisor SQLite, smoke checks
   por fase, tests, etc.).

Todos los componentes del proyecto (supervisor, dashboard, smoke checks
`phase{N}_check`, servicios de backtest…) deben leer sus rutas desde aquí en
lugar de hardcodear strings sueltos. Eso evita drift entre módulos y centraliza
la configuración para Windows / Linux y para entornos de desarrollo / cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# Prefijos S3 (cloud)
# =============================================================================


@dataclass(frozen=True)
class CloudPathSet:
    """
    Prefijos S3 vivos que se usan en el flujo cloud actual.

    Atributos retirados respecto a versiones anteriores:
    - `ppo_models`: el optimizador de cartera ya no es PPO sino un híbrido GA+PSO,
      no se persisten modelos entrenados en S3.
    - `risk_reports`: el `RiskAgent` fue sustituido por `HumanResourcesProcess`,
      que persiste sus revisiones en SQLite (no en CSV).
    """

    datos: str = "datos"
    artifacts: str = "artifacts"
    backtests: str = "backtests"
    sqlite_backups: str = "sqlite_backups"
    logs: str = "logs"
    deployment: str = "deployment"

    def join(self, *parts: str) -> str:
        clean_parts = [str(part).strip("/\\") for part in parts if str(part).strip("/\\")]
        return "/".join(clean_parts)


CLOUD_PATHS = CloudPathSet()


# =============================================================================
# Paths locales (artefactos en disco)
# =============================================================================


# Raíz de artefactos efímeros del proyecto (sandbox local).
# Convención histórica del TFM: todo lo que no está en `datos/` ni versionado
# se escribe bajo `app/.tmp/` (ya excluido de git por `.gitignore`).
ARTIFACTS_ROOT: Path = Path("app/.tmp")


@dataclass(frozen=True)
class LocalPathSet:
    """
    Paths locales canónicos del proyecto.

    Use `LOCAL_PATHS` (instancia singleton) en lugar de redefinir strings:

    >>> from app.cloud.cloud_paths import LOCAL_PATHS
    >>> LOCAL_PATHS.runtime_log
    PosixPath('app/.tmp/logs/runtime_flow.log')
    >>> LOCAL_PATHS.phase_db(5)
    PosixPath('app/.tmp/phase5/phase5.sqlite')
    """

    artifacts_root: Path = ARTIFACTS_ROOT
    logs_dir: Path = ARTIFACTS_ROOT / "logs"
    runtime_log: Path = ARTIFACTS_ROOT / "logs" / "runtime_flow.log"
    tests_dir: Path = ARTIFACTS_ROOT / "tests"
    supervisor_db: Path = ARTIFACTS_ROOT / "supervisor" / "supervisor.sqlite"
    backtests_csv_dir: Path = ARTIFACTS_ROOT / "backtests_csv"
    backtests_systems_dir: Path = ARTIFACTS_ROOT / "backtests_systems"
    sqlite_backups_dir: Path = ARTIFACTS_ROOT / "sqlite_backups"

    # Mapeo opcional de overrides por fase para los smoke checks históricos.
    # Si en el futuro alguna fase deja de usarse, basta con quitarla del
    # diccionario sin tocar los `phaseN_check.py`.
    phase_dirs: dict[int, str] = field(
        default_factory=lambda: {
            1: "phase1",
            2: "phase2",
            4: "phase4",
            5: "phase5",
            6: "phase5",  # phase6 reusa el sqlite de phase5
            8: "phase8",
            9: "phase9",
            10: "phase10",
        }
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def phase_dir(self, phase: int) -> Path:
        """Devuelve `app/.tmp/phaseN` (sin crearlo)."""
        sub = self.phase_dirs.get(phase, f"phase{phase}")
        return self.artifacts_root / sub

    def phase_db(self, phase: int) -> Path:
        """Devuelve `app/.tmp/phaseN/phaseN.sqlite` (sin crearlo)."""
        sub = self.phase_dirs.get(phase, f"phase{phase}")
        return self.artifacts_root / sub / f"{sub}.sqlite"


LOCAL_PATHS = LocalPathSet()
