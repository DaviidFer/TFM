from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    candidate_paths: list[Path] = []
    env_project_dir = str(os.getenv("TFM_PROJECT_DIR", "")).strip()
    if env_project_dir:
        candidate_paths.append(Path(env_project_dir) / ".env")
    candidate_paths.append(Path.cwd() / ".env")
    candidate_paths.append(Path(r"C:\tfm\tfm-project") / ".env")

    seen: set[str] = set()
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            break


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


@dataclass(frozen=True)
class CloudConfig:
    tfm_env: str
    project_dir: Path
    artifacts_root: Path
    db_path: Path
    logs_dir: Path
    aws_region: str
    s3_bucket: str
    s3_prefix: str
    enable_s3: bool
    enable_cloudwatch: bool
    portfolio_manager_mode: str
    trading_mode: str
    streamlit_port: int
    mt5_login: str
    mt5_password: str
    mt5_server: str
    mt5_path: str

    @property
    def has_s3_bucket(self) -> bool:
        return bool(self.s3_bucket.strip())

    @property
    def project_logs_dir(self) -> Path:
        return self.project_dir / "app" / ".tmp" / "logs"


def load_cloud_config() -> CloudConfig:
    _load_project_dotenv()
    project_dir = Path(os.getenv("TFM_PROJECT_DIR", r"C:\tfm\tfm-project")).expanduser()
    artifacts_root = Path(
        os.getenv("TFM_ARTIFACTS_ROOT", str(project_dir / "app" / ".tmp"))
    ).expanduser()
    db_path = Path(
        os.getenv("TFM_DB_PATH", str(project_dir / "app" / ".tmp" / "supervisor" / "supervisor.sqlite"))
    ).expanduser()
    logs_dir = artifacts_root / "logs"
    return CloudConfig(
        tfm_env=str(os.getenv("TFM_ENV", "local")).strip() or "local",
        project_dir=project_dir,
        artifacts_root=artifacts_root,
        db_path=db_path,
        logs_dir=logs_dir,
        aws_region=str(os.getenv("AWS_REGION", "eu-west-2")).strip() or "eu-west-2",
        s3_bucket=str(os.getenv("TFM_S3_BUCKET", "")).strip(),
        s3_prefix=str(os.getenv("TFM_S3_PREFIX", "tfm-trading")).strip() or "tfm-trading",
        enable_s3=_to_bool(os.getenv("TFM_ENABLE_S3"), default=False),
        enable_cloudwatch=_to_bool(os.getenv("TFM_ENABLE_CLOUDWATCH"), default=False),
        portfolio_manager_mode=str(os.getenv("PORTFOLIO_MANAGER_MODE", "ppo")).strip() or "ppo",
        trading_mode=str(os.getenv("TRADING_MODE", "paper")).strip() or "paper",
        streamlit_port=_to_int(os.getenv("STREAMLIT_PORT"), default=8501),
        mt5_login=str(os.getenv("MT5_LOGIN", "")).strip(),
        mt5_password=str(os.getenv("MT5_PASSWORD", "")).strip(),
        mt5_server=str(os.getenv("MT5_SERVER", "")).strip(),
        mt5_path=str(os.getenv("MT5_PATH", "")).strip(),
    )

