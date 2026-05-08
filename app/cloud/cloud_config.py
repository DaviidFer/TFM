from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _is_project_dir(path: Path) -> bool:
    try:
        return path.exists() and (path / "requirements.txt").exists() and (path / ".git").exists()
    except Exception:
        return False


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _normalized_project_path(*, env_name: str, default: Path, project_dir: Path) -> Path:
    raw = str(os.getenv(env_name, "")).strip()
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    # Si la variable heredada apunta fuera del repo activo, la tratamos como
    # obsoleta y volvemos a la ruta canónica del proyecto actual.
    if not _path_is_within(candidate, project_dir):
        return default
    return candidate


def _discover_project_dir() -> Path:
    candidate_paths: list[Path] = []
    env_project_dir = str(os.getenv("TFM_PROJECT_DIR", "")).strip()
    if env_project_dir:
        candidate_paths.append(Path(env_project_dir).expanduser())

    cwd = Path.cwd()
    candidate_paths.append(cwd)
    candidate_paths.extend(cwd.parents)

    candidate_paths.extend(
        [
            Path(r"C:\tfm\tfm-project-gitpublic"),
            Path(r"C:\tfm\tfm-project"),
            Path(r"C:\tfm-trading"),
            Path(r"C:\TFM"),
            Path(r"C:\tfm\TFM"),
        ]
    )

    seen: set[str] = set()
    for candidate in candidate_paths:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_project_dir(candidate):
            return candidate

    tfm_root = Path(r"C:\tfm")
    if tfm_root.exists():
        repos = sorted(
            [p for p in tfm_root.iterdir() if p.is_dir() and _is_project_dir(p)],
            key=lambda p: (0 if p.name == "tfm-project-gitpublic" else 1 if p.name == "tfm-project" else 2, str(p).lower()),
        )
        if repos:
            return repos[0]

    return Path(env_project_dir or r"C:\tfm\tfm-project-gitpublic").expanduser()


def _load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    candidate_paths: list[Path] = []
    env_project_dir = str(os.getenv("TFM_PROJECT_DIR", "")).strip()
    if env_project_dir:
        candidate_paths.append(Path(env_project_dir).expanduser() / ".env")
    candidate_paths.append(Path.cwd() / ".env")
    discovered = _discover_project_dir()
    candidate_paths.append(discovered / ".env")
    candidate_paths.append(Path(r"C:\tfm\tfm-project-gitpublic") / ".env")
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
    env_project_dir = str(os.getenv("TFM_PROJECT_DIR", "")).strip()
    project_dir = Path(env_project_dir).expanduser() if env_project_dir else _discover_project_dir()
    project_dir = project_dir.expanduser()
    artifacts_root_default = project_dir / "app" / ".tmp"
    db_path_default = artifacts_root_default / "supervisor" / "supervisor.sqlite"
    artifacts_root = _normalized_project_path(
        env_name="TFM_ARTIFACTS_ROOT",
        default=artifacts_root_default,
        project_dir=project_dir,
    ).expanduser()
    db_path = _normalized_project_path(
        env_name="TFM_DB_PATH",
        default=db_path_default,
        project_dir=project_dir,
    ).expanduser()
    # Reexporta valores consistentes para que cualquier subprocess/importe
    # posterior vea el mismo proyecto/SQLite, aunque la sesión arrastrase envs
    # obsoletas desde repos antiguos.
    os.environ["TFM_PROJECT_DIR"] = str(project_dir)
    os.environ["TFM_ARTIFACTS_ROOT"] = str(artifacts_root)
    os.environ["TFM_DB_PATH"] = str(db_path)
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
        portfolio_manager_mode=str(os.getenv("PORTFOLIO_MANAGER_MODE", "ga_pso")).strip() or "ga_pso",
        trading_mode=str(os.getenv("TRADING_MODE", "paper")).strip() or "paper",
        streamlit_port=_to_int(os.getenv("STREAMLIT_PORT"), default=8501),
        mt5_login=str(os.getenv("MT5_LOGIN", "")).strip(),
        mt5_password=str(os.getenv("MT5_PASSWORD", "")).strip(),
        mt5_server=str(os.getenv("MT5_SERVER", "")).strip(),
        mt5_path=str(os.getenv("MT5_PATH", "")).strip(),
    )

