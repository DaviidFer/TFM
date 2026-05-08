from __future__ import annotations

import json

from app.cloud import S3Storage, load_cloud_config
from app.cloud.heartbeat import build_heartbeat_payload, write_heartbeat


def test_load_cloud_config_reads_env(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "project"
    artifacts_root = project_dir / "app" / ".tmp"
    db_path = project_dir / "app" / "storage" / "state.sqlite"
    monkeypatch.setenv("TFM_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv("TFM_ARTIFACTS_ROOT", str(artifacts_root))
    monkeypatch.setenv("TFM_DB_PATH", str(db_path))
    monkeypatch.setenv("TFM_ENABLE_S3", "true")
    monkeypatch.setenv("TFM_S3_BUCKET", "bucket-demo")
    monkeypatch.setenv("STREAMLIT_PORT", "9999")

    config = load_cloud_config()

    assert config.project_dir == project_dir
    assert config.artifacts_root == artifacts_root
    assert config.db_path == db_path
    assert config.enable_s3 is True
    assert config.s3_bucket == "bucket-demo"
    assert config.streamlit_port == 9999


def test_heartbeat_writes_local_file(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "project"
    artifacts_root = project_dir / "app" / ".tmp"
    db_path = project_dir / "app" / "storage" / "state.sqlite"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TFM_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv("TFM_ARTIFACTS_ROOT", str(artifacts_root))
    monkeypatch.setenv("TFM_DB_PATH", str(db_path))
    monkeypatch.setenv("TFM_ENABLE_S3", "false")

    payload = build_heartbeat_payload()
    output = write_heartbeat(payload)

    assert output.exists()
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["db_path"] == str(db_path)
    assert parsed["paths"]["db_path_exists"] is True


def test_s3_storage_disabled_without_bucket(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TFM_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setenv("TFM_ENABLE_S3", "true")
    monkeypatch.delenv("TFM_S3_BUCKET", raising=False)

    storage = S3Storage(load_cloud_config())

    assert storage.enabled() is False


def test_load_cloud_config_discovers_project_from_cwd(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "tfm-project-gitpublic"
    (project_dir / ".git").mkdir(parents=True, exist_ok=True)
    (project_dir / "requirements.txt").write_text("numpy<=2.3.5\n", encoding="utf-8")
    monkeypatch.delenv("TFM_PROJECT_DIR", raising=False)
    monkeypatch.delenv("TFM_ARTIFACTS_ROOT", raising=False)
    monkeypatch.delenv("TFM_DB_PATH", raising=False)
    monkeypatch.chdir(project_dir)

    config = load_cloud_config()

    assert config.project_dir == project_dir
    assert config.artifacts_root == project_dir / "app" / ".tmp"
    assert config.db_path == project_dir / "app" / ".tmp" / "supervisor" / "supervisor.sqlite"


def test_load_cloud_config_ignores_stale_artifact_envs_outside_project(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "tfm-project-gitpublic"
    (project_dir / ".git").mkdir(parents=True, exist_ok=True)
    (project_dir / "requirements.txt").write_text("numpy<=2.3.5\n", encoding="utf-8")
    stale_root = tmp_path / "tfm-project" / "app" / ".tmp"
    stale_db = tmp_path / "tfm-project-live" / "app" / ".tmp" / "supervisor" / "supervisor.sqlite"
    monkeypatch.setenv("TFM_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv("TFM_ARTIFACTS_ROOT", str(stale_root))
    monkeypatch.setenv("TFM_DB_PATH", str(stale_db))

    config = load_cloud_config()

    assert config.project_dir == project_dir
    assert config.artifacts_root == project_dir / "app" / ".tmp"
    assert config.db_path == project_dir / "app" / ".tmp" / "supervisor" / "supervisor.sqlite"

