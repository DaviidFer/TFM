from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.cloud.cloud_config import CloudConfig


class S3Storage:
    def __init__(self, config: CloudConfig) -> None:
        self.config = config
        self._client = None

    def enabled(self) -> bool:
        return bool(self.config.enable_s3 and self.config.has_s3_bucket)

    def _require_client(self):
        if not self.enabled():
            raise RuntimeError("S3 no esta habilitado o no hay bucket configurado.")
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("boto3 no esta instalado. Instala boto3 para habilitar S3.") from exc
        self._client = boto3.client("s3", region_name=self.config.aws_region)
        return self._client

    def _full_key(self, s3_key: str) -> str:
        key = str(s3_key).strip("/\\")
        prefix = str(self.config.s3_prefix).strip("/\\")
        if prefix:
            return f"{prefix}/{key}" if key else prefix
        return key

    def upload_file(self, local_path: str | Path, s3_key: str) -> str:
        client = self._require_client()
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el fichero local: {path}")
        full_key = self._full_key(s3_key)
        client.upload_file(str(path), self.config.s3_bucket, full_key)
        return full_key

    def download_file(self, s3_key: str, local_path: str | Path) -> Path:
        client = self._require_client()
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(self.config.s3_bucket, self._full_key(s3_key), str(target))
        return target

    def upload_json(self, data: Any, s3_key: str) -> str:
        client = self._require_client()
        full_key = self._full_key(s3_key)
        body = json.dumps(data, ensure_ascii=True, indent=2).encode("utf-8")
        client.put_object(
            Bucket=self.config.s3_bucket,
            Key=full_key,
            Body=body,
            ContentType="application/json",
        )
        return full_key

    def list_prefix(self, prefix: str) -> list[str]:
        client = self._require_client()
        full_prefix = self._full_key(prefix)
        paginator = client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.config.s3_bucket, Prefix=full_prefix):
            for row in page.get("Contents", []) or []:
                key = str(row.get("Key") or "")
                if key:
                    keys.append(key)
        return keys

    def upload_directory(self, local_dir: str | Path, s3_prefix: str) -> list[str]:
        root = Path(local_dir)
        if not root.exists():
            return []
        uploaded: list[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            uploaded.append(self.upload_file(path, f"{s3_prefix.rstrip('/\\')}/{rel}"))
        return uploaded

