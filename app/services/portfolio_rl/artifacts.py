from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from .config import PPOPortfolioConfig


class PortfolioArtifactsManager:
    def __init__(self, config: PPOPortfolioConfig) -> None:
        self.config = config
        self.root = Path(config.artifact_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.root / str(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_checkpoint(
        self,
        *,
        run_id: str,
        payload: Dict[str, Any],
        filename: str = "checkpoint.pt",
    ) -> str:
        path = self.run_dir(run_id) / filename
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str, *, map_location: str = "cpu") -> Dict[str, Any]:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    def save_json(self, *, run_id: str, filename: str, payload: Dict[str, Any]) -> str:
        path = self.run_dir(run_id) / filename
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return str(path)

    def save_frame(self, *, run_id: str, filename: str, df: pd.DataFrame) -> str:
        path = self.run_dir(run_id) / filename
        df.to_csv(path, index=False, encoding="utf-8")
        return str(path)
