from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .artifacts import PortfolioArtifactsManager
from .config import PPOPortfolioConfig
from .feature_builder import PortfolioDataset
from .policy import MaskedPortfolioPolicy
from .ppo_trainer import PPOTrainer


class PPOInferenceService:
    def __init__(self, config: PPOPortfolioConfig, artifacts_manager: PortfolioArtifactsManager) -> None:
        self.config = config
        self.artifacts_manager = artifacts_manager
        self._trainer = PPOTrainer(config, artifacts_manager)

    @property
    def device(self) -> str:
        return self._trainer.device

    def load_policy(self, dataset: PortfolioDataset, checkpoint_path: str) -> MaskedPortfolioPolicy:
        policy = self._trainer.load_policy(dataset, checkpoint_path=checkpoint_path)
        policy.eval()
        return policy

    def _build_live_observation(
        self,
        dataset: PortfolioDataset,
        *,
        active_trader_ids: Iterable[str],
        rebalance_date: str | None = None,
        previous_snapshot: Mapping[str, Any] | None = None,
    ) -> Dict[str, np.ndarray]:
        if rebalance_date:
            ts = pd.Timestamp(rebalance_date)
            idx = max(0, int(np.searchsorted(np.asarray(dataset.dates, dtype="datetime64[ns]"), ts.to_datetime64(), side="right") - 1))
        else:
            idx = dataset.n_steps - 1
        idx = max(0, min(idx, dataset.n_steps - 1))

        active_ids = {str(x) for x in active_trader_ids}
        active_mask = np.asarray([1.0 if tid in active_ids else 0.0 for tid in dataset.trader_ids], dtype=np.float32)
        prev_weights = np.zeros(dataset.n_traders + 1, dtype=np.float32)
        current_weights = np.zeros(dataset.n_traders + 1, dtype=np.float32)
        current_weights[-1] = 1.0

        if previous_snapshot:
            prev_map = dict(previous_snapshot.get("target_weights") or {})
            for trader_id, weight in prev_map.items():
                trader_idx = dataset.trader_index.get(str(trader_id))
                if trader_idx is not None:
                    prev_weights[trader_idx] = float(weight)
                    current_weights[trader_idx] = float(weight)
            prev_weights[-1] = float(previous_snapshot.get("target_cash_weight") or 0.0)
            current_weights[-1] = float(previous_snapshot.get("target_cash_weight") or 0.0)
            total = float(current_weights.sum())
            if total > 0:
                current_weights = current_weights / total
                prev_weights = prev_weights / max(float(prev_weights.sum()), 1e-8)

        trader_features = np.concatenate(
            [
                dataset.trader_features[idx],
                np.column_stack(
                    [
                        (current_weights[:-1] > self.config.min_live_weight).astype(np.float32),
                        prev_weights[:-1].astype(np.float32),
                        current_weights[:-1].astype(np.float32),
                        active_mask,
                    ]
                ),
            ],
            axis=1,
        ).astype(np.float32)
        global_features = np.concatenate(
            [
                dataset.global_features[idx],
                np.asarray(
                    [
                        float(current_weights[-1]),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "trader_features": trader_features,
            "global_features": global_features,
            "active_mask": active_mask,
        }

    def infer(
        self,
        *,
        dataset: PortfolioDataset,
        checkpoint_path: str,
        active_trader_ids: Iterable[str],
        total_capital_eur: float,
        rebalance_date: str | None = None,
        previous_snapshot: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        policy = self.load_policy(dataset, checkpoint_path)
        obs = self._build_live_observation(
            dataset,
            active_trader_ids=active_trader_ids,
            rebalance_date=rebalance_date,
            previous_snapshot=previous_snapshot,
        )
        trader_t = torch.tensor(obs["trader_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
        global_t = torch.tensor(obs["global_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
        active_t = torch.tensor(obs["active_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = policy.sample_action(
                trader_t,
                global_t,
                active_t,
                max_weight_per_trader=self.config.max_weight_per_trader,
                cash_bias=self.config.cash_bias,
                deterministic=True,
            )
        weights = out["weights"].squeeze(0).detach().cpu().numpy()
        trader_weights = weights[:-1]
        cash_weight = float(weights[-1])
        selected = [
            dataset.trader_ids[i]
            for i in range(dataset.n_traders)
            if float(trader_weights[i]) >= float(self.config.min_live_weight)
        ]
        weight_map = {
            dataset.trader_ids[i]: float(trader_weights[i])
            for i in range(dataset.n_traders)
            if float(trader_weights[i]) > 0.0
        }
        euro_map = {trader_id: float(weight * total_capital_eur) for trader_id, weight in weight_map.items()}
        return {
            "selected_tickers": selected,
            "weights": weight_map,
            "euros": euro_map,
            "target_cash_weight": cash_weight,
            "active_universe_size": int(np.sum(obs["active_mask"] > 0)),
            "selected_universe_size": len(selected),
            "diagnostics": {
                "policy_value": float(out["value"].squeeze(0).cpu().item()),
                "device": self.device,
            },
            "observation_date": str(rebalance_date or dataset.dates[-1]),
        }
