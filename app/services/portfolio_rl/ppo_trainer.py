from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from .artifacts import PortfolioArtifactsManager
from .config import PPOPortfolioConfig
from .env import WeeklyPortfolioEnv
from .evaluator import PortfolioPolicyEvaluator
from .feature_builder import PortfolioDataset
from .policy import MaskedPortfolioPolicy


@dataclass
class TrajectoryBatch:
    trader_features: torch.Tensor
    global_features: torch.Tensor
    active_mask: torch.Tensor
    latent_actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    rewards: List[float]


class PPOTrainer:
    def __init__(self, config: PPOPortfolioConfig, artifacts_manager: PortfolioArtifactsManager) -> None:
        self.config = config
        self.artifacts_manager = artifacts_manager
        if config.device_preference == "cpu":
            self.device = "cpu"
        elif config.device_preference == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device_preference

    def build_policy(self, dataset: PortfolioDataset) -> MaskedPortfolioPolicy:
        return MaskedPortfolioPolicy(
            trader_feature_dim=int(dataset.trader_feature_dim + 4),
            global_feature_dim=int(dataset.global_feature_dim + 7),
            hidden_dim_encoder=self.config.hidden_dim_encoder,
            hidden_dim_head=self.config.hidden_dim_head,
            dropout=self.config.dropout,
        ).to(self.device)

    def load_policy(
        self,
        dataset: PortfolioDataset,
        *,
        checkpoint_path: str | None = None,
    ) -> MaskedPortfolioPolicy:
        policy = self.build_policy(dataset)
        if checkpoint_path:
            payload = self.artifacts_manager.load_checkpoint(checkpoint_path, map_location=self.device)
            state_dict = payload.get("policy_state_dict") or payload.get("model_state_dict")
            if isinstance(state_dict, Mapping):
                policy.load_state_dict(state_dict, strict=True)
        return policy

    @staticmethod
    def _clone_dataset(dataset: PortfolioDataset) -> PortfolioDataset:
        return PortfolioDataset(
            dates=list(dataset.dates),
            trader_ids=list(dataset.trader_ids),
            trader_features=np.asarray(dataset.trader_features, dtype=np.float32).copy(),
            global_features=np.asarray(dataset.global_features, dtype=np.float32).copy(),
            returns=np.asarray(dataset.returns, dtype=np.float32).copy(),
            active_mask=np.asarray(dataset.active_mask, dtype=np.float32).copy(),
            trader_feature_names=list(dataset.trader_feature_names),
            global_feature_names=list(dataset.global_feature_names),
            trade_metadata=dict(dataset.trade_metadata or {}),
        )

    @staticmethod
    def _safe_std(values: np.ndarray) -> np.ndarray:
        std = np.asarray(values, dtype=np.float32).copy()
        std = np.where(np.isfinite(std), std, 1.0)
        std = np.where(std < 1e-6, 1.0, std)
        return std.astype(np.float32)

    def _compute_feature_stats(self, dataset: PortfolioDataset, train_slice: slice) -> Dict[str, np.ndarray]:
        trader_train = np.asarray(dataset.trader_features[train_slice], dtype=np.float32)
        global_train = np.asarray(dataset.global_features[train_slice], dtype=np.float32)
        trader_mean = trader_train.reshape(-1, trader_train.shape[-1]).mean(axis=0, dtype=np.float32)
        trader_std = self._safe_std(trader_train.reshape(-1, trader_train.shape[-1]).std(axis=0, dtype=np.float32))
        global_mean = global_train.mean(axis=0, dtype=np.float32)
        global_std = self._safe_std(global_train.std(axis=0, dtype=np.float32))
        return {
            "trader_mean": trader_mean.astype(np.float32),
            "trader_std": trader_std.astype(np.float32),
            "global_mean": global_mean.astype(np.float32),
            "global_std": global_std.astype(np.float32),
        }

    def _apply_feature_stats(self, dataset: PortfolioDataset, feature_stats: Mapping[str, Any] | None) -> PortfolioDataset:
        work = self._clone_dataset(dataset)
        if not feature_stats:
            return work
        trader_mean = np.asarray(feature_stats.get("trader_mean", []), dtype=np.float32)
        trader_std = np.asarray(feature_stats.get("trader_std", []), dtype=np.float32)
        global_mean = np.asarray(feature_stats.get("global_mean", []), dtype=np.float32)
        global_std = np.asarray(feature_stats.get("global_std", []), dtype=np.float32)
        if trader_mean.size == work.trader_features.shape[-1] and trader_std.size == work.trader_features.shape[-1]:
            work.trader_features = ((work.trader_features - trader_mean) / trader_std).astype(np.float32)
        if global_mean.size == work.global_features.shape[-1] and global_std.size == work.global_features.shape[-1]:
            work.global_features = ((work.global_features - global_mean) / global_std).astype(np.float32)
        return work

    def prepare_dataset_for_training(self, dataset: PortfolioDataset, splits: Mapping[str, slice]) -> tuple[PortfolioDataset, Dict[str, np.ndarray]]:
        if not bool(self.config.normalize_features):
            return self._clone_dataset(dataset), {}
        feature_stats = self._compute_feature_stats(dataset, splits["train"])
        return self._apply_feature_stats(dataset, feature_stats), feature_stats

    def prepare_dataset_for_inference(self, dataset: PortfolioDataset, checkpoint_payload: Mapping[str, Any] | None) -> PortfolioDataset:
        feature_stats = dict((checkpoint_payload or {}).get("feature_stats") or {})
        return self._apply_feature_stats(dataset, feature_stats)

    def _collect_trajectory(self, policy: MaskedPortfolioPolicy, env: WeeklyPortfolioEnv) -> TrajectoryBatch:
        obs = env.reset()
        done = False
        trader_features: List[np.ndarray] = []
        global_features: List[np.ndarray] = []
        active_masks: List[np.ndarray] = []
        latent_actions: List[np.ndarray] = []
        old_log_probs: List[float] = []
        values: List[float] = []
        rewards: List[float] = []

        policy.train()
        while not done:
            trader_t = torch.tensor(obs["trader_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
            global_t = torch.tensor(obs["global_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
            active_t = torch.tensor(obs["active_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
            out = policy.sample_action(
                trader_t,
                global_t,
                active_t,
                max_weight_per_trader=self.config.max_weight_per_trader,
                cash_bias=self.config.cash_bias,
                deterministic=False,
            )
            weights = out["weights"].squeeze(0).detach().cpu().numpy()
            step = env.step(weights)
            trader_features.append(obs["trader_features"])
            global_features.append(obs["global_features"])
            active_masks.append(obs["active_mask"])
            latent_actions.append(out["latent_action"].squeeze(0).detach().cpu().numpy())
            old_log_probs.append(float(out["log_prob"].squeeze(0).detach().cpu().item()))
            values.append(float(out["value"].squeeze(0).detach().cpu().item()))
            rewards.append(float(step.reward))
            obs = env.get_observation()
            done = step.done

        returns, advantages = self._compute_gae(rewards, values)
        return TrajectoryBatch(
            trader_features=torch.tensor(np.asarray(trader_features), dtype=torch.float32, device=self.device),
            global_features=torch.tensor(np.asarray(global_features), dtype=torch.float32, device=self.device),
            active_mask=torch.tensor(np.asarray(active_masks), dtype=torch.float32, device=self.device),
            latent_actions=torch.tensor(np.asarray(latent_actions), dtype=torch.float32, device=self.device),
            old_log_probs=torch.tensor(np.asarray(old_log_probs), dtype=torch.float32, device=self.device),
            returns=torch.tensor(np.asarray(returns), dtype=torch.float32, device=self.device),
            advantages=torch.tensor(np.asarray(advantages), dtype=torch.float32, device=self.device),
            values=torch.tensor(np.asarray(values), dtype=torch.float32, device=self.device),
            rewards=rewards,
        )

    def _compute_gae(self, rewards: List[float], values: List[float]) -> tuple[np.ndarray, np.ndarray]:
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        values_arr = np.asarray(values + [0.0], dtype=np.float32)
        advantages = np.zeros_like(rewards_arr, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards_arr))):
            delta = rewards_arr[t] + self.config.gamma * values_arr[t + 1] - values_arr[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
        returns = advantages + values_arr[:-1]
        if advantages.size > 1:
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)
        return returns, advantages

    def _ppo_update(
        self,
        policy: MaskedPortfolioPolicy,
        optimizer: torch.optim.Optimizer,
        batch: TrajectoryBatch,
    ) -> Dict[str, float]:
        n = batch.trader_features.shape[0]
        batch_size = min(self.config.batch_size, n)
        idx_all = torch.arange(n, device=self.device)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        clip_fractions: List[float] = []
        stopped_by_kl = False

        for _ in range(int(self.config.ppo_epochs)):
            perm = idx_all[torch.randperm(n, device=self.device)]
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                eval_out = policy.evaluate_actions(
                    batch.trader_features[idx],
                    batch.global_features[idx],
                    batch.active_mask[idx],
                    batch.latent_actions[idx],
                    max_weight_per_trader=self.config.max_weight_per_trader,
                    cash_bias=self.config.cash_bias,
                )
                log_ratio = eval_out["log_prob"] - batch.old_log_probs[idx]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * batch.advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * batch.advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(eval_out["value"], batch.returns[idx])
                entropy = eval_out["entropy"].mean()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > float(self.config.clip_epsilon)).float().mean()
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.config.max_grad_norm)
                optimizer.step()
                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))
                approx_kls.append(float(approx_kl.detach().cpu().item()))
                clip_fractions.append(float(clip_fraction.detach().cpu().item()))
                if float(self.config.target_kl) > 0.0 and float(np.mean(approx_kls)) > float(self.config.target_kl):
                    stopped_by_kl = True
                    break
            if stopped_by_kl:
                break

        with torch.no_grad():
            eval_full = policy.evaluate_actions(
                batch.trader_features,
                batch.global_features,
                batch.active_mask,
                batch.latent_actions,
                max_weight_per_trader=self.config.max_weight_per_trader,
                cash_bias=self.config.cash_bias,
            )
        returns_np = batch.returns.detach().cpu().numpy()
        values_np = eval_full["value"].detach().cpu().numpy()
        var_returns = float(np.var(returns_np))
        explained_variance = 0.0 if var_returns <= 1e-8 else float(1.0 - (np.var(returns_np - values_np) / var_returns))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "average_reward": float(np.mean(batch.rewards)) if batch.rewards else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "explained_variance": explained_variance,
            "log_std": float(policy.log_std.detach().cpu().item()),
            "stopped_by_kl": float(1.0 if stopped_by_kl else 0.0),
        }

    def train(
        self,
        *,
        dataset: PortfolioDataset,
        splits: Dict[str, slice],
        run_id: str,
        model_version: str,
        run_type: str,
        checkpoint_path: str | None = None,
    ) -> Dict[str, Any]:
        working_dataset, feature_stats = self.prepare_dataset_for_training(dataset, splits)
        policy = self.load_policy(working_dataset, checkpoint_path=checkpoint_path)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)
        if checkpoint_path:
            payload = self.artifacts_manager.load_checkpoint(checkpoint_path, map_location=self.device)
            opt_state = payload.get("optimizer_state_dict")
            if isinstance(opt_state, dict):
                optimizer.load_state_dict(opt_state)

        evaluator = PortfolioPolicyEvaluator(self.config, self.device)
        train_env = WeeklyPortfolioEnv(working_dataset, self.config, splits["train"])
        updates = self.config.max_updates_initial if run_type == "initial_train" else self.config.max_updates_fine_tune

        history_rows: List[Dict[str, Any]] = []
        best_score = float("-inf")
        best_state: Dict[str, Any] | None = None

        for update_idx in range(1, int(updates) + 1):
            batch = self._collect_trajectory(policy, train_env)
            train_metrics = self._ppo_update(policy, optimizer, batch)
            val_eval = evaluator.evaluate_split(policy, working_dataset, splits["val"])
            row = {
                "update": int(update_idx),
                **train_metrics,
                "val_score": float(val_eval["metrics"].get("score", 0.0)),
                "val_sharpe": float(val_eval["metrics"].get("sharpe", 0.0)),
                "val_sortino": float(val_eval["metrics"].get("sortino", 0.0)),
                "val_max_drawdown": float(val_eval["metrics"].get("max_drawdown", 0.0)),
                "val_return": float(val_eval["metrics"].get("cumulative_return", 0.0)),
            }
            history_rows.append(row)
            if row["val_score"] >= best_score:
                best_score = row["val_score"]
                best_state = {
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_version": model_version,
                    "run_id": run_id,
                    "config": self.config.to_dict(),
                    "feature_stats": feature_stats,
                }

        if best_state is not None:
            policy.load_state_dict(best_state["policy_state_dict"])

        train_eval = evaluator.evaluate_split(policy, working_dataset, splits["train"])
        val_eval = evaluator.evaluate_split(policy, working_dataset, splits["val"])
        test_eval = evaluator.evaluate_split(policy, working_dataset, splits["test"])
        forward_eval = evaluator.evaluate_forward_one_year(working_dataset, test_eval["snapshots"])
        history_df = pd.DataFrame(history_rows)
        train_curve = train_eval["curve"]
        val_curve = val_eval["curve"]
        test_curve = test_eval["curve"]
        forward_df = pd.DataFrame(forward_eval)

        checkpoint_payload = best_state or {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_version": model_version,
            "run_id": run_id,
            "config": self.config.to_dict(),
            "feature_stats": feature_stats,
        }
        checkpoint_file = self.artifacts_manager.save_checkpoint(run_id=run_id, payload=checkpoint_payload)
        history_file = self.artifacts_manager.save_frame(run_id=run_id, filename="training_history.csv", df=history_df)
        train_curve_file = self.artifacts_manager.save_frame(run_id=run_id, filename="train_curve.csv", df=train_curve)
        val_curve_file = self.artifacts_manager.save_frame(run_id=run_id, filename="val_curve.csv", df=val_curve)
        test_curve_file = self.artifacts_manager.save_frame(run_id=run_id, filename="test_curve.csv", df=test_curve)
        forward_file = self.artifacts_manager.save_frame(run_id=run_id, filename="forward_eval.csv", df=forward_df)

        return {
            "run_id": run_id,
            "run_type": run_type,
            "model_version": model_version,
            "device": self.device,
            "checkpoint_path": checkpoint_file,
            "history": history_rows,
            "history_df": history_df,
            "train_eval": train_eval,
            "val_eval": val_eval,
            "test_eval": test_eval,
            "forward_eval": forward_eval,
            "artifacts": {
                "checkpoint_path": checkpoint_file,
                "history_csv": history_file,
                "train_curve_csv": train_curve_file,
                "val_curve_csv": val_curve_file,
                "test_curve_csv": test_curve_file,
                "forward_eval_csv": forward_file,
            },
        }
