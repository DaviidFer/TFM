from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from .config import PPOPortfolioConfig
from .env import WeeklyPortfolioEnv
from .feature_builder import PortfolioDataset
from .policy import MaskedPortfolioPolicy


def _curve_metrics(curve: pd.Series) -> Dict[str, float]:
    if curve.empty:
        return {"cumulative_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    rets = curve.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cumulative_return = float(curve.iloc[-1] / max(curve.iloc[0], 1e-8) - 1.0)
    sharpe = 0.0
    if float(rets.std(ddof=0)) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(52.0))
    rolling_max = curve.cummax()
    drawdown = curve / rolling_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    return {
        "cumulative_return": cumulative_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


class PortfolioPolicyEvaluator:
    def __init__(self, config: PPOPortfolioConfig, device: str) -> None:
        self.config = config
        self.device = device

    def _obs_to_tensors(self, obs: Dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trader = torch.tensor(obs["trader_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
        global_f = torch.tensor(obs["global_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
        active = torch.tensor(obs["active_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        return trader, global_f, active

    def evaluate_split(
        self,
        policy: MaskedPortfolioPolicy,
        dataset: PortfolioDataset,
        split_slice: slice,
    ) -> Dict[str, Any]:
        env = WeeklyPortfolioEnv(dataset, self.config, split_slice)
        obs = env.reset()
        done = False
        curve_rows: List[Dict[str, Any]] = [{"date": str(dataset.dates[env.current_idx]), "equity": 1.0}]
        snapshots: List[Dict[str, Any]] = []
        rewards: List[float] = []

        policy.eval()
        while not done:
            trader_t, global_t, active_t = self._obs_to_tensors(obs)
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
            step = env.step(weights)
            rewards.append(float(step.reward))
            curve_rows.append({"date": str(dataset.dates[env.current_idx]), "equity": float(env.portfolio_equity[-1])})

            trader_weights = weights[:-1]
            selected = [
                dataset.trader_ids[i]
                for i in range(dataset.n_traders)
                if float(trader_weights[i]) >= float(self.config.min_live_weight)
            ]
            snapshots.append(
                {
                    "rebalance_date": str(dataset.dates[min(env.current_idx, dataset.n_steps - 1)]),
                    "selected_traders": selected,
                    "target_weights": {
                        dataset.trader_ids[i]: float(trader_weights[i])
                        for i in range(dataset.n_traders)
                        if float(trader_weights[i]) > 0.0
                    },
                    "target_cash_weight": float(weights[-1]),
                    "n_active": int(obs["active_mask"].sum()),
                    "n_selected": int(len(selected)),
                }
            )
            obs = env.get_observation()
            done = step.done

        curve_df = pd.DataFrame(curve_rows)
        curve_df["date"] = pd.to_datetime(curve_df["date"], errors="coerce")
        curve_df = curve_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        curve = curve_df.set_index("date")["equity"] if not curve_df.empty else pd.Series(dtype=float)
        metrics = _curve_metrics(curve)
        metrics["avg_reward"] = float(np.mean(rewards)) if rewards else 0.0
        metrics["turnover_mean"] = float(np.mean([s.get("n_selected", 0) for s in snapshots])) if snapshots else 0.0
        metrics["score"] = (
            self.config.score_weights.get("net_sharpe", 0.0) * metrics.get("sharpe", 0.0)
            + self.config.score_weights.get("max_drawdown", 0.0) * abs(metrics.get("max_drawdown", 0.0))
        )
        return {"metrics": metrics, "curve": curve_df, "snapshots": snapshots}

    def evaluate_forward_one_year(
        self,
        dataset: PortfolioDataset,
        snapshots: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        dates = pd.Index(dataset.dates)
        out: List[Dict[str, Any]] = []
        for snap in snapshots:
            try:
                start_idx = int(dates.get_loc(pd.Timestamp(snap["rebalance_date"])))
            except Exception:
                continue
            end_idx = min(dataset.n_steps, start_idx + int(self.config.forward_horizon_weeks))
            if end_idx <= start_idx + 1:
                continue

            weights_vec = np.zeros(dataset.n_traders + 1, dtype=np.float32)
            for trader_id, weight in dict(snap.get("target_weights") or {}).items():
                idx = dataset.trader_index.get(str(trader_id))
                if idx is not None:
                    weights_vec[idx] = float(weight)
            weights_vec[-1] = float(snap.get("target_cash_weight") or 0.0)

            active = dataset.active_mask[start_idx]
            active_count = int(active.sum())
            active_eq = np.where(active > 0, 1.0, 0.0)
            if active_eq.sum() > 0:
                active_eq = active_eq / active_eq.sum()
            all_active_weights = np.concatenate([active_eq, [0.0]]).astype(np.float32)
            cash_weights = np.zeros(dataset.n_traders + 1, dtype=np.float32)
            cash_weights[-1] = 1.0

            bench_map = {
                "ppo": weights_vec,
                "all_active_equal_weight": all_active_weights,
                "cash": cash_weights,
            }
            for bench_name, bench_weights in bench_map.items():
                curve_values = [1.0]
                for t in range(start_idx + 1, end_idx):
                    ret_vec = dataset.returns[t]
                    gross = float(np.dot(bench_weights[:-1], ret_vec))
                    curve_values.append(curve_values[-1] * (1.0 + gross))
                curve_idx = dates[start_idx:end_idx]
                curve = pd.Series(curve_values[: len(curve_idx)], index=curve_idx, dtype=float)
                metrics = _curve_metrics(curve)
                out.append(
                    {
                        "rebalance_date": str(snap["rebalance_date"]),
                        "benchmark_name": bench_name,
                        "cumulative_return_1y": metrics["cumulative_return"],
                        "sharpe_1y": metrics["sharpe"],
                        "max_drawdown_1y": metrics["max_drawdown"],
                        "curve_points": [{"date": str(k), "equity": float(v)} for k, v in curve.items()],
                        "n_active": active_count,
                        "n_selected": int(snap.get("n_selected") or 0),
                        "target_cash_weight": float(snap.get("target_cash_weight") or 0.0),
                    }
                )
        return out
