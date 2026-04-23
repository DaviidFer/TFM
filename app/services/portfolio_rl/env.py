from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import PPOPortfolioConfig
from .feature_builder import PortfolioDataset


@dataclass
class PortfolioStepResult:
    reward: float
    done: bool
    info: Dict[str, float]


class WeeklyPortfolioEnv:
    def __init__(self, dataset: PortfolioDataset, config: PPOPortfolioConfig, split_slice: slice) -> None:
        self.dataset = dataset
        self.config = config
        self.split_slice = split_slice
        self.start = int(split_slice.start or 0)
        self.stop = int(split_slice.stop or dataset.n_steps)
        self.n_traders = dataset.n_traders
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        self.current_idx = self.start
        self.current_weights = np.zeros(self.n_traders + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        self.previous_target_weights = self.current_weights.copy()
        self.portfolio_returns: list[float] = []
        self.portfolio_equity: list[float] = [1.0]
        self.last_turnover = 0.0
        return self.get_observation()

    def _portfolio_stats(self) -> tuple[float, float, float, float]:
        rets = np.asarray(self.portfolio_returns, dtype=np.float32)
        if rets.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        ret_1w = float(rets[-1])
        ret_4w = float(np.prod(1.0 + rets[-4:]) - 1.0)
        vol_4w = float(np.std(rets[-4:])) if rets.size >= 2 else 0.0
        vol_12w = float(np.std(rets[-12:])) if rets.size >= 2 else 0.0
        return ret_1w, ret_4w, vol_4w, vol_12w

    def _drawdown_12w(self) -> float:
        equity = np.asarray(self.portfolio_equity[-12:] or [1.0], dtype=np.float32)
        peak = np.maximum.accumulate(equity)
        dd = equity / np.clip(peak, 1e-8, None) - 1.0
        return float(dd.min()) if dd.size else 0.0

    def _build_dynamic_trader_features(self, idx: int) -> np.ndarray:
        currently_in_portfolio = (self.current_weights[:-1] > self.config.min_live_weight).astype(np.float32)
        dynamic = np.column_stack(
            [
                currently_in_portfolio,
                self.previous_target_weights[:-1].astype(np.float32),
                self.current_weights[:-1].astype(np.float32),
                self.dataset.active_mask[idx].astype(np.float32),
            ]
        )
        return dynamic.astype(np.float32)

    def _build_dynamic_global_features(self) -> np.ndarray:
        ret_1w, ret_4w, vol_4w, vol_12w = self._portfolio_stats()
        drawdown_12w = self._drawdown_12w()
        return np.asarray(
            [
                float(self.current_weights[-1]),
                ret_1w,
                ret_4w,
                vol_4w,
                vol_12w,
                drawdown_12w,
                float(self.last_turnover),
            ],
            dtype=np.float32,
        )

    def get_observation(self) -> Dict[str, np.ndarray]:
        idx = min(self.current_idx, self.stop - 1)
        trader_static = self.dataset.trader_features[idx]
        dynamic = self._build_dynamic_trader_features(idx)
        trader_features = np.concatenate([trader_static, dynamic], axis=1).astype(np.float32)
        global_features = np.concatenate(
            [self.dataset.global_features[idx], self._build_dynamic_global_features()],
            axis=0,
        ).astype(np.float32)
        active_mask = self.dataset.active_mask[idx].astype(np.float32)
        return {
            "trader_features": trader_features,
            "global_features": global_features,
            "active_mask": active_mask,
        }

    def step(self, action_weights: np.ndarray) -> PortfolioStepResult:
        action = np.asarray(action_weights, dtype=np.float32).copy()
        if action.shape[0] != self.n_traders + 1:
            raise ValueError("Dimensión de acción inválida para WeeklyPortfolioEnv.")
        active = self.dataset.active_mask[self.current_idx].astype(np.float32)
        action[:-1] = action[:-1] * active
        total = float(action.sum())
        if total <= 0:
            action[:] = 0.0
            action[-1] = 1.0
        else:
            action /= total

        next_idx = min(self.current_idx + 1, self.stop - 1)
        next_returns = self.dataset.returns[next_idx].astype(np.float32)
        trader_weights = action[:-1]
        cash_weight = float(action[-1])

        gross_return = float(np.dot(trader_weights, next_returns))
        turnover = float(np.abs(action - self.current_weights).sum())
        cost = float(self.config.transaction_cost_rate) * turnover
        net_return = gross_return - cost
        hhi = float(np.sum(np.square(trader_weights)))
        dd_mag = max(0.0, -self._drawdown_12w())
        dd_penalty = max(0.0, dd_mag - float(self.config.dd_soft_limit))
        cash_penalty = max(0.0, cash_weight - float(self.config.cash_soft_limit))
        reward = (
            float(np.log(max(1e-8, 1.0 + net_return)))
            - float(self.config.lambda_turnover) * turnover
            - float(self.config.lambda_concentration) * hhi
            - float(self.config.lambda_dd) * dd_penalty
            - float(self.config.lambda_cash) * cash_penalty
        )

        post_trader = trader_weights * (1.0 + next_returns)
        post_cash = cash_weight
        denom = float(post_trader.sum() + post_cash)
        if denom <= 0:
            self.current_weights[:] = 0.0
            self.current_weights[-1] = 1.0
        else:
            self.current_weights[:-1] = post_trader / denom
            self.current_weights[-1] = post_cash / denom

        self.previous_target_weights = action
        self.portfolio_returns.append(net_return)
        self.portfolio_equity.append(self.portfolio_equity[-1] * (1.0 + net_return))
        self.last_turnover = turnover
        self.current_idx = next_idx
        done = bool(self.current_idx >= (self.stop - 1))
        return PortfolioStepResult(
            reward=reward,
            done=done,
            info={
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "cost": cost,
                "cash_weight": cash_weight,
                "hhi": hhi,
                "drawdown_penalty": dd_penalty,
            },
        )
