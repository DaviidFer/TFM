from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class PPOPortfolioConfig:
    portfolio_manager_mode: str = "ppo"
    artifact_root: str = "app/.tmp/portfolio_rl"
    seed: int = 42
    device_preference: str = "auto"
    weekly_frequency: str = "W-FRI"
    forward_horizon_weeks: int = 52
    min_history_weeks: int = 52
    initial_training_min_weeks: int = 104
    fine_tune_window_weeks: int = 104
    train_split: float = 0.65
    val_split: float = 0.20
    transaction_cost_rate: float = 0.0010
    reward_scale: float = 25.0
    lambda_turnover: float = 0.01
    lambda_concentration: float = 0.01
    lambda_dd: float = 0.20
    lambda_cash: float = 0.0
    dd_soft_limit: float = 0.15
    cash_soft_limit: float = 0.60
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_epsilon: float = 0.15
    entropy_coef: float = 0.003
    value_loss_coef: float = 0.50
    learning_rate: float = 2e-4
    batch_size: int = 64
    ppo_epochs: int = 8
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    normalize_features: bool = True
    max_updates_initial: int = 30
    max_updates_fine_tune: int = 12
    hidden_dim_encoder: int = 64
    hidden_dim_head: int = 128
    dropout: float = 0.05
    min_open_positions: int = 10
    max_weight_per_trader: float = 0.15
    min_live_weight: float = 0.01
    cash_bias: float = 0.0
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "net_sharpe": 1.0,
            "sortino": 0.5,
            "max_drawdown": -0.75,
            "turnover": -0.25,
        }
    )

    @property
    def artifact_root_path(self) -> Path:
        return Path(self.artifact_root)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
