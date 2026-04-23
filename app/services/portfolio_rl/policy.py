from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _masked_mean_pool(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    summed = (embeddings * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _weights_from_latent(
    latent_action: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    max_weight_per_trader: float,
    cash_bias: float = 0.0,
) -> torch.Tensor:
    n_traders = active_mask.shape[-1]
    trader_logits = latent_action[..., :n_traders].masked_fill(active_mask <= 0, -1e9)
    cash_logit = latent_action[..., n_traders : n_traders + 1] + float(cash_bias)
    weights = torch.softmax(torch.cat([trader_logits, cash_logit], dim=-1), dim=-1)
    trader_weights = weights[..., :n_traders] * active_mask.float()
    trader_weights = trader_weights.clamp(min=0.0, max=float(max_weight_per_trader))
    trader_sum = trader_weights.sum(dim=-1, keepdim=True)
    cash_weight = (1.0 - trader_sum).clamp(min=0.0)
    denom = trader_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    over_allocated = trader_sum > 1.0
    if over_allocated.any():
        scaled = trader_weights / denom
        trader_weights = torch.where(over_allocated, scaled, trader_weights)
        cash_weight = torch.where(over_allocated, torch.zeros_like(cash_weight), cash_weight)
    return torch.cat([trader_weights, cash_weight], dim=-1)


class MaskedPortfolioPolicy(nn.Module):
    def __init__(
        self,
        *,
        trader_feature_dim: int,
        global_feature_dim: int,
        hidden_dim_encoder: int = 64,
        hidden_dim_head: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.trader_encoder = nn.Sequential(
            nn.Linear(trader_feature_dim, hidden_dim_encoder),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_encoder, hidden_dim_encoder),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim_encoder),
            nn.ReLU(),
        )
        merged_dim = hidden_dim_encoder * 3
        self.actor_head = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim_head),
            nn.ReLU(),
            nn.Linear(hidden_dim_head, 1),
        )
        self.cash_head = nn.Sequential(
            nn.Linear(hidden_dim_encoder * 2, hidden_dim_head),
            nn.ReLU(),
            nn.Linear(hidden_dim_head, 1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim_encoder * 2, hidden_dim_head),
            nn.ReLU(),
            nn.Linear(hidden_dim_head, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(-0.5))

    def _encode(
        self,
        trader_features: torch.Tensor,
        global_features: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trader_emb = self.trader_encoder(trader_features)
        pooled = _masked_mean_pool(trader_emb, active_mask)
        global_emb = self.global_encoder(global_features)
        return trader_emb, pooled, global_emb

    def distribution_params(
        self,
        trader_features: torch.Tensor,
        global_features: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trader_emb, pooled, global_emb = self._encode(trader_features, global_features, active_mask)
        pooled_rep = pooled.unsqueeze(1).expand_as(trader_emb)
        global_rep = global_emb.unsqueeze(1).expand_as(trader_emb)
        actor_in = torch.cat([trader_emb, pooled_rep, global_rep], dim=-1)
        trader_means = self.actor_head(actor_in).squeeze(-1)
        cash_in = torch.cat([pooled, global_emb], dim=-1)
        cash_mean = self.cash_head(cash_in)
        value = self.critic_head(cash_in).squeeze(-1)
        means = torch.cat([trader_means, cash_mean], dim=-1)
        return means, value

    def sample_action(
        self,
        trader_features: torch.Tensor,
        global_features: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        max_weight_per_trader: float,
        cash_bias: float = 0.0,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        means, value = self.distribution_params(trader_features, global_features, active_mask)
        std = torch.exp(self.log_std).expand_as(means)
        if deterministic:
            latent_action = means
        else:
            latent_action = means + torch.randn_like(means) * std
        action_mask = torch.cat([active_mask, torch.ones_like(active_mask[:, :1])], dim=-1)
        dist = torch.distributions.Normal(means, std)
        log_prob = (dist.log_prob(latent_action) * action_mask).sum(dim=-1)
        weights = _weights_from_latent(
            latent_action,
            active_mask,
            max_weight_per_trader=max_weight_per_trader,
            cash_bias=cash_bias,
        )
        return {
            "latent_action": latent_action,
            "log_prob": log_prob,
            "value": value,
            "weights": weights,
            "means": means,
        }

    def evaluate_actions(
        self,
        trader_features: torch.Tensor,
        global_features: torch.Tensor,
        active_mask: torch.Tensor,
        latent_action: torch.Tensor,
        *,
        max_weight_per_trader: float,
        cash_bias: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        means, value = self.distribution_params(trader_features, global_features, active_mask)
        std = torch.exp(self.log_std).expand_as(means)
        dist = torch.distributions.Normal(means, std)
        action_mask = torch.cat([active_mask, torch.ones_like(active_mask[:, :1])], dim=-1)
        log_prob = (dist.log_prob(latent_action) * action_mask).sum(dim=-1)
        entropy = (dist.entropy() * action_mask).sum(dim=-1)
        weights = _weights_from_latent(
            latent_action,
            active_mask,
            max_weight_per_trader=max_weight_per_trader,
            cash_bias=cash_bias,
        )
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "weights": weights,
            "means": means,
        }
