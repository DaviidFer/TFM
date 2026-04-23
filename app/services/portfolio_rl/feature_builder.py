from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class PortfolioDataset:
    dates: List[pd.Timestamp]
    trader_ids: List[str]
    trader_features: np.ndarray
    global_features: np.ndarray
    returns: np.ndarray
    active_mask: np.ndarray
    trader_feature_names: List[str]
    global_feature_names: List[str]
    trade_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trader_ids = [str(x) for x in self.trader_ids]
        self.trader_index = {tid: idx for idx, tid in enumerate(self.trader_ids)}

    @property
    def n_steps(self) -> int:
        return int(len(self.dates))

    @property
    def n_traders(self) -> int:
        return int(len(self.trader_ids))

    @property
    def trader_feature_dim(self) -> int:
        return int(self.trader_features.shape[-1]) if self.trader_features.size else 0

    @property
    def global_feature_dim(self) -> int:
        return int(self.global_features.shape[-1]) if self.global_features.size else 0


def _rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    equity = (1.0 + series.fillna(0.0)).cumprod()
    rolling_peak = equity.rolling(window, min_periods=1).max()
    dd = equity / rolling_peak - 1.0
    return dd.rolling(window, min_periods=1).min().fillna(0.0)


def _rolling_hit_ratio(series: pd.Series, window: int) -> pd.Series:
    return series.gt(0.0).rolling(window, min_periods=1).mean().fillna(0.0)


def _rolling_sharpe(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=2).mean()
    std = series.rolling(window, min_periods=2).std(ddof=0).replace(0.0, np.nan)
    sharpe = (mean / std) * np.sqrt(52.0)
    return sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _turnover_proxy(active_mask: pd.Series) -> pd.Series:
    active = active_mask.astype(float)
    return active.diff().abs().fillna(0.0)


def build_weekly_feature_dataset(
    weekly_returns: pd.DataFrame,
    active_mask: pd.DataFrame,
    universe_info: Dict[str, Dict[str, Any]],
) -> PortfolioDataset:
    returns_df = weekly_returns.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    active_df = active_mask.copy().reindex_like(returns_df).fillna(0.0).astype(float)
    trader_ids = [str(c) for c in returns_df.columns]
    dates = [pd.Timestamp(x) for x in returns_df.index]

    trader_feature_names = [
        "weeks_since_promotion",
        "signal_active_this_week",
        "signal_was_active_prev_week",
        "rolling_return_1w",
        "rolling_return_4w",
        "rolling_return_12w",
        "rolling_vol_4w",
        "rolling_vol_12w",
        "rolling_sharpe_12w",
        "rolling_maxdd_12w",
        "rolling_maxdd_26w",
        "hit_ratio_12w",
        "historical_trade_count",
        "avg_trade_duration_days",
        "turnover_proxy",
        "avg_corr_active_12w",
        "beta_to_active_basket_12w",
        "confidence_score",
    ]
    global_feature_names = [
        "number_active_traders",
        "number_total_eligible_traders",
        "average_cross_corr_active",
        "equal_weight_basket_return_4w",
        "equal_weight_basket_vol_12w",
    ]

    t_steps = len(dates)
    n_traders = len(trader_ids)
    trader_features = np.zeros((t_steps, n_traders, len(trader_feature_names)), dtype=np.float32)
    global_features = np.zeros((t_steps, len(global_feature_names)), dtype=np.float32)

    eq_basket = returns_df.mean(axis=1).fillna(0.0)
    basket_ret_4w = ((1.0 + eq_basket).rolling(4, min_periods=1).apply(np.prod, raw=True) - 1.0).fillna(0.0)
    basket_vol_12w = eq_basket.rolling(12, min_periods=2).std(ddof=0).fillna(0.0)

    avg_cross_corr = pd.Series(0.0, index=returns_df.index, dtype=float)
    for idx in range(len(returns_df)):
        start = max(0, idx - 11)
        window = returns_df.iloc[start : idx + 1]
        if window.shape[0] < 2 or window.shape[1] < 2:
            avg_cross_corr.iloc[idx] = 0.0
            continue
        corr = window.corr().replace([np.inf, -np.inf], np.nan)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        avg_cross_corr.iloc[idx] = float(np.nanmean(upper.to_numpy())) if np.isfinite(upper.to_numpy()).any() else 0.0

    for j, trader_id in enumerate(trader_ids):
        ret = returns_df[trader_id].astype(float)
        active = active_df[trader_id].astype(float)
        info = dict(universe_info.get(trader_id) or {})
        promotion_date = pd.to_datetime(info.get("promotion_date"), errors="coerce")
        if pd.notna(promotion_date) and getattr(promotion_date, "tzinfo", None) is not None:
            promotion_date = promotion_date.tz_localize(None)
        if pd.isna(promotion_date):
            promotion_date = returns_df.index.min()
        if getattr(returns_df.index, "tz", None) is not None:
            work_index = returns_df.index.tz_localize(None)
        else:
            work_index = returns_df.index
        weeks_since = np.clip(((work_index - promotion_date).days / 7.0).astype(float), a_min=0.0, a_max=None)
        trade_count = float(info.get("trade_count") or 0.0)
        avg_trade_duration = float(info.get("avg_trade_duration_days") or 0.0)
        confidence_score = float(info.get("confidence_score") or info.get("win_rate_pct") or 0.0) / 100.0

        rolling_return_1w = ret.fillna(0.0)
        rolling_return_4w = ((1.0 + ret).rolling(4, min_periods=1).apply(np.prod, raw=True) - 1.0).fillna(0.0)
        rolling_return_12w = ((1.0 + ret).rolling(12, min_periods=1).apply(np.prod, raw=True) - 1.0).fillna(0.0)
        rolling_vol_4w = ret.rolling(4, min_periods=2).std(ddof=0).fillna(0.0)
        rolling_vol_12w = ret.rolling(12, min_periods=2).std(ddof=0).fillna(0.0)
        rolling_sharpe_12w = _rolling_sharpe(ret, 12)
        rolling_maxdd_12w = _rolling_drawdown(ret, 12)
        rolling_maxdd_26w = _rolling_drawdown(ret, 26)
        hit_ratio_12w = _rolling_hit_ratio(ret, 12)
        signal_prev = active.shift(1).fillna(0.0)
        turnover = _turnover_proxy(active)
        corr_proxy = ret.rolling(12, min_periods=2).corr(eq_basket).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        var_basket = eq_basket.rolling(12, min_periods=2).var(ddof=0).replace(0.0, np.nan)
        cov_tb = ret.rolling(12, min_periods=2).cov(eq_basket).replace([np.inf, -np.inf], np.nan)
        beta_proxy = (cov_tb / var_basket).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        feature_matrix = np.column_stack(
            [
                np.asarray(weeks_since, dtype=np.float32),
                active.to_numpy(dtype=np.float32),
                signal_prev.to_numpy(dtype=np.float32),
                rolling_return_1w.to_numpy(dtype=np.float32),
                rolling_return_4w.to_numpy(dtype=np.float32),
                rolling_return_12w.to_numpy(dtype=np.float32),
                rolling_vol_4w.to_numpy(dtype=np.float32),
                rolling_vol_12w.to_numpy(dtype=np.float32),
                rolling_sharpe_12w.to_numpy(dtype=np.float32),
                rolling_maxdd_12w.to_numpy(dtype=np.float32),
                rolling_maxdd_26w.to_numpy(dtype=np.float32),
                hit_ratio_12w.to_numpy(dtype=np.float32),
                np.full(t_steps, trade_count, dtype=np.float32),
                np.full(t_steps, avg_trade_duration, dtype=np.float32),
                turnover.to_numpy(dtype=np.float32),
                corr_proxy.to_numpy(dtype=np.float32),
                beta_proxy.to_numpy(dtype=np.float32),
                np.full(t_steps, confidence_score, dtype=np.float32),
            ]
        )
        trader_features[:, j, :] = feature_matrix

    global_features[:, 0] = active_df.sum(axis=1).to_numpy(dtype=np.float32)
    global_features[:, 1] = float(n_traders)
    global_features[:, 2] = avg_cross_corr.to_numpy(dtype=np.float32)
    global_features[:, 3] = basket_ret_4w.to_numpy(dtype=np.float32)
    global_features[:, 4] = basket_vol_12w.to_numpy(dtype=np.float32)

    return PortfolioDataset(
        dates=dates,
        trader_ids=trader_ids,
        trader_features=trader_features,
        global_features=global_features,
        returns=returns_df.to_numpy(dtype=np.float32),
        active_mask=active_df.to_numpy(dtype=np.float32),
        trader_feature_names=trader_feature_names,
        global_feature_names=global_feature_names,
        trade_metadata=universe_info,
    )
