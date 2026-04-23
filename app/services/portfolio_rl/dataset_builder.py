from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from app.contracts import PromotedTraderSpec
from app.storage.state_store import StateStore

from .config import PPOPortfolioConfig
from .feature_builder import PortfolioDataset, build_weekly_feature_dataset


@dataclass(frozen=True)
class PortfolioUniverseMember:
    trader_id: str
    asset: str
    timeframe: str
    promotion_date: str
    lifecycle_state: str
    metadata: Dict[str, Any]


class PortfolioDatasetBuilder:
    def __init__(self, config: PPOPortfolioConfig, store: StateStore | None = None) -> None:
        self.config = config
        self.store = store

    @staticmethod
    def _history_to_weekly_equity(history: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(history, pd.Series):
            df = history.to_frame(name="value")
        else:
            df = history.copy()
        if df.empty:
            return pd.Series(dtype="float64")
        cols_lower = {str(c).lower(): c for c in df.columns}
        if "date" in cols_lower:
            date_col = cols_lower["date"]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()
        value_col = None
        for candidate in ("equity", "balance", "value"):
            if candidate in cols_lower:
                value_col = cols_lower[candidate]
                break
        if value_col is None:
            value_col = df.columns[0]
        series = pd.to_numeric(df[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return pd.Series(dtype="float64")
        weekly = series.resample("W-FRI").last().ffill()
        return weekly.dropna()

    @staticmethod
    def _equity_to_weekly_returns(equity: pd.Series) -> pd.Series:
        if equity.empty:
            return pd.Series(dtype="float64")
        returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return returns.astype("float64")

    @staticmethod
    def _active_from_weekly_returns(returns: pd.Series) -> pd.Series:
        if returns.empty:
            return pd.Series(dtype="float64")
        active = returns.abs().rolling(2, min_periods=1).max().gt(1e-12).astype(float)
        return active

    @staticmethod
    def _load_weekly_returns_frame(path: str | Path) -> pd.Series:
        df = pd.read_csv(path)
        if df.empty:
            return pd.Series(dtype="float64")
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        df = df.dropna(subset=["week_end"]).sort_values("week_end")
        return pd.Series(
            pd.to_numeric(df["weekly_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            index=pd.DatetimeIndex(df["week_end"]),
            dtype="float64",
        )

    @staticmethod
    def _load_weekly_mask_frame(path: str | Path) -> pd.Series:
        df = pd.read_csv(path)
        if df.empty:
            return pd.Series(dtype="float64")
        df["week_end"] = pd.to_datetime(df["week_end"], errors="coerce")
        df = df.dropna(subset=["week_end"]).sort_values("week_end")
        return pd.Series(
            pd.to_numeric(df["active"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float),
            index=pd.DatetimeIndex(df["week_end"]),
            dtype="float64",
        )

    def _load_refresh_artifacts(self, trader_id: str) -> tuple[pd.Series | None, pd.Series | None, Dict[str, Any]]:
        if self.store is None:
            return None, None, {}
        latest_run = self.store.get_latest_trader_backtest_run(trader_id)
        if latest_run is None:
            return None, None, {}
        artifacts = self.store.get_trader_backtest_artifacts(str(latest_run["run_id"]))
        if artifacts is None:
            return None, None, {"latest_run": latest_run}
        weekly_returns_path = str(artifacts.get("weekly_returns_path") or "")
        weekly_mask_path = str(artifacts.get("weekly_signal_mask_path") or "")
        if not weekly_returns_path or not weekly_mask_path:
            return None, None, {"latest_run": latest_run, "artifacts": artifacts}
        returns_path = Path(weekly_returns_path)
        mask_path = Path(weekly_mask_path)
        if not returns_path.exists() or not mask_path.exists():
            return None, None, {"latest_run": latest_run, "artifacts": artifacts}
        weekly_returns = self._load_weekly_returns_frame(returns_path)
        active_mask = self._load_weekly_mask_frame(mask_path)
        metadata = {
            "latest_run": latest_run,
            "artifacts": artifacts,
            "mask_source": str((artifacts.get("metadata") or {}).get("mask_source") or "real_backtest"),
        }
        return weekly_returns, active_mask, metadata

    def build_universe_info(
        self,
        promoted_specs: Mapping[str, PromotedTraderSpec],
        backtest_registry: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        info: Dict[str, Dict[str, Any]] = {}
        for trader_id, spec in promoted_specs.items():
            bt = dict((backtest_registry or {}).get(trader_id) or {})
            latest_run = self.store.get_latest_trader_backtest_run(trader_id) if self.store is not None else None
            if latest_run is not None:
                bt = {**dict(latest_run.get("summary") or {}), **bt}
            trade_stats = dict(bt.get("trade_stats") or {})
            metadata = dict(spec.metadata or {})
            info[trader_id] = {
                "asset": spec.asset,
                "timeframe": spec.timeframe,
                "promotion_date": spec.promoted_at,
                "lifecycle_state": str(spec.lifecycle_state.value),
                "trade_count": float(trade_stats.get("total_trades") or bt.get("n_trades") or 0.0),
                "avg_trade_duration_days": float(trade_stats.get("avg_trade_duration_days") or 0.0),
                "win_rate_pct": float(trade_stats.get("win_rate_pct") or 0.0),
                "confidence_score": float(metadata.get("confidence_score") or metadata.get("robustness_score") or 0.0),
                "metadata": metadata,
            }
        return info

    def build_dataset(
        self,
        *,
        promoted_specs: Mapping[str, PromotedTraderSpec],
        history_loader: Callable[[str], pd.DataFrame | pd.Series | None] | None,
        backtest_registry: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> PortfolioDataset:
        weekly_returns_map: Dict[str, pd.Series] = {}
        active_mask_map: Dict[str, pd.Series] = {}
        universe_info = self.build_universe_info(promoted_specs, backtest_registry=backtest_registry)
        mask_source_map: Dict[str, str] = {}
        cutoff_dates: Dict[str, str] = {}

        for trader_id in sorted(promoted_specs.keys()):
            real_returns, real_mask, refresh_meta = self._load_refresh_artifacts(trader_id)
            if real_returns is not None and len(real_returns) >= int(self.config.min_history_weeks):
                weekly_returns_map[trader_id] = real_returns
                mask_series = real_mask if real_mask is not None else pd.Series(dtype="float64")
                active_mask_map[trader_id] = mask_series.reindex(real_returns.index).fillna(0.0)
                mask_source_map[trader_id] = str(refresh_meta.get("mask_source") or "real_backtest")
                cutoff_dates[trader_id] = str(((refresh_meta.get("latest_run") or {}).get("cutoff_date")) or "")
                continue

            history = history_loader(trader_id) if history_loader is not None else None
            if history is None:
                continue
            equity = self._history_to_weekly_equity(history)
            if len(equity) < int(self.config.min_history_weeks):
                continue
            returns = self._equity_to_weekly_returns(equity)
            weekly_returns_map[trader_id] = returns
            active_mask_map[trader_id] = self._active_from_weekly_returns(returns)
            mask_source_map[trader_id] = "fallback_proxy"
            cutoff_dates[trader_id] = str(returns.index.max().date().isoformat()) if len(returns.index) else ""

        if not weekly_returns_map:
            raise ValueError("No se ha podido construir dataset semanal para PPO.")

        weekly_returns = pd.concat(weekly_returns_map.values(), axis=1, join="outer").sort_index().fillna(0.0)
        weekly_returns.columns = list(weekly_returns_map.keys())
        active_mask = pd.concat(active_mask_map.values(), axis=1, join="outer").sort_index().fillna(0.0)
        active_mask.columns = list(active_mask_map.keys())
        active_mask = active_mask.reindex(index=weekly_returns.index, columns=weekly_returns.columns).fillna(0.0)
        dataset = build_weekly_feature_dataset(
            weekly_returns=weekly_returns,
            active_mask=active_mask,
            universe_info={tid: universe_info.get(tid, {}) for tid in weekly_returns.columns},
        )
        dataset.trade_metadata["dataset_refresh"] = {
            "mask_source_by_trader": mask_source_map,
            "cutoff_date_by_trader": cutoff_dates,
            "mask_source": "fallback_proxy" if any(v == "fallback_proxy" for v in mask_source_map.values()) else "real_backtest",
            "cutoff_date": max([v for v in cutoff_dates.values() if v], default=""),
        }
        return dataset

    def temporal_splits(self, dataset: PortfolioDataset) -> Dict[str, slice]:
        n = dataset.n_steps
        if n < 10:
            return {"train": slice(0, max(1, n - 2)), "val": slice(max(1, n - 2), max(1, n - 1)), "test": slice(max(1, n - 1), n)}
        train_end = max(2, int(n * self.config.train_split))
        val_end = min(n - 1, train_end + max(1, int(n * self.config.val_split)))
        return {
            "train": slice(0, train_end),
            "val": slice(train_end, val_end),
            "test": slice(val_end, n),
        }
