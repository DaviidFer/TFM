from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import warnings
import logging
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from queue import Queue
from time import sleep
from typing import Dict
from contextlib import redirect_stdout
from contextlib import redirect_stderr

import pandas as pd
from backtest_eventos.runner import run_event_backtest
from app.agents import AgentContext, DataAgent, DeveloperAgent, PortfolioManagerAgent, RiskAgent, TraderAgent, ValidationAgent
from app.contracts import PromotedTraderSpec
from app.core.structured_logging import LOG_FILE_PATH, emit_log
from app.execution.local_data_provider import LocalMarketDataProvider
from app.execution.models import ExecutionMode
from app.execution.mt5_connector import MT5Connector
from app.execution.mt5_data_provider import MT5DataProvider
from app.execution.router import ExecutionRouter
from app.runtime.live_trading_runtime import LiveTradingRuntime
from app.services.portfolio_rl import PortfolioOHLCRefreshService
from app.services.risk import build_design_risk_profile
from app.storage import StateStore

# Silencia warnings conocidos de sklearn que saturan la consola durante
# ciclos de desarrollo repetitivos sin aportar valor operativo.
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but DecisionTreeRegressor was fitted without feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*n_alphas.*deprecated.*",
    category=FutureWarning,
)


def _read_exec_mode() -> ExecutionMode:
    raw = str(os.getenv("EXECUTION_MODE", "paper")).strip().lower()
    return ExecutionMode.LIVE_MT5 if raw == "live_mt5" else ExecutionMode.PAPER


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DevelopmentOperationalSupervisor:
    _LOG_SEPARATOR = "═" * 54
    _REPORT_FORMAT_VERSION = 8

    """
    Supervisor no bloqueante para Streamlit:
    - hilo de desarrollo continuo (data->developer->validation->trader)
    - al llegar a 5 traders, conecta MT5 y activa runtime operativo D1
    """

    def __init__(self, *, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.report_format_version = self._REPORT_FORMAT_VERSION
        self.local_market_data = LocalMarketDataProvider()
        self.execution_mode = _read_exec_mode()
        self.mt5 = MT5Connector(env_path=".env")
        self.execution_router = ExecutionRouter(
            market_data=self.local_market_data,
            mode=self.execution_mode,
            mt5_connector=self.mt5,
        )
        self._shutdown = threading.Event()
        self._develop_enabled = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_lock = threading.Lock()
        self._status: Dict[str, object] = {
            "running": False,
            "develop_enabled": False,
            "current_asset": None,
            "current_stage": "idle",
            "current_cycle_steps": [],
            "developed_traders": 0,
            "target_traders": 8,
            "mt5_connected": False,
            "operational_runtime_started": False,
            "development_session_started_at": None,
            "last_cycle_completed_at": None,
            "last_cycle_asset": None,
            "last_cycle_trader_id": None,
            "portfolio_last_refresh_month": None,
            "portfolio_last_refresh_at": None,
            "portfolio_last_refresh_cutoff_date": None,
            "portfolio_last_refresh_status": None,
            "portfolio_last_refresh_traders": 0,
            "portfolio_last_refresh_mask_source": None,
            "portfolio_last_refresh_backtests_status": None,
            "portfolio_last_manual_retrain_at": None,
            "portfolio_last_manual_rebalance_at": None,
            "portfolio_last_manual_retrain_only_at": None,
            "portfolio_last_manual_retrain_and_rebalance_at": None,
            "risk_last_evaluation_at": None,
            "risk_last_evaluation_status": None,
            "risk_last_evaluation_run_id": None,
            "risk_last_evaluation_traders": 0,
            "risk_last_force_evaluation_at": None,
            "risk_last_retrain_processed_at": None,
        }
        self._promoted_registry: Dict[str, object] = {}
        self._backtest_registry: Dict[str, Dict[str, object]] = {}
        self._runtime: LiveTradingRuntime | None = None
        self._setup_context()

    def _setup_context(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ctx = AgentContext(
            store=StateStore(db_path=self.db_path),
            artifacts_root=self.db_path.parent,
            execution_router=self.execution_router,
        )
        self.data_agent = DataAgent(self.ctx)
        self.developer_agent = DeveloperAgent(self.ctx)
        self.validation_agent = ValidationAgent(self.ctx)
        self.trader_agent = TraderAgent(self.ctx)
        self.portfolio_manager_agent = PortfolioManagerAgent(self.ctx)
        self.risk_agent = RiskAgent(self.ctx, data_source=self.local_market_data)
        self.portfolio_refresh_service = PortfolioOHLCRefreshService(self.local_market_data)

    def get_status(self) -> Dict[str, object]:
        with self._status_lock:
            return dict(self._status)

    def get_backtest_registry(self) -> Dict[str, Dict[str, object]]:
        with self._status_lock:
            return {k: dict(v) for k, v in self._backtest_registry.items()}

    def get_pending_orders(self) -> list[Dict[str, object]]:
        if self._runtime is not None and hasattr(self._runtime, "get_pending_orders"):
            try:
                return list(self._runtime.get_pending_orders())
            except Exception:
                pass
        try:
            return list(self.ctx.store.list_pending_orders())
        except Exception:
            return []

    def get_portfolio_manager_snapshot(self) -> Dict[str, object]:
        signal_book: list[Dict[str, object]] = []
        last_output: Dict[str, object] | None = None
        if self._runtime is not None:
            if hasattr(self._runtime, "get_signal_book"):
                try:
                    signal_book = list(self._runtime.get_signal_book())
                except Exception:
                    signal_book = []
            if hasattr(self._runtime, "get_last_portfolio_output"):
                try:
                    last_output = self._runtime.get_last_portfolio_output()
                except Exception:
                    last_output = None
        latest_model = None
        training_runs: list[Dict[str, object]] = []
        training_metrics: list[Dict[str, object]] = []
        rebalance_rows: list[Dict[str, object]] = []
        forward_rows: list[Dict[str, object]] = []
        backtest_runs: list[Dict[str, object]] = []
        signal_audit_rows: list[Dict[str, object]] = []
        try:
            latest_model = self.ctx.store.get_latest_portfolio_model_info()
            training_runs = self.ctx.store.list_portfolio_training_runs(limit=20)
            if training_runs:
                training_metrics = self.ctx.store.list_portfolio_training_metrics(str(training_runs[0].get("run_id")))
            rebalance_rows = self.ctx.store.list_portfolio_rebalance_snapshots(limit=100)
            forward_rows = self.ctx.store.list_portfolio_forward_evaluations(limit=500)
            backtest_runs = self.ctx.store.list_trader_backtest_runs(limit=200)
            signal_audit_rows = self.ctx.store.list_trader_signal_audit(limit=5000)
        except Exception:
            pass
        return {
            "signal_book": signal_book,
            "signal_audit": signal_audit_rows,
            "last_output": last_output or {},
            "pending_orders": self.get_pending_orders(),
            "latest_model": latest_model or {},
            "training_runs": training_runs,
            "training_metrics": training_metrics,
            "rebalance_rows": rebalance_rows,
            "forward_rows": forward_rows,
            "backtest_runs": backtest_runs,
            "monthly_refresh": {
                "last_refresh_at": self.get_status().get("portfolio_last_refresh_at"),
                "cutoff_date": self.get_status().get("portfolio_last_refresh_cutoff_date"),
                "status": self.get_status().get("portfolio_last_refresh_status"),
                "n_traders": self.get_status().get("portfolio_last_refresh_traders"),
                "mask_source": self.get_status().get("portfolio_last_refresh_mask_source"),
                "backtests_status": self.get_status().get("portfolio_last_refresh_backtests_status"),
                "last_manual_retrain_at": self.get_status().get("portfolio_last_manual_retrain_at"),
                "last_manual_rebalance_at": self.get_status().get("portfolio_last_manual_rebalance_at"),
                "last_manual_retrain_only_at": self.get_status().get("portfolio_last_manual_retrain_only_at"),
                "last_manual_retrain_and_rebalance_at": self.get_status().get("portfolio_last_manual_retrain_and_rebalance_at"),
            },
        }

    def get_retrain_requests_snapshot(self) -> Dict[str, object]:
        try:
            all_requests = list(self.ctx.store.list_retrain_requests(limit=500))
        except Exception:
            all_requests = []
        pending = [row for row in all_requests if str(row.get("status")) == "pending"]
        return {"all_requests": all_requests, "pending_requests": pending}

    def get_risk_agent_snapshot(self) -> Dict[str, object]:
        runs: list[Dict[str, object]] = []
        latest_run: Dict[str, object] = {}
        details: list[Dict[str, object]] = []
        profiles: list[Dict[str, object]] = []
        forward_metrics: list[Dict[str, object]] = []
        forward_runs: list[Dict[str, object]] = []
        portfolio_checks: list[Dict[str, object]] = []
        retrain_requests: list[Dict[str, object]] = []
        try:
            runs = list(self.ctx.store.list_risk_evaluation_runs(limit=50))
            latest_run = runs[0] if runs else {}
            if latest_run:
                details = list(self.ctx.store.list_risk_evaluation_details(evaluation_run_id=str(latest_run.get("run_id")), limit=2000))
                forward_metrics = list(self.ctx.store.list_trader_forward_metrics(evaluation_run_id=str(latest_run.get("run_id")), limit=2000))
                forward_runs = list(self.ctx.store.list_trader_forward_backtest_runs(evaluation_run_id=str(latest_run.get("run_id")), limit=2000))
            profiles = list(self.ctx.store.list_trader_design_profiles(limit=2000))
            portfolio_checks = list(self.ctx.store.list_risk_portfolio_checks(limit=200))
            retrain_requests = list(self.ctx.store.list_retrain_requests(limit=500))
        except Exception:
            pass
        details_by_trader = {str(row.get("trader_id")): row for row in details}
        fwd_by_trader = {str(row.get("trader_id")): dict(row.get("metrics") or {}) for row in forward_metrics}
        profile_by_trader = {str(row.get("trader_id")): dict(row.get("profile") or {}) for row in profiles}
        run_by_trader = {str(row.get("trader_id")): row for row in forward_runs}
        trader_rows: list[Dict[str, object]] = []
        states = {str(row.trader_id): row for row in self.ctx.store.list_trader_states()}
        for trader_id in sorted(set(profile_by_trader) | set(fwd_by_trader) | set(details_by_trader)):
            state_row = states.get(trader_id)
            detail = details_by_trader.get(trader_id, {})
            profile = profile_by_trader.get(trader_id, {})
            metrics = fwd_by_trader.get(trader_id, {})
            trader_rows.append(
                {
                    "trader_id": trader_id,
                    "asset": str(profile.get("asset") or getattr(state_row, "asset", "")),
                    "timeframe": str(profile.get("timeframe") or getattr(state_row, "timeframe", "D1")),
                    "promoted_at": str(profile.get("promoted_at") or ""),
                    "current_state": str(getattr(state_row, "state", "") and state_row.state.value or detail.get("new_state") or ""),
                    "health_score": float(detail.get("health_score") or 0.0),
                    "action": str(detail.get("action") or ""),
                    "shadow_trades": int(metrics.get("shadow_trades") or 0),
                    "executed_trades": int(metrics.get("executed_trades") or 0),
                    "signal_count": int(metrics.get("signal_count") or 0),
                    "ppo_selected_count": int(metrics.get("ppo_selected_count") or 0),
                    "ppo_blocked_count": int(metrics.get("ppo_blocked_count") or 0),
                    "risk_blocked_count": int(metrics.get("risk_blocked_count") or 0),
                    "sharpe_design": profile.get("sharpe_design"),
                    "sharpe_forward": metrics.get("shadow_sharpe"),
                    "profit_factor_design": profile.get("profit_factor_design"),
                    "profit_factor_forward": metrics.get("shadow_profit_factor"),
                    "max_dd_design": profile.get("max_drawdown_design"),
                    "max_dd_forward": metrics.get("shadow_max_drawdown"),
                    "avg_loss_design": profile.get("avg_loss_design"),
                    "avg_loss_forward": metrics.get("shadow_avg_loss"),
                    "losing_streak_design": profile.get("max_losing_streak_design"),
                    "losing_streak_forward": metrics.get("shadow_losing_streak"),
                    "latest_evaluation": str(detail.get("created_at") or ""),
                    "main_reason": str((detail.get("reasons") or [""])[0] if detail.get("reasons") else ""),
                    "design_profile": profile,
                    "forward_metrics": metrics,
                    "forward_run": run_by_trader.get(trader_id, {}),
                    "risk_detail": detail,
                }
            )
        pending_retrain = [row for row in retrain_requests if str(row.get("status")) == "pending"]
        return {
            "latest_run": latest_run,
            "runs": runs,
            "details": details,
            "profiles": profiles,
            "forward_metrics": forward_metrics,
            "forward_runs": forward_runs,
            "portfolio_checks": portfolio_checks,
            "retrain_requests": retrain_requests,
            "pending_retrain_requests": pending_retrain,
            "trader_rows": trader_rows,
            "status": {
                "last_evaluation_at": self.get_status().get("risk_last_evaluation_at"),
                "last_evaluation_status": self.get_status().get("risk_last_evaluation_status"),
                "last_evaluation_run_id": self.get_status().get("risk_last_evaluation_run_id"),
                "last_evaluation_traders": self.get_status().get("risk_last_evaluation_traders"),
                "last_force_evaluation_at": self.get_status().get("risk_last_force_evaluation_at"),
                "last_retrain_processed_at": self.get_status().get("risk_last_retrain_processed_at"),
            },
        }

    def _set_status(self, **fields: object) -> None:
        with self._status_lock:
            self._status.update(fields)

    def _append_cycle_step(self, step: str) -> None:
        with self._status_lock:
            steps = list(self._status.get("current_cycle_steps", []))
            steps.append(step)
            self._status["current_cycle_steps"] = steps

    def _set_backtest_entry(self, trader_id: str, payload: Dict[str, object]) -> None:
        with self._status_lock:
            self._backtest_registry[trader_id] = payload

    @staticmethod
    def _hash_rules(promoted_spec: PromotedTraderSpec) -> str:
        payload = {
            "asset": str(promoted_spec.asset).upper(),
            "timeframe": str(promoted_spec.timeframe),
            "long_rules": list(promoted_spec.long_rules),
            "short_rules": list(promoted_spec.short_rules),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _price_data_fingerprint(csv_path: str) -> str:
        path = Path(csv_path)
        digest = hashlib.sha256()
        digest.update(path.name.encode("utf-8"))
        try:
            stat = path.stat()
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        except Exception:
            pass
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _backtest_artifact_dir(self, trader_id: str, run_id: str) -> Path:
        root = self.ctx.artifacts_root / "trader_backtests" / str(trader_id)
        root.mkdir(parents=True, exist_ok=True)
        out = root / str(run_id)
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not df.columns.duplicated().any():
            return df
        return df.loc[:, ~df.columns.duplicated()].copy()

    @classmethod
    def _normalize_pnl_frame(cls, pnl_df: pd.DataFrame) -> pd.DataFrame:
        if pnl_df.empty:
            return pd.DataFrame(columns=["date", "balance", "equity"])
        work = cls._drop_duplicate_columns(pnl_df.copy())
        if work.index.name and str(work.index.name) in {str(col) for col in work.columns}:
            work.index = work.index.rename("__index_date__")
        work = work.reset_index()
        work = cls._drop_duplicate_columns(work)
        date_col = str(work.columns[0])
        work = work.rename(columns={date_col: "date", "BALANCE": "balance", "EQUITY": "equity"})
        work = cls._drop_duplicate_columns(work)
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        if "balance" not in work.columns:
            work["balance"] = pd.NA
        if "equity" not in work.columns:
            work["equity"] = pd.NA
        work["balance"] = pd.to_numeric(work["balance"], errors="coerce")
        work["equity"] = pd.to_numeric(work["equity"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values("date")
        return work[["date", "balance", "equity"]].copy()

    @classmethod
    def _normalize_trades_frame(cls, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return pd.DataFrame(columns=["entry_time", "exit_time", "profit", "side"])
        work = cls._drop_duplicate_columns(trades_df.copy())
        norm_cols = {str(c).lower(): c for c in work.columns}
        entry_col = next((norm_cols[k] for k in ("time_entry", "entry_time", "open_time") if k in norm_cols), None)
        exit_col = next((norm_cols[k] for k in ("time_exit", "exit_time", "close_time") if k in norm_cols), None)
        profit_col = next((norm_cols[k] for k in ("pnl", "gross_profit", "profit") if k in norm_cols), None)
        side_col = next((norm_cols[k] for k in ("trade_type", "type", "side", "direction") if k in norm_cols), None)
        if entry_col is None:
            return pd.DataFrame(columns=["entry_time", "exit_time", "profit", "side"])
        rename_map = {entry_col: "entry_time"}
        if exit_col is not None:
            rename_map[exit_col] = "exit_time"
        if profit_col is not None:
            rename_map[profit_col] = "profit"
        if side_col is not None:
            rename_map[side_col] = "side"
        work = work.rename(columns=rename_map)
        work = cls._drop_duplicate_columns(work)
        work["entry_time"] = pd.to_datetime(work["entry_time"], errors="coerce")
        if "exit_time" in work.columns:
            work["exit_time"] = pd.to_datetime(work["exit_time"], errors="coerce")
        else:
            work["exit_time"] = pd.NaT
        if "profit" in work.columns:
            work["profit"] = pd.to_numeric(work["profit"], errors="coerce").fillna(0.0)
        else:
            work["profit"] = 0.0
        if "side" not in work.columns:
            work["side"] = ""
        work["side"] = work["side"].astype(str)
        work = work.dropna(subset=["entry_time"]).sort_values("entry_time")
        if "exit_time" in work.columns:
            work["exit_time"] = work["exit_time"].fillna(work["entry_time"])
        else:
            work["exit_time"] = work["entry_time"]
        return work[["entry_time", "exit_time", "profit", "side"]].copy()

    def _build_weekly_returns_rows(self, pnl_df: pd.DataFrame) -> list[Dict[str, object]]:
        work = self._normalize_pnl_frame(pnl_df)
        if work.empty:
            return []
        weekly = work.set_index("date").resample("W-FRI").last().ffill()
        weekly["weekly_return"] = (
            weekly["equity"].pct_change().replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        )
        rows: list[Dict[str, object]] = []
        for idx, row in weekly.iterrows():
            rows.append(
                {
                    "week_end": pd.Timestamp(idx).date().isoformat(),
                    "weekly_return": float(row.get("weekly_return") or 0.0),
                    "equity_close": float(row.get("equity") or 0.0),
                    "balance_close": float(row.get("balance") or 0.0),
                }
            )
        return rows

    def _build_weekly_signal_mask_rows(
        self,
        *,
        trades_df: pd.DataFrame,
        weekly_returns_rows: list[Dict[str, object]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[Dict[str, object]]:
        weekly_index = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="W-FRI")
        if len(weekly_index) == 0:
            weekly_index = pd.DatetimeIndex([pd.Timestamp(end_date).normalize()])
        trades = self._normalize_trades_frame(trades_df)
        returns_df = pd.DataFrame(weekly_returns_rows)
        if not returns_df.empty:
            returns_df["week_end"] = pd.to_datetime(returns_df["week_end"], errors="coerce")
        rows: list[Dict[str, object]] = []
        for week_end in weekly_index:
            week_start = week_end - pd.Timedelta(days=6)
            active_slice = trades[(trades["entry_time"] <= week_end) & (trades["exit_time"] >= week_start)] if not trades.empty else pd.DataFrame()
            pnl_week = 0.0
            if not returns_df.empty:
                hit = returns_df.loc[returns_df["week_end"] == week_end, "weekly_return"]
                if not hit.empty:
                    pnl_week = float(hit.iloc[-1])
            side = ""
            if not active_slice.empty and "side" in active_slice.columns:
                sides = sorted({str(v).lower() for v in active_slice["side"].astype(str).tolist() if str(v).strip()})
                if len(sides) == 1:
                    side = sides[0]
                elif len(sides) > 1:
                    side = "mixed"
            rows.append(
                {
                    "week_end": week_end.date().isoformat(),
                    "active": int(not active_slice.empty),
                    "side": side,
                    "bars_in_market": int(len(active_slice)),
                    "pnl_week": float(pnl_week),
                    "mask_source": "real_backtest",
                }
            )
        return rows

    def _persist_backtest_refresh_artifacts(
        self,
        *,
        promoted_spec: PromotedTraderSpec,
        run_id: str,
        csv_path: str,
        start_date: datetime,
        end_date: datetime,
        pnl_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        summary: Dict[str, object],
        refresh_reason: str,
    ) -> Dict[str, object]:
        trader_id = str(promoted_spec.trader_id)
        asset = str(promoted_spec.asset).upper()
        artifact_dir = self._backtest_artifact_dir(trader_id, run_id)
        pnl_norm = self._normalize_pnl_frame(pnl_df)
        trades_norm = self._normalize_trades_frame(trades_df)
        weekly_returns_rows = self._build_weekly_returns_rows(pnl_norm)
        weekly_mask_rows = self._build_weekly_signal_mask_rows(
            trades_df=trades_norm,
            weekly_returns_rows=weekly_returns_rows,
            start_date=start_date,
            end_date=end_date,
        )

        historical_pnl_path = artifact_dir / "historical_pnl.csv"
        historical_trades_path = artifact_dir / "historical_trades.csv"
        weekly_returns_path = artifact_dir / "weekly_returns.csv"
        weekly_mask_path = artifact_dir / "weekly_signal_mask.csv"
        pnl_norm.to_csv(historical_pnl_path, index=False)
        trades_norm.to_csv(historical_trades_path, index=False)
        pd.DataFrame(weekly_returns_rows).to_csv(weekly_returns_path, index=False)
        pd.DataFrame(weekly_mask_rows).to_csv(weekly_mask_path, index=False)

        cutoff_date = pd.Timestamp(end_date).date().isoformat()
        self.ctx.store.upsert_trader_backtest_run(
            run_id=run_id,
            trader_id=trader_id,
            asset=asset,
            timeframe=str(promoted_spec.timeframe),
            start_date=pd.Timestamp(start_date).date().isoformat(),
            end_date=pd.Timestamp(end_date).date().isoformat(),
            cutoff_date=cutoff_date,
            rules_hash=self._hash_rules(promoted_spec),
            price_data_fingerprint=self._price_data_fingerprint(csv_path),
            status="completed",
            summary=summary,
        )
        self.ctx.store.upsert_trader_backtest_artifacts(
            run_id=run_id,
            trader_id=trader_id,
            historical_trades_path=str(historical_trades_path),
            historical_pnl_path=str(historical_pnl_path),
            weekly_signal_mask_path=str(weekly_mask_path),
            weekly_returns_path=str(weekly_returns_path),
            metadata={
                "mask_source": "real_backtest",
                "refresh_reason": refresh_reason,
                "cutoff_date": cutoff_date,
            },
        )
        self.ctx.store.replace_trader_weekly_returns(run_id=run_id, trader_id=trader_id, rows=weekly_returns_rows)
        self.ctx.store.replace_trader_weekly_signal_mask(run_id=run_id, trader_id=trader_id, rows=weekly_mask_rows)
        return {
            "historical_trades_path": str(historical_trades_path),
            "historical_pnl_path": str(historical_pnl_path),
            "weekly_signal_mask_path": str(weekly_mask_path),
            "weekly_returns_path": str(weekly_returns_path),
            "mask_source": "real_backtest",
            "cutoff_date": cutoff_date,
        }

    def _run_backtest_for_promoted(self, promoted_spec, *, refresh_reason: str = "development_cycle") -> None:
        trader_id = str(promoted_spec.trader_id)
        asset = str(promoted_spec.asset).upper()
        csv_path = self.local_market_data.get_csv_path(asset)
        if not csv_path:
            self._set_backtest_entry(
                trader_id,
                {
                    "status": "error",
                    "asset": asset,
                    "error": f"No CSV encontrado para {asset}",
                    "updated_at": _utc_now_iso(),
                },
            )
            return

        self._set_backtest_entry(
            trader_id,
            {
                "status": "running",
                "asset": asset,
                "timeframe": str(promoted_spec.timeframe),
                "long_rules": list(promoted_spec.long_rules),
                "short_rules": list(promoted_spec.short_rules),
                "updated_at": _utc_now_iso(),
            },
        )

        try:
            initial_capital = 10000.0
            asset_df = pd.read_csv(csv_path)
            date_col = "Date" if "Date" in asset_df.columns else "date"
            if date_col not in asset_df.columns:
                raise ValueError(f"CSV sin columna Date/date: {csv_path}")
            date_series = pd.to_datetime(asset_df[date_col], errors="coerce").dropna()
            if date_series.empty:
                raise ValueError(f"CSV sin fechas validas: {csv_path}")
            start_date = date_series.min().to_pydatetime()
            end_date = date_series.max().to_pydatetime()

            sink = StringIO()
            pyeventbt_logger = logging.getLogger("pyeventbt")
            backtest_info_logger = logging.getLogger("backtest_info")
            prev_level_pyeventbt = pyeventbt_logger.level
            prev_level_backtest = backtest_info_logger.level
            prev_disable = logging.root.manager.disable
            with redirect_stdout(sink), redirect_stderr(sink):
                try:
                    logging.disable(logging.CRITICAL)
                    pyeventbt_logger.setLevel(logging.ERROR)
                    backtest_info_logger.setLevel(logging.ERROR)
                    bt = run_event_backtest(
                        csv_dir="app/.tmp/backtests_csv",
                        asset_csv_path=csv_path,
                        winners_long_stable=list(promoted_spec.long_rules),
                        winners_short_stable=list(promoted_spec.short_rules),
                        strategy_id=trader_id,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        systems_root_dir="app/.tmp/backtests_systems",
                        save_system_artifacts=False,
                        export_backtest_csv=False,
                        export_backtest_parquet=False,
                        verbose=False,
                    )
                finally:
                    logging.disable(prev_disable)
                    pyeventbt_logger.setLevel(prev_level_pyeventbt)
                    backtest_info_logger.setLevel(prev_level_backtest)

            pnl_df = bt.pnl.copy() if hasattr(bt, "pnl") else pd.DataFrame()
            chart_rows: list[Dict[str, object]] = []
            final_balance = None
            final_equity = None
            n_trades = int(len(bt.trades)) if hasattr(bt, "trades") else 0
            trade_stats: Dict[str, object] = {}
            trades_df = bt.trades.copy() if hasattr(bt, "trades") and isinstance(bt.trades, pd.DataFrame) else pd.DataFrame()
            if not trades_df.empty:
                norm_cols = {str(c).lower(): c for c in trades_df.columns}
                profit_col = None
                for name in ("pnl", "gross_profit", "profit"):
                    if name in norm_cols:
                        profit_col = norm_cols[name]
                        break
                if profit_col is not None:
                    profits = pd.to_numeric(trades_df[profit_col], errors="coerce").dropna()
                    if not profits.empty:
                        wins = profits[profits > 0]
                        losses = profits[profits < 0]
                        total = int(len(profits))
                        trade_stats["total_trades"] = total
                        trade_stats["winning_trades"] = int((profits > 0).sum())
                        trade_stats["losing_trades"] = int((profits < 0).sum())
                        trade_stats["win_rate_pct"] = float((trade_stats["winning_trades"] / total) * 100.0) if total else None
                        trade_stats["avg_win"] = float(wins.mean()) if not wins.empty else 0.0
                        trade_stats["avg_loss"] = float(losses.mean()) if not losses.empty else 0.0
                        trade_stats["profit_factor"] = (
                            float(wins.sum() / abs(losses.sum()))
                            if (not wins.empty and not losses.empty and float(losses.sum()) != 0.0)
                            else None
                        )
                        trade_stats["payoff_ratio"] = (
                            float(wins.mean() / abs(losses.mean()))
                            if (not wins.empty and not losses.empty and abs(float(losses.mean())) > 0.0)
                            else None
                        )
                        trade_stats["expectancy"] = float(profits.mean())
                        trade_stats["max_win_trade"] = float(profits.max())
                        trade_stats["max_loss_trade"] = float(profits.min())
                        streak_win = 0
                        streak_lose = 0
                        max_streak_win = 0
                        max_streak_lose = 0
                        for p in profits.tolist():
                            if p > 0:
                                streak_win += 1
                                streak_lose = 0
                            elif p < 0:
                                streak_lose += 1
                                streak_win = 0
                            else:
                                streak_win = 0
                                streak_lose = 0
                            max_streak_win = max(max_streak_win, streak_win)
                            max_streak_lose = max(max_streak_lose, streak_lose)
                        trade_stats["max_winning_streak"] = int(max_streak_win)
                        trade_stats["max_losing_streak"] = int(max_streak_lose)

                entry_col = next((norm_cols[k] for k in ("time_entry", "entry_time", "open_time") if k in norm_cols), None)
                exit_col = next((norm_cols[k] for k in ("time_exit", "exit_time", "close_time") if k in norm_cols), None)
                if entry_col and exit_col:
                    entry_ts = pd.to_datetime(trades_df[entry_col], errors="coerce")
                    exit_ts = pd.to_datetime(trades_df[exit_col], errors="coerce")
                    dur_days = (exit_ts - entry_ts).dt.total_seconds() / 86400.0
                    dur_days = pd.to_numeric(dur_days, errors="coerce").dropna()
                    if not dur_days.empty:
                        trade_stats["avg_trade_duration_days"] = float(dur_days.mean())
                        trade_stats["min_trade_duration_days"] = float(dur_days.min())
                        trade_stats["max_trade_duration_days"] = float(dur_days.max())
            if not pnl_df.empty:
                pnl_norm = self._normalize_pnl_frame(pnl_df)
                if not pnl_norm.empty:
                    final_balance = float(pnl_norm["balance"].iloc[-1]) if "balance" in pnl_norm.columns else None
                    final_equity = float(pnl_norm["equity"].iloc[-1]) if "equity" in pnl_norm.columns else None
                for _, row in pnl_norm.iterrows():
                    chart_rows.append(
                        {
                            "date": str(row["date"]),
                            "balance": float(row["balance"]) if "balance" in pnl_norm.columns else None,
                            "equity": float(row["equity"]) if "equity" in pnl_norm.columns else None,
                        }
                    )

            run_id = f"bt_{trader_id}_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}"
            summary = {
                "asset": asset,
                "timeframe": str(promoted_spec.timeframe),
                "trade_stats": trade_stats,
                "n_trades": int(n_trades),
                "initial_capital": float(initial_capital),
                "final_balance": final_balance,
                "final_equity": final_equity,
                "chart_rows_count": len(chart_rows),
            }
            artifacts = self._persist_backtest_refresh_artifacts(
                promoted_spec=promoted_spec,
                run_id=run_id,
                csv_path=csv_path,
                start_date=start_date,
                end_date=end_date,
                pnl_df=pnl_df,
                trades_df=trades_df,
                summary=summary,
                refresh_reason=refresh_reason,
            )
            try:
                design_profile = build_design_risk_profile(
                    promoted_spec=promoted_spec,
                    validation_report=None,
                    backtest_summary={
                        **summary,
                        "start_date": start_date.date().isoformat(),
                        "end_date": end_date.date().isoformat(),
                    },
                    backtest_artifacts=artifacts,
                )
                self.ctx.store.upsert_trader_design_profile(
                    trader_id=str(design_profile.trader_id),
                    asset=str(design_profile.asset),
                    timeframe=str(design_profile.timeframe),
                    promoted_at=str(design_profile.promoted_at),
                    profile=design_profile.to_dict(),
                )
            except Exception:
                pass

            self._set_backtest_entry(
                trader_id,
                {
                    "status": "ready",
                    "asset": asset,
                    "timeframe": str(promoted_spec.timeframe),
                    "long_rules": list(promoted_spec.long_rules),
                    "short_rules": list(promoted_spec.short_rules),
                    "start_date": start_date.date().isoformat(),
                    "end_date": end_date.date().isoformat(),
                    "final_balance": final_balance,
                    "final_equity": final_equity,
                    "n_trades": n_trades,
                    "initial_capital": initial_capital,
                    "trade_stats": trade_stats,
                    "chart_rows": chart_rows,
                    "run_id": run_id,
                    "cutoff_date": artifacts["cutoff_date"],
                    "mask_source": artifacts["mask_source"],
                    "historical_pnl_path": artifacts["historical_pnl_path"],
                    "historical_trades_path": artifacts["historical_trades_path"],
                    "weekly_signal_mask_path": artifacts["weekly_signal_mask_path"],
                    "weekly_returns_path": artifacts["weekly_returns_path"],
                    "updated_at": _utc_now_iso(),
                },
            )
        except Exception as exc:
            self._set_backtest_entry(
                trader_id,
                {
                    "status": "error",
                    "asset": asset,
                    "timeframe": str(promoted_spec.timeframe),
                    "long_rules": list(promoted_spec.long_rules),
                    "short_rules": list(promoted_spec.short_rules),
                    "error": str(exc),
                    "updated_at": _utc_now_iso(),
                },
            )

    def set_target_traders(self, target_traders: int) -> None:
        target = max(1, int(target_traders))
        self._set_status(target_traders=target)
        emit_log("supervisor", "target_traders_updated", console=False, target_traders=target)

    @staticmethod
    def _fmt_map_lines(mapping: Dict[str, object], *, indent: int = 8) -> list[str]:
        prefix = " " * indent
        lines: list[str] = []
        for key, value in mapping.items():
            lines.append(f"{prefix}- {key}: {value}")
        return lines

    @staticmethod
    def _pretty_model_name(name: str) -> str:
        mapping = {
            "quantile": "Quantiles",
            "rulefit": "RuleFit",
            "subgroup": "Subgroup",
            "genetico": "Genético",
            "decision_tree": "Decision Tree",
        }
        return mapping.get(str(name or "").strip().lower(), str(name or ""))

    @staticmethod
    def _pretty_param_name(name: str) -> str:
        mapping = {
            "n_bins": "n_bins",
            "combo_size": "combo_size",
            "min_coverage": "min_coverage",
            "target_n_rules": "target_n_rules",
            "n_estimators": "n_estimators",
            "max_candidate_rules": "max_candidate_rules",
            "progress_every": "progress_every",
            "population_size": "population_size",
            "n_generations": "n_generations",
            "n_monkeys": "n_monkeys",
            "is_pass_pct": "is_pass_pct",
            "oos_pass_pct": "oos_pass_pct",
            "min_coverage_is": "min_coverage_is",
            "min_coverage_oos": "min_coverage_oos",
            "corr_threshold": "corr_threshold",
            "min_ops": "min_ops",
            "target_year": "target_year",
            "top_n_long": "top_n_long",
            "top_n_short": "top_n_short",
        }
        return mapping.get(str(name or "").strip(), str(name or ""))

    @classmethod
    def _fmt_human_map_lines(cls, mapping: Dict[str, object], *, indent: int = 8) -> list[str]:
        prefix = " " * indent
        lines: list[str] = []
        for key, value in mapping.items():
            lines.append(f"{prefix}- {cls._pretty_param_name(str(key))}: {value}")
        return lines

    @staticmethod
    def _pretty_trader_name(trader_id: str | None, *, asset: str, timeframe: str = "D1") -> str:
        if asset:
            return f"{str(asset).upper()}_{str(timeframe).upper()}"
        txt = str(trader_id or "").strip()
        if txt.startswith("tr_"):
            parts = txt.split("_")
            if len(parts) >= 4:
                return f"{parts[1].upper()}_{parts[2].upper()}"
        return txt or "PENDIENTE"

    def _print_cycle_report(
        self,
        *,
        asset: str,
        dataset,
        split_ranges: Dict[str, Dict[str, str | None]] | None,
        chosen: str,
        strategy: Dict[str, object],
        generated_long: int,
        generated_short: int,
        passed_long: int,
        passed_short: int,
        trader_id: str | None,
        backtest_status: str | None = None,
    ) -> None:
        split = strategy["split"]
        val_cfg = strategy["validation"]
        model_params = strategy["params"].get(chosen, {})
        split_ranges = split_ranges or {}
        is_range = dict(split_ranges.get("data_is", {}) or {})
        oos_range = dict(split_ranges.get("data_oos", {}) or {})

        print(self._LOG_SEPARATOR)
        print("      INICIO CREACION AGENTE TRADER")
        print(self._LOG_SEPARATOR)
        print(f"ACTIVO: {asset}")
        print("")
        print("[1] Seleccion del activo")
        print(f"    • Activo: {asset}")
        print(f"    • Rango temporal: {dataset.start_date} -> {dataset.end_date}")
        print("")
        print("[2] Generacion de reglas")
        print("    • Split:")
        print(f"        - IS: {split['is_pct']}")
        print(f"        - Rango IS: {is_range.get('start') or '-'} -> {is_range.get('end') or '-'}")
        print(f"    • Modelo: {self._pretty_model_name(chosen)}")
        print("    • Parametros:")
        for line in self._fmt_human_map_lines(model_params):
            print(line)
        print("    • Reglas generadas:")
        print(f"        - LONG : {generated_long}")
        print(f"        - SHORT: {generated_short}")
        print("")
        print("[3] Validacion")
        print("    • Split:")
        print(f"        - OOS: {split['oos_pct']}")
        print(f"        - Rango OOS: {oos_range.get('start') or '-'} -> {oos_range.get('end') or '-'}")
        print("")
        print("    • Monkey IS:")
        for line in self._fmt_map_lines(val_cfg["monkey_is"]):
            print(line)
        print("")
        print("    • Monkey OOS:")
        for line in self._fmt_map_lines(val_cfg["monkey_oos"]):
            print(line)
        print("")
        print("    • Correlacion:")
        for line in self._fmt_map_lines(val_cfg["correlation_pruning"]):
            print(line)
        print("")
        print("    • Forward:")
        for line in self._fmt_map_lines(val_cfg["forward_validation"]):
            print(line)
        print("")
        print("    • Stability:")
        for line in self._fmt_map_lines(val_cfg["stability_selection"]):
            print(line)
        print("")
        print("[4] Resultado")
        print(f"    • Reglas LONG validadas : {passed_long}")
        print(f"    • Reglas SHORT validadas: {passed_short}")
        print("")
        print("[5] Creacion del agente")
        if trader_id:
            print(f"    • Trader creado: {self._pretty_trader_name(trader_id, asset=asset, timeframe='D1')}")
            print(f"    • Activo: {asset}")
            print("    • Estado: LISTO PARA OPERAR")
            print("")
            print("[6] Backtest")
            if backtest_status == "ready":
                print("    • Backtest realizado")
            elif backtest_status == "running":
                print("    • Backtest en ejecucion")
            elif backtest_status == "error":
                print("    • Backtest realizado con error")
            else:
                print("    • Backtest pendiente")
        else:
            print("    • Trader creado: NO")
            print(f"    • Activo: {asset}")
            print("    • Estado: NO CREADO (sin reglas validadas)")
            print("    • Accion: repetir proceso con otro activo/parametros")
        print(self._LOG_SEPARATOR)

    def _build_strategy(self, asset: str) -> Dict[str, object]:
        families = ("decision_tree", "rulefit", "genetico", "quantile", "subgroup")
        chosen = random.choice(families)
        if chosen == "decision_tree":
            params = {"decision_tree": {"target_n_rules": random.randint(16, 45), "progress_every": 0}}
        elif chosen == "rulefit":
            params = {
                "rulefit": {
                    "target_n_rules": random.randint(16, 45),
                    "n_estimators": random.randint(20, 55),
                    "max_candidate_rules": random.randint(120, 260),
                    "progress_every": 0,
                }
            }
        elif chosen == "genetico":
            params = {
                "genetico": {
                    "target_n_rules": random.randint(16, 45),
                    "population_size": random.randint(24, 60),
                    "n_generations": random.randint(6, 14),
                    "progress_every": 0,
                }
            }
        elif chosen == "quantile":
            params = {"quantile": {"n_bins": random.randint(3, 6), "combo_size": 1, "min_coverage": random.randint(80, 220)}}
        else:
            params = {"subgroup": {"n_bins": random.randint(4, 7), "min_coverage": random.randint(60, 180)}}

        is_pct = round(random.uniform(0.55, 0.75), 2)
        split = {
            "is_pct": is_pct,
            "oos_pct": round(1.0 - is_pct, 2),
            "holdout_year": 2025,
            "lookback_years": 10,
        }
        validation = {
            "split_assumption": {"holdout_year": 2025},
            "monkey_is": {
                "n_monkeys": random.randint(90, 160),
                "is_pass_pct": float(random.randint(85, 94)),
                "min_coverage_is": random.randint(65, 90),
                "n_jobs": 1,
            },
            "monkey_oos": {
                "n_monkeys": random.randint(90, 160),
                "oos_pass_pct": float(random.randint(70, 82)),
                "min_coverage_oos": random.randint(50, 75),
                "n_jobs": 1,
            },
            "correlation_pruning": {
                "corr_threshold": round(random.uniform(0.45, 0.6), 2),
                "min_ops": random.randint(30, 70),
                "diagnose": False,
            },
            "forward_validation": {"target_year": 2025, "min_ops": random.randint(20, 45), "verbose": False},
            "stability_selection": {
                "top_n_long": random.randint(10, 20),
                "top_n_short": random.randint(10, 20),
                "min_ops": random.randint(30, 70),
                "verbose": False,
            },
        }
        return {"chosen_family": chosen, "params": params, "split": split, "validation": validation}

    def _run_development_cycle(self) -> None:
        symbols = list(self.local_market_data.asset_csv_by_symbol.keys())
        if not symbols:
            raise RuntimeError("No hay activos disponibles en carpeta datos para desarrollo.")
        asset = random.choice(symbols)
        csv_path = self.local_market_data.get_csv_path(asset)
        if not csv_path:
            return
        strategy = self._build_strategy(asset)
        chosen = str(strategy["chosen_family"])
        self._set_status(
            current_asset=asset,
            current_stage="data_agent",
            current_cycle_steps=["data_agent"],
        )
        sink = StringIO()
        with redirect_stdout(sink):
            dataset = self.data_agent.prepare_dataset(asset=asset, timeframe="D1", asset_csv_path=csv_path)
            self._set_status(current_stage="developer_agent")
            self._append_cycle_step("developer_agent")
            dev = self.developer_agent.develop(
                dataset=dataset,
                families=(chosen,),
                family_params=strategy["params"],
                split_config=strategy["split"],
            )
            self._set_status(current_stage="validation_agent")
            self._append_cycle_step("validation_agent")
            val = self.validation_agent.validate_and_promote(
                dev,
                validation_profile=strategy["validation"],
                promote_if_empty=False,
            )

        generated_long = len(dev.candidate_rules.long_rules)
        generated_short = len(dev.candidate_rules.short_rules)
        passed_long = int(val.report.passed_long)
        passed_short = int(val.report.passed_short)

        trader_id: str | None = None
        backtest_status: str | None = None
        if val.promoted_spec is not None and (passed_long + passed_short) > 0:
            self._set_status(current_stage="trader_agent")
            self._append_cycle_step("trader_agent")
            with redirect_stdout(sink):
                self.trader_agent.activate(val.promoted_spec)
            self._promoted_registry[val.promoted_spec.trader_id] = val.promoted_spec
            trader_id = val.promoted_spec.trader_id
            self._set_status(current_stage="backtest_agent")
            self._append_cycle_step("backtest_agent")
            self._run_backtest_for_promoted(val.promoted_spec)
            backtest_status = str(self.get_backtest_registry().get(trader_id, {}).get("status", "pending"))
            if self._runtime is not None:
                self._runtime.upsert_trader(val.promoted_spec)
                self.mt5.ensure_symbols_in_marketwatch([val.promoted_spec.asset])

        self._print_cycle_report(
            asset=asset,
            dataset=dataset,
            split_ranges={k: {"start": (v.index.min().strftime("%Y-%m-%d") if len(v) else None), "end": (v.index.max().strftime("%Y-%m-%d") if len(v) else None)} for k, v in dev.blocks.items()},
            chosen=chosen,
            strategy=strategy,
            generated_long=generated_long,
            generated_short=generated_short,
            passed_long=passed_long,
            passed_short=passed_short,
            trader_id=trader_id,
            backtest_status=backtest_status,
        )
        self._set_status(
            current_stage="idle",
            current_cycle_steps=[],
            developed_traders=len(self._promoted_registry),
            last_cycle_completed_at=_utc_now_iso(),
            last_cycle_asset=asset,
            last_cycle_trader_id=trader_id,
        )

        # Si el objetivo se ha alcanzado en este mismo ciclo, desactiva desarrollo de inmediato.
        current = len(self._promoted_registry)
        target = int(self.get_status().get("target_traders", 8))
        if current >= target:
            self._develop_enabled.clear()
            self._set_status(develop_enabled=False, current_stage="idle")
            emit_log(
                "supervisor",
                "target_traders_reached",
                console=False,
                target_traders=target,
                developed_traders=current,
            )
            self._bootstrap_runtime_after_target_reached()

    def _load_historical_promoted_specs(self) -> Dict[str, PromotedTraderSpec]:
        """
        Recupera traders promovidos desde la DB para incluirlos en operativa
        aunque no pertenezcan al ciclo actual de desarrollo.
        """
        out: Dict[str, PromotedTraderSpec] = {}
        try:
            promoted_events = self.ctx.store.list_events_by_type("trader_promoted", limit=50000)
        except Exception:
            return out
        for e in promoted_events:
            payload = e.get("payload", {}) or {}
            trader_id = str(payload.get("trader_id") or "").strip()
            asset = str(payload.get("asset") or "").strip().upper()
            if not trader_id or not asset:
                continue
            timeframe = str(payload.get("timeframe") or "D1")
            long_rules = list(payload.get("long_rules", []) or [])
            short_rules = list(payload.get("short_rules", []) or [])
            origin_experiment_id = str(payload.get("origin_experiment_id") or e.get("correlation_id") or f"legacy_{trader_id}")
            promoted_at = str(payload.get("promoted_at") or e.get("occurred_at") or _utc_now_iso())
            try:
                out[trader_id] = PromotedTraderSpec(
                    trader_id=trader_id,
                    asset=asset,
                    timeframe=timeframe,
                    long_rules=long_rules,
                    short_rules=short_rules,
                    origin_experiment_id=origin_experiment_id,
                    promoted_at=promoted_at,
                    metadata=dict(payload.get("metadata", {}) or {}),
                )
            except Exception:
                continue
        return out

    def _build_operational_registry(self) -> Dict[str, PromotedTraderSpec]:
        merged = self._load_historical_promoted_specs()
        for trader_id, spec in self._promoted_registry.items():
            merged[trader_id] = spec
        return merged

    def get_all_promoted_specs(self) -> Dict[str, PromotedTraderSpec]:
        return self._build_operational_registry()

    def _portfolio_total_capital(self) -> float:
        try:
            info = self.mt5.account_info()
            balance = float(info.get("balance") or 0.0)
            if balance > 0:
                return balance
        except Exception:
            pass
        try:
            return float(os.getenv("PORTFOLIO_TOTAL_CAPITAL", "100000"))
        except Exception:
            return 100000.0

    def _portfolio_universe_ready(self) -> bool:
        status = self.get_status()
        try:
            developed = int(status.get("developed_traders") or 0)
            target = int(status.get("target_traders") or 0)
        except Exception:
            developed = 0
            target = 0
        if target <= 0:
            return True
        return developed >= target and not bool(status.get("develop_enabled"))

    def _bootstrap_runtime_after_target_reached(self) -> None:
        started = self._ensure_operational_runtime(force_start=True)
        if started and self._runtime is not None and hasattr(self._runtime, "bootstrap_now"):
            try:
                self._runtime.bootstrap_now()
            except Exception:
                pass

    def get_trader_history_frame(self, trader_id: str) -> pd.DataFrame | None:
        latest_run = self.ctx.store.get_latest_trader_backtest_run(str(trader_id))
        if latest_run is not None:
            artifacts = self.ctx.store.get_trader_backtest_artifacts(str(latest_run["run_id"]))
            path = str((artifacts or {}).get("historical_pnl_path") or "")
            if path and Path(path).exists():
                try:
                    df = pd.read_csv(path)
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df = df.dropna(subset=["date"]).sort_values("date")
                        keep_cols = ["date"] + [c for c in ("equity", "balance") if c in df.columns]
                        if len(keep_cols) > 1:
                            return df[keep_cols].copy()
                except Exception:
                    pass
        bt = self.get_backtest_registry().get(str(trader_id), {})
        if not bt or str(bt.get("status")) != "ready":
            spec = self.get_all_promoted_specs().get(str(trader_id))
            if spec is None:
                return None
            self._run_backtest_for_promoted(spec)
            bt = self.get_backtest_registry().get(str(trader_id), {})
        chart_rows = bt.get("chart_rows", []) or []
        if not chart_rows:
            return None
        df = pd.DataFrame(chart_rows)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            return None
        keep_cols = ["date"] + [c for c in ("equity", "balance") if c in df.columns]
        if len(keep_cols) == 1:
            return None
        return df[keep_cols].copy()

    @staticmethod
    def _month_refresh_token(as_of: str | None = None) -> str:
        ts = pd.Timestamp(as_of or pd.Timestamp.utcnow().isoformat())
        return f"{ts.year:04d}-{ts.month:02d}"

    def run_risk_monthly_evaluation(self, *, force: bool = False, as_of: str | None = None, force_backtest: bool = False) -> Dict[str, object]:
        if not self.risk_agent.should_run_monthly_evaluation(as_of=pd.Timestamp(as_of).to_pydatetime() if as_of else None, force=force):
            return {"status": "skipped", "reason": "not_due"}
        snapshots = self.risk_agent.evaluate_trader_universe(
            evaluation_date=pd.Timestamp(as_of).to_pydatetime() if as_of else None,
            run_type="forced" if force else "scheduled_monthly",
            force_backtest=force_backtest,
        )
        latest_run = self.ctx.store.list_risk_evaluation_runs(limit=1)
        latest = latest_run[0] if latest_run else {}
        self._set_status(
            risk_last_evaluation_at=_utc_now_iso(),
            risk_last_evaluation_status=str(latest.get("status") or "completed"),
            risk_last_evaluation_run_id=str(latest.get("run_id") or ""),
            risk_last_evaluation_traders=len(snapshots),
        )
        return {"status": str(latest.get("status") or "completed"), "run": latest, "snapshots": [snap.to_dict() for snap in snapshots]}

    def force_risk_evaluation(self, *, force_backtest: bool = True) -> Dict[str, object]:
        out = self.run_risk_monthly_evaluation(force=True, as_of=pd.Timestamp.utcnow().isoformat(), force_backtest=force_backtest)
        self._set_status(risk_last_force_evaluation_at=_utc_now_iso())
        return out

    def process_pending_retrain_requests(self) -> Dict[str, object]:
        requests = list(self.ctx.store.list_pending_retrain_requests())
        processed: list[Dict[str, object]] = []
        failed: list[Dict[str, object]] = []
        for request in requests:
            request_id = str(request.get("request_id") or "")
            asset = str(request.get("asset") or "").upper()
            timeframe = str(request.get("timeframe") or "D1")
            try:
                self.ctx.store.mark_retrain_request_running(request_id)
                csv_path = self.local_market_data.get_csv_path(asset)
                if not csv_path:
                    raise FileNotFoundError(f"No CSV encontrado para {asset}")
                strategy = self._build_strategy(asset)
                chosen = str(strategy["chosen_family"])
                dataset = self.data_agent.prepare_dataset(asset=asset, timeframe=timeframe, asset_csv_path=csv_path)
                dev = self.developer_agent.develop(
                    dataset=dataset,
                    families=(chosen,),
                    family_params=strategy["params"],
                    split_config=strategy["split"],
                )
                val = self.validation_agent.validate_and_promote(
                    dev,
                    validation_profile=strategy["validation"],
                    promote_if_empty=False,
                )
                if val.promoted_spec is None:
                    raise RuntimeError("No se pudo promover un nuevo trader durante el retraining")
                self.trader_agent.activate(val.promoted_spec)
                self._promoted_registry[val.promoted_spec.trader_id] = val.promoted_spec
                self._run_backtest_for_promoted(val.promoted_spec, refresh_reason="risk_retrain_request")
                if self._runtime is not None:
                    self._runtime.upsert_trader(val.promoted_spec)
                self.ctx.store.mark_retrain_request_completed(
                    request_id,
                    payload={"new_trader_id": str(val.promoted_spec.trader_id), "asset": asset, "timeframe": timeframe},
                )
                self.ctx.store.append_event(
                    event_id=f"evt_{uuid4().hex[:10]}",
                    event_type=EventType.RETRAIN_PROCESSED,
                    producer="supervisor",
                    payload={"request_id": request_id, "status": "completed", "new_trader_id": str(val.promoted_spec.trader_id)},
                    correlation_id=request_id,
                )
                processed.append({"request_id": request_id, "new_trader_id": str(val.promoted_spec.trader_id)})
            except Exception as exc:
                self.ctx.store.mark_retrain_request_failed(request_id, str(exc))
                self.ctx.store.append_event(
                    event_id=f"evt_{uuid4().hex[:10]}",
                    event_type=EventType.RETRAIN_PROCESSED,
                    producer="supervisor",
                    payload={"request_id": request_id, "status": "failed", "error": str(exc)},
                    correlation_id=request_id,
                )
                failed.append({"request_id": request_id, "error": str(exc)})
        if requests:
            self._set_status(risk_last_retrain_processed_at=_utc_now_iso())
        return {"processed": processed, "failed": failed, "pending_before": len(requests)}

    def _should_run_portfolio_monthly_refresh(self, *, as_of: str | None = None, force: bool = False) -> bool:
        if force:
            return True
        if self._runtime is None:
            return False
        promoted_specs = self.get_all_promoted_specs()
        if not promoted_specs:
            return False
        now = pd.Timestamp(as_of or pd.Timestamp.utcnow().isoformat())
        if int(now.day) > 3:
            return False
        last_token = str(self.get_status().get("portfolio_last_refresh_status") or "")
        if last_token.startswith("running"):
            return False
        return str(self.get_status().get("portfolio_last_refresh_month") or "") != self._month_refresh_token(as_of)

    def run_portfolio_monthly_refresh(self, *, force: bool = False, as_of: str | None = None) -> Dict[str, object]:
        promoted_specs = self.get_all_promoted_specs()
        if not promoted_specs:
            return {"status": "skipped", "reason": "no_promoted_traders"}

        month_token = self._month_refresh_token(as_of)
        self._set_status(
            portfolio_last_refresh_status=f"running:{month_token}",
            portfolio_last_refresh_backtests_status="running",
            portfolio_last_refresh_traders=len(promoted_specs),
        )
        emit_log("supervisor", "portfolio_monthly_refresh_started", console=False, month_token=month_token, n_traders=len(promoted_specs))
        try:
            symbols = sorted({str(spec.asset).upper() for spec in promoted_specs.values()})
            refresh_result = self.portfolio_refresh_service.refresh(symbols)
            refreshed_rows: list[Dict[str, object]] = []
            failures: list[Dict[str, object]] = []
            for spec in promoted_specs.values():
                try:
                    self._run_backtest_for_promoted(spec, refresh_reason="monthly_refresh")
                    row = dict(self.get_backtest_registry().get(str(spec.trader_id), {}) or {})
                    row["trader_id"] = str(spec.trader_id)
                    refreshed_rows.append(row)
                except Exception as exc:
                    failures.append({"trader_id": str(spec.trader_id), "error": str(exc)})

            risk_out = self.run_risk_monthly_evaluation(force=force, as_of=str(as_of or refresh_result.cutoff_date), force_backtest=True)
            retrain_out = self.process_pending_retrain_requests()
            self.portfolio_manager_agent.sync_universe(promoted_specs)
            model_info = self.portfolio_manager_agent.run_monthly_refresh_and_fine_tune(
                history_loader=self.get_trader_history_frame,
                as_of=str(as_of or refresh_result.cutoff_date),
                force=force,
            )
            dataset_refresh = {}
            if getattr(self.portfolio_manager_agent, "_latest_dataset", None) is not None:
                dataset_refresh = dict((self.portfolio_manager_agent._latest_dataset.trade_metadata.get("dataset_refresh") or {}))
            overall_status = "completed" if not failures else "completed_with_errors"
            self._set_status(
                portfolio_last_refresh_month=month_token,
                portfolio_last_refresh_at=_utc_now_iso(),
                portfolio_last_refresh_cutoff_date=str(refresh_result.cutoff_date),
                portfolio_last_refresh_status=overall_status,
                portfolio_last_refresh_traders=len(refreshed_rows),
                portfolio_last_refresh_mask_source=str(dataset_refresh.get("mask_source") or "unknown"),
                portfolio_last_refresh_backtests_status="ok" if not failures else "partial_error",
            )
            emit_log(
                "supervisor",
                "portfolio_monthly_refresh_completed",
                console=False,
                month_token=month_token,
                cutoff_date=str(refresh_result.cutoff_date),
                n_traders=len(refreshed_rows),
                failures=failures,
                mask_source=str(dataset_refresh.get("mask_source") or "unknown"),
                model_version=str((model_info or {}).get("model_version") or ""),
            )
            return {
                "status": overall_status,
                "month_token": month_token,
                "cutoff_date": str(refresh_result.cutoff_date),
                "refreshed_traders": refreshed_rows,
                "failures": failures,
                "refresh_result": {
                    "status": refresh_result.status,
                    "n_requested_symbols": refresh_result.n_requested_symbols,
                    "n_refreshed_symbols": refresh_result.n_refreshed_symbols,
                    "refreshed_symbols": refresh_result.refreshed_symbols,
                },
                "risk_evaluation": risk_out,
                "retrain_processing": retrain_out,
                "mask_source": str(dataset_refresh.get("mask_source") or "unknown"),
                "dataset_refresh": dataset_refresh,
                "model_info": model_info or {},
            }
        except Exception as exc:
            self._set_status(
                portfolio_last_refresh_month=month_token,
                portfolio_last_refresh_at=_utc_now_iso(),
                portfolio_last_refresh_status="error",
                portfolio_last_refresh_backtests_status="error",
            )
            emit_log("supervisor", "portfolio_monthly_refresh_error", console=False, month_token=month_token, error=str(exc))
            raise

    def force_portfolio_retraining_only(self) -> Dict[str, object]:
        now_iso = pd.Timestamp.utcnow().isoformat()
        refresh_out = self.run_portfolio_monthly_refresh(force=True, as_of=now_iso)
        self._set_status(
            portfolio_last_manual_retrain_at=_utc_now_iso(),
            portfolio_last_manual_retrain_only_at=_utc_now_iso(),
        )
        emit_log(
            "supervisor",
            "portfolio_manual_retrain_only",
            console=False,
            refresh_status=str(refresh_out.get("status") or ""),
            model_version=str((refresh_out.get("model_info") or {}).get("model_version") or ""),
        )
        return {
            "refresh": refresh_out,
            "rebalance": {"status": "not_requested", "reason": "manual_retrain_only"},
        }

    def force_portfolio_retraining_and_rebalance(self) -> Dict[str, object]:
        now_iso = pd.Timestamp.utcnow().isoformat()
        refresh_out = self.run_portfolio_monthly_refresh(force=True, as_of=now_iso)
        self._set_status(
            portfolio_last_manual_retrain_at=_utc_now_iso(),
            portfolio_last_manual_retrain_and_rebalance_at=_utc_now_iso(),
        )
        rebalance_out: Dict[str, object] = {}
        if self._runtime is not None and hasattr(self._runtime, "force_rebalance_now"):
            rebalance_out = dict(self._runtime.force_rebalance_now(reason="manual_ui_retrain") or {})
            self._set_status(portfolio_last_manual_rebalance_at=_utc_now_iso())
        else:
            rebalance_out = {"status": "runtime_not_started", "reason": "operational_runtime_required"}
        emit_log(
            "supervisor",
            "portfolio_manual_retrain_and_rebalance",
            console=False,
            refresh_status=str(refresh_out.get("status") or ""),
            rebalance_status=str(rebalance_out.get("status") or ""),
            model_version=str((refresh_out.get("model_info") or {}).get("model_version") or ""),
        )
        return {
            "refresh": refresh_out,
            "rebalance": rebalance_out,
        }

    def ensure_mt5_execution_ready(self, *, symbols: list[str] | None = None) -> Dict[str, object]:
        try:
            if not self.mt5.connected and not self.mt5.connect():
                self._set_status(mt5_connected=False)
                return {"connected": False, "reason": "mt5_connect_failed", "mode": self.execution_router.mode.value}
            self.execution_router.mode = ExecutionMode.LIVE_MT5
            self._set_status(mt5_connected=True)
            if symbols:
                clean_symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
                if clean_symbols:
                    self.mt5.ensure_symbols_in_marketwatch(clean_symbols)
            return {"connected": True, "reason": "mt5_ready", "mode": self.execution_router.mode.value}
        except Exception as exc:
            self._set_status(mt5_connected=False)
            return {"connected": False, "reason": str(exc), "mode": self.execution_router.mode.value}

    def _ensure_operational_runtime(self, *, force_start: bool = False) -> bool:
        if self._runtime is not None:
            return True
        if (len(self._promoted_registry) < 5) and (not force_start):
            return False
        operational_specs = self._build_operational_registry()
        if len(operational_specs) == 0:
            return False
        emit_log(
            "supervisor",
            "minimum_traders_reached",
            console=False,
            developed_traders=len(self._promoted_registry),
        )
        symbols = sorted({spec.asset for spec in operational_specs.values()})
        ready = self.ensure_mt5_execution_ready(symbols=symbols)
        if not bool(ready.get("connected")):
            emit_log("supervisor", "mt5_connect_failed_after_threshold")
            return False
        queue = Queue()
        provider = MT5DataProvider(events_queue=queue, symbol_list=symbols, timeframe="1d")
        self._runtime = LiveTradingRuntime(
            trader_agent=self.trader_agent,
            portfolio_manager=self.portfolio_manager_agent,
            risk_agent=self.risk_agent,
            promoted_specs=operational_specs,
            data_provider=provider,
            history_loader=self.get_trader_history_frame,
            capital_provider=self._portfolio_total_capital,
            universe_ready_provider=self._portfolio_universe_ready,
            timeframe="1d",
            bars_lookback=260,
        )
        self._set_status(operational_runtime_started=True)
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
        print(f"{ts} - STARTUP_SNAPSHOT total={len(operational_specs)} attributed={len(operational_specs)} unattributed=0")
        print("=" * 60)
        print("INICIANDO TRADING CON SISTEMAS BASADOS EN REGLAS")
        uniq_assets = sorted({spec.asset for spec in operational_specs.values()})
        print(f"Sistemas configurados: {len(operational_specs)} | activos: {len(uniq_assets)}")
        for spec in operational_specs.values():
            print(f"  - {spec.asset} {spec.timeframe}")
        print("=" * 60)
        emit_log(
            "supervisor",
            "operational_runtime_started",
            console=False,
            symbols=symbols,
            timeframe="1d",
        )
        return True

    def start_operational_runtime(self) -> Dict[str, object]:
        if self._thread is None or not self._thread.is_alive():
            self._shutdown.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        if self._runtime is not None:
            # Reinicio explícito para reconstruir símbolos/traders con el estado actual completo.
            self._runtime.stop()
            self._runtime = None
            self._set_status(operational_runtime_started=False)
        started = self._ensure_operational_runtime(force_start=True)
        if started and self._runtime is not None and hasattr(self._runtime, "bootstrap_now"):
            try:
                self._runtime.bootstrap_now()
            except Exception:
                pass
        if started:
            return {"started": True, "reason": "runtime_started", "n_traders": len(self._build_operational_registry())}
        return {"started": False, "reason": "runtime_not_started"}

    def _loop(self) -> None:
        self._set_status(running=True)
        emit_log("supervisor", "thread_started", console=False, execution_mode=self.execution_mode.value)
        while not self._shutdown.is_set():
            try:
                if self._develop_enabled.is_set():
                    current = int(self._status.get("developed_traders", 0))
                    target = int(self._status.get("target_traders", 8))
                    if current < target:
                        self._run_development_cycle()
                    else:
                        self._develop_enabled.clear()
                        self._set_status(develop_enabled=False, current_stage="idle")
                        emit_log(
                            "supervisor",
                            "target_traders_reached",
                            console=False,
                            target_traders=target,
                            developed_traders=current,
                        )
                        self._bootstrap_runtime_after_target_reached()
                self._ensure_operational_runtime()
                if self._should_run_portfolio_monthly_refresh():
                    self.run_portfolio_monthly_refresh()
                if self._runtime is not None:
                    self._runtime.poll_once()
                sleep(1)
            except Exception as exc:
                self._set_status(current_stage="error")
                emit_log("supervisor", "loop_error", console=False, error=str(exc))
                sleep(2)
        self._set_status(running=False, current_stage="stopped")
        emit_log("supervisor", "thread_stopped", console=False)

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._shutdown.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        self._develop_enabled.set()
        self._set_status(
            develop_enabled=True,
            development_session_started_at=_utc_now_iso(),
            current_stage="idle",
            current_asset=None,
            current_cycle_steps=[],
        )
        emit_log("supervisor", "development_enabled", console=False)

    def stop_development(self) -> None:
        self._develop_enabled.clear()
        self._set_status(
            develop_enabled=False,
            current_stage="idle",
            current_asset=None,
            current_cycle_steps=[],
        )
        emit_log("supervisor", "development_disabled", console=False)

    def reset_all(self) -> None:
        self._develop_enabled.clear()
        self._shutdown.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.mt5.shutdown()
        self._promoted_registry = {}
        self._runtime = None
        # Limpieza robusta: primero vacia tablas; luego intenta borrar fichero.
        try:
            self.ctx.store.clear_all()
        except Exception:
            pass
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except Exception:
                pass
        if LOG_FILE_PATH.exists():
            try:
                LOG_FILE_PATH.unlink()
            except Exception:
                pass
        self._setup_context()
        self._shutdown.clear()
        self._thread = None
        self._set_status(
            running=False,
            develop_enabled=False,
            current_asset=None,
            current_stage="idle",
            current_cycle_steps=[],
            developed_traders=0,
            target_traders=int(self._status.get("target_traders", 8)),
            mt5_connected=False,
            operational_runtime_started=False,
            development_session_started_at=None,
            last_cycle_completed_at=None,
            last_cycle_asset=None,
            last_cycle_trader_id=None,
            portfolio_last_refresh_month=None,
            portfolio_last_refresh_at=None,
            portfolio_last_refresh_cutoff_date=None,
            portfolio_last_refresh_status=None,
            portfolio_last_refresh_traders=0,
            portfolio_last_refresh_mask_source=None,
            portfolio_last_refresh_backtests_status=None,
            portfolio_last_manual_retrain_at=None,
            portfolio_last_manual_rebalance_at=None,
            portfolio_last_manual_retrain_only_at=None,
            portfolio_last_manual_retrain_and_rebalance_at=None,
            risk_last_evaluation_at=None,
            risk_last_evaluation_status=None,
            risk_last_evaluation_run_id=None,
            risk_last_evaluation_traders=0,
            risk_last_force_evaluation_at=None,
            risk_last_retrain_processed_at=None,
        )
        with self._status_lock:
            self._backtest_registry = {}
        emit_log("supervisor", "reset_completed", console=False, db_path=str(self.db_path))

