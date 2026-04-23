from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping
from uuid import uuid4

import pandas as pd

from backtest_eventos.runner import run_event_backtest
from app.contracts import DesignRiskProfile, PromotedTraderSpec, TraderForwardMetrics
from app.core.structured_logging import emit_log
from app.services.risk.risk_metrics import (
    compute_avg_loss,
    compute_avg_win,
    compute_expectancy,
    compute_losing_streak,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    compute_trade_frequency,
    compute_winrate,
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _normalize_pnl_frame(pnl_df: pd.DataFrame) -> pd.DataFrame:
    if pnl_df is None or pnl_df.empty:
        return pd.DataFrame(columns=["date", "balance", "equity"])
    work = pnl_df.copy().reset_index()
    date_col = str(work.columns[0])
    work = work.rename(columns={date_col: "date", "BALANCE": "balance", "EQUITY": "equity"})
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in ("balance", "equity"):
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.dropna(subset=["date"]).sort_values("date")[["date", "balance", "equity"]]


def _normalize_trades_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "profit", "side"])
    work = trades_df.copy()
    norm = {str(c).lower(): c for c in work.columns}
    entry_col = next((norm[k] for k in ("time_entry", "entry_time", "open_time") if k in norm), None)
    exit_col = next((norm[k] for k in ("time_exit", "exit_time", "close_time") if k in norm), None)
    profit_col = next((norm[k] for k in ("pnl", "gross_profit", "profit") if k in norm), None)
    side_col = next((norm[k] for k in ("trade_type", "type", "side", "direction") if k in norm), None)
    if entry_col is None:
        return pd.DataFrame(columns=["entry_time", "exit_time", "profit", "side"])
    rename_map = {entry_col: "entry_time"}
    if exit_col:
        rename_map[exit_col] = "exit_time"
    if profit_col:
        rename_map[profit_col] = "profit"
    if side_col:
        rename_map[side_col] = "side"
    work = work.rename(columns=rename_map)
    work["entry_time"] = pd.to_datetime(work["entry_time"], errors="coerce")
    work["exit_time"] = pd.to_datetime(work.get("exit_time"), errors="coerce").fillna(work["entry_time"])
    work["profit"] = pd.to_numeric(work.get("profit"), errors="coerce").fillna(0.0)
    work["side"] = work.get("side", "").astype(str) if "side" in work.columns else ""
    return work.dropna(subset=["entry_time"]).sort_values("entry_time")[["entry_time", "exit_time", "profit", "side"]]


def _compute_profile_metrics(
    *,
    pnl_df: pd.DataFrame | None,
    trades_df: pd.DataFrame | None,
    initial_capital: float | None = None,
) -> Dict[str, Any]:
    pnl_norm = _normalize_pnl_frame(pnl_df if pnl_df is not None else pd.DataFrame())
    trades_norm = _normalize_trades_frame(trades_df if trades_df is not None else pd.DataFrame())
    returns = pd.Series(dtype="float64")
    equity_series = pd.Series(dtype="float64")
    if not pnl_norm.empty:
        value_col = "equity" if pnl_norm["equity"].notna().any() else "balance"
        equity_series = pd.to_numeric(pnl_norm[value_col], errors="coerce").dropna()
        returns = equity_series.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    net_pnl = 0.0
    if not equity_series.empty:
        base = float(initial_capital or equity_series.iloc[0] or 0.0)
        net_pnl = float(equity_series.iloc[-1] - base)
    elif not trades_norm.empty:
        net_pnl = float(trades_norm["profit"].sum())
    return {
        "trades": int(len(trades_norm)),
        "return": (float(net_pnl / float(initial_capital)) if initial_capital and initial_capital != 0 else None),
        "sharpe": compute_sharpe(returns, annualization=52.0),
        "profit_factor": compute_profit_factor(trades_norm),
        "max_drawdown": compute_max_drawdown(equity_series),
        "avg_loss": compute_avg_loss(trades_norm),
        "avg_win": compute_avg_win(trades_norm),
        "winrate": compute_winrate(trades_norm),
        "expectancy": compute_expectancy(trades_norm),
        "max_losing_streak": compute_losing_streak(trades_norm),
        "monthly_trade_frequency": compute_trade_frequency(trades_norm),
        "returns_mean": _safe_float(returns.mean()) if not returns.empty else None,
        "returns_std": _safe_float(returns.std(ddof=0)) if not returns.empty else None,
    }


def build_design_risk_profile(
    promoted_spec: PromotedTraderSpec,
    validation_report: Mapping[str, Any] | None,
    backtest_summary: Mapping[str, Any] | None,
    backtest_artifacts: Mapping[str, Any] | None,
) -> DesignRiskProfile:
    summary = dict(backtest_summary or {})
    artifacts = dict(backtest_artifacts or {})
    metadata = {
        "validation_report_present": bool(validation_report),
        "artifact_paths": artifacts,
        "missing_fields": [],
    }
    pnl_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    try:
        pnl_path = str(artifacts.get("historical_pnl_path") or "")
        if pnl_path and Path(pnl_path).exists():
            pnl_df = pd.read_csv(pnl_path)
    except Exception:
        metadata["missing_fields"].append("historical_pnl_path")
    try:
        trades_path = str(artifacts.get("historical_trades_path") or "")
        if trades_path and Path(trades_path).exists():
            trades_df = pd.read_csv(trades_path)
    except Exception:
        metadata["missing_fields"].append("historical_trades_path")

    initial_capital = _safe_float(summary.get("initial_capital"))
    metrics = _compute_profile_metrics(pnl_df=pnl_df, trades_df=trades_df, initial_capital=initial_capital)
    trade_stats = dict(summary.get("trade_stats") or {})

    return DesignRiskProfile(
        trader_id=str(promoted_spec.trader_id),
        asset=str(promoted_spec.asset).upper(),
        timeframe=str(promoted_spec.timeframe).upper(),
        promoted_at=str(promoted_spec.promoted_at),
        design_start=str(summary.get("start_date") or "") or None,
        design_end=str(summary.get("end_date") or "") or None,
        sharpe_design=_safe_float(metrics.get("sharpe")),
        profit_factor_design=_safe_float(trade_stats.get("profit_factor", metrics.get("profit_factor"))),
        max_drawdown_design=_safe_float(metrics.get("max_drawdown")),
        avg_loss_design=_safe_float(trade_stats.get("avg_loss", metrics.get("avg_loss"))),
        avg_win_design=_safe_float(trade_stats.get("avg_win", metrics.get("avg_win"))),
        winrate_design=_safe_float(trade_stats.get("win_rate_pct")) / 100.0 if trade_stats.get("win_rate_pct") is not None else _safe_float(metrics.get("winrate")),
        expectancy_design=_safe_float(trade_stats.get("expectancy", metrics.get("expectancy"))),
        max_losing_streak_design=_safe_int(trade_stats.get("max_losing_streak", metrics.get("max_losing_streak"))),
        trades_design=_safe_int(summary.get("n_trades", metrics.get("trades"))),
        monthly_trade_frequency_design=_safe_float(metrics.get("monthly_trade_frequency")),
        returns_mean_design=_safe_float(metrics.get("returns_mean")),
        returns_std_design=_safe_float(metrics.get("returns_std")),
        metadata=metadata,
    )


class ForwardBacktestService:
    def __init__(self, *, store: Any, artifacts_root: Path) -> None:
        self.store = store
        self.artifacts_root = Path(artifacts_root)

    def _resolve_csv_path(self, promoted_spec: PromotedTraderSpec, data_source: Any) -> str:
        if data_source is None:
            raise ValueError("data_source_not_available")
        if hasattr(data_source, "get_csv_path"):
            path = data_source.get_csv_path(str(promoted_spec.asset).upper())
            if path:
                return str(path)
        if callable(data_source):
            path = data_source(str(promoted_spec.asset).upper())
            if path:
                return str(path)
        raise FileNotFoundError(f"No CSV encontrado para {promoted_spec.asset}")

    def _executed_metrics(self, trader_id: str, evaluation_date: str) -> Dict[str, Any]:
        audits = list(self.store.list_trader_signal_audit(trader_id=trader_id, limit=5000))
        if not audits:
            return {}
        eval_ts = pd.Timestamp(evaluation_date)
        filtered = [row for row in audits if pd.Timestamp(row.get("timestamp")) <= eval_ts]
        if not filtered:
            return {}
        executed = [row for row in filtered if bool(row.get("executed"))]
        selected = [row for row in filtered if bool(row.get("ppo_selected"))]
        blocked = [row for row in filtered if not bool(row.get("risk_approved"))]
        executed_returns = pd.Series([float(r.get("executed_return") or 0.0) for r in executed], dtype="float64")
        return {
            "executed_trades": int(len(executed)),
            "executed_pnl": float(executed_returns.sum()) if not executed_returns.empty else 0.0,
            "executed_return": float(executed_returns.sum()) if not executed_returns.empty else 0.0,
            "executed_sharpe": compute_sharpe(executed_returns, annualization=52.0) if not executed_returns.empty else None,
            "executed_profit_factor": compute_profit_factor(executed_returns) if not executed_returns.empty else None,
            "executed_max_drawdown": compute_max_drawdown((1.0 + executed_returns).cumprod()) if not executed_returns.empty else None,
            "ppo_selected_count": int(len(selected)),
            "risk_blocked_count": int(len(blocked)),
        }

    def run_forward_backtest_for_trader(
        self,
        trader_id: str,
        promoted_spec: PromotedTraderSpec,
        design_profile: DesignRiskProfile,
        evaluation_date: str,
        data_source: Any,
        artifacts_root: Path | None = None,
        evaluation_run_id: str | None = None,
    ) -> TraderForwardMetrics:
        eval_ts = pd.Timestamp(evaluation_date)
        promoted_ts = pd.Timestamp(promoted_spec.promoted_at)
        csv_path = self._resolve_csv_path(promoted_spec, data_source)
        raw_df = pd.read_csv(csv_path)
        date_col = "Date" if "Date" in raw_df.columns else "date"
        raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
        raw_df = raw_df.dropna(subset=[date_col]).sort_values(date_col)
        if raw_df.empty:
            raise ValueError(f"CSV sin fechas válidas para {promoted_spec.asset}")
        last_date = min(pd.Timestamp(raw_df[date_col].max()), eval_ts)
        forward_start = promoted_ts.normalize()
        evaluation_run_id = str(evaluation_run_id or f"risk_{uuid4().hex[:10]}")
        run_id = f"fw_{trader_id}_{uuid4().hex[:8]}"
        output_root = Path(artifacts_root or self.artifacts_root) / "risk_forward" / trader_id / run_id
        output_root.mkdir(parents=True, exist_ok=True)

        if last_date <= forward_start or raw_df.loc[raw_df[date_col] >= forward_start].shape[0] < 15:
            metrics = TraderForwardMetrics(
                trader_id=trader_id,
                asset=str(promoted_spec.asset).upper(),
                timeframe=str(promoted_spec.timeframe).upper(),
                evaluation_run_id=evaluation_run_id,
                promoted_at=str(promoted_spec.promoted_at),
                evaluation_date=str(eval_ts.isoformat()),
                forward_start=str(forward_start.date().isoformat()),
                forward_end=str(last_date.date().isoformat()),
                insufficient_evidence=True,
                metadata={"reason": "insufficient_forward_history"},
            )
            self.store.save_trader_forward_backtest_run(
                run_id=run_id,
                evaluation_run_id=evaluation_run_id,
                trader_id=trader_id,
                asset=str(promoted_spec.asset).upper(),
                timeframe=str(promoted_spec.timeframe).upper(),
                promoted_at=str(promoted_spec.promoted_at),
                forward_start=str(forward_start.date().isoformat()),
                forward_end=str(last_date.date().isoformat()),
                status="insufficient_evidence",
                artifact_paths={},
                metrics=metrics.to_dict(),
            )
            self.store.save_trader_forward_metrics(
                evaluation_run_id=evaluation_run_id,
                trader_id=trader_id,
                asset=str(promoted_spec.asset).upper(),
                timeframe=str(promoted_spec.timeframe).upper(),
                evaluation_date=str(eval_ts.isoformat()),
                metrics=metrics.to_dict(),
            )
            return metrics

        bt = run_event_backtest(
            csv_dir="app/.tmp/backtests_csv",
            asset_csv_path=csv_path,
            winners_long_stable=list(promoted_spec.long_rules),
            winners_short_stable=list(promoted_spec.short_rules),
            strategy_id=f"{trader_id}_forward",
            start_date=forward_start.to_pydatetime(),
            end_date=last_date.to_pydatetime(),
            initial_capital=float(10000.0),
            systems_root_dir="app/.tmp/backtests_systems",
            save_system_artifacts=False,
            export_backtest_csv=False,
            export_backtest_parquet=False,
            verbose=False,
        )
        pnl_df = bt.pnl.copy() if hasattr(bt, "pnl") else pd.DataFrame()
        trades_df = bt.trades.copy() if hasattr(bt, "trades") and isinstance(bt.trades, pd.DataFrame) else pd.DataFrame()
        pnl_norm = _normalize_pnl_frame(pnl_df)
        trades_norm = _normalize_trades_frame(trades_df)
        historical_pnl_path = output_root / "forward_pnl.csv"
        historical_trades_path = output_root / "forward_trades.csv"
        pnl_norm.to_csv(historical_pnl_path, index=False)
        trades_norm.to_csv(historical_trades_path, index=False)

        returns = pd.Series(dtype="float64")
        shadow_return = 0.0
        if not pnl_norm.empty:
            series_name = "equity" if pnl_norm["equity"].notna().any() else "balance"
            equity = pd.to_numeric(pnl_norm[series_name], errors="coerce").dropna()
            if not equity.empty:
                returns = equity.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
                if float(equity.iloc[0]) != 0:
                    shadow_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)

        executed = self._executed_metrics(trader_id, str(eval_ts.isoformat()))
        metrics = TraderForwardMetrics(
            trader_id=trader_id,
            asset=str(promoted_spec.asset).upper(),
            timeframe=str(promoted_spec.timeframe).upper(),
            evaluation_run_id=evaluation_run_id,
            promoted_at=str(promoted_spec.promoted_at),
            evaluation_date=str(eval_ts.isoformat()),
            forward_start=str(forward_start.date().isoformat()),
            forward_end=str(last_date.date().isoformat()),
            shadow_trades=int(len(trades_norm)),
            executed_trades=int(executed.get("executed_trades") or 0),
            shadow_pnl=float(trades_norm["profit"].sum()) if not trades_norm.empty else 0.0,
            executed_pnl=float(executed.get("executed_pnl") or 0.0),
            shadow_return=float(shadow_return),
            executed_return=_safe_float(executed.get("executed_return")),
            shadow_sharpe=compute_sharpe(returns, annualization=52.0),
            executed_sharpe=_safe_float(executed.get("executed_sharpe")),
            shadow_profit_factor=compute_profit_factor(trades_norm),
            executed_profit_factor=_safe_float(executed.get("executed_profit_factor")),
            shadow_max_drawdown=compute_max_drawdown(pnl_norm["equity"] if "equity" in pnl_norm.columns else pnl_norm["balance"]),
            executed_max_drawdown=_safe_float(executed.get("executed_max_drawdown")),
            shadow_avg_loss=compute_avg_loss(trades_norm),
            shadow_avg_win=compute_avg_win(trades_norm),
            shadow_winrate=compute_winrate(trades_norm),
            shadow_expectancy=compute_expectancy(trades_norm),
            shadow_losing_streak=compute_losing_streak(trades_norm),
            signal_count=int(len(trades_norm)),
            ppo_selected_count=int(executed.get("ppo_selected_count") or 0),
            ppo_blocked_count=max(0, int(len(trades_norm)) - int(executed.get("ppo_selected_count") or 0)),
            risk_blocked_count=int(executed.get("risk_blocked_count") or 0),
            insufficient_evidence=bool(len(trades_norm) == 0),
            metadata={
                "design_profile": design_profile.to_dict(),
                "historical_pnl_path": str(historical_pnl_path),
                "historical_trades_path": str(historical_trades_path),
            },
        )

        self.store.save_trader_forward_backtest_run(
            run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            trader_id=trader_id,
            asset=str(promoted_spec.asset).upper(),
            timeframe=str(promoted_spec.timeframe).upper(),
            promoted_at=str(promoted_spec.promoted_at),
            forward_start=str(forward_start.date().isoformat()),
            forward_end=str(last_date.date().isoformat()),
            status="completed",
            artifact_paths={
                "historical_pnl_path": str(historical_pnl_path),
                "historical_trades_path": str(historical_trades_path),
            },
            metrics=metrics.to_dict(),
        )
        self.store.save_trader_forward_metrics(
            evaluation_run_id=evaluation_run_id,
            trader_id=trader_id,
            asset=str(promoted_spec.asset).upper(),
            timeframe=str(promoted_spec.timeframe).upper(),
            evaluation_date=str(eval_ts.isoformat()),
            metrics=metrics.to_dict(),
        )
        emit_log(
            "risk_forward",
            "forward_backtest_completed",
            console=False,
            trader_id=trader_id,
            evaluation_run_id=evaluation_run_id,
            shadow_trades=int(metrics.shadow_trades),
            shadow_sharpe=metrics.shadow_sharpe,
            insufficient_evidence=bool(metrics.insufficient_evidence),
        )
        return metrics
