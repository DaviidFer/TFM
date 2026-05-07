from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _to_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, pd.DataFrame):
        if values.empty:
            return pd.Series(dtype="float64")
        series = pd.to_numeric(values.iloc[:, 0], errors="coerce")
    elif values is None:
        return pd.Series(dtype="float64")
    else:
        series = pd.Series(list(values) if isinstance(values, (list, tuple, np.ndarray)) else [values], dtype="float64")
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return series.astype(float)


def _extract_trade_pnl(trades: Any) -> pd.Series:
    if trades is None:
        return pd.Series(dtype="float64")
    if isinstance(trades, pd.DataFrame):
        norm = {str(c).lower(): c for c in trades.columns}
        pnl_col = next((norm[k] for k in ("profit", "pnl", "gross_profit") if k in norm), None)
        if pnl_col is None:
            return pd.Series(dtype="float64")
        return _to_series(trades[pnl_col])
    return _to_series(trades)


def compute_sharpe(returns: Any, annualization: float = 52.0) -> float | None:
    series = _to_series(returns)
    if len(series) < 2:
        return None
    sigma = float(series.std(ddof=0))
    if sigma <= 0:
        return 0.0
    return float((series.mean() / sigma) * np.sqrt(float(max(annualization, 1.0))))


def compute_profit_factor(trades: Any) -> float | None:
    pnl = _extract_trade_pnl(trades)
    if pnl.empty:
        return None
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if wins.empty and losses.empty:
        return None
    if losses.empty:
        return None
    loss_sum = abs(float(losses.sum()))
    if loss_sum <= 0:
        return None
    return float(wins.sum() / loss_sum)


def compute_max_drawdown(equity_curve: Any) -> float | None:
    series = _to_series(equity_curve)
    if series.empty:
        return None
    rolling_max = series.cummax().replace(0.0, np.nan)
    dd = (series / rolling_max) - 1.0
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        return None
    return float(abs(dd.min()))


def compute_winrate(trades: Any) -> float | None:
    pnl = _extract_trade_pnl(trades)
    if pnl.empty:
        return None
    return float((pnl > 0).sum() / len(pnl))


def compute_avg_win(trades: Any) -> float | None:
    pnl = _extract_trade_pnl(trades)
    wins = pnl[pnl > 0]
    if wins.empty:
        return None
    return float(wins.mean())


def compute_avg_loss(trades: Any) -> float | None:
    pnl = _extract_trade_pnl(trades)
    losses = pnl[pnl < 0]
    if losses.empty:
        return None
    return float(losses.mean())


def compute_expectancy(trades: Any) -> float | None:
    pnl = _extract_trade_pnl(trades)
    if pnl.empty:
        return None
    return float(pnl.mean())


def compute_losing_streak(trades: Any) -> int | None:
    pnl = _extract_trade_pnl(trades)
    if pnl.empty:
        return None
    current = 0
    best = 0
    for value in pnl.tolist():
        if float(value) < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def compute_trade_frequency(trades: Any) -> float | None:
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return None
    norm = {str(c).lower(): c for c in trades.columns}
    entry_col = next((norm[k] for k in ("entry_time", "time_entry", "open_time") if k in norm), None)
    if entry_col is None:
        return None
    entry = pd.to_datetime(trades[entry_col], errors="coerce").dropna()
    if entry.empty:
        return None
    span_days = max(float((entry.max() - entry.min()).days), 1.0)
    months = max(span_days / 30.4375, 1e-9)
    return float(len(entry) / months)


def align_development_and_forward_curves(
    development_curve: pd.DataFrame | None,
    forward_curve: pd.DataFrame | None,
    *,
    promoted_at: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for df, label in ((development_curve, "development_equity"), (forward_curve, "forward_equity")):
        if df is None or df.empty:
            continue
        work = df.copy()
        if "date" not in work.columns:
            if work.index.name:
                work = work.reset_index()
            else:
                continue
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values("date")
        value_col = "equity" if "equity" in work.columns else ("balance" if "balance" in work.columns else None)
        if value_col is None:
            continue
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[value_col]).rename(columns={value_col: label})
        frames.append(work[["date", label]].set_index("date"))
    if not frames:
        return pd.DataFrame(columns=["date", "development_equity", "forward_equity", "promoted"])
    out = pd.concat(frames, axis=1).sort_index().reset_index()
    promoted_ts = pd.to_datetime(promoted_at, errors="coerce")
    out["promoted"] = bool(pd.notna(promoted_ts))
    if pd.notna(promoted_ts):
        out["promotion_marker"] = promoted_ts
    return out


def build_metric_comparison_table(
    *,
    design_profile: Dict[str, Any] | None,
    forward_metrics: Dict[str, Any] | None,
    executed_metrics: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Tabla legible para el dashboard que confronta tres columnas:
    diseno historico vs. comportamiento forward shadow vs. comportamiento
    realmente ejecutado por la operativa live.
    """
    design = dict(design_profile or {})
    forward = dict(forward_metrics or {})
    executed = dict(executed_metrics or {})
    rows = [
        ("Trades", design.get("trades_design"), forward.get("shadow_trades"), forward.get("executed_trades", executed.get("executed_trades"))),
        ("Retorno", design.get("returns_mean_design"), forward.get("shadow_return"), forward.get("executed_return", executed.get("executed_return"))),
        ("Sharpe", design.get("sharpe_design"), forward.get("shadow_sharpe"), forward.get("executed_sharpe", executed.get("executed_sharpe"))),
        ("Profit factor", design.get("profit_factor_design"), forward.get("shadow_profit_factor"), forward.get("executed_profit_factor", executed.get("executed_profit_factor"))),
        ("Max drawdown", design.get("max_drawdown_design"), forward.get("shadow_max_drawdown"), forward.get("executed_max_drawdown", executed.get("executed_max_drawdown"))),
        ("Winrate", design.get("winrate_design"), forward.get("shadow_winrate"), executed.get("executed_winrate")),
        ("Expectancy", design.get("expectancy_design"), forward.get("shadow_expectancy"), executed.get("executed_expectancy")),
        ("Avg win", design.get("avg_win_design"), forward.get("shadow_avg_win"), executed.get("executed_avg_win")),
        ("Avg loss", design.get("avg_loss_design"), forward.get("shadow_avg_loss"), executed.get("executed_avg_loss")),
        ("Losing streak", design.get("max_losing_streak_design"), forward.get("shadow_losing_streak"), executed.get("executed_losing_streak")),
        ("Signal count", None, forward.get("signal_count"), None),
        ("PM selected count", None, forward.get("pm_selected_count"), None),
    ]
    return pd.DataFrame(rows, columns=["Metrica", "Diseno / OOS / Holdout", "Forward post-promocion", "Ejecutado real"])
