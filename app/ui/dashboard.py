from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.cloud import LOCAL_PATHS
from app.runtime import DevelopmentOperationalSupervisor
from app.services.trader_health import align_development_and_forward_curves, build_metric_comparison_table
from app.ui.dashboard_data import load_dashboard_snapshot


DEV_EVENT_TYPES = {
    "dataset_ready",
    "development_started",
    "split_and_target_ready",
    "candidate_rules_ready",
    "validation_completed",
    "trader_promoted",
    "trader_promoted_with_rules",
    "trader_state_changed",
}
OPS_EVENT_TYPES = {
    "trader_metrics_updated",
    "broker_order_routed",
    "broker_order_rejected",
    "broker_access_denied",
}
OPS_COMPONENTS = {"mt5_connector", "mt5_data_provider", "live_runtime", "execution_router", "trader_agent"}
DEFAULT_EVENT_LIMIT = 300
DEFAULT_AUTO_REFRESH_MS = 2500


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "-"
    out = ts.replace("T", " ")
    out = out.split("+")[0]
    out = out.split(".")[0]
    return out


def _fmt_obj(x: Any) -> str:
    if isinstance(x, dict):
        return "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}" for k, v in x.items())
    if isinstance(x, list):
        return "\n".join(f"- {i}" for i in x)
    return str(x)


def _event_to_row(e: Dict[str, Any]) -> Dict[str, Any]:
    payload = e.get("payload", {}) or {}
    return {
        "fecha_hora": _fmt_ts(str(e.get("occurred_at", ""))),
        "agente": str(e.get("producer", "-")),
        "paso": str(e.get("event_type", "-")).replace("_", " "),
        "detalle": str(e.get("event_type", "-")),
        "parametros_clave": _fmt_obj(payload),
    }


def _fmt_kv_lines(data: Dict[str, Any], *, keys: List[str]) -> List[str]:
    out: List[str] = []
    for k in keys:
        if k in data and data.get(k) is not None:
            out.append(f"{k}: {data.get(k)}")
    return out


def _pretty_trader_name(trader_id: Any, *, asset: Any = None, timeframe: Any = None) -> str:
    asset_txt = str(asset or "").strip().upper()
    tf_txt = str(timeframe or "").strip().upper()
    trader_txt = str(trader_id or "").strip()
    if asset_txt and tf_txt:
        return f"{asset_txt}_{tf_txt}"
    if trader_txt.startswith("tr_"):
        parts = trader_txt.split("_")
        if len(parts) >= 4:
            return f"{parts[1].upper()}_{parts[2].upper()}"
    return trader_txt or "pendiente"


def _pretty_trader_from_row(row: Dict[str, Any]) -> str:
    return _pretty_trader_name(
        row.get("trader_id"),
        asset=row.get("asset") or row.get("symbol"),
        timeframe=row.get("timeframe") or "D1",
    )


def _human_pm_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename_map = {
        "trader": "Trader",
        "symbol": "Símbolo",
        "side": "Lado",
        "fecha_señal": "Fecha señal",
        "fase_pm": "Fase PM",
        "decision_pm": "Decisión PM",
        "estado_orden": "Estado orden",
        "peso_pct": "Peso (%)",
        "euros_asignados": "Euros asignados",
        "acciones_estimadas": "Acciones estimadas",
        "motivo_interpretado": "Motivo",
    }
    return df.rename(columns=rename_map)


def _equity_frame_to_return_frame(df: pd.DataFrame, *, series_name: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns or "equity" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    if work.empty:
        return pd.DataFrame()
    base = float(work["equity"].iloc[0] or 1.0)
    work[series_name] = ((work["equity"] / max(base, 1e-8)) - 1.0) * 100.0
    out = work.set_index("date")[[series_name]]
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")
    return out


def _sanitize_line_chart_frame(df: pd.DataFrame, *, index_name: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work = work.replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")
    if work.empty:
        return pd.DataFrame()
    if index_name and index_name in work.columns:
        work[index_name] = pd.to_numeric(work[index_name], errors="coerce")
        work = work.dropna(subset=[index_name])
        if work.empty:
            return pd.DataFrame()
        work = work.sort_values(index_name).set_index(index_name)
    if isinstance(work.index, pd.DatetimeIndex):
        work = work[~work.index.isna()]
    return work.dropna(axis=1, how="all")


def _build_portfolio_curve_figures(
    *,
    portfolio_curve: pd.DataFrame,
    latest_weights: Dict[str, float],
) -> List[Any]:
    """
    Genera las figuras del Portfolio Manager para el modo GA+PSO:
    - curva historica de la cartera seleccionada vs. equal-weight,
    - asignacion de pesos por trader.
    """
    figures: List[Any] = []

    fig_curves, ax_curves = plt.subplots(figsize=(6, 3))
    plotted = False
    if portfolio_curve is not None and not portfolio_curve.empty:
        for col in portfolio_curve.columns:
            ax_curves.plot(
                portfolio_curve.index,
                portfolio_curve[col].astype(float) * 100.0,
                label=str(col),
            )
            plotted = True
    if plotted:
        ax_curves.set_title("Curva acumulada cartera GA+PSO vs. equal weight (%)")
        ax_curves.set_ylabel("%")
        ax_curves.legend(fontsize=8)
        ax_curves.grid(True, alpha=0.2)
    else:
        ax_curves.text(0.5, 0.5, "Sin curvas historicas", ha="center", va="center")
        ax_curves.set_axis_off()
    figures.append(fig_curves)

    fig_weights, ax_weights = plt.subplots(figsize=(6, 3))
    if latest_weights:
        keys = list(latest_weights.keys())
        vals = [float(latest_weights[k]) * 100.0 for k in keys]
        ax_weights.bar(keys, vals, color="tab:green")
        ax_weights.set_title("Asignacion GA+PSO (%)")
        ax_weights.tick_params(axis="x", rotation=90, labelsize=8)
    else:
        ax_weights.text(0.5, 0.5, "Sin pesos disponibles", ha="center", va="center")
        ax_weights.set_axis_off()
    figures.append(fig_weights)

    return figures


def _render_dev_forward_equity_chart(
    *,
    development_curve: pd.DataFrame | None,
    forward_curve: pd.DataFrame | None,
    promoted_at: str | None,
) -> None:
    aligned = align_development_and_forward_curves(
        development_curve,
        forward_curve,
        promoted_at=str(promoted_at or ""),
    )
    if aligned.empty or "date" not in aligned.columns:
        st.info("No hay curva combinada desarrollo + real para este trader.")
        return
    plot_long = aligned.melt(
        id_vars=[c for c in ["date", "promotion_marker"] if c in aligned.columns],
        value_vars=[c for c in ["development_equity", "forward_equity"] if c in aligned.columns],
        var_name="serie",
        value_name="equity",
    ).dropna(subset=["equity"])
    if plot_long.empty:
        st.info("No hay puntos suficientes para la curva combinada.")
        return
    plot_long["serie"] = plot_long["serie"].replace(
        {"development_equity": "Backtest desarrollo", "forward_equity": "Operativa real"}
    )
    base_chart = (
        alt.Chart(plot_long)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Fecha"),
            y=alt.Y("equity:Q", title="Equity / Balance"),
            color=alt.Color("serie:N", title="Serie"),
            tooltip=["date:T", "serie:N", "equity:Q"],
        )
        .properties(height=320)
    )
    if "promotion_marker" in aligned.columns and aligned["promotion_marker"].notna().any():
        promo_ts = pd.to_datetime(aligned["promotion_marker"].dropna().iloc[0], errors="coerce")
        if pd.notna(promo_ts):
            rule_df = pd.DataFrame({"promotion_marker": [promo_ts]})
            rule = alt.Chart(rule_df).mark_rule(strokeDash=[6, 4]).encode(x="promotion_marker:T")
            st.altair_chart(base_chart + rule, width="stretch")
            return
    st.altair_chart(base_chart, width="stretch")


def _pm_signal_row_from_live(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rebalance_id": str(row.get("rebalance_id") or ""),
        "trader": _pretty_trader_name(row.get("trader_id"), asset=row.get("symbol"), timeframe="D1"),
        "symbol": row.get("symbol"),
        "side": _human_side_label(row.get("side")),
        "fecha_señal": _fmt_ts(str(row.get("detected_at", ""))),
        "fase_pm": _interpret_pm_phase(row.get("portfolio_phase")),
        "decision_pm": "seleccionado" if bool(row.get("selected")) else "descartado",
        "estado_orden": row.get("status"),
        "peso_pct": round(float(row.get("weight") or 0.0) * 100.0, 3),
        "euros_asignados": round(float(row.get("euros") or 0.0), 2),
        "acciones_estimadas": int(float(row.get("volume") or 0.0)) if row.get("volume") is not None else 0,
        "motivo_interpretado": _interpret_pm_reason(row.get("reason") or row.get("status")),
        "_priority": 3,
        "_raw_ts": str(row.get("detected_at") or ""),
    }


def _pm_signal_row_from_audit(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    selected = bool(row.get("pm_selected"))
    status = "executed" if bool(row.get("executed")) else ("selected" if selected else "discarded")
    return {
        "rebalance_id": str(metadata.get("rebalance_id") or ""),
        "trader": _pretty_trader_name(row.get("trader_id"), asset=row.get("asset"), timeframe=row.get("timeframe") or "D1"),
        "symbol": row.get("asset"),
        "side": _human_side_label(row.get("signal_side")),
        "fecha_señal": _fmt_ts(str(metadata.get("detected_at") or row.get("timestamp") or "")),
        "fase_pm": _interpret_pm_phase(metadata.get("portfolio_phase")),
        "decision_pm": "seleccionado" if selected else "descartado",
        "estado_orden": status,
        "peso_pct": round(float(row.get("pm_weight") or 0.0) * 100.0, 3),
        "euros_asignados": round(float(metadata.get("portfolio_euros") or 0.0), 2),
        "acciones_estimadas": int(float(metadata.get("volume") or 0.0)) if metadata.get("volume") is not None else 0,
        "motivo_interpretado": _interpret_pm_reason(row.get("reason_if_blocked") or metadata.get("source") or status),
        "_priority": 2 if bool(row.get("executed")) else 1,
        "_raw_ts": str(row.get("timestamp") or ""),
    }


def _normalize_pm_signal_rows(signal_book: List[Dict[str, Any]], signal_audit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keyed: Dict[str, Dict[str, Any]] = {}

    def _merge(item: Dict[str, Any]) -> None:
        key = "|".join(
            [
                str(item.get("rebalance_id") or ""),
                str(item.get("trader") or ""),
                str(item.get("symbol") or ""),
                str(item.get("side") or ""),
            ]
        )
        current = keyed.get(key)
        if current is None:
            keyed[key] = item
            return
        current_priority = int(current.get("_priority") or 0)
        item_priority = int(item.get("_priority") or 0)
        if item_priority > current_priority or (item_priority == current_priority and str(item.get("_raw_ts") or "") > str(current.get("_raw_ts") or "")):
            keyed[key] = item

    for row in signal_book:
        _merge(_pm_signal_row_from_live(dict(row)))
    for row in signal_audit:
        _merge(_pm_signal_row_from_audit(dict(row)))
    return list(keyed.values())


def _render_pm_signal_tables(signal_rows: List[Dict[str, Any]], *, key_prefix: str) -> None:
    if not signal_rows:
        st.info("No hay señales auditadas para este entrenamiento/rebalanceo.")
        return
    df_signals = pd.DataFrame(signal_rows)
    pm_cols = [
        "trader",
        "symbol",
        "side",
        "fecha_señal",
        "fase_pm",
        "decision_pm",
        "estado_orden",
        "peso_pct",
        "euros_asignados",
        "acciones_estimadas",
        "motivo_interpretado",
    ]
    df_selected = df_signals[df_signals["decision_pm"] == "seleccionado"].copy()
    df_discarded = df_signals[df_signals["decision_pm"] == "descartado"].copy()
    df_exec = df_signals[df_signals["estado_orden"] == "executed"].copy()
    df_selected_show = _human_pm_signal_columns(df_selected[pm_cols].copy()) if not df_selected.empty else pd.DataFrame()
    df_discarded_show = _human_pm_signal_columns(df_discarded[pm_cols].copy()) if not df_discarded.empty else pd.DataFrame()
    df_exec_show = (
        _human_pm_signal_columns(df_exec[["trader", "symbol", "side", "fecha_señal", "fase_pm", "peso_pct", "euros_asignados", "acciones_estimadas"]].copy())
        if not df_exec.empty
        else pd.DataFrame()
    )

    st.markdown("**Señales seleccionadas por el portfolio manager**")
    if df_selected.empty:
        st.info("No hay señales seleccionadas en este entrenamiento.")
    else:
        st.dataframe(
            df_selected_show.style
            .map(lambda v: _pm_style_cell(v, column="fase_pm"), subset=["Fase PM"])
            .map(lambda v: _pm_style_cell(v, column="decision_pm"), subset=["Decisión PM"])
            .map(lambda v: _pm_style_cell(v, column="estado_orden"), subset=["Estado orden"]),
            width="stretch",
            hide_index=True,
        )

    st.markdown("**Señales descartadas por el portfolio manager**")
    if df_discarded.empty:
        st.info("No hay señales descartadas en este entrenamiento.")
    else:
        st.dataframe(
            df_discarded_show.style
            .map(lambda v: _pm_style_cell(v, column="fase_pm"), subset=["Fase PM"])
            .map(lambda v: _pm_style_cell(v, column="decision_pm"), subset=["Decisión PM"])
            .map(lambda v: _pm_style_cell(v, column="estado_orden"), subset=["Estado orden"]),
            width="stretch",
            hide_index=True,
        )

    st.markdown("**Señales transformadas en operación ejecutada**")
    if df_exec.empty:
        st.info("Ninguna señal seleccionada se ha ejecutado todavía.")
    else:
        st.dataframe(
            df_exec_show.style.map(lambda v: _pm_style_cell(v, column="fase_pm"), subset=["Fase PM"]),
            width="stretch",
            hide_index=True,
        )


def _is_live_portfolio_rebalance(row: Dict[str, Any]) -> bool:
    """
    Determina si un snapshot persistido corresponde a un rebalanceo real
    del runtime live (modo GA+PSO). Se ignoran los registros marcados como
    `offline_test_eval`, que corresponden a rebalanceos sinteticos generados
    fuera del runtime operativo (smoke tests y evaluaciones offline).
    """
    metadata = dict(row.get("metadata") or {})
    if str(metadata.get("source") or "").strip().lower() == "offline_test_eval":
        return False
    return bool(row.get("rebalance_date"))


def _human_review_action(action: Any) -> str:
    """Etiqueta humana para `TraderReviewAction` (KEEP / RETRAINING)."""
    mapping = {
        "keep": "Mantener",
        "retraining": "Reentrenando",
    }
    return mapping.get(str(action or "").strip().lower(), str(action or "-"))


def _human_trader_state(state: Any) -> str:
    mapping = {
        "live": "LIVE",
        "retraining": "RETRAINING",
    }
    return mapping.get(str(state or "").strip().lower(), str(state or "-"))


def _fmt_short_date(ts: Any) -> str:
    txt = str(ts or "").strip()
    if not txt:
        return "-"
    return txt.replace("T", " ").split(" ")[0]


def _model_name(raw: Any) -> str:
    txt = str(raw or "").strip().lower()
    mapping = {
        "quantile": "Quantiles",
        "rulefit": "RuleFit",
        "genetic": "Genético",
    }
    return mapping.get(txt, txt.capitalize() if txt else "-")


def _human_param_name(name: Any) -> str:
    txt = str(name or "").strip()
    mapping = {
        "is_pct": "IS %",
        "oos_pct": "OOS %",
        "holdout_year": "Holdout year",
        "lookback_years": "Lookback years",
        "n_bins": "N bins",
        "combo_size": "Tamaño combinación",
        "min_coverage": "Cobertura mínima",
        "target_n_rules": "Objetivo N reglas",
        "n_estimators": "N estimators",
        "max_candidate_rules": "Máx reglas candidatas",
        "progress_every": "Progreso cada",
        "n_monkeys": "N monkeys",
        "is_pass_pct": "IS pass %",
        "oos_pass_pct": "OOS pass %",
        "min_coverage_is": "Cobertura mínima IS",
        "min_coverage_oos": "Cobertura mínima OOS",
        "n_jobs": "N jobs",
        "corr_threshold": "Umbral correlación",
        "min_ops": "Mín operaciones",
        "target_year": "Año objetivo",
        "top_n_long": "Top N LONG",
        "top_n_short": "Top N SHORT",
        "diagnose": "Diagnóstico",
        "verbose": "Verbose",
    }
    if txt in mapping:
        return mapping[txt]
    clean = txt.replace("_", " ").strip()
    return clean[:1].upper() + clean[1:] if clean else "-"


def _format_params_inline(data: Any) -> str:
    if not isinstance(data, dict) or not data:
        return "-"
    parts: List[str] = []
    for key, value in data.items():
        if str(key) == "progress_every" and str(value).strip() in {"", "0", "0.0", "None", "none", "False", "false"}:
            continue
        parts.append(f"{_human_param_name(key)}: {value}")
    return " | ".join(parts)


def _format_validation_params_inline(data: Any, *, drop_keys: List[str] | None = None) -> str:
    if not isinstance(data, dict) or not data:
        return "-"
    excluded = set(drop_keys or [])
    parts: List[str] = []
    for key, value in data.items():
        if str(key) in excluded:
            continue
        parts.append(f"{_human_param_name(key)}: {value}")
    return " | ".join(parts) if parts else "-"


def _pretty_dev_event_title(e: Dict[str, Any]) -> str:
    producer = str(e.get("producer", "") or "")
    event_type = str(e.get("event_type", "") or "")
    mapping = {
        ("data_process", "dataset_ready"): "DataProcess -> Dataset preparado",
        ("data_agent", "dataset_ready"): "DataProcess -> Dataset preparado",
        ("developer_agent", "development_started"): "Developer Agent -> Configuración del desarrollo",
        ("developer_agent", "split_and_target_ready"): "Developer Agent -> Split preparado",
        ("developer_agent", "candidate_rules_ready"): "Developer Agent -> Generación de reglas completada",
        ("validation_agent", "validation_completed"): "Validation Agent -> Validación completada",
        ("validation_agent", "trader_promoted"): "Validation Agent -> Trader promovido",
        ("validation_agent", "trader_promoted_with_rules"): "Validation Agent -> Trader promovido",
        ("trader_agent", "trader_state_changed"): "Trader Agent -> Estado del trader actualizado",
    }
    return mapping.get((producer, event_type), f"{producer} -> {event_type}".replace("_", " "))


def _build_dev_group_context(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for ev in events:
        payload = ev.get("payload", {}) or {}
        event_type = str(ev.get("event_type", ""))
        if payload.get("asset") and not ctx.get("asset"):
            ctx["asset"] = payload.get("asset")
        if payload.get("timeframe") and not ctx.get("timeframe"):
            ctx["timeframe"] = payload.get("timeframe")
        if event_type == "dataset_ready":
            ctx["dataset"] = payload
        elif event_type == "development_started":
            ctx["development_started"] = payload
        elif event_type == "split_and_target_ready":
            ctx["split_ready"] = payload
        elif event_type == "validation_completed":
            ctx["validation_completed"] = payload
        elif event_type in {"trader_promoted", "trader_promoted_with_rules"}:
            ctx["trader_promoted"] = payload
    return ctx


def _enrich_dev_group_context(
    group_ctx: Dict[str, Any],
    *,
    all_events: List[Dict[str, Any]],
    asset: str,
    reference_ts: str,
) -> Dict[str, Any]:
    if group_ctx.get("development_started") and group_ctx.get("split_ready"):
        return group_ctx
    ref_dt = _parse_iso(reference_ts)
    if ref_dt is None:
        return group_ctx
    enriched = dict(group_ctx)
    for ev in reversed(all_events):
        payload = ev.get("payload", {}) or {}
        ev_dt = _parse_iso(str(ev.get("occurred_at", "")))
        if ev_dt is None or ev_dt > ref_dt:
            continue
        delta_sec = abs((ref_dt - ev_dt).total_seconds())
        if delta_sec > 300:
            continue
        event_type = str(ev.get("event_type", ""))
        payload_asset = str(payload.get("asset") or "").upper()
        if event_type in {"development_started", "dataset_ready"} and payload_asset != str(asset or "").upper():
            continue
        if event_type == "split_and_target_ready" and payload_asset and payload_asset != str(asset or "").upper():
            continue
        if event_type == "development_started" and not enriched.get("development_started"):
            enriched["development_started"] = payload
        elif event_type == "split_and_target_ready" and not enriched.get("split_ready"):
            enriched["split_ready"] = payload
        elif event_type == "dataset_ready" and not enriched.get("dataset"):
            enriched["dataset"] = payload
        if enriched.get("development_started") and enriched.get("split_ready"):
            break
    return enriched


def _pretty_dev_event_lines(e: Dict[str, Any], group_ctx: Dict[str, Any]) -> List[str]:
    event_type = str(e.get("event_type", ""))
    payload = e.get("payload", {}) or {}
    if event_type == "dataset_ready":
        return [
            f"Activo: {payload.get('asset')}",
            f"Timeframe: {payload.get('timeframe')}",
            f"Rango temporal: {_fmt_short_date(payload.get('start_date'))} -> {_fmt_short_date(payload.get('end_date'))}",
            f"Filas del dataset: {payload.get('rows')}",
        ]
    if event_type == "development_started":
        families = list(payload.get("families", []) or [])
        chosen_family = families[0] if families else ""
        family_params = dict(payload.get("family_params", {}) or {})
        split_config = dict(payload.get("split_config", {}) or {})
        return [
            f"Activo: {payload.get('asset')}",
            f"Modelo de Generación de reglas: {_model_name(chosen_family)}",
            f"Parámetros: {_format_params_inline(family_params.get(chosen_family, family_params))}",
            f"Split: IS {split_config.get('is_pct')} | OOS {split_config.get('oos_pct')} | Holdout year {split_config.get('holdout_year')}",
        ]
    if event_type == "split_and_target_ready":
        split_detail = dict(payload.get("split_detail", {}) or {})
        block_date_ranges = dict(payload.get("block_date_ranges", {}) or {})
        data_is = block_date_ranges.get("data_is", {}) or {}
        data_oos = block_date_ranges.get("data_oos", {}) or {}
        return [
            f"Split IS: {split_detail.get('is_pct')} | {_fmt_short_date(data_is.get('start'))} -> {_fmt_short_date(data_is.get('end'))}",
            f"Split OOS: {split_detail.get('oos_pct')} | {_fmt_short_date(data_oos.get('start'))} -> {_fmt_short_date(data_oos.get('end'))}",
            f"Holdout year: {split_detail.get('holdout_year')}",
            f"Lookback years: {split_detail.get('lookback_years')}",
        ]
    if event_type == "candidate_rules_ready":
        dev_started = dict(group_ctx.get("development_started", {}) or {})
        split_ready = dict(group_ctx.get("split_ready", {}) or {})
        summary = dict(payload.get("summary", {}) or payload.get("candidate_summary", {}) or {})
        families = list(summary.get("families", []) or dev_started.get("families", []) or [])
        chosen_family = families[0] if families else ""
        family_params = dict(dev_started.get("family_params", {}) or {})
        split_config = dict(dev_started.get("split_config", {}) or {})
        block_date_ranges = dict(split_ready.get("block_date_ranges", {}) or {})
        data_is = block_date_ranges.get("data_is", {}) or {}
        data_oos = block_date_ranges.get("data_oos", {}) or {}
        return [
            f"Activo: {payload.get('asset') or dev_started.get('asset') or group_ctx.get('asset')}",
            f"Modelo de Generación de reglas: {_model_name(chosen_family)}",
            f"Parámetros: {_format_params_inline(family_params.get(chosen_family, family_params))}",
            f"Split IS: {split_config.get('is_pct')} | {_fmt_short_date(data_is.get('start'))} -> {_fmt_short_date(data_is.get('end'))}",
            f"Split OOS: {split_config.get('oos_pct')} | {_fmt_short_date(data_oos.get('start'))} -> {_fmt_short_date(data_oos.get('end'))}",
            f"Reglas Generadas LONG={summary.get('n_long')} SHORT={summary.get('n_short')}",
        ]
    if event_type == "validation_completed":
        lines = [
            f"Activo: {payload.get('asset') or group_ctx.get('asset')}",
            f"Reglas LONG validadas: {payload.get('passed_long')}",
            f"Reglas SHORT validadas: {payload.get('passed_short')}",
        ]
        profile = payload.get("validation_profile", {}) or {}
        split_ready = dict(group_ctx.get("split_ready", {}) or {})
        block_date_ranges = dict(split_ready.get("block_date_ranges", {}) or {})
        data_is = block_date_ranges.get("data_is", {}) or {}
        data_oos = block_date_ranges.get("data_oos", {}) or {}
        if profile:
            split_assumption = dict(profile.get("split_assumption", {}) or {})
            lines.append(
                f"Split: IS {_fmt_short_date(data_is.get('start'))} -> {_fmt_short_date(data_is.get('end'))} | "
                f"OOS {_fmt_short_date(data_oos.get('start'))} -> {_fmt_short_date(data_oos.get('end'))} | "
                f"Holdout year {split_assumption.get('holdout_year')}"
            )
            lines.append(
                f"Monkey IS: {_format_validation_params_inline(profile.get('monkey_is'), drop_keys=['n_jobs'])}"
            )
            lines.append(
                f"Monkey OOS: {_format_validation_params_inline(profile.get('monkey_oos'), drop_keys=['n_jobs'])}"
            )
            if profile.get("correlation_pruning"):
                lines.append(
                    f"Poda por correlación: "
                    f"{_format_validation_params_inline(profile.get('correlation_pruning'), drop_keys=['min_ops', 'diagnose'])}"
                )
            if profile.get("forward_validation"):
                lines.append(
                    f"Validación forward: "
                    f"{_format_validation_params_inline(profile.get('forward_validation'), drop_keys=['min_ops', 'verbose'])}"
                )
            if profile.get("stability_selection"):
                lines.append(
                    f"Selección por estabilidad: "
                    f"{_format_validation_params_inline(profile.get('stability_selection'), drop_keys=['min_ops', 'verbose'])}"
                )
        return lines
    if event_type in {"trader_promoted", "trader_promoted_with_rules"}:
        lines = [
            f"Trader: {_pretty_trader_name(payload.get('trader_id'), asset=payload.get('asset'), timeframe=payload.get('timeframe'))}",
            f"Activo: {payload.get('asset')}",
            f"Timeframe: {payload.get('timeframe')}",
        ]
        long_rules = payload.get("long_rules", []) or []
        short_rules = payload.get("short_rules", []) or []
        lines.append(f"Reglas Validadas LONG={len(long_rules)} SHORT={len(short_rules)}")
        return lines
    if event_type == "trader_state_changed":
        return [
            f"Trader: {_pretty_trader_name(payload.get('trader_id'), asset=group_ctx.get('asset'), timeframe=group_ctx.get('timeframe', 'D1'))}",
            f"Estado: {str(payload.get('new_state') or '').capitalize()}",
        ]

    # fallback compacto para eventos no contemplados
    lines: List[str] = []
    for key, value in list(payload.items())[:5]:
        lines.append(f"{_human_param_name(key)}: {value}")
    return lines


def _render_pretty_dev_events(
    events: List[Dict[str, Any]],
    *,
    title: str,
    max_items: int = 80,
    source_events: List[Dict[str, Any]] | None = None,
) -> None:
    st.markdown(f"### {title}")
    if not events:
        st.info("Sin eventos de desarrollo todavía.")
        return
    shown = events[:max_items]
    lookup_events = list(source_events or events)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for e in shown:
        key = str(e.get("correlation_id") or f"evt::{e.get('event_id', '')}")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(e)

    for idx, key in enumerate(order):
        g = groups[key]
        first = g[0]
        ts = _fmt_ts(str(first.get("occurred_at", "")))
        group_ctx = _build_dev_group_context(g)
        asset = str(group_ctx.get("asset") or "-")
        group_ctx = _enrich_dev_group_context(
            group_ctx,
            all_events=lookup_events,
            asset=asset,
            reference_ts=str(first.get("occurred_at", "")),
        )
        asset = str(group_ctx.get("asset") or asset or "-")
        trader_payload = dict(group_ctx.get("trader_promoted", {}) or {})
        trader_id = _pretty_trader_name(
            trader_payload.get("trader_id"),
            asset=trader_payload.get("asset") or asset,
            timeframe=trader_payload.get("timeframe") or group_ctx.get("timeframe") or "D1",
        )
        title_exp = f"[{ts}] Activo: {asset} | Trader: {trader_id if trader_id != 'pendiente' else 'pendiente'}"
        with st.expander(title_exp, expanded=(idx == len(order) - 1)):
            lines: List[str] = []
            for ev in g:
                event_ts = _fmt_ts(str(ev.get("occurred_at", "")))
                lines.append(f"[{event_ts}] {_pretty_dev_event_title(ev)}")
                for detail in _pretty_dev_event_lines(ev, group_ctx):
                    lines.append(f"  - {detail}")
                lines.append("")
            st.text("\n".join(lines).strip())


def _get_supervisor() -> DevelopmentOperationalSupervisor:
    default_db_path = Path(os.getenv("TFM_DB_PATH", str(LOCAL_PATHS.supervisor_db)))
    if "tfm_supervisor" not in st.session_state:
        st.session_state["tfm_supervisor"] = DevelopmentOperationalSupervisor(db_path=default_db_path)
        return st.session_state["tfm_supervisor"]

    sup = st.session_state["tfm_supervisor"]
    # Compatibilidad hot-reload: si hay una instancia antigua en memoria,
    # la reemplazamos por una nueva con la version actual de la clase.
    needs_rebuild = (
        (not hasattr(sup, "get_backtest_registry"))
        or int(getattr(sup, "dashboard_snapshot_schema_version", 0)) < 10
    )
    if needs_rebuild:
        db_path = Path(getattr(sup, "db_path", default_db_path))
        prev_status = {}
        resume_development = False
        resume_runtime = False
        try:
            prev_status = sup.get_status() if hasattr(sup, "get_status") else {}
            resume_development = bool(prev_status.get("develop_enabled"))
            resume_runtime = bool(prev_status.get("operational_runtime_started")) or bool(prev_status.get("running"))
            if hasattr(sup, "stop_development"):
                sup.stop_development()
            if hasattr(sup, "_shutdown"):
                sup._shutdown.set()
            if hasattr(sup, "_thread") and sup._thread is not None and sup._thread.is_alive():
                sup._thread.join(timeout=2)
        except Exception:
            pass
        rebuilt = DevelopmentOperationalSupervisor(db_path=db_path)
        try:
            if resume_development:
                rebuilt.start()
            elif resume_runtime:
                rebuilt.start_operational_runtime()
        except Exception:
            pass
        st.session_state["tfm_supervisor"] = rebuilt
    return st.session_state["tfm_supervisor"]


@st.cache_data(ttl=2, show_spinner=False)
def _load_cached_dashboard_snapshot(db_path_str: str, event_limit: int) -> Dict[str, Any]:
    snap = load_dashboard_snapshot(db_path=Path(db_path_str), event_limit=event_limit)
    return {
        "events": list(reversed(snap.events)),
        "trader_states": list(snap.trader_states),
    }


def _load_events(db_path: Path, event_limit: int) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    snap = _load_cached_dashboard_snapshot(str(db_path), int(event_limit))
    return list(snap.get("events", []))


def _load_trader_states(db_path: Path) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    snap = _load_cached_dashboard_snapshot(str(db_path), 200)
    return list(snap.get("trader_states", []))


def _filter_dev_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in events:
        if str(e.get("event_type")) in DEV_EVENT_TYPES or str(e.get("producer")) in {"data_process", "data_agent", "developer_agent", "validation_agent", "supervisor"}:
            out.append(e)
    return out


def _filter_completed_dev_cycles(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mantiene solo ciclos que acaban con trader_promoted para evitar
    mostrar intentos parciales/fallidos en la vista principal.
    """
    promoted_corr_ids = {
        str(e.get("correlation_id"))
        for e in events
        if str(e.get("event_type")) in {"trader_promoted", "trader_promoted_with_rules"} and e.get("correlation_id")
    }
    if not promoted_corr_ids:
        return []
    out: List[Dict[str, Any]] = []
    for e in events:
        corr = e.get("correlation_id")
        if corr and str(corr) in promoted_corr_ids and str(e.get("event_type")) in DEV_EVENT_TYPES:
            out.append(e)
    return out


def _build_promoted_rules_index(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for e in events:
        if str(e.get("event_type")) not in {"trader_promoted", "trader_promoted_with_rules"}:
            continue
        payload = e.get("payload", {}) or {}
        trader_id = str(payload.get("trader_id") or "")
        if not trader_id:
            continue
        out[trader_id] = {
            "asset": payload.get("asset"),
            "long_rules": list(payload.get("long_rules", []) or []),
            "short_rules": list(payload.get("short_rules", []) or []),
        }
    return out


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _filter_events_by_session(events: List[Dict[str, Any]], *, session_started_at: str | None) -> List[Dict[str, Any]]:
    if not session_started_at:
        return events
    start_dt = _parse_iso(session_started_at)
    if start_dt is None:
        return events
    out: List[Dict[str, Any]] = []
    for e in events:
        ev_dt = _parse_iso(str(e.get("occurred_at", "")))
        if ev_dt is None:
            continue
        if ev_dt >= start_dt:
            out.append(e)
    return out


def _filter_ops_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in events:
        if str(e.get("event_type")) in OPS_EVENT_TYPES or str(e.get("producer")) in OPS_COMPONENTS:
            out.append(e)
    return out


def _render_flow_table(events: List[Dict[str, Any]], *, title: str) -> None:
    st.markdown(f"### {title}")
    if not events:
        st.info("Sin eventos todavía.")
        return
    rows = [_event_to_row(e) for e in events]
    st.table(pd.DataFrame(rows)[["fecha_hora", "agente", "paso", "detalle", "parametros_clave"]])


def _latest_order_events_by_trader(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for e in events:
        et = str(e.get("event_type", ""))
        if et not in {"broker_order_routed", "broker_order_rejected"}:
            continue
        payload = e.get("payload", {}) or {}
        trader_id = str(payload.get("trader_id") or "")
        if not trader_id:
            continue
        out[trader_id] = e
    return out


def _safe_mt5_positions(supervisor: DevelopmentOperationalSupervisor) -> List[Dict[str, Any]]:
    try:
        if not bool(supervisor.get_status().get("mt5_connected")):
            return []
        if not hasattr(supervisor, "mt5"):
            return []
        return list(supervisor.mt5.get_open_positions())
    except Exception:
        return []


def _market_status_text() -> str:
    now = datetime.now(timezone.utc)
    return "Mercado cerrado (fin de semana)" if now.weekday() >= 5 else "Mercado potencialmente abierto"


def _interpret_pm_reason(reason: Any) -> str:
    txt = str(reason or "").strip()
    if not txt:
        return "-"
    upper = txt.upper()
    if "TRADE_RETCODE_INVALID_FILL" in upper or "(10030)" in upper:
        return "MT5 rechaza el filling mode usado para ese simbolo."
    if "TRADE_RETCODE_MARKET_CLOSED" in upper or "(10018)" in upper:
        return "Mercado cerrado para ese simbolo en ese momento."
    if "TRADE_RETCODE_INVALID_VOLUME" in upper or "(10014)" in upper:
        return "Volumen invalido para ese simbolo."
    if "TRADE_RETCODE_NO_MONEY" in upper or "(10019)" in upper:
        return "Fondos insuficientes."
    if "ORDER_SEND_NONE" in upper:
        return "La API devolvio None al enviar la orden."
    if txt == "executed":
        return "Orden ejecutada correctamente."
    if txt == "selected":
        return "Seleccionada por el portfolio manager y pendiente de ejecución."
    if txt == "discarded":
        return "Descartada por el portfolio manager en este rebalanceo."
    if txt == "waiting_full_universe" or txt == "portfolio_manager_waiting_full_universe":
        return "La señal queda en espera hasta terminar de generar todo el universo de traders."
    if txt == "waiting_next_monday" or txt == "portfolio_manager_weekly_rebalance_only":
        return "La señal se revisará en el próximo rebalanceo semanal del lunes."
    if txt == "already_open" or txt == "position_already_open":
        return "Ese trader ya tiene una posición abierta en la misma dirección."
    if txt == "closed":
        return "La posición se ha cerrado porque la señal ya no sigue activa."
    if txt == "close_rejected":
        return "Se intentó cerrar la posición pero MT5 la rechazó; se reintentará."
    if txt == "signal_inactive":
        return "La posición se cierra porque la señal ha desaparecido."
    if txt == "signal_side_changed":
        return "La posición actual se cierra porque la nueva señal cambió de dirección."
    return txt


def _human_side_label(side: Any) -> str:
    txt = str(side or "").strip().lower()
    mapping = {
        "buy": "Compra",
        "sell": "Venta",
    }
    return mapping.get(txt, str(side or "-"))


def _human_signal_label(signal: Any) -> str:
    txt = str(signal or "").strip()
    upper = txt.upper()
    mapping = {
        "SIGNALTYPE.BUY": "Señal de compra",
        "SIGNALTYPE.SELL": "Señal de venta",
        "CLOSE_BUY": "Cerrar compra",
        "CLOSE_SELL": "Cerrar venta",
        "CLOSE_POSITION": "Cerrar posición",
    }
    return mapping.get(upper, txt or "-")


def _interpret_pm_phase(phase: Any) -> str:
    txt = str(phase or "").strip()
    if txt == "despliegue_inicial":
        return "Despliegue inicial"
    if txt == "rebalanceo_semanal":
        return "Rebalanceo semanal"
    return "-"


def _badge_html(label: str, *, bg: str, fg: str = "#ffffff") -> str:
    return (
        f"<span style='display:inline-block;padding:0.2rem 0.55rem;margin:0.1rem 0.25rem 0.1rem 0;"
        f"border-radius:999px;background:{bg};color:{fg};font-size:0.82rem;font-weight:600;'>{label}</span>"
    )


def _pm_phase_badge_html(phase: Any) -> str:
    txt = str(phase or "").strip()
    if txt == "Despliegue inicial":
        return _badge_html("Despliegue inicial", bg="#1d4ed8")
    if txt == "Rebalanceo semanal":
        return _badge_html("Rebalanceo semanal", bg="#7c3aed")
    return _badge_html("-", bg="#6b7280")


def _pm_decision_badge_html(decision: Any) -> str:
    txt = str(decision or "").strip().lower()
    if txt == "seleccionado":
        return _badge_html("Seleccionado", bg="#15803d")
    if txt == "descartado":
        return _badge_html("Descartado", bg="#6b7280")
    return _badge_html(str(decision or "-"), bg="#6b7280")


def _pm_status_badge_html(status: Any) -> str:
    txt = str(status or "").strip().lower()
    mapping = {
        "executed": ("Ejecutada", "#15803d"),
        "selected": ("Pendiente envío", "#2563eb"),
        "rejected": ("Rechazada", "#dc2626"),
        "waiting_next_monday": ("Espera lunes", "#d97706"),
        "waiting_full_universe": ("Espera universo", "#d97706"),
        "already_open": ("Ya abierta", "#0f766e"),
        "discarded": ("Descartada", "#6b7280"),
        "closed": ("Cerrada", "#475569"),
        "close_rejected": ("Cierre rechazado", "#b91c1c"),
    }
    label, color = mapping.get(txt, (str(status or "-"), "#6b7280"))
    return _badge_html(label, bg=color)


def _ops_status_badge_html(is_operating: bool) -> str:
    if is_operating:
        return _badge_html("Operando", bg="#15803d")
    return _badge_html("Sin operación abierta", bg="#6b7280")


def _latest_order_badge_html(event: Dict[str, Any] | None) -> str:
    if not event:
        return _badge_html("Sin orden reciente", bg="#6b7280")
    payload = event.get("payload", {}) or {}
    result = payload.get("result", {}) or {}
    accepted = bool(result.get("accepted"))
    event_type = str(event.get("event_type") or "")
    if accepted:
        return _badge_html("Última orden aceptada", bg="#15803d")
    if event_type == "broker_order_rejected":
        return _badge_html("Última orden rechazada", bg="#dc2626")
    return _badge_html("Última orden registrada", bg="#2563eb")


def _pm_style_cell(value: Any, *, column: str) -> str:
    txt = str(value or "").strip()
    if column == "fase_pm":
        if txt == "Despliegue inicial":
            return "background-color: #dbeafe; color: #1e3a8a; font-weight: 600;"
        if txt == "Rebalanceo semanal":
            return "background-color: #ede9fe; color: #5b21b6; font-weight: 600;"
    if column == "decision_pm":
        if txt == "seleccionado":
            return "background-color: #dcfce7; color: #166534; font-weight: 600;"
        if txt == "descartado":
            return "background-color: #f3f4f6; color: #374151; font-weight: 600;"
    if column == "estado_orden":
        mapping = {
            "executed": "background-color: #dcfce7; color: #166534; font-weight: 700;",
            "selected": "background-color: #dbeafe; color: #1d4ed8; font-weight: 700;",
            "rejected": "background-color: #fee2e2; color: #991b1b; font-weight: 700;",
            "waiting_next_monday": "background-color: #fef3c7; color: #92400e; font-weight: 700;",
            "waiting_full_universe": "background-color: #fef3c7; color: #92400e; font-weight: 700;",
            "already_open": "background-color: #ccfbf1; color: #115e59; font-weight: 700;",
            "discarded": "background-color: #f3f4f6; color: #374151; font-weight: 700;",
            "closed": "background-color: #e2e8f0; color: #334155; font-weight: 700;",
            "close_rejected": "background-color: #fee2e2; color: #991b1b; font-weight: 700;",
        }
        return mapping.get(txt, "")
    return ""


def _build_live_signature(
    *,
    status: Dict[str, Any],
    all_events: List[Dict[str, Any]],
    all_states: List[Dict[str, Any]],
    backtests: Dict[str, Dict[str, Any]],
    pm_snapshot: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
    open_positions: List[Dict[str, Any]],
    pending_orders: List[Dict[str, Any]],
) -> str:
    last_event_ts = str(all_events[-1].get("occurred_at")) if all_events else "-"
    last_state_ts = str(all_states[0].get("updated_at")) if all_states else "-"
    bt_sig_parts: List[str] = []
    for trader_id in sorted(backtests.keys()):
        b = backtests.get(trader_id, {})
        bt_sig_parts.append(f"{trader_id}:{b.get('status')}:{b.get('updated_at')}")
    signal_book = list(pm_snapshot.get("signal_book", []) or [])
    last_output = dict(pm_snapshot.get("last_output", {}) or {})
    rebalance_rows = list(pm_snapshot.get("rebalance_rows", []) or [])
    monthly_refresh = dict(pm_snapshot.get("monthly_refresh", {}) or {})
    hr_status = dict(risk_snapshot.get("status", {}) or {})
    hr_runs = list(risk_snapshot.get("runs", []) or [])
    pending_retrain = list(risk_snapshot.get("pending_retrain_requests", []) or [])
    signal_sig = ",".join(
        sorted(
            f"{row.get('trader_id')}:{row.get('status')}:{row.get('ticket')}:{row.get('detected_at')}"
            for row in signal_book
        )
    )
    position_sig = ",".join(
        sorted(
            f"{row.get('ticket')}:{row.get('symbol')}:{row.get('volume')}:{row.get('price_open')}"
            for row in open_positions
        )
    )
    pending_sig = ",".join(
        sorted(
            f"{row.get('pending_key', '')}:{row.get('symbol')}:{row.get('side')}:{row.get('next_retry_at')}"
            for row in pending_orders
        )
    )
    return "|".join(
        [
            str(bool(status.get("running"))),
            str(bool(status.get("develop_enabled"))),
            str(status.get("developed_traders", 0)),
            str(bool(status.get("mt5_connected"))),
            str(bool(status.get("operational_runtime_started"))),
            str(status.get("current_stage", "")),
            str(status.get("current_asset", "")),
            str(len(all_states)),
            str(len(all_events)),
            last_event_ts,
            last_state_ts,
            ",".join(bt_sig_parts),
            str(last_output.get("status", "")),
            str(last_output.get("portfolio_phase", "")),
            str(len(signal_book)),
            signal_sig,
            str(len(open_positions)),
            position_sig,
            str(len(pending_orders)),
            pending_sig,
            str(len(rebalance_rows)),
            str(monthly_refresh.get("cutoff_date", "")),
            str(monthly_refresh.get("status", "")),
            str(monthly_refresh.get("mask_source", "")),
            str(monthly_refresh.get("last_manual_rebalance_at", "")),
            str(hr_status.get("last_run_at", "")),
            str(hr_status.get("last_status", "")),
            str(hr_status.get("last_force_run_at", "")),
            str(len(hr_runs)),
            str(len(pending_retrain)),
        ]
    )


def _mount_auto_refresh_watcher(*, enabled: bool, interval_ms: int, signature: str) -> None:
    if not enabled:
        st.session_state["_ui_last_signature"] = signature
        return

    run_every = f"{max(1, int(interval_ms) // 1000)}s"

    @st.fragment(run_every=run_every)
    def _watch_changes() -> None:
        last_sig = st.session_state.get("_ui_last_signature")
        if last_sig is None:
            st.session_state["_ui_last_signature"] = signature
            return
        if str(last_sig) != str(signature):
            st.session_state["_ui_last_signature"] = signature
            st.rerun()

    _watch_changes()


def _compute_backtest_metrics(bt: Dict[str, Any]) -> Dict[str, Any]:
    df = _prepare_backtest_chart_frame(bt.get("chart_rows", []) or [])
    if df.empty:
        return {}

    series_name = "equity" if "equity" in df.columns else ("balance" if "balance" in df.columns else None)
    if series_name is None:
        return {}
    eq_df = df[["date", series_name]].copy()
    eq_df[series_name] = pd.to_numeric(eq_df[series_name], errors="coerce")
    eq_df = eq_df.dropna(subset=[series_name]).sort_values("date").set_index("date")
    if eq_df.empty:
        return {}
    equity = eq_df[series_name]

    initial_capital = float(bt.get("initial_capital") or equity.iloc[0] or 1.0)
    if initial_capital == 0:
        initial_capital = 1.0
    final_equity = float(equity.iloc[-1])
    net_profit = final_equity - initial_capital
    return_pct = (net_profit / initial_capital) * 100.0

    days = max((df["date"].iloc[-1] - df["date"].iloc[0]).days, 1)
    years = max(days / 365.25, 1e-9)
    cagr_pct = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else float("nan")

    ret = equity.pct_change().dropna()
    mu = float(ret.mean()) if not ret.empty else 0.0
    sigma = float(ret.std(ddof=0)) if not ret.empty else 0.0
    sharpe = (mu / sigma) * math.sqrt(252.0) if sigma > 0 else 0.0
    downside = ret[ret < 0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    sortino = (mu / downside_std) * math.sqrt(252.0) if downside_std > 0 else 0.0
    vol_ann_pct = sigma * math.sqrt(252.0) * 100.0

    rolling_max = equity.cummax()
    dd = (equity / rolling_max) - 1.0
    max_dd_pct = float(dd.min() * 100.0) if not dd.empty else 0.0
    max_dd_abs = float((equity - rolling_max).min()) if not dd.empty else 0.0
    calmar = ((cagr_pct / 100.0) / abs(max_dd_pct / 100.0)) if max_dd_pct < 0 else 0.0
    ulcer_index = float((((dd * 100.0) ** 2).mean() ** 0.5)) if not dd.empty else 0.0
    recovery_factor = (net_profit / abs(max_dd_abs)) if max_dd_abs < 0 else 0.0
    # Duración de drawdowns en días naturales.
    dd_durations: List[int] = []
    in_dd = False
    dd_start = None
    for ts, is_dd in (dd < 0).items():
        if bool(is_dd) and not in_dd:
            in_dd = True
            dd_start = ts
        elif (not bool(is_dd)) and in_dd:
            in_dd = False
            dd_durations.append(max((ts - dd_start).days, 0))
            dd_start = None
    current_dd_duration_days = 0
    if in_dd and dd_start is not None:
        current_dd_duration_days = max((equity.index[-1] - dd_start).days, 0)
        dd_durations.append(current_dd_duration_days)
    max_dd_duration_days = max(dd_durations) if dd_durations else 0
    avg_dd_duration_days = (sum(dd_durations) / len(dd_durations)) if dd_durations else 0.0

    # Años positivos/negativos/planos.
    yearly_equity = equity.resample("YE").last()
    yearly_ret = yearly_equity.pct_change().dropna()
    positive_years = int((yearly_ret > 0).sum()) if not yearly_ret.empty else 0
    negative_years = int((yearly_ret < 0).sum()) if not yearly_ret.empty else 0
    flat_years = int((yearly_ret == 0).sum()) if not yearly_ret.empty else 0

    trade_stats = bt.get("trade_stats", {}) or {}
    total_trades = int(trade_stats.get("total_trades", bt.get("n_trades", 0) or 0))
    trades_per_year = (total_trades / years) if years > 0 else 0.0
    expectancy = float(trade_stats.get("expectancy", (net_profit / total_trades) if total_trades > 0 else 0.0))

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "net_profit": net_profit,
        "return_pct": return_pct,
        "cagr_pct": cagr_pct,
        "max_dd_pct": max_dd_pct,
        "max_dd_abs": max_dd_abs,
        "vol_ann_pct": vol_ann_pct,
        "ulcer_index": ulcer_index,
        "recovery_factor": recovery_factor,
        "max_dd_duration_days": int(max_dd_duration_days),
        "avg_dd_duration_days": float(avg_dd_duration_days),
        "current_dd_duration_days": int(current_dd_duration_days),
        "positive_years": positive_years,
        "negative_years": negative_years,
        "flat_years": flat_years,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "total_trades": total_trades,
        "winning_trades": trade_stats.get("winning_trades"),
        "losing_trades": trade_stats.get("losing_trades"),
        "win_rate_pct": trade_stats.get("win_rate_pct"),
        "profit_factor": trade_stats.get("profit_factor"),
        "payoff_ratio": trade_stats.get("payoff_ratio"),
        "avg_win": trade_stats.get("avg_win"),
        "avg_loss": trade_stats.get("avg_loss"),
        "expectancy": expectancy,
        "trades_per_year": trades_per_year,
        "avg_trade_duration_days": trade_stats.get("avg_trade_duration_days"),
        "min_trade_duration_days": trade_stats.get("min_trade_duration_days"),
        "max_trade_duration_days": trade_stats.get("max_trade_duration_days"),
        "max_winning_streak": trade_stats.get("max_winning_streak"),
        "max_losing_streak": trade_stats.get("max_losing_streak"),
        "max_win_trade": trade_stats.get("max_win_trade"),
        "max_loss_trade": trade_stats.get("max_loss_trade"),
    }


def _prepare_backtest_chart_frame(chart_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not chart_rows:
        return pd.DataFrame()
    df = pd.DataFrame(chart_rows)
    if df.empty:
        return df
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def main() -> None:
    st.set_page_config(page_title="TFM Multiagent Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.0rem;
            padding-left: 4.5rem;
            padding-right: 4.5rem;
            max-width: 1700px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Dashboard Multiagente TFM")
    st.write("Desarrollo y operativa en tiempo real (MT5 D1).")

    supervisor = _get_supervisor()

    status = supervisor.get_status()
    section_options = ["Desarrollo", "Backtest", "Operativa", "Portfolio manager", "Recursos Humanos"]
    selected_section = st.radio(
        "Sección",
        options=section_options,
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_section",
    )

    need_events = selected_section in {"Desarrollo", "Backtest", "Operativa"}
    need_states = True
    need_backtests = selected_section == "Backtest"
    need_pm_history = selected_section == "Portfolio manager"
    need_hr_history = selected_section == "Recursos Humanos"
    need_open_positions = bool(status.get("mt5_connected")) and selected_section == "Operativa"

    all_events = _load_events(Path(supervisor.db_path), int(DEFAULT_EVENT_LIMIT)) if need_events else []
    all_states = _load_trader_states(Path(supervisor.db_path)) if need_states else []
    backtests = supervisor.get_backtest_registry() if (need_backtests and hasattr(supervisor, "get_backtest_registry")) else {}
    pm_snapshot = (
        supervisor.get_portfolio_manager_snapshot(include_history=need_pm_history)
        if hasattr(supervisor, "get_portfolio_manager_snapshot")
        else {}
    )
    hr_snapshot = (
        supervisor.get_human_resources_snapshot(include_history=need_hr_history)
        if hasattr(supervisor, "get_human_resources_snapshot")
        else {}
    )
    pending_orders = list(pm_snapshot.get("pending_orders", []) or [])
    open_positions = _safe_mt5_positions(supervisor) if need_open_positions else []
    signal_book = list(pm_snapshot.get("signal_book", []) or [])
    signal_audit = list(pm_snapshot.get("signal_audit", []) or [])
    last_output = dict(pm_snapshot.get("last_output", {}) or {})
    rebalance_rows = list(pm_snapshot.get("rebalance_rows", []) or [])
    monthly_refresh = dict(pm_snapshot.get("monthly_refresh", {}) or {})
    backtest_runs = list(pm_snapshot.get("backtest_runs", []) or [])
    normalized_pm_signals = _normalize_pm_signal_rows(signal_book, signal_audit)
    live_rebalance_rows = [dict(row) for row in rebalance_rows if _is_live_portfolio_rebalance(dict(row))]
    live_signature = _build_live_signature(
        status=status,
        all_events=all_events,
        all_states=all_states,
        backtests=backtests,
        pm_snapshot=pm_snapshot,
        risk_snapshot=hr_snapshot,
        open_positions=open_positions,
        pending_orders=pending_orders,
    )
    auto_refresh_enabled = selected_section != "Desarrollo"
    refresh_ms = 3000 if selected_section == "Operativa" else int(DEFAULT_AUTO_REFRESH_MS)
    _mount_auto_refresh_watcher(enabled=auto_refresh_enabled, interval_ms=refresh_ms, signature=live_signature)
    st.caption(f"DB: `{supervisor.db_path}`")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Supervisor", "running" if bool(status.get("running")) else "stopped")
    c2.metric("Desarrollo", "activo" if bool(status.get("develop_enabled")) else "parado")
    # Fuente robusta para UI: DB persistida, no solo estado en memoria.
    c3.metric("Traders desarrollados", int(len(all_states)))
    runtime_mode = str(status.get("operational_runtime_mode") or "-")
    mt5_label = "conectado" if bool(status.get("mt5_connected")) else f"desconectado ({runtime_mode})"
    c4.metric("MT5", mt5_label)
    c5.metric("Señales PM", int(len(signal_book)))
    c6.metric("Posiciones abiertas", int(len(open_positions)))
    c7.metric("Retries pendientes", int(len(pending_orders)))
    st.caption(f"Objetivo traders: {int(status.get('target_traders', 8))}")
    c_global_1, c_global_2, c_global_3 = st.columns([1, 5, 1])
    with c_global_1:
        if st.button("Refrescar dashboard", key="btn_global_refresh"):
            st.rerun()

    if selected_section == "Desarrollo":
        st.markdown("### Controles de desarrollo")
        pre_status = supervisor.get_status()
        c_ctl_1, c_ctl_2 = st.columns([1, 2])
        with c_ctl_1:
            target_traders = st.number_input(
                "Objetivo de traders a desarrollar",
                min_value=1,
                max_value=200,
                value=int(pre_status.get("target_traders", 8)),
                step=1,
                key="dev_target_traders",
            )
            if int(pre_status.get("target_traders", 8)) != int(target_traders):
                supervisor.set_target_traders(int(target_traders))
                st.rerun()
        with c_ctl_2:
            b1, b2, b3 = st.columns(3)
            if b1.button("Iniciar desarrollo de agentes trader", key="btn_dev_start"):
                supervisor.start()
                st.success("Desarrollo iniciado.")
                st.rerun()
            if b2.button("Parar desarrollo de agentes trader", key="btn_dev_stop"):
                supervisor.stop_development()
                st.info("Desarrollo detenido.")
                st.rerun()
            if b3.button("Borrar todos los traders y reiniciar", key="btn_dev_reset"):
                supervisor.reset_all()
                st.warning("Sistema reiniciado.")
                st.rerun()

        st.markdown("### Estado actual de desarrollo")
        current_asset = status.get("current_asset")
        current_stage = status.get("current_stage")
        current_steps = [str(x) for x in list(status.get("current_cycle_steps", []))]
        if bool(status.get("develop_enabled")):
            if current_asset:
                st.info(f"Se esta desarrollando un agente sobre `{current_asset}` (etapa: `{current_stage}`).")
            else:
                st.info("Desarrollo activo: seleccionando activo y configuracion.")
        else:
            last_asset = status.get("last_cycle_asset")
            last_trader = status.get("last_cycle_trader_id")
            if last_asset:
                if last_trader:
                    st.success(
                        f"Desarrollo finalizado sobre `{last_asset}`. Trader creado: "
                        f"`{_pretty_trader_name(last_trader, asset=last_asset, timeframe='D1')}`."
                    )
                else:
                    st.warning(f"Desarrollo finalizado sobre `{last_asset}` sin trader promovido.")
            else:
                st.info("No hay desarrollos en curso.")

        if current_steps:
            st.markdown("**Progreso del ciclo actual**")
            human_steps = {
                "data_process": "1) DataProcess",
                "data_agent": "1) DataProcess",
                "developer_agent": "2) DeveloperAgent",
                "validation_agent": "3) ValidationAgent",
                "trader_agent": "4) TraderAgent",
                "backtest_agent": "5) BacktestAgent",
            }
            for step in current_steps:
                st.markdown(f"- {human_steps.get(step, step)}")

        session_events = _filter_events_by_session(
            all_events,
            session_started_at=(status.get("development_session_started_at") or None),
        )
        # 1) preferimos ciclos completados (con trader_promoted)
        visible_dev_events = _filter_completed_dev_cycles(_filter_dev_events(session_events))
        dev_source_events = _filter_dev_events(session_events)
        # 2) fallback: si todavía no hay promoción, mostrar eventos de sesión en curso
        if not visible_dev_events:
            visible_dev_events = _filter_dev_events(session_events)
            dev_source_events = visible_dev_events
        # 3) fallback extra: si la sesión quedó desincronizada, mostrar últimos eventos globales
        if not visible_dev_events:
            visible_dev_events = _filter_dev_events(all_events)
            dev_source_events = visible_dev_events
        _render_pretty_dev_events(
            visible_dev_events,
            title="Eventos de desarrollo (Data -> Developer -> Validation -> Trader)",
            max_items=int(DEFAULT_EVENT_LIMIT),
            source_events=dev_source_events,
        )
        states = list(all_states)
        if states:
            for row in states:
                row["updated_at"] = _fmt_ts(str(row.get("updated_at", "")))
                row["trader"] = _pretty_trader_from_row(row)
            st.markdown("### Traders desarrollados")
            st.table(pd.DataFrame(states)[["trader", "asset", "timeframe", "state", "updated_at"]])

    if selected_section == "Backtest":
        st.markdown("### Backtests por trader")
        states = list(all_states)
        promoted_rules = _build_promoted_rules_index(all_events)
        if not states:
            st.info("Todavia no hay traders desarrollados.")
        else:
            for row in states:
                trader_id = str(row.get("trader_id"))
                asset = str(row.get("asset"))
                bt = backtests.get(trader_id, {})
                pretty_trader = _pretty_trader_from_row(row)
                with st.expander(f"{pretty_trader} ({asset})", expanded=False):
                    st.markdown("**1) Reglas de entrada**")
                    fallback = promoted_rules.get(trader_id, {})
                    long_rules = list(bt.get("long_rules", fallback.get("long_rules", [])))
                    short_rules = list(bt.get("short_rules", fallback.get("short_rules", [])))
                    if long_rules:
                        st.markdown("LONG")
                        for r in long_rules:
                            st.markdown(f"- `{r}`")
                    else:
                        st.markdown("- LONG: sin reglas")
                    if short_rules:
                        st.markdown("SHORT")
                        for r in short_rules:
                            st.markdown(f"- `{r}`")
                    else:
                        st.markdown("- SHORT: sin reglas")

                    st.markdown("**2) Grafica resultados (historico completo)**")
                    status_bt = str(bt.get("status", "pending"))
                    if status_bt == "ready":
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Trades", int(bt.get("n_trades", 0)))
                        c2.metric("Balance final", f"{float(bt.get('final_balance', 0.0)):.2f}" if bt.get("final_balance") is not None else "-")
                        c3.metric("Equity final", f"{float(bt.get('final_equity', 0.0)):.2f}" if bt.get("final_equity") is not None else "-")
                        st.caption(
                            f"Periodo backtest: {bt.get('start_date', '-')} -> {bt.get('end_date', '-')}"
                        )
                        chart_rows = bt.get("chart_rows", []) or []
                        if chart_rows:
                            chart_df = _prepare_backtest_chart_frame(chart_rows)
                            y_cols = [c for c in ["equity", "balance"] if c in chart_df.columns]
                            if y_cols and not chart_df.empty:
                                # Visualización relativa (%), para ver mejor fluctuaciones
                                # incluso cuando el capital absoluto varía poco.
                                base_capital = float(bt.get("initial_capital") or chart_df[y_cols[0]].iloc[0] or 1.0)
                                if base_capital == 0:
                                    base_capital = 1.0
                                for col in y_cols:
                                    chart_df[f"{col}_pct"] = ((chart_df[col] / base_capital) - 1.0) * 100.0

                                pct_cols = [f"{c}_pct" for c in y_cols]
                                pct_min = float(chart_df[pct_cols].min().min())
                                pct_max = float(chart_df[pct_cols].max().max())
                                if pct_min == pct_max:
                                    pad = 0.1
                                else:
                                    pad = max((pct_max - pct_min) * 0.15, 0.05)
                                y_domain = [pct_min - pad, pct_max + pad]

                                chart_long = chart_df.melt(
                                    id_vars=["date"],
                                    value_vars=pct_cols,
                                    var_name="serie",
                                    value_name="pct",
                                )
                                chart_long["serie"] = chart_long["serie"].str.replace("_pct", "", regex=False)
                                rel_chart = (
                                    alt.Chart(chart_long)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("date:T", title="Fecha"),
                                        y=alt.Y("pct:Q", title="Variación sobre capital inicial (%)", scale=alt.Scale(domain=y_domain)),
                                        color=alt.Color("serie:N", title="Serie"),
                                        tooltip=[
                                            alt.Tooltip("date:T", title="Fecha"),
                                            alt.Tooltip("serie:N", title="Serie"),
                                            alt.Tooltip("pct:Q", title="% variación", format=".4f"),
                                        ],
                                    )
                                    .properties(height=320)
                                )
                                st.altair_chart(rel_chart, width="stretch")
                                metrics = _compute_backtest_metrics(bt)
                                if metrics:
                                    st.markdown("**Resumen cuantitativo**")
                                    m1, m2, m3, m4 = st.columns(4)
                                    m1.metric("Net Profit", f"{metrics['net_profit']:.2f}")
                                    m2.metric("Return %", f"{metrics['return_pct']:.2f}%")
                                    m3.metric("Max Drawdown", f"{metrics['max_dd_pct']:.2f}%")
                                    m4.metric("Sharpe", f"{metrics['sharpe']:.3f}")

                                    with st.expander("Ver métricas en profundidad", expanded=False):
                                        c1, c2, c3, c4 = st.columns(4)
                                        with c1:
                                            st.caption("Rendimiento")
                                            st.caption(f"CAGR: {metrics['cagr_pct']:.3f}%")
                                            st.caption(f"Initial capital: {metrics['initial_capital']:.2f}")
                                            st.caption(f"Final equity: {metrics['final_equity']:.2f}")
                                            st.caption(f"Años positivos: {metrics['positive_years']}")
                                            st.caption(f"Años negativos: {metrics['negative_years']}")
                                            st.caption(f"Años planos: {metrics['flat_years']}")
                                        with c2:
                                            st.caption("Riesgo")
                                            st.caption(f"Max DD %: {metrics['max_dd_pct']:.3f}%")
                                            st.caption(f"Max DD abs: {metrics['max_dd_abs']:.2f}")
                                            st.caption(f"Duración DD max (días): {metrics['max_dd_duration_days']}")
                                            st.caption(f"Duración DD media (días): {metrics['avg_dd_duration_days']:.2f}")
                                            st.caption(f"Duración DD actual (días): {metrics['current_dd_duration_days']}")
                                            st.caption(f"Vol anual: {metrics['vol_ann_pct']:.3f}%")
                                            st.caption(f"Ulcer index: {metrics['ulcer_index']:.3f}")
                                            st.caption(f"Recovery factor: {metrics['recovery_factor']:.3f}")
                                        with c3:
                                            st.caption("Ratios")
                                            st.caption(f"Sharpe: {metrics['sharpe']:.3f}")
                                            st.caption(f"Sortino: {metrics['sortino']:.3f}")
                                            st.caption(f"Calmar: {metrics['calmar']:.3f}")
                                            if metrics.get("profit_factor") is not None:
                                                st.caption(f"Profit factor: {float(metrics['profit_factor']):.3f}")
                                            if metrics.get("payoff_ratio") is not None:
                                                st.caption(f"Payoff ratio: {float(metrics['payoff_ratio']):.3f}")
                                        with c4:
                                            st.caption("Trades")
                                            st.caption(f"Total: {metrics['total_trades']}")
                                            if metrics.get("winning_trades") is not None:
                                                st.caption(f"Winners: {metrics['winning_trades']}")
                                            if metrics.get("losing_trades") is not None:
                                                st.caption(f"Losers: {metrics['losing_trades']}")
                                            if metrics.get("win_rate_pct") is not None:
                                                st.caption(f"Win rate: {float(metrics['win_rate_pct']):.2f}%")
                                            st.caption(f"Trades/año: {metrics['trades_per_year']:.2f}")
                                            st.caption(f"Expectancy: {metrics['expectancy']:.3f}")
                                            if metrics.get("avg_win") is not None:
                                                st.caption(f"Avg win: {float(metrics['avg_win']):.3f}")
                                            if metrics.get("avg_loss") is not None:
                                                st.caption(f"Avg loss: {float(metrics['avg_loss']):.3f}")
                                            if metrics.get("avg_trade_duration_days") is not None:
                                                st.caption(f"Duración media trade (días): {float(metrics['avg_trade_duration_days']):.2f}")
                                            if metrics.get("min_trade_duration_days") is not None:
                                                st.caption(f"Duración mínima trade (días): {float(metrics['min_trade_duration_days']):.2f}")
                                            if metrics.get("max_trade_duration_days") is not None:
                                                st.caption(f"Duración máxima trade (días): {float(metrics['max_trade_duration_days']):.2f}")
                                            if metrics.get("max_winning_streak") is not None:
                                                st.caption(f"Racha máxima ganadora: {int(metrics['max_winning_streak'])}")
                                            if metrics.get("max_losing_streak") is not None:
                                                st.caption(f"Racha máxima perdedora: {int(metrics['max_losing_streak'])}")
                                            if metrics.get("max_win_trade") is not None:
                                                st.caption(f"Mayor ganancia trade: {float(metrics['max_win_trade']):.3f}")
                                            if metrics.get("max_loss_trade") is not None:
                                                st.caption(f"Mayor pérdida trade: {float(metrics['max_loss_trade']):.3f}")
                            else:
                                st.info("No hay columnas de equity/balance para graficar.")
                        else:
                            st.info("Backtest completado sin serie de resultados.")
                    elif status_bt == "running":
                        st.info("Backtest en ejecucion. Esta pestaña se actualizara automaticamente.")
                    elif status_bt == "error":
                        st.error(f"Backtest con error: {bt.get('error', 'desconocido')}")
                    else:
                        st.info("Backtest pendiente. Se ejecutara al crear/promover el trader.")

    if selected_section == "Portfolio manager":
        st.markdown("### Portfolio manager (GA + PSO)")
        pm_btn_1, _ = st.columns(2)
        if pm_btn_1.button("Forzar rebalanceo ahora", key="btn_pm_force_rebalance"):
            out = supervisor.force_portfolio_rebalance() if hasattr(supervisor, "force_portfolio_rebalance") else {}
            refresh_status = str(((out.get("refresh") or {}).get("status")) or "")
            rebalance_status = str(((out.get("rebalance") or {}).get("status")) or "")
            rebalance_reason = str(((out.get("rebalance") or {}).get("reason")) or "")
            st.session_state["pm_manual_action_message"] = (
                f"Rebalanceo ejecutado. Refresh=`{refresh_status or '-'}` | "
                f"rebalanceo=`{rebalance_status or '-'}`"
                f"{f' | motivo=`{rebalance_reason}`' if rebalance_reason else ''}."
            )
            st.rerun()
        if st.session_state.get("pm_manual_action_message"):
            st.success(str(st.session_state.get("pm_manual_action_message")))
            if st.button("Limpiar aviso portfolio", key="btn_pm_clear_manual_msg"):
                st.session_state.pop("pm_manual_action_message", None)
                st.rerun()

        decision = dict(last_output.get("decision") or {})
        decision_metadata = dict(decision.get("metadata") or {})
        ga_config = dict(decision_metadata.get("ga_config") or {})
        pso_config = dict(decision_metadata.get("pso_config") or {})
        baselines = dict(decision_metadata.get("baselines") or {}) or dict(last_output.get("baselines") or {})
        ew_baseline = dict(baselines.get("equal_weight_all_valid") or {})
        pm_status = str(last_output.get("status") or "")
        pm_phase = _interpret_pm_phase(last_output.get("portfolio_phase"))

        s1, s2, s3, s4, s5, s6, s7, s8 = st.columns(8)
        s1.metric("Modo", str(last_output.get("optimizer_mode") or decision.get("optimizer_mode") or "ga_pso").upper())
        s2.metric("Fase PM", pm_phase)
        s3.metric("Estado PM", pm_status or "-")
        s4.metric("Activos / validos", f"{int(last_output.get('active_universe_size') or 0)} / {int(last_output.get('valid_universe_size') or 0)}")
        s5.metric("Seleccionados", int(last_output.get("selected_universe_size") or decision.get("selected_universe_size") or 0))
        s6.metric("Cash target", f"{float(last_output.get('target_cash_weight') or decision.get('target_cash_weight') or 0.0) * 100.0:.2f}%")
        s7.metric("Fitness", f"{float(last_output.get('fitness') or decision.get('fitness') or 0.0):.4f}")
        s8.metric("Tiempo (s)", f"{float(last_output.get('execution_time_seconds') or decision_metadata.get('execution_time_seconds') or 0.0):.2f}")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Sharpe neto", f"{float(last_output.get('sharpe_neto') or decision.get('sharpe_neto') or 0.0):.4f}")
        m2.metric("MDD", f"{float(last_output.get('mdd') or decision.get('mdd') or 0.0) * 100.0:.2f}%")
        m3.metric("Corr media", f"{float(last_output.get('corr_media') or decision.get('corr_media') or 0.0):.4f}")
        m4.metric("lambda_dd", f"{float(decision_metadata.get('lambda_dd') or 0.0):.2f}")
        m5.metric("lambda_corr", f"{float(decision_metadata.get('lambda_corr') or 0.0):.2f}")
        m6.metric("Lookback (semanas)", int(decision_metadata.get("lookback_weeks") or 0))

        if ew_baseline:
            st.caption(
                "Baseline equal weight (todos los traders validos): "
                f"fitness=`{float(ew_baseline.get('fitness') or 0.0):.4f}` | "
                f"Sharpe=`{float(ew_baseline.get('sharpe_neto') or 0.0):.4f}` | "
                f"MDD=`{float(ew_baseline.get('mdd') or 0.0) * 100.0:.2f}%` | "
                f"Corr media=`{float(ew_baseline.get('corr_media') or 0.0):.4f}`"
            )
        excluded_traders = list(decision_metadata.get("excluded_traders") or [])
        if excluded_traders:
            st.caption(
                f"Traders excluidos por falta de historico ({len(excluded_traders)}): "
                + ", ".join(_pretty_trader_name(t) for t in excluded_traders[:20])
                + ("..." if len(excluded_traders) > 20 else "")
            )

        with st.expander("Configuracion GA / PSO", expanded=False):
            cg1, cg2 = st.columns(2)
            with cg1:
                st.markdown("**GA**")
                if ga_config:
                    st.dataframe(pd.DataFrame([ga_config]).T.rename(columns={0: "valor"}), width="stretch")
                else:
                    st.info("No hay configuracion GA registrada todavia.")
            with cg2:
                st.markdown("**PSO**")
                if pso_config:
                    st.dataframe(pd.DataFrame([pso_config]).T.rename(columns={0: "valor"}), width="stretch")
                else:
                    st.info("No hay configuracion PSO registrada todavia.")

        st.caption(
            f"Ultimo refresh mensual: {str(monthly_refresh.get('last_refresh_at') or '-')}"
            f" | Fecha de corte: {str(monthly_refresh.get('cutoff_date') or '-')}"
            f" | Traders refrescados: {int(monthly_refresh.get('n_traders') or 0)}"
        )
        st.caption(
            f"Estado refresh backtests: {str(monthly_refresh.get('backtests_status') or '-')}"
            f" | Estado refresh mensual: {str(monthly_refresh.get('status') or '-')}"
            f" | Ultimo rebalanceo manual: {str(monthly_refresh.get('last_manual_rebalance_at') or '-')}"
        )

        st.markdown(
            "".join(
                [
                    _pm_phase_badge_html(pm_phase),
                    _pm_decision_badge_html("seleccionado"),
                    _pm_decision_badge_html("descartado"),
                    _pm_status_badge_html("executed"),
                    _pm_status_badge_html("selected"),
                    _pm_status_badge_html("waiting_next_monday"),
                    _pm_status_badge_html("rejected"),
                ]
            ),
            unsafe_allow_html=True,
        )

        # Curva historica de la cartera GA+PSO vs equal-weight (de la ultima ejecucion).
        portfolio_curve = last_output.get("portfolio_curve")
        weights_for_chart = dict(last_output.get("weights") or decision.get("weights") or {})
        if isinstance(portfolio_curve, pd.DataFrame) and not portfolio_curve.empty:
            curve_chart = _sanitize_line_chart_frame(portfolio_curve.copy() * 100.0)
            if not curve_chart.empty:
                st.markdown("**Curva acumulada (cartera vs. equal weight)**")
                st.line_chart(curve_chart, width="stretch")
        elif weights_for_chart:
            figs = _build_portfolio_curve_figures(
                portfolio_curve=pd.DataFrame(),
                latest_weights=weights_for_chart,
            )
            for fig in figs:
                st.pyplot(fig, clear_figure=False, width="stretch")

        if weights_for_chart:
            top = sorted(weights_for_chart.items(), key=lambda kv: float(kv[1]), reverse=True)[:20]
            top_df = pd.DataFrame(
                [
                    {
                        "Trader": _pretty_trader_name(tid),
                        "Peso (%)": float(w) * 100.0,
                        "Capital (EUR)": float(dict(last_output.get("euros") or {}).get(tid, 0.0)),
                    }
                    for tid, w in top
                ]
            )
            st.markdown("**Top traders por peso (ultima decision)**")
            st.dataframe(top_df, width="stretch", hide_index=True)

        # Senales de la ultima rebalanceo
        _render_pm_signal_tables(normalized_pm_signals, key_prefix="pm_ga_pso")

        if backtest_runs:
            st.markdown("**Refresh mensual de backtests promovidos**")
            df_runs = pd.DataFrame(backtest_runs)
            if not df_runs.empty:
                df_runs = df_runs.sort_values(["cutoff_date", "updated_at"], ascending=[False, False]).head(20).copy()
                df_runs["trader"] = df_runs.apply(
                    lambda row: _pretty_trader_name(row.get("trader_id"), asset=row.get("asset"), timeframe=row.get("timeframe")),
                    axis=1,
                )
                df_runs["trade_count"] = df_runs["summary"].map(lambda x: (x or {}).get("n_trades"))
                st.dataframe(
                    df_runs[["trader", "asset", "timeframe", "cutoff_date", "status", "trade_count", "updated_at"]],
                    width="stretch",
                    hide_index=True,
                )
                hr_rows_by_trader = {
                    str(row.get("trader_id") or ""): dict(row)
                    for row in list(hr_snapshot.get("trader_rows", []) or [])
                }
                for _, bt_row in df_runs.iterrows():
                    trader_id = str(bt_row.get("trader_id") or "")
                    title = f"{str(bt_row.get('trader') or trader_id)} | refresh {str(bt_row.get('cutoff_date') or '-')}"
                    with st.expander(title, expanded=False):
                        dev_curve = supervisor.get_trader_history_frame(trader_id) if hasattr(supervisor, "get_trader_history_frame") else None
                        forward_curve = pd.DataFrame()
                        promoted_at = ""
                        hr_row = hr_rows_by_trader.get(trader_id, {})
                        if hr_row:
                            promoted_at = str(hr_row.get("promoted_at") or "")
                            forward_run = dict(hr_row.get("forward_run") or {})
                            forward_pnl_path = str((forward_run.get("artifact_paths") or {}).get("historical_pnl_path") or "")
                            if forward_pnl_path:
                                try:
                                    forward_curve = pd.read_csv(forward_pnl_path)
                                except Exception:
                                    forward_curve = pd.DataFrame()
                        if not promoted_at:
                            promoted_at = str(bt_row.get("updated_at") or "")
                        _render_dev_forward_equity_chart(
                            development_curve=dev_curve,
                            forward_curve=forward_curve,
                            promoted_at=promoted_at,
                        )

        if live_rebalance_rows:
            st.markdown("**Historico de rebalanceos GA+PSO**")
            df_reb = pd.DataFrame(live_rebalance_rows)
            if not df_reb.empty:
                table_rows: list[Dict[str, Any]] = []
                for _, row in df_reb.iterrows():
                    metadata_map = dict(row.get("metadata") or {})
                    weights_map = dict(row.get("target_weights") or {})
                    top_weights = sorted(weights_map.items(), key=lambda x: float(x[1]), reverse=True)[:5]
                    table_rows.append(
                        {
                            "rebalance_date": row.get("rebalance_date"),
                            "n_active": len(row.get("active_traders") or []),
                            "n_valid": int(metadata_map.get("valid_universe_size") or 0),
                            "n_selected": len(row.get("selected_traders") or []),
                            "cash": float(row.get("target_cash_weight") or 0.0),
                            "fitness": float(metadata_map.get("fitness") or 0.0),
                            "sharpe_neto": float(metadata_map.get("sharpe_neto") or 0.0),
                            "mdd": float(metadata_map.get("mdd") or 0.0),
                            "corr_media": float(metadata_map.get("corr_media") or 0.0),
                            "top_selected_traders": ", ".join([_pretty_trader_name(k) for k, _ in top_weights]),
                            "top_weights": ", ".join([f"{_pretty_trader_name(k)}={float(v) * 100.0:.1f}%" for k, v in top_weights]),
                        }
                    )
                table_df = pd.DataFrame(table_rows).sort_values("rebalance_date", ascending=False).reset_index(drop=True)
                st.dataframe(table_df, width="stretch", hide_index=True)

    if selected_section == "Recursos Humanos":
        st.markdown("### Recursos Humanos (HumanResourcesProcess)")
        st.caption(
            "Este proceso compara el comportamiento forward de cada trader promovido con su perfil "
            "de diseno y decide si sigue valido (KEEP) o si debe pasar a reentrenamiento (RETRAINING)."
        )
        hr_runs = list(hr_snapshot.get("runs", []) or [])
        latest_hr_run = dict(hr_snapshot.get("latest_run", {}) or {})
        hr_rows = list(hr_snapshot.get("trader_rows", []) or [])
        hr_details = list(hr_snapshot.get("details", []) or [])
        retrain_requests = list(hr_snapshot.get("retrain_requests", []) or [])
        pending_retrain_requests = list(hr_snapshot.get("pending_retrain_requests", []) or [])
        hr_status = dict(hr_snapshot.get("status", {}) or {})

        r_btn_1, r_btn_2, r_btn_3 = st.columns(3)
        if r_btn_1.button("Forzar revision de salud ahora", key="btn_hr_force_eval"):
            out = supervisor.force_trader_health_evaluation() if hasattr(supervisor, "force_trader_health_evaluation") else {}
            st.session_state["hr_manual_action_message"] = (
                f"Revision de salud ejecutada. Estado=`{str(out.get('status') or '-')}`."
            )
            st.rerun()
        if r_btn_2.button("Forzar backtest forward de todos los traders", key="btn_hr_force_backtest"):
            out = supervisor.force_trader_health_evaluation(force_backtest=True) if hasattr(supervisor, "force_trader_health_evaluation") else {}
            st.session_state["hr_manual_action_message"] = (
                f"Backtest forward + revision de salud ejecutados. Estado=`{str(out.get('status') or '-')}`."
            )
            st.rerun()
        if r_btn_3.button("Procesar RetrainRequests pendientes", key="btn_hr_process_retrain"):
            out = supervisor.process_pending_retrain_requests() if hasattr(supervisor, "process_pending_retrain_requests") else {}
            st.session_state["hr_manual_action_message"] = (
                f"RetrainRequests procesadas. OK=`{len(list(out.get('processed') or []))}` | errores=`{len(list(out.get('failed') or []))}`."
            )
            st.rerun()
        if st.session_state.get("hr_manual_action_message"):
            st.success(str(st.session_state.get("hr_manual_action_message")))
            if st.button("Limpiar aviso", key="btn_hr_clear_manual_msg"):
                st.session_state.pop("hr_manual_action_message", None)
                st.rerun()

        live_count = sum(1 for row in hr_rows if str(row.get("current_state")) == "live")
        retraining_count = sum(1 for row in hr_rows if str(row.get("current_state")) == "retraining")
        insufficient_count = sum(1 for row in hr_rows if bool((row.get("forward_metrics") or {}).get("insufficient_evidence")))
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Ultima revision", _fmt_ts(str(hr_status.get("last_run_at") or latest_hr_run.get("completed_at") or "-")))
        k2.metric("Proxima revision", "Dias 1-3 / 30 dias")
        k3.metric("LIVE", live_count)
        k4.metric("RETRAINING", retraining_count)
        k5.metric("Retrain pendientes", len(pending_retrain_requests))
        k6.metric("Evidencia insuficiente", insufficient_count)
        st.caption(
            f"Estado revision: {str(hr_status.get('last_status') or latest_hr_run.get('status') or '-')}"
            f" | Run ID: {str(hr_status.get('last_run_id') or latest_hr_run.get('run_id') or '-')}"
            f" | Traders evaluados: {int(hr_status.get('last_traders') or latest_hr_run.get('evaluated_traders') or 0)}"
        )

        if hr_rows:
            summary_rows: list[Dict[str, Any]] = []
            for row in hr_rows:
                metrics = dict(row.get("forward_metrics") or {})
                profile = dict(row.get("design_profile") or {})
                summary_rows.append(
                    {
                        "Trader": _pretty_trader_name(row.get("trader_id"), asset=row.get("asset"), timeframe=row.get("timeframe")),
                        "Activo": row.get("asset"),
                        "Timeframe": row.get("timeframe"),
                        "Promoted at": _fmt_ts(str(row.get("promoted_at") or "")),
                        "Estado actual": _human_trader_state(row.get("current_state")),
                        "Health score": round(float(row.get("health_score") or 0.0), 2),
                        "Ultima accion": _human_review_action(row.get("action")),
                        "Shadow trades": int(metrics.get("shadow_trades") or 0),
                        "Executed trades": int(metrics.get("executed_trades") or 0),
                        "Signal count": int(metrics.get("signal_count") or 0),
                        "PM selected count": int(metrics.get("pm_selected_count") or 0),
                        "Sharpe diseno": profile.get("sharpe_design"),
                        "Sharpe forward": metrics.get("shadow_sharpe"),
                        "PF diseno": profile.get("profit_factor_design"),
                        "PF forward": metrics.get("shadow_profit_factor"),
                        "Max DD diseno": profile.get("max_drawdown_design"),
                        "Max DD forward": metrics.get("shadow_max_drawdown"),
                        "Avg loss diseno": profile.get("avg_loss_design"),
                        "Avg loss forward": metrics.get("shadow_avg_loss"),
                        "Losing streak diseno": profile.get("max_losing_streak_design"),
                        "Losing streak forward": metrics.get("shadow_losing_streak"),
                        "Ultima revision": _fmt_ts(str(row.get("latest_evaluation") or "")),
                        "Motivo principal": row.get("main_reason"),
                    }
                )
            with st.expander("Resumen de traders evaluados", expanded=False):
                st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

            for row in hr_rows:
                trader_id = str(row.get("trader_id") or "")
                asset = str(row.get("asset") or "")
                timeframe = str(row.get("timeframe") or "D1")
                profile = dict(row.get("design_profile") or {})
                metrics = dict(row.get("forward_metrics") or {})
                detail = dict(row.get("review_detail") or {})
                forward_run = dict(row.get("forward_run") or {})
                title = f"{_pretty_trader_name(trader_id, asset=asset, timeframe=timeframe)} | {_human_trader_state(row.get('current_state'))} | score={round(float(row.get('health_score') or 0.0), 2)}"
                with st.expander(title, expanded=False):
                    i1, i2, i3, i4 = st.columns(4)
                    i1.metric("Estado", _human_trader_state(row.get("current_state")))
                    i2.metric("Ultima accion", _human_review_action(row.get("action")))
                    i3.metric("Health score", f"{float(row.get('health_score') or 0.0):.2f}")
                    i4.metric("Promocion", _fmt_short_date(row.get("promoted_at")))
                    st.caption(f"Ultima revision: {_fmt_ts(str(row.get('latest_evaluation') or ''))}")
                    for reason in list(detail.get("reasons") or []):
                        st.markdown(f"- {reason}")

                    dev_curve = supervisor.get_trader_history_frame(trader_id) if hasattr(supervisor, "get_trader_history_frame") else None
                    fwd_curve = pd.DataFrame()
                    forward_pnl_path = str((forward_run.get("artifact_paths") or {}).get("historical_pnl_path") or "")
                    if forward_pnl_path:
                        try:
                            fwd_curve = pd.read_csv(forward_pnl_path)
                        except Exception:
                            fwd_curve = pd.DataFrame()
                    aligned = align_development_and_forward_curves(dev_curve, fwd_curve, promoted_at=str(row.get("promoted_at") or ""))
                    if not aligned.empty and "date" in aligned.columns:
                        plot_long = aligned.melt(
                            id_vars=[c for c in ["date", "promotion_marker"] if c in aligned.columns],
                            value_vars=[c for c in ["development_equity", "forward_equity"] if c in aligned.columns],
                            var_name="serie",
                            value_name="equity",
                        ).dropna(subset=["equity"])
                        base_chart = alt.Chart(plot_long).mark_line().encode(
                            x=alt.X("date:T", title="Fecha"),
                            y=alt.Y("equity:Q", title="Equity / Balance"),
                            color=alt.Color("serie:N", title="Curva"),
                            tooltip=["date:T", "serie:N", "equity:Q"],
                        ).properties(height=300)
                        if "promotion_marker" in aligned.columns and aligned["promotion_marker"].notna().any():
                            promo_ts = pd.to_datetime(aligned["promotion_marker"].dropna().iloc[0], errors="coerce")
                            if pd.notna(promo_ts):
                                rule_df = pd.DataFrame({"promotion_marker": [promo_ts]})
                                rule = alt.Chart(rule_df).mark_rule(strokeDash=[6, 4]).encode(x="promotion_marker:T")
                                st.altair_chart(base_chart + rule, width="stretch")
                            else:
                                st.altair_chart(base_chart, width="stretch")
                        else:
                            st.altair_chart(base_chart, width="stretch")

                        dd_plot = aligned.copy()
                        for col in ["development_equity", "forward_equity"]:
                            if col in dd_plot.columns:
                                series = pd.to_numeric(dd_plot[col], errors="coerce")
                                dd_plot[col] = (series / series.cummax()) - 1.0
                        dd_long = dd_plot.melt(
                            id_vars=[c for c in ["date", "promotion_marker"] if c in dd_plot.columns],
                            value_vars=[c for c in ["development_equity", "forward_equity"] if c in dd_plot.columns],
                            var_name="serie",
                            value_name="drawdown",
                        ).dropna(subset=["drawdown"])
                        dd_chart = alt.Chart(dd_long).mark_line().encode(
                            x=alt.X("date:T", title="Fecha"),
                            y=alt.Y("drawdown:Q", title="Drawdown"),
                            color=alt.Color("serie:N", title="Curva"),
                            tooltip=["date:T", "serie:N", "drawdown:Q"],
                        ).properties(height=260)
                        if "promotion_marker" in dd_plot.columns and dd_plot["promotion_marker"].notna().any():
                            promo_ts = pd.to_datetime(dd_plot["promotion_marker"].dropna().iloc[0], errors="coerce")
                            if pd.notna(promo_ts):
                                dd_rule = alt.Chart(pd.DataFrame({"promotion_marker": [promo_ts]})).mark_rule(strokeDash=[6, 4]).encode(x="promotion_marker:T")
                                st.altair_chart(dd_chart + dd_rule, width="stretch")
                            else:
                                st.altair_chart(dd_chart, width="stretch")
                        else:
                            st.altair_chart(dd_chart, width="stretch")

                    comparison_df = build_metric_comparison_table(
                        design_profile=profile,
                        forward_metrics=metrics,
                        executed_metrics=metrics,
                    )
                    st.markdown("**Metricas comparativas**")
                    st.dataframe(comparison_df, width="stretch", hide_index=True)

                    detail_rows = [
                        {
                            "Fecha": _fmt_ts(str(item.get("created_at") or "")),
                            "Accion": _human_review_action(item.get("action")),
                            "Estado anterior": _human_trader_state(item.get("previous_state")),
                            "Estado nuevo": _human_trader_state(item.get("new_state")),
                            "Health score": float(item.get("health_score") or 0.0),
                            "Reasons": "; ".join(list(item.get("reasons") or [])),
                        }
                        for item in hr_details
                        if str(item.get("trader_id") or "") == trader_id
                    ]
                    if detail_rows:
                        st.markdown("**Historial de revisiones de salud**")
                        st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)
        else:
            st.info("Todavia no hay revisiones de salud guardadas.")

    if selected_section == "Operativa":
        st.markdown("### Operativa por trader")
        runtime_mode = str(status.get("operational_runtime_mode") or "paper")
        runtime_active = bool(status.get("operational_runtime_started"))
        mt5_connected = bool(status.get("mt5_connected"))
        if runtime_active and runtime_mode != "live_mt5":
            st.warning(
                "La operativa está corriendo en modo degradado `paper` porque MT5 no está listo. "
                "Aun así, el runtime puede generar señales y auditoría de portfolio/risk."
            )
        elif (not runtime_active) and (not mt5_connected):
            st.info(
                "El runtime operativo está parado y MT5 aparece desconectado. "
                "Pulsa **Lanzar operativa MT5 con traders actuales** para reintentar la conexión."
            )
        if st.button("Lanzar operativa MT5 con traders actuales", key="btn_start_ops_mt5"):
            out = supervisor.start_operational_runtime() if hasattr(supervisor, "start_operational_runtime") else {"started": False}
            if bool(out.get("started")):
                st.session_state["ops_launch_feedback"] = {
                    "level": "success",
                    "text": (
                        f"Operativa iniciada con {int(out.get('n_traders', 0))} traders "
                        f"en modo `{str(out.get('runtime_mode') or runtime_mode)}`."
                    ),
                }
            else:
                reason = str(out.get("reason") or "runtime_not_started")
                if reason == "no_promoted_traders":
                    msg = (
                        "No se puede iniciar la operativa porque no hay traders promovidos activos. "
                        "Desarrolla traders en la pestaña Desarrollo y vuelve a intentarlo."
                    )
                else:
                    msg = f"No se pudo iniciar operativa MT5. Motivo: `{reason}`."
                if str(out.get("mt5_reason") or "").strip():
                    msg += f" Detalle MT5: `{str(out.get('mt5_reason'))}`."
                st.session_state["ops_launch_feedback"] = {"level": "warning", "text": msg}
            st.rerun()
        launch_feedback = st.session_state.get("ops_launch_feedback")
        if isinstance(launch_feedback, dict) and str(launch_feedback.get("text") or "").strip():
            level = str(launch_feedback.get("level") or "info")
            text = str(launch_feedback.get("text"))
            if level == "success":
                st.success(text)
            elif level == "error":
                st.error(text)
            else:
                st.warning(text)
            if st.button("Limpiar aviso operativa", key="btn_ops_clear_feedback"):
                st.session_state.pop("ops_launch_feedback", None)
                st.rerun()
        st.markdown("### Test manual API MT5 (AAPL)")
        c_m1, c_m2, c_m3 = st.columns(3)
        side_manual = c_m1.selectbox("Direccion", options=["buy", "sell"], key="ops_manual_side_aapl")
        vol_manual = float(c_m2.number_input("Volumen", min_value=1.0, max_value=1000.0, value=1.0, step=1.0, key="ops_manual_vol_aapl"))
        if c_m3.button("Enviar orden manual AAPL", key="btn_ops_manual_aapl"):
            try:
                prep = supervisor.ensure_mt5_execution_ready(symbols=["AAPL"]) if hasattr(supervisor, "ensure_mt5_execution_ready") else {"connected": False}
                if not bool(prep.get("connected")) or str(prep.get("mode")) != "live_mt5":
                    st.error(f"MT5 no está listo para ejecución LIVE. Estado={prep}")
                else:
                    res_manual = supervisor.trader_agent.route_order(
                        trader_id="manual_aapl",
                        symbol="AAPL",
                        side=side_manual,
                        volume=vol_manual,
                        comment="MANUAL_AAPL",
                        correlation_id="manual_aapl_order",
                    )
                    accepted = bool(res_manual.get("accepted"))
                    mode_manual = str(res_manual.get("mode"))
                    if accepted and "live_mt5" in mode_manual.lower():
                        st.success(f"Orden manual AAPL enviada correctamente. Ticket={res_manual.get('ticket')}")
                    elif accepted:
                        st.error(f"La orden no salió por LIVE_MT5. mode={mode_manual} reason={res_manual.get('reason')}")
                    else:
                        st.error(f"Orden manual AAPL rechazada. Motivo={res_manual.get('reason')}")
                    st.caption(f"Execution mode: {mode_manual}")
                    broker_payload = res_manual.get("broker_payload", {}) if isinstance(res_manual.get("broker_payload"), dict) else {}
                    if broker_payload:
                        st.caption(f"MT5 retcode: {broker_payload.get('retcode')}")
                        st.caption(f"MT5 retcode_name: {broker_payload.get('retcode_name')}")
                        st.caption(f"MT5 comment: {broker_payload.get('comment')}")
                        st.caption(f"MT5 retcode_external: {broker_payload.get('retcode_external')}")
                        st.caption(f"MT5 last_error: {broker_payload.get('last_error')}")
            except Exception as exc:
                st.error(f"Error enviando orden manual AAPL: {exc}")
        st.caption(
            f"{_market_status_text()} | MT5: {'conectado' if bool(status.get('mt5_connected')) else 'desconectado'} | "
            f"Runtime: {'activo' if bool(status.get('operational_runtime_started')) else 'parado'} | "
            f"Modo runtime: {runtime_mode}"
        )
        st.markdown("### Ordenes pendientes de retry")
        if not pending_orders:
            st.info("No hay ordenes pendientes de retry.")
        else:
            rows_retry: list[Dict[str, Any]] = []
            for row in pending_orders:
                rows_retry.append(
                    {
                        "Trader": _pretty_trader_name(row.get("trader_id"), asset=row.get("symbol"), timeframe="D1"),
                        "Símbolo": row.get("symbol"),
                        "Lado": _human_side_label(row.get("side")),
                        "Señal": _human_signal_label(row.get("signal_label")),
                        "Volumen": row.get("volume"),
                        "Próximo intento": _fmt_ts(str(row.get("next_retry_at", ""))),
                        "Intentos acumulados": row.get("attempts"),
                        "Último motivo de rechazo": _interpret_pm_reason(row.get("last_reason")),
                    }
                )
            st.table(
                pd.DataFrame(rows_retry)[
                    [
                        "Trader",
                        "Símbolo",
                        "Lado",
                        "Señal",
                        "Volumen",
                        "Próximo intento",
                        "Intentos acumulados",
                        "Último motivo de rechazo",
                    ]
                ]
            )

        states = list(all_states)
        if not states:
            st.info("No hay traders disponibles para operativa.")
        else:
            latest_orders = _latest_order_events_by_trader(all_events)
            for row in states:
                trader_id = str(row.get("trader_id") or "")
                asset = str(row.get("asset") or "").upper()
                timeframe = str(row.get("timeframe") or "D1")
                trader_positions = [
                    p
                    for p in open_positions
                    if (
                        str(p.get("symbol", "")).upper() == asset
                        and (
                            (str(p.get("comment", "")).find(trader_id) >= 0)
                            if str(p.get("comment", "")).strip()
                            else True
                        )
                    )
                ]
                # Fallback pragmático: si no hay comentario/magic por trader,
                # al menos mostramos posición abierta por activo.
                is_operating = len(trader_positions) > 0
                status_label = "OPERANDO" if is_operating else "SIN OPERACION ABIERTA"
                last_evt = latest_orders.get(trader_id)
                pretty_trader = _pretty_trader_from_row(row)
                with st.expander(f"{pretty_trader} ({asset})", expanded=False):
                    st.markdown(
                        _ops_status_badge_html(is_operating) + _latest_order_badge_html(last_evt),
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Estado:** `{status_label}`")
                    st.caption(f"Activo: {asset} | Timeframe: {timeframe}")

                    if is_operating:
                        for i, pos in enumerate(trader_positions, start=1):
                            side_num = int(pos.get("type", -1)) if str(pos.get("type", "")).strip() != "" else -1
                            side_txt = "BUY" if side_num == 0 else ("SELL" if side_num == 1 else str(pos.get("type")))
                            t_raw = pos.get("time_update") or pos.get("time")
                            dt_txt = "-"
                            try:
                                if t_raw is not None:
                                    dt_txt = datetime.fromtimestamp(int(t_raw)).strftime("%d/%m/%Y %H:%M:%S")
                            except Exception:
                                dt_txt = str(t_raw)
                            st.markdown(f"**Orden abierta #{i}**")
                            st.markdown(
                                _badge_html(side_txt, bg="#1d4ed8" if side_txt == "BUY" else "#b91c1c"),
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Fecha: {dt_txt}")
                            st.caption(f"Direccion: {side_txt}")
                            st.caption(f"Volumen: {pos.get('volume')}")
                            st.caption(f"Precio apertura: {pos.get('price_open')}")
                            if pos.get("sl") is not None:
                                st.caption(f"SL: {pos.get('sl')}")
                            if pos.get("tp") is not None:
                                st.caption(f"TP: {pos.get('tp')}")
                            st.caption(f"Ticket: {pos.get('ticket')}")
                            st.caption(f"Profit flotante: {pos.get('profit')}")
                    else:
                        st.caption("Sin posicion abierta en este momento.")

                    if last_evt:
                        payload = last_evt.get("payload", {}) or {}
                        result = payload.get("result", {}) or {}
                        broker_payload = result.get("broker_payload", {}) or {}
                        st.markdown("**Ultima orden registrada**")
                        st.caption(f"Fecha: {_fmt_ts(str(last_evt.get('occurred_at', '')))}")
                        st.caption(f"Tipo evento: {last_evt.get('event_type')}")
                        st.caption(f"Direccion: {payload.get('side')}")
                        st.caption(f"Volumen: {payload.get('volume')}")
                        st.caption(f"Ticket: {result.get('ticket')}")
                        st.caption(f"Aceptada: {result.get('accepted')}")
                        if result.get("reason"):
                            st.caption(f"Motivo: {result.get('reason')}")
                        if isinstance(broker_payload, dict):
                            if broker_payload.get("retcode") is not None:
                                st.caption(f"MT5 retcode: {broker_payload.get('retcode')}")
                            if broker_payload.get("retcode_name"):
                                st.caption(f"MT5 retcode_name: {broker_payload.get('retcode_name')}")
                            if broker_payload.get("comment"):
                                st.caption(f"MT5 comment: {broker_payload.get('comment')}")
                            if broker_payload.get("retcode_external") is not None:
                                st.caption(f"MT5 retcode_external: {broker_payload.get('retcode_external')}")
                            if broker_payload.get("last_error"):
                                st.caption(f"MT5 last_error: {broker_payload.get('last_error')}")


if __name__ == "__main__":
    main()


