from __future__ import annotations

import json
import math
import os
import threading
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

# Fragment en vivo: solo esta region se re-ejecuta periodicamente.
LIVE_FRAGMENT_INTERVAL_DEV_OPS = "2s"
EVENT_LIMIT_DESARROLLO_LIVE = 120
EVENT_LIMIT_OPERATIVA_LIVE = 220


def _event_limit_for_section(section: str) -> int:
    if section == "Desarrollo":
        return int(EVENT_LIMIT_DESARROLLO_LIVE)
    if section == "Operativa":
        return int(EVENT_LIMIT_OPERATIVA_LIVE)
    return int(DEFAULT_EVENT_LIMIT)


def _render_desarrollo_controls(supervisor: DevelopmentOperationalSupervisor) -> None:
    """Botones/objetivo fuera del fragment para widgets estables."""
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
            if supervisor.start():
                st.success("Desarrollo iniciado.")
            else:
                st.warning(
                    "No se inicia desarrollo: ya hay tantos traders promovidos como el objetivo configurado "
                    "(o más). Reduzca traders con reinicio / HR o suba el objetivo."
                )
            st.rerun()
        if b2.button("Parar desarrollo de agentes trader", key="btn_dev_stop"):
            supervisor.stop_development()
            st.info("Desarrollo detenido.")
            st.rerun()
        if b3.button("Borrar todos los traders y reiniciar", key="btn_dev_reset"):
            supervisor.reset_all()
            st.warning("Sistema reiniciado.")
            st.rerun()




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


def _pm_latest_row_per_trader(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por trader con la señal más reciente (para tabla resumen)."""
    if df.empty or "trader" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    key = work["fecha_señal"].astype(str)
    work = work.assign(_sort_key=key)
    work = work.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"], errors="ignore")
    return work.drop_duplicates(subset=["trader"], keep="first").reset_index(drop=True)


def _pm_developed_trader_legend_label(signal_row: Dict[str, Any] | None) -> str:
    """Misma familia de textos que la leyenda PM (estado de orden / decisión), no Sí/No."""
    if signal_row is None:
        return "Sin señal en libro PM"
    estado = str(signal_row.get("estado_orden") or "").strip().lower()
    decision = str(signal_row.get("decision_pm") or "").strip().lower()

    if estado == "executed":
        return "Ejecutada"
    if estado == "selected":
        return "Pendiente envío"
    if estado == "rejected":
        return "Rechazada"
    if estado == "waiting_next_monday":
        return "Espera lunes"
    if estado == "waiting_full_universe":
        return "Espera universo"
    if estado == "already_open":
        return "Ya abierta"
    if estado == "discarded":
        return "Descartada"
    if estado == "closed":
        return "Cerrada"
    if estado == "close_rejected":
        return "Cierre rechazado"
    if decision == "descartado":
        return "Descartado"
    if decision == "seleccionado":
        return "Seleccionado"
    if estado:
        return estado
    return "—"


def _try_mt5_broker_open_positions(supervisor: DevelopmentOperationalSupervisor) -> List[Dict[str, Any]]:
    """
    Posiciones abiertas leyendo el terminal MT5 SOLO si ya estaba conectado.

    El dashboard no debe abrir/inicializar MT5 por su cuenta: la conexión debe
    nacer exclusivamente del flujo operativo del supervisor cuando corresponde.
    """
    mt5 = getattr(supervisor, "mt5", None)
    if mt5 is None or not bool(getattr(mt5, "connected", False)):
        return []
    try:
        return list(mt5.get_open_positions())
    except Exception:
        return []


def _pm_signals_for_latest_rebalance(df_signals: pd.DataFrame) -> pd.DataFrame:
    """Filas del último ciclo de rebalanceo PM (por rebalance_id más reciente en el tiempo)."""
    if df_signals.empty or "rebalance_id" not in df_signals.columns:
        return df_signals
    work = df_signals.copy()
    rid = work["rebalance_id"].astype(str).str.strip()
    work = work.assign(_rid=rid)
    non_empty = work[work["_rid"] != ""]
    if non_empty.empty:
        return df_signals
    if "fecha_señal" not in work.columns:
        latest = str(non_empty.iloc[-1]["_rid"])
    else:
        tmp = non_empty.copy()
        tmp["_ts"] = tmp["fecha_señal"].astype(str)
        latest = str(tmp.sort_values("_ts", ascending=False).iloc[0]["_rid"])
    out = work[work["_rid"] == latest].drop(columns=["_rid"], errors="ignore")
    return out.reset_index(drop=True)


def _pm_has_effective_allocation(row: pd.Series | Dict[str, Any], *, min_weight_pct: float = 2.0) -> bool:
    weight_pct = float(pd.to_numeric((row.get("peso_pct") if isinstance(row, dict) else row.get("peso_pct")), errors="coerce") or 0.0)
    euros = float(pd.to_numeric((row.get("euros_asignados") if isinstance(row, dict) else row.get("euros_asignados")), errors="coerce") or 0.0)
    return weight_pct >= float(min_weight_pct) and euros > 0.0


def _pm_filter_effective_allocations(df: pd.DataFrame, *, min_weight_pct: float = 2.0) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = df.apply(lambda row: _pm_has_effective_allocation(row, min_weight_pct=min_weight_pct), axis=1)
    return df[mask].copy().reset_index(drop=True)


def _pm_filter_zero_or_tiny_allocations(df: pd.DataFrame, *, min_weight_pct: float = 2.0) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = ~df.apply(lambda row: _pm_has_effective_allocation(row, min_weight_pct=min_weight_pct), axis=1)
    out = df[mask].copy()
    if out.empty:
        return out
    out["estado_orden"] = "rejected"
    out["decision_pm"] = "descartado"
    out["motivo_interpretado"] = out["motivo_interpretado"].map(
        lambda txt: "Sin asignación PM (<2%)" if not str(txt or "").strip() or str(txt).strip().lower() == "signal" else f"Sin asignación PM (<2%) · {txt}"
    )
    return out.reset_index(drop=True)


def _count_ui_open_positions(signal_rows: List[Dict[str, Any]]) -> int:
    """
    Contador puramente visual: número de posiciones con asignación efectiva
    en el último rebalanceo mostrado por el PM.
    """
    if not signal_rows:
        return 0
    df_latest = _pm_signals_for_latest_rebalance(pd.DataFrame(signal_rows))
    if df_latest.empty:
        return 0
    df_effective = _pm_filter_effective_allocations(
        df_latest[df_latest["estado_orden"].astype(str).str.lower() == "executed"].copy()
    )
    if df_effective.empty:
        return 0
    if "symbol" not in df_effective.columns:
        return int(len(df_effective))
    return int(df_effective["symbol"].astype(str).str.strip().str.upper().nunique())


def _pm_symbol_volume_hints(df_latest_by_trader: pd.DataFrame) -> Dict[str, int]:
    """
    Por símbolo: acciones estimadas PM agregadas.
    """
    if df_latest_by_trader.empty:
        return {}
    sub = df_latest_by_trader[df_latest_by_trader["estado_orden"].astype(str).str.lower() == "executed"].copy()
    if sub.empty or "symbol" not in sub.columns:
        return {}
    out: Dict[str, int] = {}
    for sym, grp in sub.groupby(sub["symbol"].astype(str).str.strip().str.upper()):
        if "acciones_estimadas" in grp.columns:
            vols = pd.to_numeric(grp["acciones_estimadas"], errors="coerce").fillna(0.0)
        else:
            vols = pd.Series(0.0, index=grp.index, dtype=float)
        total_vol = int(round(float(vols.sum())))
        out[str(sym)] = max(0, total_vol)
    return out


def _render_pm_company_allocation_pie(
    *,
    df_latest_by_trader: pd.DataFrame,
    last_portfolio_output: Dict[str, Any] | None,
) -> None:
    """Donut interactivo (Plotly): % por símbolo + cash; hover con acciones y LONG/SHORT."""
    st.markdown("**Asignación por compañía / símbolo (última cartera PM)**")
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.caption("Instala plotly (`pip install plotly`) para el gráfico interactivo de cartera.")
        return

    lo = dict(last_portfolio_output or {})
    sym_weights_pct: Dict[str, float] = {}
    df_effective_latest = _pm_filter_effective_allocations(
        df_latest_by_trader[df_latest_by_trader["estado_orden"].astype(str).str.lower() == "executed"].copy()
    )

    if not df_effective_latest.empty and "symbol" in df_effective_latest.columns and "peso_pct" in df_effective_latest.columns:
        for sym, grp in df_effective_latest.groupby(df_effective_latest["symbol"].astype(str).str.strip().str.upper()):
            sym_weights_pct[str(sym)] = float(pd.to_numeric(grp["peso_pct"], errors="coerce").fillna(0.0).sum())

    executed_weight_total = float(sum(sym_weights_pct.values()))
    cash_pct = max(0.0, 100.0 - executed_weight_total)

    if not sym_weights_pct and cash_pct <= 1e-9:
        st.caption(
            "Sin pesos de cartera para diagramar (sin última optimización en memoria o todos los pesos a cero). "
            "Tras un rebalanceo con selección, aquí aparecerán los porcentajes por símbolo."
        )
        return

    hints = _pm_symbol_volume_hints(df_effective_latest)

    rows: list[tuple[str, float, int]] = []
    for sym, pct in sorted(sym_weights_pct.items(), key=lambda kv: float(kv[1]), reverse=True):
        if float(pct) <= 1e-6:
            continue
        vol = hints.get(sym, 0)
        rows.append((sym, float(pct), int(vol)))

    tiny_thr = 0.05
    main: list[tuple[str, float, int]] = []
    tiny_sum = 0.0
    for sym, pct, vol in rows:
        if pct < tiny_thr:
            tiny_sum += pct
        else:
            main.append((sym, pct, vol))
    if tiny_sum > 1e-9:
        main.append(("Otros (pesos <0.05%)", tiny_sum, 0))

    if cash_pct > 1e-6:
        main.append(("Cash (efectivo)", float(cash_pct), 0))

    total = sum(p for _, p, _ in main)
    if total <= 1e-12:
        st.caption("Suma de pesos nula; no se muestra el diagrama.")
        return

    labels = [r[0] for r in main]
    values = [r[1] for r in main]
    customdata = [[r[2]] for r in main]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                pull=[0.03] * len(labels),
                sort=False,
                direction="clockwise",
                marker=dict(line=dict(color="white", width=1.2)),
                texttemplate="%{label}<br>%{value:.1f}%",
                textposition="outside",
                textfont=dict(size=10),
                insidetextorientation="horizontal",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Acciones (estim. PM): %{customdata[0]}<extra></extra>"
                ),
                customdata=customdata,
            )
        ]
    )
    fig.update_layout(
        title=dict(text="% cartera por símbolo (incluye cash)", font=dict(size=12)),
        height=600,
        width=800,
        margin=dict(l=10, r=10, t=42, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.caption("El diagrama muestra el % real del capital total: posiciones ejecutadas + cash no invertido.")
    _, c_mid, _ = st.columns([0.7, 1.8, 0.7])
    with c_mid:
        st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})


def _render_pm_allocation_pie_chart(last_output: Dict[str, Any]) -> None:
    """Quesito: pesos por trader + cash de la última decisión PM."""
    decision = dict(last_output.get("decision") or {})
    weights = dict(last_output.get("weights") or decision.get("weights") or {})
    cash = float(last_output.get("target_cash_weight") or decision.get("target_cash_weight") or 0.0)
    if not weights and cash <= 1e-12:
        return

    labels: list[str] = []
    sizes: list[float] = []
    for tid, w in sorted(weights.items(), key=lambda kv: float(kv[1]), reverse=True):
        labels.append(_pretty_trader_name(tid))
        sizes.append(float(w) * 100.0)
    if cash > 1e-9:
        labels.append("Cash (efectivo)")
        sizes.append(float(cash) * 100.0)

    s = sum(sizes)
    if s <= 1e-12:
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title("Asignación de cartera (último rebalanceo PM)", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)


def _render_pm_ga_pso_optimization_chart(last_output: Dict[str, Any]) -> None:
    """
    Barras: fitness de los subconjuntos que el GA prioriza (evaluación rápida).
    Línea: fitness de la cartera final tras PSO sobre el mejor candidato.
    """
    diagnostics = dict(last_output.get("diagnostics") or {})
    subsets = list(diagnostics.get("ga_top_subsets") or [])
    decision = dict(last_output.get("decision") or {})
    final_fitness = float(last_output.get("fitness") or decision.get("fitness") or 0.0)

    if not subsets:
        st.caption(
            "No hay candidatos GA en diagnóstico (universo pequeño con PSO directo, rebalanceo degradado o sin última optimización en memoria)."
        )
        return

    ordered = sorted(subsets, key=lambda s: float(s.get("fitness", 0.0)), reverse=True)
    xs = [f"#{i}" for i in range(1, len(ordered) + 1)]
    fs = [float(s.get("fitness", 0.0)) for s in ordered]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(xs, fs, color="#93c5fd", edgecolor="#2563eb", linewidth=0.8, label="Fitness candidato GA (eval. rápida)")
    ax.axhline(
        final_fitness,
        color="#15803d",
        linestyle="--",
        linewidth=2,
        label=f"Fitness final (tras PSO): {final_fitness:.4f}",
    )
    ax.set_xlabel("Candidatos GA ordenados por fitness (mejor a la izquierda)")
    ax.set_ylabel("Fitness")
    ax.set_title("GA explora subconjuntos · PSO refina pesos hacia el óptimo")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)
    st.caption(
        "Cada barra es un subconjunto de traders evaluado en la fase GA; "
        "la línea verde es la fitness de la cartera ya optimizada con PSO que se aplica en vivo."
    )


def _render_pm_signal_tables(
    signal_rows: List[Dict[str, Any]],
    *,
    developed_states: List[Dict[str, Any]],
    key_prefix: str,
    last_portfolio_output: Dict[str, Any] | None = None,
) -> None:
    pm_cols = [
        "trader",
        "symbol",
        "side",
        "fecha_señal",
        "decision_pm",
        "estado_orden",
        "peso_pct",
        "euros_asignados",
        "acciones_estimadas",
        "motivo_interpretado",
    ]

    latest_signal_by_trader: Dict[str, Dict[str, Any]] = {}
    if signal_rows:
        _df_sig_latest = _pm_latest_row_per_trader(pd.DataFrame(signal_rows))
        if not _df_sig_latest.empty:
            for _, rec in _df_sig_latest.iterrows():
                k = str(rec.get("trader") or "").strip()
                if k:
                    latest_signal_by_trader[k] = rec.to_dict()

    # 1) Traders persistidos (se actualiza al promover / cambiar estado / HR)
    st.markdown("**Traders desarrollados (fases previas, persistidos en BD)**")
    if developed_states:
        dev_rows = []
        for row in developed_states:
            pretty = _pretty_trader_from_row(row)
            dev_rows.append(
                {
                    "trader": pretty,
                    "asset": row.get("asset"),
                    "Señal en PM": _pm_developed_trader_legend_label(latest_signal_by_trader.get(pretty)),
                    "actualizado": _fmt_ts(str(row.get("updated_at", ""))),
                }
            )
        df_dev = pd.DataFrame(dev_rows)
        st.dataframe(
            df_dev.style.map(lambda v: _pm_style_cell(v, column="señal_en_pm"), subset=["Señal en PM"]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No hay traders registrados en la base de datos.")

    if not signal_rows:
        st.info("No hay señales auditadas para este entrenamiento/rebalanceo (mercado cerrado o PM sin ciclo reciente).")
        return

    df_signals = pd.DataFrame(signal_rows)
    df_latest_rebalance = _pm_signals_for_latest_rebalance(df_signals)
    df_exec = df_latest_rebalance[df_latest_rebalance["estado_orden"] == "executed"].copy()
    df_exec_effective = _pm_filter_effective_allocations(df_exec)
    df_zero_alloc_latest = _pm_filter_zero_or_tiny_allocations(df_latest_rebalance)
    df_discarded = df_latest_rebalance[df_latest_rebalance["decision_pm"] == "descartado"].copy()
    if not df_zero_alloc_latest.empty:
        df_discarded = pd.concat([df_discarded, df_zero_alloc_latest], ignore_index=True)
        df_discarded = (
            df_discarded.sort_values(["fecha_señal", "symbol", "trader"], ascending=[False, True, True])
            .drop_duplicates(subset=["trader", "symbol", "side"], keep="first")
            .reset_index(drop=True)
        )
    df_discarded_show = _human_pm_signal_columns(df_discarded[pm_cols].copy()) if not df_discarded.empty else pd.DataFrame()
    df_exec_show = (
        _human_pm_signal_columns(df_exec_effective[["trader", "symbol", "side", "fecha_señal", "peso_pct", "euros_asignados", "acciones_estimadas"]].copy())
        if not df_exec_effective.empty
        else pd.DataFrame()
    )

    # 2) Traders que han emitido señal en el libro PM (última fila por trader)
    st.markdown("**Traders con señal registrada (último evento por trader)**")
    df_with_signal = _pm_latest_row_per_trader(df_signals)
    if not df_with_signal.empty:
        zero_alloc_latest = _pm_filter_zero_or_tiny_allocations(df_with_signal)
        if not zero_alloc_latest.empty:
            zero_keys = {
                (str(r.get("trader")), str(r.get("symbol")), str(r.get("side")))
                for _, r in zero_alloc_latest.iterrows()
            }
            work = df_with_signal.copy()
            for idx, rec in work.iterrows():
                key = (str(rec.get("trader")), str(rec.get("symbol")), str(rec.get("side")))
                if key in zero_keys:
                    work.at[idx, "decision_pm"] = "descartado"
                    work.at[idx, "estado_orden"] = "rejected"
                    work.at[idx, "motivo_interpretado"] = "Sin asignación PM (<2%)"
            df_with_signal = work
    if df_with_signal.empty:
        st.info("Ningún trader figura aún en el libro de señales.")
    else:
        df_sig_show = _human_pm_signal_columns(df_with_signal[pm_cols].copy())
        st.dataframe(
            df_sig_show.style.map(lambda v: _pm_style_cell(v, column="decision_pm"), subset=["Decisión PM"]).map(
                lambda v: _pm_style_cell(v, column="estado_orden"), subset=["Estado orden"]
            ),
            width="stretch",
            hide_index=True,
        )
    _render_pm_company_allocation_pie(
        df_latest_by_trader=df_with_signal,
        last_portfolio_output=last_portfolio_output,
    )

    # 3) Ejecutadas por el PM
    st.markdown("**Señales ejecutadas por el portfolio manager**")
    if df_exec_effective.empty:
        st.info("Ninguna señal se ha ejecutado todavía (mercado cerrado, pendiente o rechazada).")
    else:
        st.dataframe(df_exec_show, width="stretch", hide_index=True)

    lo = last_portfolio_output or {}
    if lo:
        dec_lo = dict(lo.get("decision") or {})
        wmap = dict(lo.get("weights") or dec_lo.get("weights") or {})
        cash_w = float(lo.get("target_cash_weight") or dec_lo.get("target_cash_weight") or 0.0)
        if wmap or cash_w > 1e-9:
            st.markdown("**Distribución de cartera (pesos + cash)**")
            _render_pm_allocation_pie_chart(lo)
        st.markdown("**Optimización GA + PSO (candidatos vs. cartera final)**")
        _render_pm_ga_pso_optimization_chart(lo)

    # 4) Rechazadas / sin asignación (incluye descartadas y filas con asignación nula o <2%)
    st.markdown("**Órdenes rechazadas o sin asignación por el portfolio manager**")
    st.caption(
        "Lista basada en el último ciclo de rebalanceo registrado (mismo `rebalance_id` más reciente). "
        "Incluye señales descartadas y también las que quedaron con asignación nula o inferior al 2%."
    )
    if df_discarded.empty:
        st.info("No hay órdenes rechazadas ni señales sin asignación en el último rebalanceo del portfolio manager.")
    else:
        st.dataframe(
            df_discarded_show.style.map(lambda v: _pm_style_cell(v, column="decision_pm"), subset=["Decisión PM"]).map(
                lambda v: _pm_style_cell(v, column="estado_orden"), subset=["Estado orden"]
            ),
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


_SUPERVISOR_SINGLETON_LOCK = threading.Lock()
_SUPERVISOR_BY_DB_KEY: Dict[str, DevelopmentOperationalSupervisor] = {}


def _supervisor_db_cache_key(db_path: Path) -> str:
    try:
        return str(db_path.resolve())
    except Exception:
        return str(db_path)


def _maybe_warn_numpy_abi() -> None:
    """Un aviso global si NumPy es demasiado nuevo para Numba/PyEventBT."""
    if st.session_state.get("_tfm_numpy_abi_checked"):
        return
    st.session_state["_tfm_numpy_abi_checked"] = True
    try:
        from app.core.numpy_numba_abi import numpy_numba_abi_fail_message

        msg = numpy_numba_abi_fail_message()
        if msg:
            st.error(msg)
    except Exception:
        pass


def _get_supervisor() -> DevelopmentOperationalSupervisor:
    """Un supervisor por BD compartido entre todas las sesiones Streamlit (mismo proceso)."""
    default_db_path = Path(os.getenv("TFM_DB_PATH", str(LOCAL_PATHS.supervisor_db)))
    cache_key = _supervisor_db_cache_key(default_db_path)

    with _SUPERVISOR_SINGLETON_LOCK:
        sup = _SUPERVISOR_BY_DB_KEY.get(cache_key)
        if sup is None:
            sup = DevelopmentOperationalSupervisor(db_path=default_db_path)
            _SUPERVISOR_BY_DB_KEY[cache_key] = sup

    needs_rebuild = (
        (not hasattr(sup, "get_backtest_registry"))
        or int(getattr(sup, "dashboard_snapshot_schema_version", 0)) < DevelopmentOperationalSupervisor._DASHBOARD_SNAPSHOT_SCHEMA_VERSION
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
        with _SUPERVISOR_SINGLETON_LOCK:
            _SUPERVISOR_BY_DB_KEY[_supervisor_db_cache_key(db_path)] = rebuilt
        sup = rebuilt

    st.session_state["tfm_supervisor"] = sup
    return sup


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


@st.cache_data(ttl=2, show_spinner=False)
def _load_trader_states_light(db_path_str: str) -> List[Dict[str, Any]]:
    """Solo `trader_states`; no arrastra `list_events` del snapshot completo."""
    p = Path(db_path_str)
    if not p.exists():
        return []
    from app.storage import StateStore

    out: List[Dict[str, Any]] = []
    for r in StateStore(db_path=p).list_trader_states():
        out.append(
            {
                "trader_id": r.trader_id,
                "asset": r.asset,
                "timeframe": r.timeframe,
                "state": r.state.value,
                "updated_at": r.updated_at,
                "notes": r.notes,
            }
        )
    return out


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


def _dashboard_open_positions(supervisor: DevelopmentOperationalSupervisor, status: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Posiciones abiertas mostradas en el dashboard.

    La UI nunca debe lanzar MT5 ni intentar conectar al broker. Solo refleja el
    estado ya existente:
    - si el router está en PAPER, muestra el ledger simulado;
    - si MT5 ya estaba conectado por el flujo operativo, muestra sus posiciones;
    - en caso contrario, devuelve vacío sin efectos laterales.
    """
    try:
        router = getattr(supervisor, "execution_router", None)
        if router is None:
            return []
        mode = str(getattr(router.mode, "value", router.mode) or "").strip().lower()

        if mode == "paper":
            return list(router.get_open_positions(actor="trader_agent"))

        broker = _try_mt5_broker_open_positions(supervisor)
        if broker:
            return broker

        mt5 = getattr(router, "mt5", None)
        if mt5 is not None and bool(getattr(mt5, "connected", False)):
            return list(router.get_open_positions(actor="trader_agent"))
        return []
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
    # Leyenda PM (mismos textos que Estado orden / Decisión PM / badges antiguos).
    if column == "señal_en_pm":
        legend_styles = {
            "Ejecutada": "background-color: #dcfce7; color: #166534; font-weight: 700;",
            "Pendiente envío": "background-color: #dbeafe; color: #1d4ed8; font-weight: 700;",
            "Rechazada": "background-color: #fee2e2; color: #991b1b; font-weight: 700;",
            "Espera lunes": "background-color: #fef3c7; color: #92400e; font-weight: 700;",
            "Espera universo": "background-color: #fef3c7; color: #92400e; font-weight: 700;",
            "Ya abierta": "background-color: #ccfbf1; color: #115e59; font-weight: 700;",
            "Descartada": "background-color: #f3f4f6; color: #374151; font-weight: 700;",
            "Cerrada": "background-color: #e2e8f0; color: #334155; font-weight: 700;",
            "Cierre rechazado": "background-color: #fee2e2; color: #991b1b; font-weight: 700;",
            "Seleccionado": "background-color: #dcfce7; color: #166534; font-weight: 600;",
            "Descartado": "background-color: #f3f4f6; color: #374151; font-weight: 600;",
            "Sin señal en libro PM": "background-color: #f9fafb; color: #6b7280; font-weight: 600;",
            "—": "background-color: #f9fafb; color: #6b7280; font-weight: 500;",
        }
        return legend_styles.get(txt, "background-color: #f9fafb; color: #4b5563; font-weight: 500;")
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



def _render_dashboard_content(supervisor: DevelopmentOperationalSupervisor, selected_section: str) -> None:
    """Contenido por pestaña; Desarrollo/Backtest desde fragment en vivo."""
    status = supervisor.get_status()
    need_events = selected_section in {"Desarrollo", "Backtest"}
    need_states = True
    need_backtests = selected_section == "Backtest"
    # Barra superior: contador visual de posiciones abiertas derivado del PM actual.
    need_open_positions = True
    need_pm_snapshot = selected_section == "Portfolio manager" or need_open_positions
    need_pm_history = selected_section == "Portfolio manager" or need_open_positions
    need_hr_snapshot = selected_section == "Recursos Humanos"
    need_hr_history = selected_section == "Recursos Humanos"

    ev_limit = _event_limit_for_section(selected_section)
    all_events = _load_events(Path(supervisor.db_path), ev_limit) if need_events else []
    all_states = _load_trader_states_light(str(supervisor.db_path)) if need_states else []
    backtests = supervisor.get_backtest_registry() if (need_backtests and hasattr(supervisor, "get_backtest_registry")) else {}
    pm_snapshot = {}
    if need_pm_snapshot and hasattr(supervisor, "get_portfolio_manager_snapshot"):
        pm_snapshot = supervisor.get_portfolio_manager_snapshot(include_history=need_pm_history)
    hr_snapshot = {}
    if need_hr_snapshot and hasattr(supervisor, "get_human_resources_snapshot"):
        hr_snapshot = supervisor.get_human_resources_snapshot(include_history=need_hr_history)
    pending_orders = list(pm_snapshot.get("pending_orders", []) or [])
    signal_book = list(pm_snapshot.get("signal_book", []) or [])
    signal_audit = list(pm_snapshot.get("signal_audit", []) or [])
    last_output = dict(pm_snapshot.get("last_output", {}) or {})
    rebalance_rows = list(pm_snapshot.get("rebalance_rows", []) or [])
    normalized_pm_signals = _normalize_pm_signal_rows(signal_book, signal_audit)
    open_positions = (
        _pm_filter_effective_allocations(
            _pm_signals_for_latest_rebalance(pd.DataFrame(normalized_pm_signals)).query("estado_orden == 'executed'")
        ).to_dict("records")
        if need_open_positions and normalized_pm_signals
        else []
    )
    open_positions_count = _count_ui_open_positions(normalized_pm_signals) if need_open_positions else 0
    live_rebalance_rows = [dict(row) for row in rebalance_rows if _is_live_portfolio_rebalance(dict(row))]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Supervisor", "running" if bool(status.get("running")) else "stopped")
    c2.metric("Desarrollo", "activo" if bool(status.get("develop_enabled")) else "parado")
    # Fuente robusta para UI: DB persistida, no solo estado en memoria.
    c3.metric("Traders desarrollados", int(len(all_states)))
    c4.metric("Señales PM", int(status.get("pm_signal_count", len(signal_book))))
    c5.metric("Retries pendientes", int(status.get("pending_orders_count", len(pending_orders))))
    c6.metric(
        "Posiciones abiertas",
        int(open_positions_count),
        help="Contador puramente visual: número de símbolos con asignación efectiva (>=2%) "
        "en el último rebalanceo mostrado por el Portfolio Manager.",
    )
    st.caption(f"Objetivo traders: {int(status.get('target_traders', 8))}")
    c_global_1, c_global_2, c_global_3 = st.columns([1, 5, 1])
    with c_global_1:
        if st.button("Refrescar dashboard", key="btn_global_refresh"):
            st.rerun()

    if selected_section == "Desarrollo":
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
            max_items=min(80, ev_limit),
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
                bt = (
                    supervisor.get_backtest_entry(trader_id)
                    if hasattr(supervisor, "get_backtest_entry")
                    else backtests.get(trader_id, {})
                )
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
                        st.info("Backtest en ejecucion. Esta pestaña se actualiza automaticamente.")
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

        # Señales de la última rebalanceo (orden: desarrollados → con señal → ejecutadas → descartadas)
        _render_pm_signal_tables(
            normalized_pm_signals,
            developed_states=list(all_states),
            key_prefix="pm_ga_pso",
            last_portfolio_output=dict(last_output) if last_output else None,
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
                        dd_long["serie"] = dd_long["serie"].replace(
                            {"development_equity": "Backtest desarrollo", "forward_equity": "Operativa real"}
                        )
                        dd_chart = alt.Chart(dd_long).mark_line().encode(
                            x=alt.X("date:T", title="Fecha"),
                            y=alt.Y("drawdown:Q", title="Drawdown"),
                            color=alt.Color(
                                "serie:N",
                                title="Curva",
                                scale=alt.Scale(
                                    domain=["Backtest desarrollo", "Operativa real"],
                                    range=["#2563eb", "#f59e0b"],
                                ),
                            ),
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
                "Pulsa **Lanzar operativa MT5 con traders actuales** para reintentar la conexión "
                "cuando haya al menos 5 traders promovidos."
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
                elif reason == "minimum_traders_not_reached":
                    min_required = int(out.get("min_traders_required") or 5)
                    msg = (
                        f"No se puede iniciar la operativa porque solo hay {int(out.get('n_traders', 0))} traders "
                        f"y el mínimo para lanzar MT5 es {min_required}."
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
                min_runtime = int(status.get("min_traders_for_runtime") or 5)
                developed = int(status.get("developed_traders") or 0)
                if developed < min_runtime:
                    st.error(
                        f"MT5 no debe lanzarse con menos de {min_runtime} traders promovidos. "
                        f"Ahora mismo hay {developed}."
                    )
                else:
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
    st.title("Dashboard TFM")
    st.write("Desarrollo, backtest y supervision del portfolio en tiempo real.")

    _maybe_warn_numpy_abi()

    supervisor = _get_supervisor()

    section_options = ["Desarrollo", "Backtest", "Portfolio manager", "Recursos Humanos"]
    selected_section = st.radio(
        "Sección",
        options=section_options,
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_section",
    )

    if selected_section == "Desarrollo":
        _render_desarrollo_controls(supervisor)

    if selected_section in {"Desarrollo", "Backtest"}:
        _live_dashboard_fragment()
    else:
        _render_dashboard_content(supervisor, selected_section)


@st.fragment(run_every=LIVE_FRAGMENT_INTERVAL_DEV_OPS)
def _live_dashboard_fragment() -> None:
    """Solo esta region hace polling; el script completo ya no usa st.rerun() por firma."""
    tab = str(st.session_state.get("dashboard_section") or "Desarrollo")
    if tab not in {"Desarrollo", "Backtest"}:
        return
    _render_dashboard_content(_get_supervisor(), tab)


if __name__ == "__main__":
    main()


