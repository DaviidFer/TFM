from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from app.toolbox.indicators import build_feature_library


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_COND_RE = re.compile(r"(?:`([^`]+)`|([A-Za-z_]\w*))\s*(>=|<=|>|<)\s*(" + _FLOAT_RE + r")")


@dataclass(frozen=True)
class CompiledRule:
    rule_id: int
    rule_str: str
    bounds: Dict[str, Tuple[float, bool, float, bool]]


def _merge_bound(existing: Tuple[float, bool, float, bool], op: str, val: float) -> Tuple[float, bool, float, bool]:
    lower, l_inc, upper, u_inc = existing

    if op == ">":
        if val > lower or (val == lower and l_inc):
            lower, l_inc = val, False
    elif op == ">=":
        if val > lower or (val == lower and not l_inc):
            lower, l_inc = val, True
    elif op == "<":
        if val < upper or (val == upper and u_inc):
            upper, u_inc = val, False
    elif op == "<=":
        if val < upper or (val == upper and not u_inc):
            upper, u_inc = val, True
    else:
        raise ValueError(f"Operador no soportado: {op}")

    return lower, l_inc, upper, u_inc


def _compile_rule_bounds(rule_str: str) -> Dict[str, Tuple[float, bool, float, bool]]:
    matches = _COND_RE.findall(str(rule_str))
    if not matches:
        raise ValueError(f"No se han encontrado condiciones parseables en: {rule_str}")

    bounds: Dict[str, Tuple[float, bool, float, bool]] = {}
    for feat_bt, feat_plain, op, val_str in matches:
        feat = feat_bt if feat_bt else feat_plain
        val = float(val_str)
        if feat not in bounds:
            bounds[feat] = (-math.inf, False, math.inf, False)
        bounds[feat] = _merge_bound(bounds[feat], op, val)

    return bounds


def _extract_rules_list(rules_obj: Any, name: str) -> List[str]:
    """
    Acepta:
      - list/tuple/set/Series de reglas (strings)
      - DataFrame con columna 'regla' o 'rule'
      - path CSV con columnas 'regla' o 'rule'
    """
    if rules_obj is None:
        return []

    if isinstance(rules_obj, (str, Path)):
        p = Path(rules_obj)
        if not p.exists():
            raise FileNotFoundError(f"{name}: no existe archivo {p}")
        df = pd.read_csv(p)
        if "regla" in df.columns:
            s = df["regla"]
        elif "rule" in df.columns:
            s = df["rule"]
        else:
            raise ValueError(f"{name}: CSV sin columna 'regla' ni 'rule': {p}")
        out = s.dropna().astype(str).tolist()
        return list(pd.unique(pd.Series(out, dtype=str)))

    if isinstance(rules_obj, pd.DataFrame):
        if "regla" in rules_obj.columns:
            s = rules_obj["regla"]
        elif "rule" in rules_obj.columns:
            s = rules_obj["rule"]
        else:
            raise ValueError(f"{name}: DataFrame sin columna 'regla' ni 'rule'.")
        out = s.dropna().astype(str).tolist()
        return list(pd.unique(pd.Series(out, dtype=str)))

    if isinstance(rules_obj, (pd.Series, list, tuple, set)):
        out = pd.Series(list(rules_obj)).dropna().astype(str).tolist()
        return list(pd.unique(pd.Series(out, dtype=str)))

    raise ValueError(f"{name}: tipo no soportado {type(rules_obj)}")


def compile_rules(rules_obj: Any, start_rule_id: int = 0, name: str = "rules") -> List[CompiledRule]:
    rules = _extract_rules_list(rules_obj, name=name)
    compiled: List[CompiledRule] = []
    for i, rstr in enumerate(rules):
        compiled.append(
            CompiledRule(
                rule_id=start_rule_id + i,
                rule_str=rstr,
                bounds=_compile_rule_bounds(rstr),
            )
        )
    return compiled


def _extract_needed_periods(compiled_rules: Sequence[CompiledRule]) -> Dict[str, Set[int]]:
    """
    Devuelve los periodos usados en nombres de features para acotar
    el coste de build_feature_library. La librería predictora actual
    usa exclusivamente nombres del tipo `Indicador_<periodo>`.
    """
    periods: Set[int] = set()

    for cr in compiled_rules:
        for feat in cr.bounds.keys():
            m_gen = re.match(r"^[A-Za-z][A-Za-z0-9]*_(\d+)$", feat)
            if m_gen:
                periods.add(int(m_gen.group(1)))

    return {
        "periods": periods,
        "breakout_periods": set(),
        "seq_periods": set(),
    }


def _compute_bars_needed(specs: Dict[str, Set[int]], safety_margin: int = 20) -> int:
    max_period = max(specs["periods"]) if specs["periods"] else 2
    # Conservador para indicadores tipo RVI y osciladores con medias móviles.
    return int(max(80, max_period + 8) + safety_margin)


def _compute_feature_values_lastrow(
    df_ohlc: pd.DataFrame,
    specs: Dict[str, Set[int]],
) -> Dict[str, float]:
    df = df_ohlc.copy()
    df.columns = [str(c).lower() for c in df.columns]

    periods = sorted(specs["periods"]) if specs["periods"] else []
    breakouts = sorted(specs["breakout_periods"]) if specs["breakout_periods"] else []
    seqs = sorted(specs["seq_periods"]) if specs["seq_periods"] else []

    feat_df = build_feature_library(
        data_ohlc=df[["open", "high", "low", "close"]],
        periods=periods,
        breakout_periods=breakouts,
        seq_periods=seqs,
        include_rvi=True,
        dropna=False,
    )
    last = feat_df.iloc[-1]
    return {str(k): float(v) if pd.notna(v) else float("nan") for k, v in last.items()}


def evaluate_any_rule(compiled_rules: Sequence[CompiledRule], feature_values: Mapping[str, float]) -> bool:
    """
    True si al menos una regla esta activa.
    Cada regla es AND de bounds; conjunto de reglas es OR.
    """
    for cr in compiled_rules:
        ok = True
        for feat, (lb, lb_inc, ub, ub_inc) in cr.bounds.items():
            v = feature_values.get(feat, None)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                ok = False
                break

            if lb != -math.inf:
                if lb_inc:
                    if v < lb:
                        ok = False
                        break
                else:
                    if v <= lb:
                        ok = False
                        break

            if ub != math.inf:
                if ub_inc:
                    if v > ub:
                        ok = False
                        break
                else:
                    if v >= ub:
                        ok = False
                        break

        if ok:
            return True

    return False


def _pip_size(symbol: str) -> float:
    s = str(symbol).upper()
    return 0.01 if "JPY" in s else 0.0001


def _round_price(px: float, symbol: str) -> float:
    s = str(symbol).upper()
    digits = 3 if "JPY" in s else 5
    return float(round(px, digits))


def _sl_tp_prices_from_pips(
    ref_price: float,
    symbol: str,
    side: str,
    sl_pips: float,
    tp_pips: float,
) -> Tuple[Decimal, Decimal]:
    pip = _pip_size(symbol)
    sl_dec = Decimal("0")
    tp_dec = Decimal("0")

    if ref_price is None or (isinstance(ref_price, float) and math.isnan(ref_price)):
        return sl_dec, tp_dec

    if side == "BUY":
        if sl_pips > 0:
            sl_px = _round_price(ref_price - sl_pips * pip, symbol)
            if sl_px > 0 and sl_px < ref_price:
                sl_dec = Decimal(str(sl_px))
        if tp_pips > 0:
            tp_px = _round_price(ref_price + tp_pips * pip, symbol)
            if tp_px > 0 and tp_px > ref_price:
                tp_dec = Decimal(str(tp_px))
    elif side == "SELL":
        if sl_pips > 0:
            sl_px = _round_price(ref_price + sl_pips * pip, symbol)
            if sl_px > 0 and sl_px > ref_price:
                sl_dec = Decimal(str(sl_px))
        if tp_pips > 0:
            tp_px = _round_price(ref_price - tp_pips * pip, symbol)
            if tp_px > 0 and tp_px < ref_price:
                tp_dec = Decimal(str(tp_px))
    else:
        raise ValueError(f"side no soportado: {side}")

    return sl_dec, tp_dec


def _import_pyeventbt_runtime():
    from pyeventbt import SignalEvent
    return SignalEvent


def make_rules_engine(
    *,
    strategy,
    strategy_id: str,
    signal_timeframe,
    rules_long,
    rules_short,
    enable_long: bool = True,
    enable_short: bool = True,
    collision_mode: str = "skip",  # "skip" | "close_all"
    one_trade_per_day: bool = False,
    sl_pips: float = 0.0,
    tp_pips: float = 0.0,
    atr_filter_enabled: bool = False,
    atr_period: int = 14,
    atr_method: str = "sma",
    atr_mode: str = "pct",  # "pct" or "raw"
    atr_bin_edges: Optional[List[float]] = None,
    atr_allowed_bins: Optional[Set[int]] = None,
    adx_filter_enabled: bool = False,
    adx_period: int = 14,
    adx_bin_edges: Optional[List[float]] = None,
    adx_allowed_bins: Optional[Set[int]] = None,
    returns_filter_enabled: bool = False,
    returns_lookback: int = 150,
    returns_percentile: float = 0.80,
    volatility_filter_enabled: bool = False,
    volatility_lookback: int = 20,
    volatility_percentile: float = 0.80,
    volatility_threshold_lookback: int = 150,
    apply_filters_to_open_positions: bool = False,
    log_filter_blocks: bool = False,
    precomputed_features: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Registra en `strategy` un custom signal engine basado en reglas.

    Semantica de trading:
    - LONG activo si cualquier regla long activa (OR).
    - SHORT activo si cualquier regla short activa (OR).
    - Si long y short activos a la vez: `collision_mode`.
    """
    if not enable_long and not enable_short:
        raise ValueError("enable_long y enable_short no pueden ser ambos False.")

    if collision_mode not in {"skip", "close_all"}:
        raise ValueError("collision_mode debe ser 'skip' o 'close_all'.")

    if logger is None:
        logger = logging.getLogger("pyeventbt")

    SignalEvent = _import_pyeventbt_runtime()

    compiled_long = compile_rules(rules_long, name="rules_long") if enable_long else []
    compiled_short = compile_rules(rules_short, name="rules_short") if enable_short else []

    specs_long = _extract_needed_periods(compiled_long)
    specs_short = _extract_needed_periods(compiled_short)
    specs = {
        "periods": set().union(specs_long["periods"], specs_short["periods"]),
        "breakout_periods": set().union(specs_long["breakout_periods"], specs_short["breakout_periods"]),
        "seq_periods": set().union(specs_long["seq_periods"], specs_short["seq_periods"]),
    }
    bars_needed = _compute_bars_needed(specs, safety_margin=20)

    ATR = None
    ADX = None
    if atr_filter_enabled or adx_filter_enabled:
        from pyeventbt.indicators.indicators import ATR as _ATR, ADX as _ADX
        ATR, ADX = _ATR, _ADX

    if atr_bin_edges is None:
        atr_bin_edges = [0.35, 0.80, 1.30] if atr_mode == "pct" else []
    if adx_bin_edges is None:
        adx_bin_edges = [15, 20, 25, 35, 50]

    precomp_df: Optional[pd.DataFrame] = None
    if precomputed_features is not None and not precomputed_features.empty:
        precomp_df = precomputed_features.copy()
        if not isinstance(precomp_df.index, pd.DatetimeIndex):
            precomp_df.index = pd.to_datetime(precomp_df.index, errors="coerce")
        precomp_df = precomp_df[~precomp_df.index.isna()]
        precomp_df = precomp_df.sort_index()
        # Dict de acceso O(1) por timestamp
        precomp_df = precomp_df.replace([np.inf, -np.inf], np.nan)

    state = {
        "position_side": defaultdict(lambda: None),
        "current_trading_date": defaultdict(lambda: None),
        "orders_placed_today": defaultdict(lambda: False),
        "cache": {},
    }

    def _bin_index(value: float, edges: List[float]) -> int:
        b = 0
        for e in edges:
            if value > e:
                b += 1
            else:
                break
        return b

    @strategy.custom_signal_engine(strategy_id=strategy_id, strategy_timeframes=[signal_timeframe])
    def rules_engine(event, modules):
        symbol = event.symbol
        signal_events = []

        open_positions = modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(symbol)
        pending_orders = modules.PORTFOLIO.get_number_of_strategy_pending_orders_by_symbol(symbol)

        if open_positions["TOTAL"] == 0:
            state["position_side"][symbol] = None

        if one_trade_per_day:
            current_date = event.datetime.date()
            if state["current_trading_date"][symbol] != current_date:
                state["current_trading_date"][symbol] = current_date
                state["orders_placed_today"][symbol] = False
            if open_positions["TOTAL"] == 0 and state["orders_placed_today"][symbol]:
                return signal_events

        cache_key = (symbol, event.datetime)
        if cache_key in state["cache"]:
            cached = state["cache"][cache_key]
            feature_values = cached["feature_values"]
            ref_price = cached["ref_price"]
            atr_val = cached["atr_val"]
            atr_bin = cached["atr_bin"]
            adx_val = cached["adx_val"]
            adx_bin = cached["adx_bin"]
            ret_abs = cached["ret_abs"]
            ret_thr = cached["ret_thr"]
            ret_extreme = cached["ret_extreme"]
            vol_now = cached["vol_now"]
            vol_thr = cached["vol_thr"]
            vol_extreme = cached["vol_extreme"]
        else:
            bars = modules.DATA_PROVIDER.get_latest_bars(symbol, signal_timeframe, bars_needed)
            if bars is None or bars.height < bars_needed:
                return signal_events

            df = bars.to_pandas()
            df.columns = [str(c).lower() for c in df.columns]
            if not {"open", "high", "low", "close"}.issubset(set(df.columns)):
                return signal_events

            if precomp_df is not None and event.datetime in precomp_df.index:
                row = precomp_df.loc[event.datetime]
                # Si hay duplicados de índice, coger la última fila
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                feature_values = {
                    str(k): (float(v) if pd.notna(v) else float("nan"))
                    for k, v in row.items()
                }
            else:
                feature_values = _compute_feature_values_lastrow(df, specs)
            close = df["close"].to_numpy(dtype=float)
            high = df["high"].to_numpy(dtype=float)
            low = df["low"].to_numpy(dtype=float)
            ref_price = float(close[-1]) if close.size else float("nan")

            atr_val = None
            atr_bin = None
            if atr_filter_enabled:
                atr_arr = ATR.compute(high, low, close, period=int(atr_period), method=str(atr_method))
                atr_raw = float(atr_arr[-1])
                if atr_mode == "pct":
                    close_last = float(close[-1])
                    atr_val = (atr_raw / close_last) * 100.0 if close_last != 0 else float("nan")
                else:
                    atr_val = atr_raw
                atr_bin = _bin_index(atr_val, atr_bin_edges) if atr_bin_edges and not math.isnan(atr_val) else 0

            adx_val = None
            adx_bin = None
            if adx_filter_enabled:
                adx_arr, _plus_di, _minus_di = ADX.compute(high, low, close, period=int(adx_period))
                adx_val = float(adx_arr[-1])
                adx_bin = _bin_index(adx_val, adx_bin_edges) if adx_bin_edges and not math.isnan(adx_val) else 0

            ret_abs = None
            ret_thr = None
            ret_extreme = None
            if returns_filter_enabled:
                if close.size >= int(returns_lookback) + 2:
                    logret = np.diff(np.log(close))
                    abs_lr = np.abs(logret)
                    ret_abs = float(abs_lr[-1])
                    window = abs_lr[-int(returns_lookback):]
                    q = float(returns_percentile)
                    if not (0.0 <= q <= 1.0):
                        raise ValueError(f"returns_percentile debe estar en [0,1], recibido {returns_percentile}")
                    ret_thr = float(np.quantile(window, q))
                    ret_extreme = bool(ret_abs > ret_thr) if not (math.isnan(ret_abs) or math.isnan(ret_thr)) else False
                else:
                    ret_extreme = False

            vol_now = None
            vol_thr = None
            vol_extreme = None
            if volatility_filter_enabled:
                vol_lb = int(volatility_lookback)
                thr_lb = int(volatility_threshold_lookback)
                qv = float(volatility_percentile)
                if not (0.0 <= qv <= 1.0):
                    raise ValueError(
                        f"volatility_percentile debe estar en [0,1], recibido {volatility_percentile}"
                    )
                if vol_lb < 2:
                    raise ValueError(f"volatility_lookback debe ser >= 2, recibido {volatility_lookback}")
                if thr_lb < 10:
                    raise ValueError(
                        f"volatility_threshold_lookback debe ser >= 10, recibido {volatility_threshold_lookback}"
                    )

                if close.size >= vol_lb + 2:
                    logret = np.diff(np.log(close))
                    vol_series = (
                        pd.Series(logret, copy=False)
                        .rolling(vol_lb, min_periods=vol_lb)
                        .std(ddof=0)
                        .to_numpy(dtype=float)
                    )
                    vol_series = vol_series[np.isfinite(vol_series)]
                    if vol_series.size > 0:
                        vol_now = float(vol_series[-1])
                        hist_n = min(int(thr_lb), int(vol_series.size))
                        hist = vol_series[-hist_n:]
                        if hist.size >= 10:
                            vol_thr = float(np.quantile(hist, qv))
                            vol_extreme = bool(vol_now > vol_thr) if not math.isnan(vol_thr) else False
                        else:
                            vol_extreme = False
                    else:
                        vol_extreme = False
                else:
                    vol_extreme = False

            state["cache"][cache_key] = {
                "feature_values": feature_values,
                "ref_price": ref_price,
                "atr_val": atr_val,
                "atr_bin": atr_bin,
                "adx_val": adx_val,
                "adx_bin": adx_bin,
                "ret_abs": ret_abs,
                "ret_thr": ret_thr,
                "ret_extreme": ret_extreme,
                "vol_now": vol_now,
                "vol_thr": vol_thr,
                "vol_extreme": vol_extreme,
            }

        long_active = evaluate_any_rule(compiled_long, feature_values) if enable_long else False
        short_active = evaluate_any_rule(compiled_short, feature_values) if enable_short else False

        trade_allowed = True
        if atr_filter_enabled and atr_allowed_bins is not None:
            trade_allowed = trade_allowed and (atr_bin in atr_allowed_bins)
        if adx_filter_enabled and adx_allowed_bins is not None:
            trade_allowed = trade_allowed and (adx_bin in adx_allowed_bins)
        if returns_filter_enabled:
            trade_allowed = trade_allowed and (not bool(ret_extreme))
        if volatility_filter_enabled:
            trade_allowed = trade_allowed and (not bool(vol_extreme))

        # Colision de senales long/short
        if enable_long and enable_short and long_active and short_active:
            if collision_mode == "close_all":
                if open_positions["TOTAL"] > 0:
                    modules.EXECUTION_ENGINE.close_all_strategy_positions()
                    state["position_side"][symbol] = None
                if pending_orders["TOTAL"] > 0:
                    modules.EXECUTION_ENGINE.cancel_all_strategy_pending_orders()
            return signal_events

        # Gestion de posicion abierta
        if open_positions["TOTAL"] > 0:
            current_side = state["position_side"][symbol]
            if apply_filters_to_open_positions and (not trade_allowed):
                modules.EXECUTION_ENGINE.close_all_strategy_positions()
                state["position_side"][symbol] = None
                return signal_events

            if current_side == "LONG":
                if (not enable_long) or (not long_active):
                    modules.EXECUTION_ENGINE.close_all_strategy_positions()
                    state["position_side"][symbol] = None
                return signal_events

            if current_side == "SHORT":
                if (not enable_short) or (not short_active):
                    modules.EXECUTION_ENGINE.close_all_strategy_positions()
                    state["position_side"][symbol] = None
                return signal_events

            return signal_events

        # Flat: no abrir si hay pendientes o filtro bloquea
        if pending_orders["TOTAL"] > 0:
            return signal_events
        if not trade_allowed:
            if log_filter_blocks and (long_active or short_active):
                logger.info(
                    f"{event.datetime} - BLOQUEADO filtro {symbol} "
                    f"ATR={atr_val} bin={atr_bin} | ADX={adx_val} bin={adx_bin} | "
                    f"ret_abs={ret_abs} thr={ret_thr} extreme={ret_extreme} | "
                    f"vol_now={vol_now} vol_thr={vol_thr} vol_extreme={vol_extreme}"
                )
            return signal_events

        if enable_long and long_active and (not short_active):
            time_generated = (
                event.datetime + signal_timeframe.to_timedelta()
                if modules.TRADING_CONTEXT == "BACKTEST"
                else datetime.now()
            )
            sl_dec, tp_dec = _sl_tp_prices_from_pips(
                ref_price=ref_price,
                symbol=symbol,
                side="BUY",
                sl_pips=float(sl_pips),
                tp_pips=float(tp_pips),
            )
            signal_events.append(
                SignalEvent(
                    symbol=symbol,
                    time_generated=time_generated,
                    strategy_id=strategy_id,
                    signal_type="BUY",
                    order_type="MARKET",
                    sl=sl_dec,
                    tp=tp_dec,
                )
            )
            state["position_side"][symbol] = "LONG"
            if one_trade_per_day:
                state["orders_placed_today"][symbol] = True
            return signal_events

        if enable_short and short_active and (not long_active):
            time_generated = (
                event.datetime + signal_timeframe.to_timedelta()
                if modules.TRADING_CONTEXT == "BACKTEST"
                else datetime.now()
            )
            sl_dec, tp_dec = _sl_tp_prices_from_pips(
                ref_price=ref_price,
                symbol=symbol,
                side="SELL",
                sl_pips=float(sl_pips),
                tp_pips=float(tp_pips),
            )
            signal_events.append(
                SignalEvent(
                    symbol=symbol,
                    time_generated=time_generated,
                    strategy_id=strategy_id,
                    signal_type="SELL",
                    order_type="MARKET",
                    sl=sl_dec,
                    tp=tp_dec,
                )
            )
            state["position_side"][symbol] = "SHORT"
            if one_trade_per_day:
                state["orders_placed_today"][symbol] = True
            return signal_events

        return signal_events

    return rules_engine

