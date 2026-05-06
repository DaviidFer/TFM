from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime
from decimal import Decimal
import logging
import shutil
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .rules_engine import make_rules_engine
from app.toolbox.indicators import build_feature_library


def _patch_symbol_info_for_non_fx(symbol: str, account_currency: str = "USD") -> None:
    """
    PyEventBT (CSV provider) asume que mt5.symbol_info(symbol) siempre existe.
    Para acciones como AAPL puede devolver None en el simulador y romper.
    Este parche devuelve un SymbolInfo minimo para simbolos desconocidos.
    """
    from pyeventbt.broker.mt5_broker.mt5_simulator_wrapper import Mt5SimulatorWrapper as mt5

    def _make_stock_stub(sym: str, spread_points: int = 1) -> SimpleNamespace:
        # Usar Decimal en campos numéricos de trading para evitar TypeError
        # en operaciones internas de PyEventBT (Decimal * float, comparaciones, etc.)
        return SimpleNamespace(
            name=sym,
            digits=2,
            visible=True,
            select=True,
            currency_margin=account_currency,
            currency_profit=account_currency,
            spread=int(spread_points),
            volume_min=Decimal("1.0"),
            volume_max=Decimal("1000000.0"),
            volume_step=Decimal("1.0"),
            trade_contract_size=Decimal("1.0"),
            margin_initial=Decimal("1.0"),
        )

    # Evitar parchear multiples veces
    if getattr(mt5, "_cursor_symbol_info_patched", False):
        # actualizar/insertar symbol solicitado por si no estaba
        fallback_map = getattr(mt5, "_cursor_fallback_symbol_info", {})
        # refrescar siempre para evitar stubs antiguos con floats
        fallback_map[symbol] = _make_stock_stub(symbol)
        mt5._cursor_fallback_symbol_info = fallback_map
        return

    original_symbol_info = mt5.symbol_info
    fallback_map = {
        symbol: _make_stock_stub(symbol)
    }

    def symbol_info_patched(sym: str):
        info = original_symbol_info(sym)
        if info is not None:
            return info
        return fallback_map.get(sym, _make_stock_stub(sym))

    mt5.symbol_info = staticmethod(symbol_info_patched)
    mt5._cursor_symbol_info_patched = True
    mt5._cursor_original_symbol_info = original_symbol_info
    mt5._cursor_fallback_symbol_info = fallback_map


def get_timeframe_from_filename(filename: str):
    """
    Extrae symbol_base y timeframe de nombre tipo:
      EURUSD_D1, AUDCHF_H4, GBPUSD_H1, XAUUSD_M30
    """
    from pyeventbt import StrategyTimeframes

    symbol_base = filename.split("_")[0].upper()
    fup = filename.upper()

    if "_H4" in fup:
        tf = StrategyTimeframes.FOUR_HOUR
    elif "_H1" in fup:
        tf = StrategyTimeframes.ONE_HOUR
    elif "_H2" in fup:
        tf = StrategyTimeframes.TWO_HOUR
    elif "_H3" in fup:
        tf = StrategyTimeframes.THREE_HOUR
    elif "_H6" in fup:
        tf = StrategyTimeframes.SIX_HOUR
    elif "_H8" in fup:
        tf = StrategyTimeframes.EIGHT_HOUR
    elif "_H12" in fup:
        tf = StrategyTimeframes.TWELVE_HOUR
    elif "_M30" in fup:
        tf = StrategyTimeframes.THIRTY_MIN
    elif "_M15" in fup:
        tf = StrategyTimeframes.FIFTEEN_MIN
    elif "_M10" in fup:
        tf = StrategyTimeframes.TEN_MIN
    elif "_M5" in fup:
        tf = StrategyTimeframes.FIVE_MIN
    elif "_D1" in fup or ("_H" not in fup and "_M" not in fup):
        tf = StrategyTimeframes.ONE_DAY
    else:
        raise ValueError(f"No se pudo inferir timeframe de '{filename}'.")

    return symbol_base, tf


def infer_csv_filename_from_asset_path(asset_csv_path: str, timeframe_suffix: str = "D1") -> str:
    """
    Convierte una ruta de activo (ej. datos/Stocks/AAPL.csv) al formato usado
    por PyEventBT en este proyecto (ej. AAPL_D1).
    """
    stem = Path(asset_csv_path).stem.upper()
    return f"{stem}_{timeframe_suffix}"


def _dedupe_logger_handlers() -> None:
    """
    Evita logs duplicados por acumulación de handlers entre ejecuciones en notebook.
    """
    for name in ("pyeventbt", "backtest_info"):
        lg = logging.getLogger(name)
        # limpiar handlers previos; pyeventbt volverá a configurar lo necesario
        lg.handlers.clear()
        lg.propagate = True


def _load_ohlc_from_asset_csv(asset_csv_path: str) -> pd.DataFrame:
    p = Path(asset_csv_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    df = pd.read_csv(p)
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    }
    df = df.rename(columns=rename_map)
    needed = ["date", "open", "high", "low", "close"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"CSV activo sin columnas {miss}: {p}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _asset_name_from_asset_csv(asset_csv_path: str) -> str:
    p = Path(asset_csv_path)
    return p.stem.upper()


def _rules_to_list(rules_obj: Any) -> List[str]:
    if rules_obj is None:
        return []
    if isinstance(rules_obj, pd.DataFrame):
        if "regla" in rules_obj.columns:
            s = rules_obj["regla"]
        elif "rule" in rules_obj.columns:
            s = rules_obj["rule"]
        else:
            raise ValueError("DataFrame de reglas sin columna 'regla' ni 'rule'.")
        return s.dropna().astype(str).tolist()
    if isinstance(rules_obj, pd.Series):
        return rules_obj.dropna().astype(str).tolist()
    if isinstance(rules_obj, (list, tuple, set)):
        return pd.Series(list(rules_obj)).dropna().astype(str).tolist()
    raise ValueError(f"Formato de reglas no soportado: {type(rules_obj)}")


def _normalize_numeric_strategy_id(strategy_id: Any) -> str:
    """
    PyEventBT exige strategy_id convertible a int.
    Si llega un id no numerico (ej. "F1_0"), lo convertimos a un id numerico estable.
    """
    s = str(strategy_id).strip()
    if s.isdigit():
        return s
    # ID estable y reproducible en [100000, 999999999]
    numeric = 100000 + (zlib.crc32(s.encode("utf-8")) % 900000000)
    return str(int(numeric))


def _write_rules_csv(path: Path, rules: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "rule_id": range(len(rules)),
            "rule": rules,
        }
    ).to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


@contextmanager
def prepare_backtest_data(
    csv_dir: str,
    csv_filename: str,
    asset_csv_path: Optional[str] = None,
    stock_spread_usd: float = 0.01,
):
    """
    Copia temporalmente `{csv_filename}.csv` como `{symbol_base}.csv`
    para el formato esperado por PyEventBT.
    """
    import shutil

    csv_path = Path(csv_dir)
    raw = csv_filename[:-4] if csv_filename.lower().endswith(".csv") else csv_filename
    symbol_base, timeframe = get_timeframe_from_filename(raw)

    src = csv_path / f"{raw}.csv"
    dst = csv_path / f"{symbol_base}.csv"
    bak = csv_path / f"{symbol_base}.csv.backup"

    def _build_pyeventbt_csv_from_asset(asset_path: Path, out_path: Path, spread_usd: float) -> None:
        """
        Convierte CSV estilo Yahoo/Stocks a formato PyEventBT sin cabecera:
        date,time,open,high,low,close,tickvol,volume,spread
        """
        if not asset_path.exists():
            raise FileNotFoundError(f"No se encuentra el CSV del activo: {asset_path}")

        df = pd.read_csv(asset_path)
        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        df = df.rename(columns=rename_map)

        needed = ["date", "open", "high", "low", "close"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"El CSV del activo no tiene columnas requeridas {missing}: {asset_path}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df = df.sort_values("date")

        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).copy()

        if "volume" in df.columns:
            vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        else:
            vol = pd.Series(0, index=df.index, dtype="int64")

        # PyEventBT espera date/time separados y sin cabecera.
        out = pd.DataFrame(
            {
                "date": df["date"].dt.strftime("%Y.%m.%d"),
                "time": "00:00:00",
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "tickvol": vol.clip(lower=0).astype("int64"),
                "volume": vol.clip(lower=0).astype("int64"),
                # spread en "points" con 2 dígitos para stocks (0.01 USD = 1 point)
                "spread": max(0, int(round(float(spread_usd) * 100.0))),
            }
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, header=False)

    # Si hay CSV del activo, regenerar siempre el CSV de backtest (evita stale y asegura spread actual)
    if asset_csv_path is not None:
        asset_path = Path(asset_csv_path)
        if not asset_path.is_absolute():
            asset_path = Path.cwd() / asset_path
        _build_pyeventbt_csv_from_asset(asset_path=asset_path, out_path=src, spread_usd=stock_spread_usd)
    elif not src.exists():
        raise FileNotFoundError(
            f"No se encuentra el CSV fuente: {src}. "
            "Pasa asset_csv_path para generarlo automáticamente."
        )

    try:
        if dst.exists() and not bak.exists():
            shutil.copy2(dst, bak)
        shutil.copy2(src, dst)
        yield symbol_base, timeframe
    finally:
        if bak.exists():
            shutil.copy2(bak, dst)
            bak.unlink()


def run_event_backtest(
    *,
    csv_dir: str,
    winners_long_stable,
    winners_short_stable,
    csv_filename: Optional[str] = None,
    asset_csv_path: str = "datos/Stocks/AAPL.csv",
    timeframe_suffix: str = "D1",
    # backward compatibility (si venias usando estos nombres):
    rules_long=None,
    rules_short=None,
    strategy_id: str = "1234",
    start_date: datetime = datetime(2017, 1, 1),
    end_date: datetime = datetime(2025, 12, 31),
    initial_capital: float = 100000.0,
    account_currency: str = "USD",
    backtest_name: Optional[str] = None,
    export_backtest_csv: bool = True,
    export_backtest_parquet: bool = True,
    # Engine params passthrough
    enable_long: bool = True,
    enable_short: bool = True,
    collision_mode: str = "close_all",
    one_trade_per_day: bool = False,
    sl_pips: float = 0.0,
    tp_pips: float = 0.0,
    atr_filter_enabled: bool = False,
    adx_filter_enabled: bool = False,
    returns_filter_enabled: bool = False,
    returns_lookback: int = 150,
    returns_percentile: float = 0.80,
    volatility_filter_enabled: bool = False,
    volatility_lookback: int = 20,
    volatility_percentile: float = 0.80,
    volatility_threshold_lookback: int = 150,
    stock_spread_usd: float = 0.01,
    systems_root_dir: str = "systems",
    save_system_artifacts: bool = True,
    system_name: Optional[str] = None,
    verbose: bool = True,
):
    """
    Runner high-level:
    - prepara CSV para symbol base
    - crea Strategy
    - registra engine de reglas OR (long/short)
    - ejecuta backtest
    """
    # Politica fija de colision para este proyecto:
    # solo se admite close_all (skip queda descartado).
    if str(collision_mode) != "close_all":
        if verbose:
            print(
                f"[backtest_eventos] collision_mode='{collision_mode}' ignorado; "
                "se fuerza 'close_all'."
            )
        collision_mode = "close_all"

    strategy_id_raw = str(strategy_id)
    strategy_id_num = _normalize_numeric_strategy_id(strategy_id_raw)
    if verbose and strategy_id_raw != strategy_id_num:
        print(
            f"[backtest_eventos] strategy_id='{strategy_id_raw}' no numerico; "
            f"usando strategy_id='{strategy_id_num}'."
        )

    # Reglas finales a usar en backtest: winners_*_stable
    rules_long_final = winners_long_stable if winners_long_stable is not None else rules_long
    rules_short_final = winners_short_stable if winners_short_stable is not None else rules_short
    if rules_long_final is None or rules_short_final is None:
        raise ValueError(
            "Debes pasar winners_long_stable y winners_short_stable "
            "(o rules_long/rules_short para compatibilidad)."
        )

    # Hardcode seguro desde el activo cargado en notebook, salvo override explicito.
    # Ej: datos/Stocks/AAPL.csv -> AAPL_D1
    if csv_filename is None:
        csv_filename = infer_csv_filename_from_asset_path(
            asset_csv_path=asset_csv_path,
            timeframe_suffix=timeframe_suffix,
        )

    asset_name = _asset_name_from_asset_csv(asset_csv_path)
    system_dir = Path(systems_root_dir) / asset_name
    if system_name is not None and len(str(system_name).strip()) > 0:
        system_dir = system_dir / str(system_name)

    rules_dir = system_dir / "rules"
    backtests_dir = system_dir / "backtests"

    # Guardar reglas usadas por el sistema (formato rule_id, rule)
    if save_system_artifacts:
        _write_rules_csv(rules_dir / "winners_long_stable.csv", _rules_to_list(rules_long_final))
        _write_rules_csv(rules_dir / "winners_short_stable.csv", _rules_to_list(rules_short_final))
        backtests_dir.mkdir(parents=True, exist_ok=True)

    _dedupe_logger_handlers()

    from pyeventbt import (
        MinSizingConfig,
        PassthroughRiskConfig,
        Strategy,
    )

    with prepare_backtest_data(
        csv_dir=csv_dir,
        csv_filename=csv_filename,
        asset_csv_path=asset_csv_path,
        stock_spread_usd=stock_spread_usd,
    ) as (symbol_base, signal_timeframe):
        # Asegura symbol_info valido para activos no-FX (AAPL, MSFT, etc.)
        _patch_symbol_info_for_non_fx(symbol=symbol_base, account_currency=account_currency)

        # Precompute de features para acelerar el engine (evita recalcular indicadores por barra)
        asset_ohlc = _load_ohlc_from_asset_csv(asset_csv_path)
        precomp_features = build_feature_library(
            data_ohlc=asset_ohlc[["open", "high", "low", "close"]],
            dropna=False,
        )

        # Ajuste de spread realista de stocks: convertir USD a "points" según dígitos (2)
        spread_points = max(0, int(round(float(stock_spread_usd) * 100.0)))
        from pyeventbt.broker.mt5_broker.mt5_simulator_wrapper import Mt5SimulatorWrapper as mt5
        s_info = mt5.symbol_info(symbol_base)
        if s_info is not None:
            try:
                s_info.spread = spread_points
            except Exception:
                pass

        strategy = Strategy()

        make_rules_engine(
            strategy=strategy,
            strategy_id=strategy_id_num,
            signal_timeframe=signal_timeframe,
            rules_long=rules_long_final,
            rules_short=rules_short_final,
            enable_long=enable_long,
            enable_short=enable_short,
            collision_mode=collision_mode,
            one_trade_per_day=one_trade_per_day,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            atr_filter_enabled=atr_filter_enabled,
            adx_filter_enabled=adx_filter_enabled,
            returns_filter_enabled=returns_filter_enabled,
            returns_lookback=returns_lookback,
            returns_percentile=returns_percentile,
            volatility_filter_enabled=volatility_filter_enabled,
            volatility_lookback=volatility_lookback,
            volatility_percentile=volatility_percentile,
            volatility_threshold_lookback=volatility_threshold_lookback,
            precomputed_features=precomp_features,
        )

        strategy.configure_predefined_sizing_engine(MinSizingConfig())
        strategy.configure_predefined_risk_engine(PassthroughRiskConfig())

        bt_name = backtest_name or strategy_id
        if verbose:
            print(
                f"[backtest_eventos] symbol={symbol_base} tf={signal_timeframe.value} "
                f"from={start_date.date()} to={end_date.date()} | csv_filename={csv_filename}"
            )

        backtest = strategy.backtest(
            strategy_id=strategy_id_num,
            initial_capital=initial_capital,
            symbols_to_trade=[symbol_base],
            csv_dir=csv_dir,
            backtest_name=bt_name,
            start_date=start_date,
            end_date=end_date,
            export_backtest_csv=export_backtest_csv,
            export_backtest_parquet=export_backtest_parquet,
            backtest_results_dir=(str(backtests_dir) if save_system_artifacts else None),
            account_currency=account_currency,
        )
        return backtest


def run_integration_grid_backtests(
    *,
    csv_dir: str,
    asset_csv_path: str,
    winners_long_stable,
    winners_short_stable,
    collision_modes: Sequence[str] = ("close_all",),
    systems_root_dir: str = "systems",
    initial_capital: float = 100000.0,
    start_date: datetime = datetime(2017, 1, 1),
    end_date: datetime = datetime(2025, 12, 31),
    stock_spread_usd: float = 0.01,
    export_backtest_csv: bool = True,
    export_backtest_parquet: bool = True,
    one_trade_per_day: bool = False,
    sl_pips: float = 0.0,
    tp_pips: float = 0.0,
    atr_filter_enabled: bool = False,
    adx_filter_enabled: bool = False,
    returns_filter_enabled: bool = False,
    returns_lookback: int = 150,
    returns_percentile: float = 0.80,
    volatility_filter_enabled: bool = False,
    volatility_lookback: int = 20,
    volatility_percentile: float = 0.80,
    volatility_threshold_lookback: int = 150,
    save_system_artifacts: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta grid:
      LONG_ONLY / SHORT_ONLY / COMBINADO  x  collision_mode(s)
    y guarda artefactos en systems/<ACTIVO>/...
    """
    configs = [
        {"name": "LONG_ONLY", "enable_long": True, "enable_short": False, "long_rules": winners_long_stable, "short_rules": []},
        {"name": "SHORT_ONLY", "enable_long": False, "enable_short": True, "long_rules": [], "short_rules": winners_short_stable},
        {"name": "COMBINADO", "enable_long": True, "enable_short": True, "long_rules": winners_long_stable, "short_rules": winners_short_stable},
    ]

    rows: List[Dict[str, Any]] = []
    asset_name = _asset_name_from_asset_csv(asset_csv_path)

    # collision_mode fijo: close_all
    coll = "close_all"
    for i, cfg in enumerate(configs):
        bt_name = f"{asset_name}_{cfg['name']}_{coll}"
        bt = run_event_backtest(
            csv_dir=csv_dir,
            asset_csv_path=asset_csv_path,
            winners_long_stable=cfg["long_rules"],
            winners_short_stable=cfg["short_rules"],
            enable_long=cfg["enable_long"],
            enable_short=cfg["enable_short"],
            collision_mode=coll,
            strategy_id=f"93{i}1",
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            one_trade_per_day=one_trade_per_day,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            atr_filter_enabled=atr_filter_enabled,
            adx_filter_enabled=adx_filter_enabled,
            returns_filter_enabled=returns_filter_enabled,
            returns_lookback=returns_lookback,
            returns_percentile=returns_percentile,
            volatility_filter_enabled=volatility_filter_enabled,
            volatility_lookback=volatility_lookback,
            volatility_percentile=volatility_percentile,
            volatility_threshold_lookback=volatility_threshold_lookback,
            stock_spread_usd=stock_spread_usd,
            export_backtest_csv=export_backtest_csv,
            export_backtest_parquet=export_backtest_parquet,
            systems_root_dir=systems_root_dir,
            save_system_artifacts=save_system_artifacts,
            system_name=(f"{cfg['name']}" if save_system_artifacts else None),
            backtest_name=bt_name,
            verbose=verbose,
        )

        pnl = bt.pnl.copy()
        final_balance = float(pnl["BALANCE"].iloc[-1]) if "BALANCE" in pnl.columns and len(pnl) else float("nan")
        final_equity = float(pnl["EQUITY"].iloc[-1]) if "EQUITY" in pnl.columns and len(pnl) else float("nan")
        n_trades = len(bt.trades) if hasattr(bt, "trades") else None

        rows.append(
            {
                "config": cfg["name"],
                "collision_mode": coll,
                "n_long_rules": len(_rules_to_list(cfg["long_rules"])),
                "n_short_rules": len(_rules_to_list(cfg["short_rules"])),
                "final_balance": final_balance,
                "final_equity": final_equity,
                "realized_pnl": (final_balance - float(initial_capital)) if pd.notna(final_balance) else float("nan"),
                "n_trades": n_trades,
                "system_path": str((Path(systems_root_dir) / asset_name / cfg["name"]).resolve()),
            }
        )

    df = pd.DataFrame(rows).sort_values(["config", "collision_mode"]).reset_index(drop=True)
    return df


def run_volatility_filter_grid_backtests(
    *,
    csv_dir: str,
    asset_csv_path: str,
    winners_long_stable,
    winners_short_stable,
    volatility_lookbacks: Sequence[int] = (10, 20, 30, 50),
    volatility_percentiles: Sequence[float] = (0.70, 0.80, 0.90),
    include_no_filter: bool = True,
    collision_modes: Sequence[str] = ("close_all",),
    systems_root_dir: str = "systems",
    initial_capital: float = 100000.0,
    start_date: datetime = datetime(2017, 1, 1),
    end_date: datetime = datetime(2025, 12, 31),
    stock_spread_usd: float = 0.01,
    export_backtest_csv: bool = True,
    export_backtest_parquet: bool = True,
    one_trade_per_day: bool = False,
    sl_pips: float = 0.0,
    tp_pips: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid search del filtro de volatilidad (desviacion estandar de log-returns).
    Ejecuta para cada combinacion de (lookback, percentile) el grid de integracion:
      LONG_ONLY / SHORT_ONLY / COMBINADO x collision_mode(s).
    """
    rows: List[pd.DataFrame] = []

    if include_no_filter:
        base = run_integration_grid_backtests(
            csv_dir=csv_dir,
            asset_csv_path=asset_csv_path,
            winners_long_stable=winners_long_stable,
            winners_short_stable=winners_short_stable,
            collision_modes=("close_all",),
            systems_root_dir=systems_root_dir,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            stock_spread_usd=stock_spread_usd,
            export_backtest_csv=export_backtest_csv,
            export_backtest_parquet=export_backtest_parquet,
            one_trade_per_day=one_trade_per_day,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            volatility_filter_enabled=False,
            verbose=verbose,
        )
        base["volatility_filter_enabled"] = False
        base["volatility_lookback"] = None
        base["volatility_percentile"] = None
        rows.append(base)

    for lb in volatility_lookbacks:
        for q in volatility_percentiles:
            df_run = run_integration_grid_backtests(
                csv_dir=csv_dir,
                asset_csv_path=asset_csv_path,
                winners_long_stable=winners_long_stable,
                winners_short_stable=winners_short_stable,
                collision_modes=("close_all",),
                systems_root_dir=systems_root_dir,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
                stock_spread_usd=stock_spread_usd,
                export_backtest_csv=export_backtest_csv,
                export_backtest_parquet=export_backtest_parquet,
                one_trade_per_day=one_trade_per_day,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                volatility_filter_enabled=True,
                volatility_lookback=int(lb),
                volatility_percentile=float(q),
                volatility_threshold_lookback=max(150, int(lb) * 3),
                verbose=verbose,
            )
            df_run["volatility_filter_enabled"] = True
            df_run["volatility_lookback"] = int(lb)
            df_run["volatility_percentile"] = float(q)
            rows.append(df_run)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, axis=0, ignore_index=True)
    sort_cols = ["realized_pnl", "final_balance"]
    out = out.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    return out


def run_two_stage_best_system_backtest(
    *,
    csv_dir: str,
    asset_csv_path: str,
    winners_long_stable,
    winners_short_stable,
    volatility_lookbacks: Sequence[int] = (20, 60, 90, 120),
    volatility_percentiles: Sequence[float] = (0.70, 0.80, 0.90),
    systems_root_dir: str = "systems",
    initial_capital: float = 100000.0,
    start_date: datetime = datetime(2017, 1, 1),
    end_date: datetime = datetime(2025, 12, 31),
    stock_spread_usd: float = 0.01,
    one_trade_per_day: bool = False,
    sl_pips: float = 0.0,
    tp_pips: float = 0.0,
    export_final_csv: bool = True,
    export_final_parquet: bool = True,
    clean_asset_dir_before_save: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Flujo en 2 fases:
      1) Seleccion de configuracion base: LONG_ONLY / SHORT_ONLY / COMBINADO
         (collision fija: close_all)
      2) Grid de filtro de volatilidad sobre la mejor configuracion de fase 1.
    Solo se guarda artefacto final en systems/<ASSET>/.
    """
    configs = [
        {
            "name": "LONG_ONLY",
            "enable_long": True,
            "enable_short": False,
            "long_rules": winners_long_stable,
            "short_rules": [],
        },
        {
            "name": "SHORT_ONLY",
            "enable_long": False,
            "enable_short": True,
            "long_rules": [],
            "short_rules": winners_short_stable,
        },
        {
            "name": "COMBINADO",
            "enable_long": True,
            "enable_short": True,
            "long_rules": winners_long_stable,
            "short_rules": winners_short_stable,
        },
    ]

    asset_name = _asset_name_from_asset_csv(asset_csv_path)

    # ===== FASE 1: mejor configuracion base =====
    phase1_rows: List[Dict[str, Any]] = []
    for i, cfg in enumerate(configs):
        bt = run_event_backtest(
            csv_dir=csv_dir,
            asset_csv_path=asset_csv_path,
            winners_long_stable=cfg["long_rules"],
            winners_short_stable=cfg["short_rules"],
            enable_long=cfg["enable_long"],
            enable_short=cfg["enable_short"],
            collision_mode="close_all",
            strategy_id=f"F1_{i}",
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            one_trade_per_day=one_trade_per_day,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            stock_spread_usd=stock_spread_usd,
            export_backtest_csv=False,
            export_backtest_parquet=False,
            systems_root_dir=systems_root_dir,
            save_system_artifacts=False,
            verbose=False,
        )
        pnl = bt.pnl.copy()
        final_balance = float(pnl["BALANCE"].iloc[-1]) if "BALANCE" in pnl.columns and len(pnl) else float("nan")
        final_equity = float(pnl["EQUITY"].iloc[-1]) if "EQUITY" in pnl.columns and len(pnl) else float("nan")
        n_trades = len(bt.trades) if hasattr(bt, "trades") else None
        phase1_rows.append(
            {
                "config": cfg["name"],
                "collision_mode": "close_all",
                "final_balance": final_balance,
                "final_equity": final_equity,
                "realized_pnl": (final_balance - float(initial_capital)) if pd.notna(final_balance) else float("nan"),
                "n_trades": n_trades,
            }
        )
    phase1_df = pd.DataFrame(phase1_rows).sort_values("realized_pnl", ascending=False).reset_index(drop=True)
    best_cfg_name = str(phase1_df.iloc[0]["config"])
    best_cfg = next(c for c in configs if c["name"] == best_cfg_name)

    if verbose:
        print(f"[fase1] mejor config: {best_cfg_name}")

    # ===== FASE 2: grid volatilidad solo sobre config ganadora =====
    phase2_rows: List[Dict[str, Any]] = []
    for lb in volatility_lookbacks:
        for q in volatility_percentiles:
            bt = run_event_backtest(
                csv_dir=csv_dir,
                asset_csv_path=asset_csv_path,
                winners_long_stable=best_cfg["long_rules"],
                winners_short_stable=best_cfg["short_rules"],
                enable_long=best_cfg["enable_long"],
                enable_short=best_cfg["enable_short"],
                collision_mode="close_all",
                strategy_id=f"F2_{int(lb)}_{int(round(float(q)*100))}",
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                one_trade_per_day=one_trade_per_day,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                volatility_filter_enabled=True,
                volatility_lookback=int(lb),
                volatility_percentile=float(q),
                volatility_threshold_lookback=max(150, int(lb) * 3),
                stock_spread_usd=stock_spread_usd,
                export_backtest_csv=False,
                export_backtest_parquet=False,
                systems_root_dir=systems_root_dir,
                save_system_artifacts=False,
                verbose=False,
            )
            pnl = bt.pnl.copy()
            final_balance = float(pnl["BALANCE"].iloc[-1]) if "BALANCE" in pnl.columns and len(pnl) else float("nan")
            final_equity = float(pnl["EQUITY"].iloc[-1]) if "EQUITY" in pnl.columns and len(pnl) else float("nan")
            n_trades = len(bt.trades) if hasattr(bt, "trades") else None
            phase2_rows.append(
                {
                    "config": best_cfg["name"],
                    "collision_mode": "close_all",
                    "volatility_lookback": int(lb),
                    "volatility_percentile": float(q),
                    "final_balance": final_balance,
                    "final_equity": final_equity,
                    "realized_pnl": (final_balance - float(initial_capital)) if pd.notna(final_balance) else float("nan"),
                    "n_trades": n_trades,
                }
            )
    phase2_df = pd.DataFrame(phase2_rows).sort_values("realized_pnl", ascending=False).reset_index(drop=True)
    best_vol = phase2_df.iloc[0].to_dict()

    if verbose:
        print(
            "[fase2] mejor volatilidad: "
            f"lookback={int(best_vol['volatility_lookback'])}, "
            f"percentile={float(best_vol['volatility_percentile']):.2f}"
        )

    # ===== Guardar SOLO sistema final =====
    asset_dir = Path(systems_root_dir) / asset_name
    if clean_asset_dir_before_save and asset_dir.exists():
        shutil.rmtree(asset_dir, ignore_errors=True)

    final_bt = run_event_backtest(
        csv_dir=csv_dir,
        asset_csv_path=asset_csv_path,
        winners_long_stable=best_cfg["long_rules"],
        winners_short_stable=best_cfg["short_rules"],
        enable_long=best_cfg["enable_long"],
        enable_short=best_cfg["enable_short"],
        collision_mode="close_all",
        strategy_id="FINAL",
        backtest_name=f"{asset_name}_{best_cfg['name']}_VOL",
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        one_trade_per_day=one_trade_per_day,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
        volatility_filter_enabled=True,
        volatility_lookback=int(best_vol["volatility_lookback"]),
        volatility_percentile=float(best_vol["volatility_percentile"]),
        volatility_threshold_lookback=max(150, int(best_vol["volatility_lookback"]) * 3),
        stock_spread_usd=stock_spread_usd,
        export_backtest_csv=export_final_csv,
        export_backtest_parquet=export_final_parquet,
        systems_root_dir=systems_root_dir,
        save_system_artifacts=True,
        system_name=None,
        verbose=verbose,
    )

    final_system_path = str((Path(systems_root_dir) / asset_name).resolve())
    return {
        "phase1_results": phase1_df,
        "phase2_results": phase2_df,
        "best_phase1": phase1_df.iloc[0].to_dict(),
        "best_volatility": best_vol,
        "final_system_path": final_system_path,
        "final_backtest": final_bt,
    }

