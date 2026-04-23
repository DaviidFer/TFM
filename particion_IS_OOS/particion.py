"""
Partición temporal IS/OOS para series con holdout opcional.
Requiere: pandas.
"""
import pandas as pd


def _pct_to_float(x):
    """
    Admite porcentajes en formato 0-1 o 0-100.
    """
    if isinstance(x, (int, float)):
        return float(x) / 100.0 if x > 1 else float(x)
    raise TypeError("IS_PCT y OOS_PCT deben ser numéricos (0-1 o 0-100).")


def split_is_oos_from_ready_df(
    df_full: pd.DataFrame,
    is_pct=0.50,
    oos_pct=0.50,
    holdout_year: int = 2025,
    holdout_enabled: bool = True,
    lookback_years: int = 10,
):
    """
    Split temporal para series diarias con esta lógica:

    Si holdout_enabled=True:
      - main_df    = los `lookback_years` años ANTERIORES al holdout_year
      - holdout_df = año holdout_year completo
      - final_df   = desde holdout_year + 1 en adelante

    Si no hay histórico suficiente para main_df,
    usa todo lo disponible antes del holdout.

    Devuelve
    --------
    is_df : pd.DataFrame
        IS (mitad reciente de main_df)
    oos_df : pd.DataFrame
        OOS (mitad antigua de main_df)
    holdout_df : pd.DataFrame
        Año holdout
    main_df : pd.DataFrame
        Bloque total de modelado (10 años previos al holdout)
    final_df : pd.DataFrame
        Datos posteriores al holdout (por ejemplo 2026+)
    """

    if not isinstance(df_full, pd.DataFrame) or df_full.empty:
        raise ValueError("df_full debe ser un DataFrame no vacío.")

    if not isinstance(df_full.index, pd.DatetimeIndex):
        raise TypeError("El índice de df_full debe ser DatetimeIndex.")

    if not isinstance(lookback_years, int) or lookback_years <= 0:
        raise ValueError("lookback_years debe ser un entero > 0.")

    is_pct = _pct_to_float(is_pct)
    oos_pct = _pct_to_float(oos_pct)

    if not (0 < is_pct < 1):
        raise ValueError(f"is_pct debe estar en (0,1). Valor recibido: {is_pct}")
    if not (0 < oos_pct < 1):
        raise ValueError(f"oos_pct debe estar en (0,1). Valor recibido: {oos_pct}")
    if abs((is_pct + oos_pct) - 1.0) > 1e-12:
        raise ValueError(f"is_pct + oos_pct debe sumar 1.0. Suma actual: {is_pct + oos_pct:.12f}")

    df_full = df_full.sort_index().copy()

    if holdout_enabled:
        main_start = pd.Timestamp(f"{holdout_year - lookback_years}-01-01")
        main_end = pd.Timestamp(f"{holdout_year - 1}-12-31")

        available_pre_holdout = df_full.loc[:main_end].copy()

        if available_pre_holdout.empty:
            raise ValueError("No hay datos anteriores al holdout para construir main_df.")

        main_df = available_pre_holdout.loc[available_pre_holdout.index >= main_start].copy()
        if main_df.empty:
            main_df = available_pre_holdout.copy()

        holdout_df = df_full.loc[
            f"{holdout_year}-01-01":f"{holdout_year}-12-31"
        ].copy()

        final_df = df_full.loc[
            f"{holdout_year + 1}-01-01":
        ].copy()

    else:
        holdout_df = df_full.iloc[0:0].copy()
        final_df = df_full.iloc[0:0].copy()

        max_date = df_full.index.max()
        cutoff_date = max_date - pd.DateOffset(years=lookback_years)
        main_df = df_full.loc[df_full.index >= cutoff_date].copy()

        if main_df.empty:
            main_df = df_full.copy()

    if len(main_df) < 10:
        raise ValueError(
            f"Datos insuficientes en main para hacer split IS/OOS. "
            f"Filas disponibles en main: {len(main_df)}"
        )

    n = len(main_df)
    n_oos = int(round(n * oos_pct))
    n_oos = max(1, min(n - 1, n_oos))

    oos_df = main_df.iloc[:n_oos].copy()
    is_df = main_df.iloc[n_oos:].copy()

    return is_df, oos_df, holdout_df, main_df, final_df


def run_particion_is_oos(
    df_full: pd.DataFrame,
    is_pct=0.50,
    oos_pct=0.50,
    holdout_year: int = 2025,
    holdout_enabled: bool = True,
    lookback_years: int = 10,
):
    """
    Ejecuta la partición IS/OOS con los parámetros por defecto (50% IS, 50% OOS,
    holdout 2025, 10 años de lookback). Devuelve (data, data_oos, data_2025, data_main, data_final).
    """
    return split_is_oos_from_ready_df(
        df_full=df_full,
        is_pct=is_pct,
        oos_pct=oos_pct,
        holdout_year=holdout_year,
        holdout_enabled=holdout_enabled,
        lookback_years=lookback_years,
    )
