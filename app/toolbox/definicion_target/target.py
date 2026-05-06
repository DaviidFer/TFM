"""
Definición del target a 1 vela vista (retorno open-to-open).
Requiere: pandas.
"""
import pandas as pd


def target_retorno_1vela(
    df: pd.DataFrame,
    out_col: str = "Target"
) -> pd.DataFrame:
    """
    Genera un target continuo simple a 1 vela vista.

    Definición
    ----------
    Para datos diarios:
        Target[t] = (Open[t+1] - Open[t]) / Open[t]

    Parámetros
    ----------
    df : DataFrame
        Debe contener al menos la columna 'open'.
    out_col : str
        Nombre de la columna target final.

    Notas
    -----
    - La última fila tendrá NaN, porque no existe t+1.
    - Solo crea la columna Target.
    - Pensado para datos diarios.

    Retorna
    -------
    DataFrame con la columna Target añadida.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df debe ser un DataFrame no vacío.")

    req = {"open"}
    faltan = req - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas requeridas: {faltan}")

    df = df.copy()
    open_now = pd.to_numeric(df["open"], errors="coerce")
    open_next = open_now.shift(-1)

    df[out_col] = (open_next - open_now) / open_now
    return df


def run_target_para_bloques(
    data: pd.DataFrame,
    data_oos: pd.DataFrame,
    data_2025: pd.DataFrame,
    data_final: pd.DataFrame,
    out_col: str = "Target",
    dropna_target: bool = True,
):
    """
    Aplica target_retorno_1vela a los cuatro bloques (data, data_oos, data_2025, data_final)
    y opcionalmente elimina las filas con Target NaN.

    Retorna
    -------
    data, data_oos, data_2025, data_final : DataFrames con columna Target.
    """
    data = target_retorno_1vela(data, out_col=out_col)
    data_oos = target_retorno_1vela(data_oos, out_col=out_col)
    data_2025 = target_retorno_1vela(data_2025, out_col=out_col)
    data_final = target_retorno_1vela(data_final, out_col=out_col)

    if dropna_target:
        data = data.dropna(subset=[out_col]).copy()
        data_oos = data_oos.dropna(subset=[out_col]).copy()
        data_2025 = data_2025.dropna(subset=[out_col]).copy()
        data_final = data_final.dropna(subset=[out_col]).copy()

    return data, data_oos, data_2025, data_final
