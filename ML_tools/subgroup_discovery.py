# ============================================================
# Generación de reglas: Subgroup Discovery
# Misma interfaz que el resto de modelos: (df_alcistas, df_bajistas)
# Implementación: subgrupos por indicador (mejor y peor bin por target)
# ============================================================

import numpy as np
import pandas as pd

WANTED_COLS = ["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]


def _rule_string(col: str, left: float, right: float, closed: str = "right") -> str:
    l_f = format(float(left), ".12g")
    r_f = format(float(right), ".12g")
    if np.isclose(left, right, equal_nan=False):
        return f"({col} == {l_f})"
    if closed == "right":
        return f"({col} > {l_f}) & ({col} <= {r_f})"
    return f"({col} >= {l_f}) & ({col} < {r_f})"


def build_subgroup_discovery_rules(
    data: pd.DataFrame,
    target_col: str = "Target",
    exclude_cols=None,
    min_coverage: int = 50,
    n_bins: int = 5,
    **kwargs
):
    """
    Subgroup discovery: por cada indicador numérico, se discretiza en n_bins,
    se calcula la media del target por subgrupo y se devuelve el mejor (alcista)
    y el peor (bajista) como reglas. Mismo formato que el resto de modelos.
    """
    if exclude_cols is None:
        exclude_cols = ["open", "high", "low", "close", "Target", "Return"]

    candidate_cols = [
        c for c in data.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])
    ]
    y = pd.to_numeric(data[target_col], errors="coerce").to_numpy(dtype=np.float64)

    rows_alcistas = []
    rows_bajistas = []

    for col in candidate_cols:
        x = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < min_coverage:
            continue

        x_valid = x[valid]
        y_valid = y[valid]
        n_unique = len(np.unique(x_valid))
        if n_unique < 2:
            continue

        q = min(n_bins, n_unique)
        try:
            bins = pd.qcut(x_valid, q=q, duplicates="drop")
        except (ValueError, TypeError):
            continue

        grp = pd.DataFrame({"y": y_valid, "_bin": bins}).groupby("_bin", observed=True)["y"].agg(["mean", "count"])
        if grp.empty or grp["count"].lt(min_coverage).all():
            continue

        grp = grp[grp["count"] >= min_coverage]
        if grp.empty:
            continue

        best_idx = grp["mean"].idxmax()
        worst_idx = grp["mean"].idxmin()

        for interval in grp.index:
            left = getattr(interval, "left", np.nan)
            right = getattr(interval, "right", np.nan)
            if np.isnan(left) or np.isnan(right):
                continue
            mean_val = float(grp.loc[interval, "mean"])
            count_val = int(grp.loc[interval, "count"])
            if count_val < min_coverage:
                continue

            if np.isclose(right, np.nanmax(x_valid)):
                mask = np.isfinite(x) & (x >= left) & (x <= right)
            else:
                mask = np.isfinite(x) & (x >= left) & (x < right)
            idx = np.flatnonzero(mask)
            if idx.size < min_coverage:
                continue

            regla = _rule_string(col, left, right, "right")
            label = f"({format(float(left), '.12g')}, {format(float(right), '.12g')}]"
            row = {
                "indicators": (col,),
                "bin_labels": (label,),
                "coverage": int(idx.size),
                "indices": idx,
                "target_promedio": mean_val,
                "regla": regla,
            }

            if interval == best_idx:
                rows_alcistas.append(row)
            if interval == worst_idx and worst_idx != best_idx:
                rows_bajistas.append(row)

    df_alcistas = pd.DataFrame(rows_alcistas)
    df_bajistas = pd.DataFrame(rows_bajistas)

    if not df_alcistas.empty:
        df_alcistas = df_alcistas.sort_values(by=["target_promedio", "coverage"], ascending=[False, False]).reset_index(drop=True)
    if not df_bajistas.empty:
        df_bajistas = df_bajistas.sort_values(by=["target_promedio", "coverage"], ascending=[True, False]).reset_index(drop=True)

    df_alcistas = df_alcistas[WANTED_COLS] if not df_alcistas.empty else pd.DataFrame(columns=WANTED_COLS)
    df_bajistas = df_bajistas[WANTED_COLS] if not df_bajistas.empty else pd.DataFrame(columns=WANTED_COLS)
    return df_alcistas, df_bajistas


def run_subgroup_discovery_rules(
    data: pd.DataFrame,
    target_col: str = "Target",
    exclude_cols=None,
    min_coverage: int = 50,
    n_bins: int = 5,
    **kwargs
):
    """
    Wrapper para el notebook. Devuelve (df_alcistas_sd, df_bajistas_sd).
    """
    return build_subgroup_discovery_rules(
        data=data,
        target_col=target_col,
        exclude_cols=exclude_cols,
        min_coverage=min_coverage,
        n_bins=n_bins,
        **kwargs
    )
