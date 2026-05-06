# ============================================================
# Generación de reglas: Combinaciones de bins por cuantiles
# Indicadores binarios 0/1 -> bins exactos; continuos -> qcut
# ============================================================

from itertools import combinations, product
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def _is_binary_01(s: pd.Series) -> bool:
    vals = pd.Series(s).dropna().unique()
    if len(vals) == 0:
        return False
    return set(vals).issubset({0, 1})


def _mask_for_bin(values: np.ndarray, left: float, right: float, closed: str = "right") -> np.ndarray:
    if np.isclose(left, right, equal_nan=False):
        m = np.isfinite(values) & np.isclose(values, left)
        return m
    if closed == "both":
        m = (values >= left) & (values <= right)
    elif closed == "right":
        m = (values > left) & (values <= right)
    elif closed == "left":
        m = (values >= left) & (values < right)
    else:
        m = (values > left) & (values < right)
    return m & ~np.isnan(values)


def _rule_string(ind_name: str, left: float, right: float, closed: str = "right") -> str:
    col = f"{ind_name}"
    l = format(float(left), ".12g")
    r = format(float(right), ".12g")
    if np.isclose(left, right, equal_nan=False):
        return f"({col} == {l})"
    if closed == "both":
        return f"({col} >= {l}) & ({col} <= {r})"
    elif closed == "right":
        return f"({col} > {l}) & ({col} <= {r})"
    elif closed == "left":
        return f"({col} >= {l}) & ({col} < {r})"
    else:
        return f"({col} > {l}) & ({col} < {r})"


def _build_bin_catalog_with_means(
    data: pd.DataFrame,
    rules_df: pd.DataFrame,
    indicator_col: str = "indicador",
    left_col: str = "bin_left",
    right_col: str = "bin_right",
    label_col: str = "bin_label",
    mean_col: str = "target_promedio",
    closed: str = "right"
) -> Dict[str, List[Dict[str, Any]]]:
    rules_df = rules_df[rules_df[indicator_col].isin(data.columns)].copy()
    catalog: Dict[str, List[Dict[str, Any]]] = {}
    for ind, grp in rules_df.groupby(indicator_col, sort=False):
        col_vals = pd.to_numeric(data[ind], errors="coerce").to_numpy(dtype=float)
        bins_list: List[Dict[str, Any]] = []
        for _, r in grp.iterrows():
            left = float(r[left_col])
            right = float(r[right_col])
            mask = _mask_for_bin(col_vals, left, right, closed=closed)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            bins_list.append({
                "indicator": ind,
                "label": str(r.get(label_col, f"[{left}, {right}]")),
                "left": left,
                "right": right,
                "indices": idx,
                "component_target_promedio": float(r.get(mean_col, np.nan)),
            })
        if bins_list:
            catalog[ind] = bins_list
    return catalog


def _combine_catalog_with_target(
    data: pd.DataFrame,
    catalog: Dict[str, List[Dict[str, Any]]],
    target_col: str,
    combo_size: int,
    min_coverage: int,
    closed: str = "right",
    max_combos: Optional[int] = None
) -> pd.DataFrame:
    indicators = list(catalog.keys())
    rows: List[Dict[str, Any]] = []
    emitted = 0
    target_arr = pd.to_numeric(data[target_col], errors="coerce").to_numpy(dtype=float)

    for inds in combinations(indicators, combo_size):
        lists = [catalog[i] for i in inds]
        for combo_bins in product(*lists):
            idx = combo_bins[0]["indices"]
            for b in combo_bins[1:]:
                idx = np.intersect1d(idx, b["indices"], assume_unique=True)
                if idx.size == 0:
                    break
            coverage = int(idx.size)
            if coverage < min_coverage or coverage == 0:
                continue
            target_slice = target_arr[idx]
            combo_mean = float(np.nanmean(target_slice)) if np.isfinite(target_slice).any() else np.nan
            reglas = [_rule_string(b["indicator"], b["left"], b["right"], closed=closed) for b in combo_bins]
            regla_compuesta = " & ".join(reglas)
            rows.append({
                "indicators": tuple(b["indicator"] for b in combo_bins),
                "bin_labels": tuple(b["label"] for b in combo_bins),
                "coverage": coverage,
                "indices": idx,
                "target_promedio": combo_mean,
                "regla": regla_compuesta,
            })
            emitted += 1
            if max_combos is not None and emitted >= max_combos:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def build_quantile_bin_combinations(
    data: pd.DataFrame,
    n_bins: int = 10,
    target_col: str = "Target",
    exclude_cols: List[str] = None,
    combo_size: int = 2,
    min_coverage: int = 200,
    closed: str = "right"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if exclude_cols is None:
        exclude_cols = ['open', 'high', 'low', 'close', 'Target', 'Return']
    if target_col not in data.columns:
        raise ValueError(f"No se encuentra la columna target '{target_col}' en el DataFrame.")
    if combo_size < 1:
        raise ValueError("combo_size debe ser >= 1")

    mejores_rows, peores_rows = [], []
    candidate_cols = [
        c for c in data.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])
    ]

    for col in candidate_cols:
        dfc = data[[col, target_col]].dropna()
        if dfc.empty:
            continue
        nunique = dfc[col].nunique(dropna=True)
        if nunique < 2:
            continue

        if _is_binary_01(dfc[col]):
            grp = (
                dfc.groupby(col, dropna=True)[target_col]
                .agg(['mean', 'count'])
                .reset_index()
                .rename(columns={col: '_bin_value'})
            )
            if grp.empty:
                continue
            max_mean = grp['mean'].max()
            min_mean = grp['mean'].min()
            top_bins = grp[grp['mean'] == max_mean]
            bottom_bins = grp[grp['mean'] == min_mean]
            for _, row in top_bins.iterrows():
                val = float(row['_bin_value'])
                mejores_rows.append({
                    'indicador': col,
                    'bin_label': f"[{int(val)}, {int(val)}]",
                    'bin_left': val,
                    'bin_right': val,
                    'target_promedio': float(row['mean']),
                })
            for _, row in bottom_bins.iterrows():
                val = float(row['_bin_value'])
                peores_rows.append({
                    'indicador': col,
                    'bin_label': f"[{int(val)}, {int(val)}]",
                    'bin_left': val,
                    'bin_right': val,
                    'target_promedio': float(row['mean']),
                })
            continue

        q = min(n_bins, nunique)
        try:
            bins = pd.qcut(dfc[col], q=q, duplicates='drop')
        except ValueError:
            q = min(q, max(2, nunique))
            try:
                bins = pd.qcut(dfc[col], q=q, duplicates='drop')
            except Exception:
                continue
        dfc = dfc.assign(_bin=bins)
        grp = dfc.groupby('_bin', observed=True)[target_col].agg(['mean', 'count']).reset_index()
        if grp.empty:
            continue
        max_mean = grp['mean'].max()
        min_mean = grp['mean'].min()
        top_bins = grp[grp['mean'] == max_mean]
        bottom_bins = grp[grp['mean'] == min_mean]
        for _, row in top_bins.iterrows():
            interval = row['_bin']
            left = getattr(interval, 'left', np.nan)
            right = getattr(interval, 'right', np.nan)
            mejores_rows.append({
                'indicador': col,
                'bin_label': f"[{left}, {right}]",
                'bin_left': float(left) if pd.notna(left) else np.nan,
                'bin_right': float(right) if pd.notna(right) else np.nan,
                'target_promedio': float(row['mean']),
            })
        for _, row in bottom_bins.iterrows():
            interval = row['_bin']
            left = getattr(interval, 'left', np.nan)
            right = getattr(interval, 'right', np.nan)
            peores_rows.append({
                'indicador': col,
                'bin_label': f"[{left}, {right}]",
                'bin_left': float(left) if pd.notna(left) else np.nan,
                'bin_right': float(right) if pd.notna(right) else np.nan,
                'target_promedio': float(row['mean']),
            })

    alcistas_df = pd.DataFrame(mejores_rows)
    bajistas_df = pd.DataFrame(peores_rows)
    alc_catalog = _build_bin_catalog_with_means(
        data, alcistas_df, mean_col="target_promedio", closed=closed
    )
    baj_catalog = _build_bin_catalog_with_means(
        data, bajistas_df, mean_col="target_promedio", closed=closed
    )
    df_alcistas = _combine_catalog_with_target(
        data, alc_catalog, target_col=target_col, combo_size=combo_size,
        min_coverage=min_coverage, closed=closed
    )
    df_bajistas = _combine_catalog_with_target(
        data, baj_catalog, target_col=target_col, combo_size=combo_size,
        min_coverage=min_coverage, closed=closed
    )
    if not df_alcistas.empty:
        df_alcistas = df_alcistas.sort_values(
            by=["target_promedio", "coverage"], ascending=[False, False]
        ).reset_index(drop=True)
    if not df_bajistas.empty:
        df_bajistas = df_bajistas.sort_values(
            by=["target_promedio", "coverage"], ascending=[True, False]
        ).reset_index(drop=True)
    wanted_cols = ["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]
    df_alcistas = df_alcistas[wanted_cols] if not df_alcistas.empty else df_alcistas
    df_bajistas = df_bajistas[wanted_cols] if not df_bajistas.empty else df_bajistas
    return df_alcistas, df_bajistas
