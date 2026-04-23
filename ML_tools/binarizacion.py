# ============================================================
# Generación de reglas: Binarización (solo IS)
# Features 0/1: NO se discretizan a >2 bins
# Features continuas: sí se discretizan con N bins
# ============================================================

import numpy as np
import pandas as pd


def _get_numeric_feature_cols(df: pd.DataFrame, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = set()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude_cols]


def _is_binary_01(s: pd.Series) -> bool:
    vals = pd.Series(s).dropna().unique()
    if len(vals) == 0:
        return False
    return set(vals).issubset({0, 1})


def fit_rule_binarizer_is(
    df_is: pd.DataFrame,
    bins: int = 4,
    exclude_cols=None,
    binary_include_zero: bool = True,
    min_non_nan: int = 50,
):
    """
    Ajusta las reglas SOLO con IS.

    Devuelve:
    - specs: especificaciones para reutilizar luego en OOS/2025/final
    - rulebook: tabla descriptiva de reglas
    - X_is_rules: matriz binaria de reglas generada SOLO en IS
    """
    if exclude_cols is None:
        exclude_cols = set()

    if not isinstance(df_is, pd.DataFrame) or df_is.empty:
        raise ValueError("df_is debe ser un DataFrame no vacío.")
    if bins < 2:
        raise ValueError("bins debe ser >= 2")

    feature_cols = _get_numeric_feature_cols(df_is, exclude_cols=exclude_cols)

    specs = []
    meta_rows = []
    out = {}
    idx = df_is.index

    for col in feature_cols:
        s = pd.to_numeric(df_is[col], errors="coerce")
        n_non_nan = int(s.notna().sum())

        if n_non_nan < min_non_nan:
            continue

        # CASO 1: FEATURE BINARIA 0/1
        if _is_binary_01(s):
            states = [1]
            if binary_include_zero:
                states = [0, 1]

            spec = {
                "feature": col,
                "kind": "binary",
                "states": states
            }
            specs.append(spec)

            x = s.to_numpy(dtype=float)

            for st in states:
                rule_name = f"{col}__EQ_{st}"
                out[rule_name] = (np.isfinite(x) & (x == st)).astype("int8")

                meta_rows.append({
                    "rule_name": rule_name,
                    "feature": col,
                    "kind": "binary",
                    "state": st,
                    "bin_idx": np.nan,
                    "bin_from": np.nan,
                    "bin_to": np.nan
                })

            continue

        # CASO 2: FEATURE CONTINUA
        vals = s.dropna().to_numpy(dtype=float)

        if len(np.unique(vals)) < 2:
            continue

        q = np.linspace(0.0, 1.0, bins + 1)
        edges = np.nanquantile(vals, q)
        edges = np.unique(edges)

        if len(edges) < 3:
            edges = np.unique(np.nanquantile(vals, [0.0, 0.5, 1.0]))

        if len(edges) < 3:
            continue

        spec = {
            "feature": col,
            "kind": "continuous",
            "edges": edges
        }
        specs.append(spec)

        x = s.to_numpy(dtype=float)
        n_intervals = len(edges) - 1

        for i in range(n_intervals):
            lo = edges[i]
            hi = edges[i + 1]
            rule_name = f"{col}__BIN_{i+1}_{n_intervals}"

            if i < n_intervals - 1:
                mask = np.isfinite(x) & (x >= lo) & (x < hi)
            else:
                mask = np.isfinite(x) & (x >= lo) & (x <= hi)

            out[rule_name] = mask.astype("int8")

            meta_rows.append({
                "rule_name": rule_name,
                "feature": col,
                "kind": "continuous",
                "state": np.nan,
                "bin_idx": i + 1,
                "bin_from": float(lo),
                "bin_to": float(hi)
            })

    X_is_rules = pd.DataFrame(out, index=idx)
    rulebook = pd.DataFrame(meta_rows)

    return specs, rulebook, X_is_rules


def apply_rule_binarizer_from_specs(df: pd.DataFrame, specs):
    """
    Aplica las especificaciones aprendidas en IS a otro DataFrame
    (OOS / 2025 / final).
    """
    idx = df.index
    out = {}

    for spec in specs:
        col = spec["feature"]
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if spec["kind"] == "binary":
            for st in spec["states"]:
                name = f"{col}__EQ_{st}"
                out[name] = (np.isfinite(x) & (x == st)).astype("int8")

        elif spec["kind"] == "continuous":
            edges = spec["edges"]
            n_intervals = len(edges) - 1

            for i in range(n_intervals):
                lo = edges[i]
                hi = edges[i + 1]
                name = f"{col}__BIN_{i+1}_{n_intervals}"

                if i < n_intervals - 1:
                    mask = np.isfinite(x) & (x >= lo) & (x < hi)
                else:
                    mask = np.isfinite(x) & (x >= lo) & (x <= hi)

                out[name] = mask.astype("int8")

    return pd.DataFrame(out, index=idx)
