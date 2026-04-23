# ============================================================
# Generación de reglas: Decision Tree Multi-Seed
# Reglas homogéneas; splitter="random" + múltiples random_state
# ============================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def _build_decision_tree_rules_single_seed(
    data: pd.DataFrame,
    target_col: str = "Target",
    exclude_cols=None,
    min_coverage: int = 100,
    max_depth: int = 3,
    min_samples_leaf: int = 100,
    min_samples_split: int = 200,
    top_k_features: int = None,
    max_features="sqrt",
    splitter: str = "random",
    ccp_alpha: float = 0.0,
    random_state: int = 42
):
    """
    Genera reglas de UN solo árbol.
    Mantiene el mismo formato de salida homogéneo.
    """
    if exclude_cols is None:
        exclude_cols = ["open", "high", "low", "close", "Target", "Return"]

    if target_col not in data.columns:
        raise ValueError(f"No existe la columna target '{target_col}'.")

    df = data.copy()
    candidate_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(candidate_cols) == 0:
        raise ValueError("No hay columnas numéricas candidatas.")

    y = pd.to_numeric(df[target_col], errors="coerce")

    if top_k_features is None:
        feature_cols = candidate_cols.copy()
    else:
        corrs = df[candidate_cols].corrwith(y, method="spearman").abs().fillna(0.0)
        feature_cols = corrs.sort_values(ascending=False).head(min(top_k_features, len(corrs))).index.tolist()

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = pd.to_numeric(df[target_col], errors="coerce")

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        splitter=splitter,
        ccp_alpha=ccp_alpha,
        random_state=random_state
    )
    model.fit(X, y)

    tree = model.tree_
    feature_names = X.columns.tolist()
    node_indicator = model.decision_path(X).tocsc()

    NEG_INF = -np.inf
    POS_INF = np.inf

    def _fmt(x):
        if np.isneginf(x):
            return "-inf"
        if np.isposinf(x):
            return "inf"
        return format(float(x), ".12g")

    def _rule_piece(col, lo, hi):
        if np.isneginf(lo) and np.isposinf(hi):
            return None
        if np.isneginf(lo):
            return f"({col} <= {_fmt(hi)})"
        if np.isposinf(hi):
            return f"({col} > {_fmt(lo)})"
        if np.isclose(lo, hi):
            return f"({col} == {_fmt(lo)})"
        return f"({col} > {_fmt(lo)}) & ({col} <= {_fmt(hi)})"

    def _label_piece(lo, hi):
        return f"({_fmt(lo)}, {_fmt(hi)}]"

    rows = []

    def walk(node_id, bounds):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        is_leaf = (left == right)

        idx = node_indicator[:, node_id].indices
        coverage = int(len(idx))

        if coverage >= min_coverage and len(bounds) > 0:
            y_slice = y.iloc[idx].dropna()
            if not y_slice.empty:
                mean_target = float(y_slice.mean())
                ordered_features = [f for f in feature_cols if f in bounds]
                indicators = tuple(ordered_features)
                bin_labels = tuple(_label_piece(bounds[f][0], bounds[f][1]) for f in ordered_features)
                rule_parts = []
                for f in ordered_features:
                    part = _rule_piece(f, bounds[f][0], bounds[f][1])
                    if part is not None:
                        rule_parts.append(part)
                if len(rule_parts) > 0:
                    regla = " & ".join(rule_parts)
                    rows.append({
                        "indicators": indicators,
                        "bin_labels": bin_labels,
                        "coverage": coverage,
                        "indices": idx,
                        "target_promedio": mean_target,
                        "regla": regla
                    })

        if is_leaf:
            return

        feat_idx = tree.feature[node_id]
        thr = float(tree.threshold[node_id])
        feat = feature_names[feat_idx]

        bounds_left = {k: [v[0], v[1]] for k, v in bounds.items()}
        if feat not in bounds_left:
            bounds_left[feat] = [NEG_INF, POS_INF]
        bounds_left[feat][1] = min(bounds_left[feat][1], thr)
        walk(left, bounds_left)

        bounds_right = {k: [v[0], v[1]] for k, v in bounds.items()}
        if feat not in bounds_right:
            bounds_right[feat] = [NEG_INF, POS_INF]
        bounds_right[feat][0] = max(bounds_right[feat][0], thr)
        walk(right, bounds_right)

    walk(0, {})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["regla"]).reset_index(drop=True)
    return out


def build_decision_tree_rules_multiseed(
    data: pd.DataFrame,
    target_col: str = "Target",
    exclude_cols=None,
    min_coverage: int = 100,
    max_depth: int = 3,
    min_samples_leaf: int = 100,
    min_samples_split: int = 200,
    top_k_features: int = None,
    max_features="sqrt",
    splitter: str = "random",
    ccp_alpha: float = 0.0,
    start_random_state: int = 1,
    target_n_rules: int = 3000,
    progress_every: int = 25
):
    """
    Itera sobre random_state hasta alcanzar target_n_rules reglas únicas.
    Devuelve (df_alcistas_dt, df_bajistas_dt) con columnas homogéneas.
    """
    if target_n_rules < 1:
        raise ValueError("target_n_rules debe ser >= 1")

    all_rules = []
    seen = set()
    rs = start_random_state

    while len(seen) < target_n_rules:
        out_seed = _build_decision_tree_rules_single_seed(
            data=data,
            target_col=target_col,
            exclude_cols=exclude_cols,
            min_coverage=min_coverage,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            top_k_features=top_k_features,
            max_features=max_features,
            splitter=splitter,
            ccp_alpha=ccp_alpha,
            random_state=rs
        )

        if not out_seed.empty:
            for _, row in out_seed.iterrows():
                regla = row["regla"]
                if regla not in seen:
                    seen.add(regla)
                    all_rules.append(row.to_dict())

        if progress_every and rs % progress_every == 0:
            print(f"random_state={rs} | reglas únicas acumuladas={len(seen)}")

        rs += 1

    out = pd.DataFrame(all_rules).drop_duplicates(subset=["regla"]).reset_index(drop=True)

    if out.empty:
        empty = pd.DataFrame(columns=["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"])
        return empty.copy(), empty.copy()

    df_alcistas_dt = out[out["target_promedio"] > 0].copy()
    df_bajistas_dt = out[out["target_promedio"] < 0].copy()

    if not df_alcistas_dt.empty:
        df_alcistas_dt = df_alcistas_dt.sort_values(
            by=["target_promedio", "coverage"], ascending=[False, False]
        ).reset_index(drop=True)
    if not df_bajistas_dt.empty:
        df_bajistas_dt = df_bajistas_dt.sort_values(
            by=["target_promedio", "coverage"], ascending=[True, False]
        ).reset_index(drop=True)

    wanted_cols = ["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]
    return df_alcistas_dt[wanted_cols], df_bajistas_dt[wanted_cols]
