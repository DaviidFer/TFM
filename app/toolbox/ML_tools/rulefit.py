# ============================================================
# Generación de reglas: RuleFit Multi-Seed
# Gradient Boosting -> extracción de reglas -> LassoCV
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def _build_rulefit_rules_single_seed(
    data: pd.DataFrame,
    target_col: str = "Target",
    exclude_cols=None,
    min_coverage: int = 100,
    top_k_features: int = 120,
    n_estimators: int = 80,
    tree_depth: int = 3,
    min_samples_leaf_tree: int = 40,
    max_candidate_rules: int = 1200,
    max_rule_indicators: int = 2,
    subsample: float = 0.8,
    max_features="sqrt",
    random_state: int = 42
):
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

    y = pd.to_numeric(df[target_col], errors="coerce").astype(np.float64)

    corrs = df[candidate_cols].corrwith(y, method="spearman").abs().fillna(0.0)
    feature_cols = corrs.sort_values(ascending=False).head(min(top_k_features, len(corrs))).index.tolist()

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).astype(np.float64)
    y_arr = y.to_numpy(dtype=np.float64)

    gbr = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=tree_depth,
        min_samples_leaf=min_samples_leaf_tree,
        subsample=subsample,
        max_features=max_features,
        random_state=random_state
    )
    gbr.fit(X, y_arr)

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

    def _extract_rules_from_tree(tree_model, X_df):
        tree = tree_model.tree_
        feature_names = X_df.columns.tolist()
        leaf_ids = tree_model.apply(X_df)
        rows_local = []

        def walk(node_id, bounds):
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]

            if left == right:
                idx = np.flatnonzero(leaf_ids == node_id)
                coverage = int(idx.size)

                if coverage >= min_coverage:
                    ordered_features = [f for f in feature_cols if f in bounds]

                    if len(ordered_features) == 0 or len(ordered_features) > max_rule_indicators:
                        return

                    indicators = tuple(ordered_features)
                    bin_labels = tuple(_label_piece(bounds[f][0], bounds[f][1]) for f in ordered_features)

                    rule_parts = []
                    for f in ordered_features:
                        part = _rule_piece(f, bounds[f][0], bounds[f][1])
                        if part is not None:
                            rule_parts.append(part)

                    regla = " & ".join(rule_parts) if rule_parts else "(TRUE)"

                    rows_local.append({
                        "indicators": indicators,
                        "bin_labels": bin_labels,
                        "coverage": coverage,
                        "indices": idx,
                        "regla": regla
                    })
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
        return rows_local

    candidate_rows = []
    for est in gbr.estimators_.ravel():
        candidate_rows.extend(_extract_rules_from_tree(est, X))

    if len(candidate_rows) == 0:
        empty = pd.DataFrame(columns=["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"])
        return empty.copy(), empty.copy()

    cand = pd.DataFrame(candidate_rows).drop_duplicates(subset=["regla"]).reset_index(drop=True)

    prelim_scores = []
    for _, r in cand.iterrows():
        idx = r["indices"]
        yt = y_arr[idx]
        mean_t = float(np.nanmean(yt)) if np.isfinite(yt).any() else np.nan
        score = abs(mean_t) * np.log1p(len(idx))
        prelim_scores.append(score)

    cand["pre_score"] = prelim_scores
    cand = cand.sort_values("pre_score", ascending=False).head(max_candidate_rules).reset_index(drop=True)

    n = len(df)
    R = np.zeros((n, len(cand)), dtype=np.float64)

    for j, idx in enumerate(cand["indices"]):
        R[idx, j] = 1.0

    scaler = StandardScaler(with_mean=True, with_std=True)
    R_scaled = scaler.fit_transform(R).astype(np.float64, copy=False)

    lasso = LassoCV(
        cv=5,
        random_state=random_state,
        max_iter=5000,
        n_alphas=50,
        selection="random",
        precompute=False
    )
    lasso.fit(R_scaled, y_arr)

    coefs = lasso.coef_
    selected = np.flatnonzero(np.abs(coefs) > 1e-12)

    if selected.size == 0:
        selected = np.arange(min(100, len(cand)))

    rows = []
    for j in selected:
        r = cand.iloc[j]
        idx = r["indices"]
        yt = y_arr[idx]
        mean_t = float(np.nanmean(yt)) if np.isfinite(yt).any() else np.nan

        rows.append({
            "indicators": r["indicators"],
            "bin_labels": r["bin_labels"],
            "coverage": int(len(idx)),
            "indices": idx,
            "target_promedio": mean_t,
            "regla": r["regla"]
        })

    out = pd.DataFrame(rows).drop_duplicates(subset=["regla"]).reset_index(drop=True)

    if out.empty:
        empty = pd.DataFrame(columns=["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"])
        return empty.copy(), empty.copy()

    df_alcistas_rulefit = out[out["target_promedio"] > 0].copy()
    df_bajistas_rulefit = out[out["target_promedio"] < 0].copy()

    if not df_alcistas_rulefit.empty:
        df_alcistas_rulefit = df_alcistas_rulefit.sort_values(
            by=["target_promedio", "coverage"], ascending=[False, False]
        ).reset_index(drop=True)

    if not df_bajistas_rulefit.empty:
        df_bajistas_rulefit = df_bajistas_rulefit.sort_values(
            by=["target_promedio", "coverage"], ascending=[True, False]
        ).reset_index(drop=True)

    wanted_cols = ["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]
    return df_alcistas_rulefit[wanted_cols], df_bajistas_rulefit[wanted_cols]


def build_rulefit_rules_multiseed(
    data: pd.DataFrame,
    target_col: str = "Target",
    exclude_cols=None,
    min_coverage: int = 100,
    top_k_features: int = 120,
    n_estimators: int = 80,
    tree_depth: int = 3,
    min_samples_leaf_tree: int = 40,
    max_candidate_rules: int = 1200,
    max_rule_indicators: int = 2,
    subsample: float = 0.8,
    max_features="sqrt",
    start_random_state: int = 1,
    target_n_rules: int = 3000,
    progress_every: int = 25,
    max_seed_attempts: int | None = 8,
    max_stalled_seeds: int | None = 3,
):
    """
    Itera sobre random_state hasta acumular target_n_rules reglas únicas.
    Devuelve (df_alcistas_rulefit, df_bajistas_rulefit).
    """
    if target_n_rules < 1:
        raise ValueError("target_n_rules debe ser >= 1")

    all_rules = []
    seen = set()
    rs = start_random_state
    seed_attempts = 0
    stalled_seeds = 0

    while len(seen) < target_n_rules:
        if max_seed_attempts is not None and seed_attempts >= int(max_seed_attempts):
            break
        if max_stalled_seeds is not None and stalled_seeds >= int(max_stalled_seeds):
            break
        before_seed = len(seen)
        df_alc_seed, df_baj_seed = _build_rulefit_rules_single_seed(
            data=data,
            target_col=target_col,
            exclude_cols=exclude_cols,
            min_coverage=min_coverage,
            top_k_features=top_k_features,
            n_estimators=n_estimators,
            tree_depth=tree_depth,
            min_samples_leaf_tree=min_samples_leaf_tree,
            max_candidate_rules=max_candidate_rules,
            max_rule_indicators=max_rule_indicators,
            subsample=subsample,
            max_features=max_features,
            random_state=rs
        )

        out_seed = pd.concat([df_alc_seed, df_baj_seed], axis=0, ignore_index=True)

        if not out_seed.empty:
            for _, row in out_seed.iterrows():
                regla = row["regla"]
                if regla not in seen:
                    seen.add(regla)
                    all_rules.append(row.to_dict())

        if progress_every and rs % progress_every == 0:
            print(f"random_state={rs} | reglas únicas acumuladas={len(seen)}")

        seed_attempts += 1
        if len(seen) == before_seed:
            stalled_seeds += 1
        else:
            stalled_seeds = 0
        rs += 1

    out = pd.DataFrame(all_rules).drop_duplicates(subset=["regla"]).reset_index(drop=True)

    if out.empty:
        empty = pd.DataFrame(columns=["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"])
        return empty.copy(), empty.copy()

    df_alcistas_rulefit = out[out["target_promedio"] > 0].copy()
    df_bajistas_rulefit = out[out["target_promedio"] < 0].copy()

    if not df_alcistas_rulefit.empty:
        df_alcistas_rulefit = df_alcistas_rulefit.sort_values(
            by=["target_promedio", "coverage"], ascending=[False, False]
        ).reset_index(drop=True)

    if not df_bajistas_rulefit.empty:
        df_bajistas_rulefit = df_bajistas_rulefit.sort_values(
            by=["target_promedio", "coverage"], ascending=[True, False]
        ).reset_index(drop=True)

    wanted_cols = ["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]
    return df_alcistas_rulefit[wanted_cols], df_bajistas_rulefit[wanted_cols]
