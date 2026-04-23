from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PAT = re.compile(r"(?:`([^`]+)`|([A-Za-z_]\w*))\s*(>=|>|<=|<)\s*(" + _FLOAT + r")")


def _ensure_return_series_simple(df: pd.DataFrame, return_col: str = "Target") -> pd.Series:
    if return_col in df.columns:
        return pd.to_numeric(df[return_col], errors="coerce").astype(float)

    if "open" not in df.columns:
        raise ValueError(f"No existe '{return_col}' ni columna 'open' para calcularlo.")

    o = pd.to_numeric(df["open"], errors="coerce").astype(float)
    return ((o.shift(-1) - o) / o).astype(float)


def _parse_regla_to_bins(regla: str) -> List[Tuple[str, float, float, bool, bool]]:
    parts = _PAT.findall(str(regla))
    if not parts:
        return []

    by_col: Dict[str, List[Tuple[str, float]]] = {}
    for col_bt, col_plain, op, val in parts:
        col = col_bt if col_bt else col_plain
        by_col.setdefault(col, []).append((op, float(val)))

    bins: List[Tuple[str, float, float, bool, bool]] = []
    for col, lst in by_col.items():
        left, right = -np.inf, +np.inf
        inc_left, inc_right = False, False

        for op, val in lst:
            if op in (">", ">="):
                if val > left or (val == left and op == ">=" and not inc_left):
                    left = val
                    inc_left = (op == ">=")
            else:
                if val < right or (val == right and op == "<=" and not inc_right):
                    right = val
                    inc_right = (op == "<=")

        if (not np.isfinite(left)) and (not np.isfinite(right)):
            continue

        bins.append((col, float(left), float(right), bool(inc_left), bool(inc_right)))

    return bins


def _build_global_bin_registry(
    rules_series: pd.Series,
) -> Tuple[Dict[str, Dict[Tuple[float, float, bool, bool], int]], List[Tuple[str, Tuple[float, float, bool, bool]]]]:
    per_ind: Dict[str, Dict[Tuple[float, float, bool, bool], int]] = {}
    reverse: List[Tuple[str, Tuple[float, float, bool, bool]]] = []

    for regla in rules_series.dropna().astype(str):
        bins = _parse_regla_to_bins(regla)
        for col, left, right, incL, incR in bins:
            key = (left, right, incL, incR)
            if col not in per_ind:
                per_ind[col] = {}
            if key not in per_ind[col]:
                per_ind[col][key] = len(reverse)
                reverse.append((col, key))

    return per_ind, reverse


def _build_mask_matrix(
    data: pd.DataFrame,
    reverse_registry: List[Tuple[str, Tuple[float, float, bool, bool]]],
    dtype: str = "bool",
) -> np.ndarray:
    T = len(data)
    M = np.zeros((len(reverse_registry), T), dtype=dtype)

    for i, (col, (left, right, incL, incR)) in enumerate(reverse_registry):
        x = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x)

        if np.isfinite(left):
            mask &= (x >= left) if incL else (x > left)
        if np.isfinite(right):
            mask &= (x <= right) if incR else (x < right)

        M[i, :] = mask

    return M


def _rules_to_bin_ids(
    rules_series: pd.Series,
    registry: Dict[str, Dict[Tuple[float, float, bool, bool], int]],
) -> Tuple[np.ndarray, List[str], int]:
    parsed_rules = []
    rule_names = []

    max_len = 0
    for regla in rules_series.dropna().astype(str):
        bins = _parse_regla_to_bins(regla)
        ids = []
        for col, left, right, incL, incR in bins:
            ids.append(registry[col][(left, right, incL, incR)])
        if len(ids) == 0:
            continue
        parsed_rules.append(ids)
        rule_names.append(regla)
        max_len = max(max_len, len(ids))

    if max_len == 0 or len(parsed_rules) == 0:
        return np.empty((0, 0), dtype=np.int64), [], 0

    bin_ids = -np.ones((len(parsed_rules), max_len), dtype=np.int64)
    for i, ids in enumerate(parsed_rules):
        bin_ids[i, :len(ids)] = ids

    return bin_ids, rule_names, max_len


def build_rule_return_matrix(
    data: pd.DataFrame,
    df_rules: pd.DataFrame,
    direction: str = "long",
    return_col: str = "Target",
    chunk_size: int = 1000,
    dtype: str = "float32",
    drop_empty: bool = True,
) -> pd.DataFrame:
    if df_rules is None or df_rules.empty or "regla" not in df_rules.columns:
        return pd.DataFrame(index=data.index)

    direction = direction.lower().strip()
    if direction not in {"long", "short"}:
        raise ValueError("direction debe ser 'long' o 'short'.")

    sign = 1.0 if direction == "long" else -1.0
    tgt = (sign * _ensure_return_series_simple(data, return_col=return_col)).to_numpy(dtype=np.float32)
    dates = data.index

    registry, reverse_registry = _build_global_bin_registry(df_rules["regla"])
    if len(reverse_registry) == 0:
        return pd.DataFrame(index=dates)

    M = _build_mask_matrix(data, reverse_registry, dtype="bool")
    bin_ids, rule_names, k_max = _rules_to_bin_ids(df_rules["regla"], registry)

    if bin_ids.size == 0 or k_max == 0:
        return pd.DataFrame(index=dates)

    n_rules = bin_ids.shape[0]
    T = len(data)
    out = np.zeros((T, n_rules), dtype=dtype)

    for start in range(0, n_rules, chunk_size):
        end = min(start + chunk_size, n_rules)
        ids_chunk = bin_ids[start:end, :]

        combined = np.ones((end - start, T), dtype=bool)
        for j in range(k_max):
            idxj = ids_chunk[:, j]
            valid = idxj >= 0
            if not valid.any():
                continue
            tmp = np.ones((end - start, T), dtype=bool)
            tmp[valid] = M[idxj[valid], :]
            combined &= tmp

        out[:, start:end] = (combined.T * tgt[:, None]).astype(dtype, copy=False)

    df_out = pd.DataFrame(out, index=dates, columns=rule_names)

    if drop_empty:
        nonzero_cols = df_out.columns[(df_out.abs().sum(axis=0) != 0).values]
        df_out = df_out[nonzero_cols]

    return df_out


def diagnose_rule_returns(rule_returns: pd.DataFrame, name: str = "rule_returns", sample_cols: int = 200) -> None:
    if rule_returns is None or rule_returns.empty:
        print(f"[{name}] vacío.")
        return

    T, N = rule_returns.shape
    print(f"[{name}] cols={N} | filas={T}")

    A = rule_returns.to_numpy(copy=False)
    n_check = min(sample_cols, N)
    base = A[:, 0]

    same = 0
    for j in range(n_check):
        if np.array_equal(A[:, j], base):
            same += 1
    print(f"[{name}] columnas idénticas a la col0 (en las primeras {n_check}): {same}")

    m = min(20, N)
    if m >= 2:
        x = base.astype(np.float64)
        x = x - x.mean()
        xstd = x.std()
        if xstd < 1e-12:
            print(f"[{name}] col0 casi constante; corr no informativa.")
            return
        x = x / xstd

        corrs = []
        for j in range(m):
            y = A[:, j].astype(np.float64)
            y = y - y.mean()
            ystd = y.std()
            if ystd < 1e-12:
                corrs.append(np.nan)
            else:
                y = y / ystd
                corrs.append(float((x @ y) / len(x)))

        corrs = np.array(corrs, dtype=float)
        finite = np.isfinite(corrs)
        if finite.any():
            print(f"[{name}] corr(col0, cols[:{m}]): min={np.nanmin(corrs):.4f} | max={np.nanmax(corrs):.4f} | p50={np.nanmedian(corrs):.4f}")
        else:
            print(f"[{name}] corr(col0, cols[:{m}]): todas NaN")


def prune_correlated_rules_fast(
    rule_returns: pd.DataFrame,
    corr_threshold: float = 0.90,
    score_df: Optional[pd.DataFrame] = None,
    score_col: str = "target_promedio",
    metric: str = "mean",
    min_ops: int = 100,
    absolute_corr: bool = False,
    dtype: str = "float32",
):
    if rule_returns is None or rule_returns.empty:
        empty_keep = pd.DataFrame(columns=["regla", "score", "ops", "mean", "std", "cluster_id"])
        empty_drop = pd.DataFrame(columns=["regla", "dropped_due_to", "corr_with", "corr_val", "score", "ops"])
        empty_rank = pd.DataFrame(columns=["regla", "score", "ops", "mean", "std"])
        return empty_keep, empty_drop, empty_rank

    R = rule_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    A0 = R.to_numpy(copy=False)
    ops = (A0 != 0.0).sum(axis=0)

    valid = ops >= int(min_ops)
    if not valid.any():
        empty_keep = pd.DataFrame(columns=["regla", "score", "ops", "mean", "std", "cluster_id"])
        empty_drop = pd.DataFrame(columns=["regla", "dropped_due_to", "corr_with", "corr_val", "score", "ops"])
        empty_rank = pd.DataFrame(columns=["regla", "score", "ops", "mean", "std"])
        return empty_keep, empty_drop, empty_rank

    R = R.loc[:, valid]
    cols = R.columns.to_list()
    ops = ops[valid]
    A = R.to_numpy().astype(dtype, copy=False)

    scores = None
    if score_df is not None and ("regla" in score_df.columns) and (score_col in score_df.columns):
        score_map = dict(zip(score_df["regla"].astype(str), score_df[score_col]))
        scores = np.array([score_map.get(c, np.nan) for c in cols], dtype=float)

    col_mean = A.mean(axis=0)
    col_std = A.std(axis=0, ddof=0)
    col_std[col_std < 1e-12] = 1.0
    fallback = {
        "mean": col_mean,
        "sum": A.sum(axis=0),
        "sharpe": col_mean / col_std,
    }.get(metric, col_mean)

    if scores is None:
        scores = fallback.astype(float, copy=False)
    else:
        mask_nan = ~np.isfinite(scores)
        if mask_nan.any():
            scores[mask_nan] = fallback[mask_nan]

    means = col_mean.astype(float, copy=False)
    stds = A.std(axis=0, ddof=0).astype(float, copy=False)

    ranking_df = pd.DataFrame({
        "regla": cols,
        "score": scores,
        "ops": ops,
        "mean": means,
        "std": stds,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    std = A.std(axis=0, ddof=0)
    std[std < 1e-12] = 1.0
    Z = (A - A.mean(axis=0, keepdims=True)) / std

    order = np.argsort(-scores)
    removed = np.zeros(Z.shape[1], dtype=bool)
    keep_list = []
    dropped_rows = []
    cluster_id = 0
    owners = {}

    T = Z.shape[0]
    for j in order:
        if removed[j]:
            continue

        keep_list.append(j)
        owners[j] = cluster_id

        corr = (Z[:, j].T @ Z) / T
        if absolute_corr:
            corr = np.abs(corr)

        mask = corr >= corr_threshold
        mask[j] = False

        drop_idx = np.where(~removed & mask)[0]
        for k in drop_idx:
            dropped_rows.append({
                "regla": cols[k],
                "dropped_due_to": cols[j],
                "corr_with": cols[j],
                "corr_val": float(corr[k]),
                "score": float(scores[k]),
                "ops": int(ops[k]),
            })

        removed |= mask
        cluster_id += 1

    kept_idx = np.array(keep_list, dtype=int)

    kept_df = pd.DataFrame({
        "regla": [cols[i] for i in kept_idx],
        "score": scores[kept_idx],
        "ops": ops[kept_idx],
        "mean": means[kept_idx],
        "std": stds[kept_idx],
        "cluster_id": [owners[i] for i in kept_idx],
    }).sort_values("score", ascending=False).reset_index(drop=True)

    if len(dropped_rows) == 0:
        dropped_df = pd.DataFrame(columns=["regla", "dropped_due_to", "corr_with", "corr_val", "score", "ops"])
    else:
        dropped_df = pd.DataFrame(dropped_rows).sort_values("score", ascending=False).reset_index(drop=True)

    return kept_df, dropped_df, ranking_df


def _pick_score_col(df: pd.DataFrame) -> Optional[str]:
    """Prefer score_pruning (orientado: long=target_promedio, short=-target_promedio)."""
    for c in ["score_pruning", "original_profit", "mean_synthetic", "target_promedio"]:
        if c in df.columns:
            return c
    return None


def run_pl_correlation_pruning(
    data_all: pd.DataFrame,
    df_rules_long: Optional[pd.DataFrame] = None,
    df_rules_short: Optional[pd.DataFrame] = None,
    return_col: str = "Target",
    corr_threshold: float = 0.50,
    min_ops: int = 50,
    absolute_corr: bool = False,
    metric: str = "mean",
    score_col_long: Optional[str] = None,
    score_col_short: Optional[str] = None,
    chunk_size: int = 1000,
    dtype: str = "float32",
    diagnose: bool = True,
):
    out = {}

    if df_rules_long is not None and not df_rules_long.empty:
        rr_long = build_rule_return_matrix(
            data=data_all,
            df_rules=df_rules_long,
            direction="long",
            return_col=return_col,
            chunk_size=chunk_size,
            dtype=dtype,
        )
        if diagnose:
            diagnose_rule_returns(rr_long, name="rule_returns_long")
        score_col_long = score_col_long or _pick_score_col(df_rules_long)
        kept_long, dropped_long, ranking_long = prune_correlated_rules_fast(
            rule_returns=rr_long,
            corr_threshold=corr_threshold,
            score_df=df_rules_long,
            score_col=score_col_long or "target_promedio",
            metric=metric,
            min_ops=min_ops,
            absolute_corr=absolute_corr,
            dtype=dtype,
        )
        final_rules_long = df_rules_long[df_rules_long["regla"].isin(kept_long["regla"])].copy()
        out.update({
            "rule_returns_long": rr_long,
            "kept_long": kept_long,
            "dropped_long": dropped_long,
            "ranking_long": ranking_long,
            "final_rules_long": final_rules_long,
        })

    if df_rules_short is not None and not df_rules_short.empty:
        rr_short = build_rule_return_matrix(
            data=data_all,
            df_rules=df_rules_short,
            direction="short",
            return_col=return_col,
            chunk_size=chunk_size,
            dtype=dtype,
        )
        if diagnose:
            diagnose_rule_returns(rr_short, name="rule_returns_short")
        score_col_short = score_col_short or _pick_score_col(df_rules_short)
        kept_short, dropped_short, ranking_short = prune_correlated_rules_fast(
            rule_returns=rr_short,
            corr_threshold=corr_threshold,
            score_df=df_rules_short,
            score_col=score_col_short or "target_promedio",
            metric=metric,
            min_ops=min_ops,
            absolute_corr=absolute_corr,
            dtype=dtype,
        )
        final_rules_short = df_rules_short[df_rules_short["regla"].isin(kept_short["regla"])].copy()
        out.update({
            "rule_returns_short": rr_short,
            "kept_short": kept_short,
            "dropped_short": dropped_short,
            "ranking_short": ranking_short,
            "final_rules_short": final_rules_short,
        })

    return out