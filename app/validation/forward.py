from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.validation.correlation import build_rule_return_matrix


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception as e:
            raise ValueError("El indice debe ser DatetimeIndex o parseable a fechas.") from e
    return out


def _ensure_target(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    out = df.copy()
    if return_col in out.columns:
        out[return_col] = pd.to_numeric(out[return_col], errors="coerce").astype(float)
        return out
    if "open" not in out.columns:
        raise ValueError(f"No existe '{return_col}' ni columna 'open' para calcularla.")
    o = pd.to_numeric(out["open"], errors="coerce").astype(float)
    out[return_col] = ((o.shift(-1) - o) / o).astype(float)
    return out


def _prep_rules(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["regla"])
    if "regla" not in df.columns:
        raise ValueError("Los df_rules deben contener la columna 'regla'.")
    out = df.copy()
    out = out.drop_duplicates(subset=["regla"]).reset_index(drop=True)
    return out


def _profit_sum_and_coverage(rr_year: pd.DataFrame, eps: float = 1e-12):
    if rr_year is None or rr_year.empty:
        return np.array([], dtype=float), np.array([], dtype=int)
    arr = rr_year.to_numpy(copy=False)
    arr = np.nan_to_num(arr, nan=0.0)
    coverage = (np.abs(arr) > eps).sum(axis=0).astype(int)
    profit_sum = arr.sum(axis=0).astype(np.float64)
    return profit_sum, coverage


def validate_forward_year_profitability(
    data_target_year: pd.DataFrame,
    df_rules_long: Optional[pd.DataFrame] = None,
    df_rules_short: Optional[pd.DataFrame] = None,
    target_year: Optional[int] = None,
    return_col: str = "Target",
    min_ops: int = 0,
    chunk_size: int = 1000,
    dtype: str = "float32",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    data = _ensure_datetime_index(data_target_year)
    data = _ensure_target(data, return_col=return_col)
    if target_year is not None:
        year_mask = data.index.year == int(target_year)
        data = data.loc[year_mask]

    df_rules_long = _prep_rules(df_rules_long)
    df_rules_short = _prep_rules(df_rules_short)
    out: Dict[str, pd.DataFrame] = {}

    if not df_rules_long.empty:
        rr_long_year = build_rule_return_matrix(
            data=data, df_rules=df_rules_long, direction="long", return_col=return_col, chunk_size=chunk_size, dtype=dtype
        )
        profit_sum, coverage = _profit_sum_and_coverage(rr_long_year)
        metrics_long = pd.DataFrame({"regla": list(rr_long_year.columns), "profit_sum_year": profit_sum, "coverage_year": coverage})
        merged_long = df_rules_long.merge(metrics_long, on="regla", how="left")
        passed_long = merged_long[merged_long["profit_sum_year"] > 0.0]
        if min_ops and min_ops > 0:
            passed_long = passed_long[passed_long["coverage_year"] >= int(min_ops)]
        failed_long = merged_long.drop(index=passed_long.index)
        out["passed_long_forward"] = passed_long.sort_values("profit_sum_year", ascending=False).reset_index(drop=True)
        out["failed_long_forward"] = failed_long.sort_values("profit_sum_year", ascending=False).reset_index(drop=True)
        out["rr_long_year"] = rr_long_year

    if not df_rules_short.empty:
        rr_short_year = build_rule_return_matrix(
            data=data, df_rules=df_rules_short, direction="short", return_col=return_col, chunk_size=chunk_size, dtype=dtype
        )
        profit_sum, coverage = _profit_sum_and_coverage(rr_short_year)
        metrics_short = pd.DataFrame({"regla": list(rr_short_year.columns), "profit_sum_year": profit_sum, "coverage_year": coverage})
        merged_short = df_rules_short.merge(metrics_short, on="regla", how="left")
        passed_short = merged_short[merged_short["profit_sum_year"] > 0.0]
        if min_ops and min_ops > 0:
            passed_short = passed_short[passed_short["coverage_year"] >= int(min_ops)]
        failed_short = merged_short.drop(index=passed_short.index)
        out["passed_short_forward"] = passed_short.sort_values("profit_sum_year", ascending=False).reset_index(drop=True)
        out["failed_short_forward"] = failed_short.sort_values("profit_sum_year", ascending=False).reset_index(drop=True)
        out["rr_short_year"] = rr_short_year

    if verbose:
        pln = len(out.get("passed_long_forward", []))
        psn = len(out.get("passed_short_forward", []))
        print(f"[forward] passed_long_forward={pln} | passed_short_forward={psn}")
    return out


__all__ = ["validate_forward_year_profitability"]
