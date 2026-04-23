# ============================================================
# Validación tipo "monos" IS/OOS
# - En OOS exige min_coverage_oos (configurable)
# ============================================================

from __future__ import annotations

import re
from typing import Optional, Iterable, Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

WANTED_COLS = ["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]


def combine_rule_dfs(rule_dfs: Iterable[pd.DataFrame] | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combina varios DataFrames de reglas y elimina duplicados por 'regla'.

    Acepta:
    - iterable de DataFrames
    - dict nombre_modelo -> DataFrame
    """
    if isinstance(rule_dfs, Mapping):
        items = list(rule_dfs.items())
        dfs = [df for _, df in items]
    else:
        dfs = list(rule_dfs)

    frames = []
    for i, df in enumerate(dfs):
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "regla" not in df.columns:
            raise ValueError(f"El DataFrame en posición {i} no contiene la columna 'regla'.")
        # nos quedamos con columnas estándar si existen, pero toleramos extras
        cols = [c for c in WANTED_COLS if c in df.columns]
        frames.append(df[cols].copy() if cols else df.copy())

    if len(frames) == 0:
        return pd.DataFrame(columns=WANTED_COLS)

    out = pd.concat(frames, axis=0, ignore_index=True)
    if "regla" not in out.columns:
        raise ValueError("No se encontró columna 'regla' tras combinar rule_dfs.")
    out = out.drop_duplicates(subset=["regla"]).reset_index(drop=True)
    return out


def monkey_validate_is_multi(
    dataset_is: pd.DataFrame,
    rule_dfs: Iterable[pd.DataFrame] | Mapping[str, pd.DataFrame],
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper: combina N dfs (p.ej. alcistas de varios modelos) y valida en IS.
    """
    df_reglas = combine_rule_dfs(rule_dfs)
    return monkey_validate_is(dataset_is=dataset_is, df_reglas=df_reglas, **kwargs)


def monkey_validate_oos_multi(
    dataset_oos: pd.DataFrame,
    df_reglas_is_ok: pd.DataFrame,
    min_coverage_oos: int = 100,
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper: valida en OOS imponiendo min_coverage_oos (configurable).
    """
    return monkey_validate_oos(
        dataset_oos=dataset_oos,
        df_reglas_is_ok=df_reglas_is_ok,
        min_coverage_oos=min_coverage_oos,
        **kwargs,
    )


def _pct_to_float01(x: float) -> float:
    x = float(x)
    return x / 100.0 if x > 1 else x


def _ensure_return_series_simple(df: pd.DataFrame, return_col: str = "Target") -> pd.Series:
    if return_col in df.columns:
        return pd.to_numeric(df[return_col], errors="coerce").astype(float)

    if "open" not in df.columns:
        raise ValueError(f"No existe '{return_col}' ni columna 'open' para calcularlo.")

    o = pd.to_numeric(df["open"], errors="coerce").astype(float)
    return ((o.shift(-1) - o) / o).astype(float)


def _sanitize_indices_fast(idx_raw, n_total: int, valid_mask: np.ndarray) -> np.ndarray:
    if isinstance(idx_raw, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        arr = np.asarray(idx_raw, dtype=float).ravel()
    else:
        return np.empty(0, dtype=np.int64)

    if arr.size == 0:
        return np.empty(0, dtype=np.int64)

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)

    arr = arr.astype(np.int64, copy=False)
    arr = arr[(arr >= 0) & (arr < n_total)]
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)

    arr = arr[valid_mask[arr]]
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)

    return np.unique(arr)


def _simulate_monkeys_same_direction_same_cadence_fast(
    valid_returns: np.ndarray,
    n_ops: int,
    sign: float,
    n_monkeys: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_ops <= 0 or valid_returns.size == 0 or n_monkeys <= 0:
        return np.array([], dtype=float)

    sample = rng.choice(valid_returns, size=(n_monkeys, n_ops), replace=True)
    return (sign * sample.sum(axis=1)).astype(float)


_CLAUSE_RE = re.compile(r"\([^()]+\)")
_SIMPLE_CLAUSE_RE = re.compile(
    r"^\(\s*([A-Za-z_]\w*)\s*(<=|>=|==|<|>)\s*([-+]?inf|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)$"
)


def _extract_clauses(rule: str):
    if not isinstance(rule, str) or len(rule.strip()) == 0:
        return []
    return _CLAUSE_RE.findall(rule)


def _parse_simple_clause(clause: str):
    m = _SIMPLE_CLAUSE_RE.match(clause.strip())
    if m is None:
        return None

    col, op, raw_val = m.groups()

    if raw_val.lower() == "inf":
        val = np.inf
    elif raw_val.lower() == "-inf":
        val = -np.inf
    else:
        val = float(raw_val)

    return col, op, val


def _build_clause_mask_cache(dataset: pd.DataFrame, rules, valid_mask: np.ndarray):
    rule_clauses_map = {}
    unique_clauses = set()

    for r in rules:
        clauses = tuple(_extract_clauses(r))
        rule_clauses_map[r] = clauses
        unique_clauses.update(clauses)

    needed_cols = set()
    parsed_clauses = {}

    for clause in unique_clauses:
        parsed = _parse_simple_clause(clause)
        if parsed is not None:
            col, op, val = parsed
            parsed_clauses[clause] = parsed
            needed_cols.add(col)

    col_arrays = {}
    for col in needed_cols:
        if col in dataset.columns:
            col_arrays[col] = pd.to_numeric(dataset[col], errors="coerce").to_numpy(dtype=float)

    clause_cache = {}

    for clause in unique_clauses:
        parsed = parsed_clauses.get(clause, None)

        if parsed is None:
            try:
                mask = dataset.eval(clause, engine="numexpr")
                if isinstance(mask, (pd.Series, pd.Index)):
                    mask = mask.to_numpy(dtype=bool, copy=False)
                else:
                    mask = np.asarray(mask, dtype=bool)
                mask = mask & valid_mask
            except Exception:
                mask = np.zeros(len(dataset), dtype=bool)

            clause_cache[clause] = mask
            continue

        col, op, val = parsed

        if col not in col_arrays:
            clause_cache[clause] = np.zeros(len(dataset), dtype=bool)
            continue

        arr = col_arrays[col]
        finite_mask = np.isfinite(arr)

        if op == ">":
            mask = finite_mask & (arr > val)
        elif op == ">=":
            mask = finite_mask & (arr >= val)
        elif op == "<":
            mask = finite_mask & (arr < val)
        elif op == "<=":
            mask = finite_mask & (arr <= val)
        elif op == "==":
            if np.isinf(val):
                mask = finite_mask & (arr == val)
            else:
                mask = finite_mask & np.isclose(arr, val)
        else:
            mask = np.zeros(len(dataset), dtype=bool)

        clause_cache[clause] = mask & valid_mask

    return clause_cache, rule_clauses_map


def _rule_to_indices_from_clause_cache(rule: str, clause_cache: dict, rule_clauses_map: dict, n_rows: int):
    clauses = rule_clauses_map.get(rule, ())
    if len(clauses) == 0:
        return np.empty(0, dtype=np.int64)

    mask = np.ones(n_rows, dtype=bool)
    for clause in clauses:
        mask &= clause_cache.get(clause, False)
        if not mask.any():
            return np.empty(0, dtype=np.int64)

    return np.flatnonzero(mask).astype(np.int64, copy=False)


def _process_is_batch(
    rows_batch,
    ret_vals: np.ndarray,
    valid_returns: np.ndarray,
    valid_mask: np.ndarray,
    sign: float,
    n_monkeys: int,
    is_pass_pct_01: float,
    seed_base: int,
    min_coverage_is: int,
):
    passed_rows = []
    n_total = len(ret_vals)
    rng_master = np.random.default_rng(seed_base)

    for row in rows_batch:
        idx_is = _sanitize_indices_fast(row.indices, n_total, valid_mask)
        n_ops_is = int(idx_is.size)

        if n_ops_is < int(min_coverage_is):
            continue

        pnl_is = sign * float(ret_vals[idx_is].sum())

        rng_is = np.random.default_rng(rng_master.integers(0, 10_000_000_000))
        monkeys_is = _simulate_monkeys_same_direction_same_cadence_fast(
            valid_returns=valid_returns,
            n_ops=n_ops_is,
            sign=sign,
            n_monkeys=n_monkeys,
            rng=rng_is,
        )

        if monkeys_is.size == 0:
            continue

        pct_monkeys_beaten_is = float(np.mean(pnl_is > monkeys_is))
        if pct_monkeys_beaten_is < is_pass_pct_01:
            continue

        target_mean_is = float(ret_vals[idx_is].mean())
        passed_rows.append(
            {
                "indicators": row.indicators,
                "bin_labels": row.bin_labels,
                "coverage": n_ops_is,
                "indices": idx_is,
                "target_promedio": target_mean_is,
                "regla": row.regla,
            }
        )

    return passed_rows


def _process_oos_batch_fast(
    rows_batch,
    clause_cache: dict,
    rule_clauses_map: dict,
    n_rows_oos: int,
    ret_oos_vals: np.ndarray,
    valid_returns_oos: np.ndarray,
    sign: float,
    n_monkeys: int,
    oos_pass_pct_01: float,
    seed_base: int,
    min_coverage_oos: int,
):
    passed_rows = []
    rng_master = np.random.default_rng(seed_base)

    for row in rows_batch:
        idx_oos = _rule_to_indices_from_clause_cache(
            rule=row.regla, clause_cache=clause_cache, rule_clauses_map=rule_clauses_map, n_rows=n_rows_oos
        )

        n_ops_oos = int(idx_oos.size)
        if n_ops_oos < int(min_coverage_oos):
            continue

        pnl_oos = sign * float(ret_oos_vals[idx_oos].sum())

        rng_oos = np.random.default_rng(rng_master.integers(0, 10_000_000_000))
        monkeys_oos = _simulate_monkeys_same_direction_same_cadence_fast(
            valid_returns=valid_returns_oos,
            n_ops=n_ops_oos,
            sign=sign,
            n_monkeys=n_monkeys,
            rng=rng_oos,
        )

        if monkeys_oos.size == 0:
            continue

        pct_monkeys_beaten_oos = float(np.mean(pnl_oos > monkeys_oos))
        if pct_monkeys_beaten_oos < oos_pass_pct_01:
            continue

        target_mean_oos = float(ret_oos_vals[idx_oos].mean())
        passed_rows.append(
            {
                "indicators": row.indicators,
                "bin_labels": row.bin_labels,
                "coverage": n_ops_oos,
                "indices": idx_oos,
                "target_promedio": target_mean_oos,
                "regla": row.regla,
            }
        )

    return passed_rows


def monkey_validate_is(
    dataset_is: pd.DataFrame,
    df_reglas: pd.DataFrame,
    direction: str = "long",
    return_col: str = "Target",
    n_monkeys: int = 2000,
    is_pass_pct: float = 95.0,
    random_state: Optional[int] = 42,
    n_jobs: int = -1,
    batch_size: int = 250,
    min_coverage_is: int = 100,
) -> pd.DataFrame:
    if df_reglas.empty:
        return df_reglas.copy()

    direction = str(direction).lower().strip()
    if direction not in {"long", "short"}:
        raise ValueError("direction debe ser 'long' o 'short'.")

    sign = 1.0 if direction == "long" else -1.0
    is_pass_pct_01 = _pct_to_float01(is_pass_pct)

    ret_is = _ensure_return_series_simple(dataset_is, return_col=return_col)
    ret_is_vals = ret_is.to_numpy(dtype=float)

    valid_mask_is = np.isfinite(ret_is_vals)
    if not valid_mask_is.any():
        raise ValueError("No hay returns válidos en IS.")

    valid_returns_is = ret_is_vals[valid_mask_is]

    rows = list(
        df_reglas[["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]].itertuples(
            index=False
        )
    )
    n_input = len(rows)

    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    base_seed = int(random_state if random_state is not None else 42)

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_is_batch)(
            rows_batch=batch,
            ret_vals=ret_is_vals,
            valid_returns=valid_returns_is,
            valid_mask=valid_mask_is,
            sign=sign,
            n_monkeys=n_monkeys,
            is_pass_pct_01=is_pass_pct_01,
            seed_base=base_seed + 100_000 * b,
            min_coverage_is=min_coverage_is,
        )
        for b, batch in enumerate(batches)
    )

    passed_rows = [item for sublist in results for item in sublist]
    passed_df = pd.DataFrame(passed_rows).drop_duplicates(subset=["regla"]).reset_index(drop=True)

    if not passed_df.empty:
        ascending = False if direction == "long" else True
        passed_df = passed_df.sort_values(by=["target_promedio", "coverage"], ascending=[ascending, False]).reset_index(
            drop=True
        )

    print("=" * 90)
    print(f"VALIDACIÓN IS | {direction.upper()}")
    print("=" * 90)
    print(f"Reglas entrada : {n_input}")
    print(f"Reglas aprobadas: {len(passed_df)}")
    print(f"Reglas eliminadas: {n_input - len(passed_df)}")
    print("=" * 90)

    return passed_df


def monkey_validate_oos(
    dataset_oos: pd.DataFrame,
    df_reglas_is_ok: pd.DataFrame,
    direction: str = "long",
    return_col: str = "Target",
    n_monkeys: int = 2000,
    oos_pass_pct: float = 80.0,
    random_state: Optional[int] = 42,
    n_jobs: int = -1,
    batch_size: int = 250,
    min_coverage_oos: int = 100,
) -> pd.DataFrame:
    """
    Valida en OOS SOLO reglas que ya pasaron IS.
    Devuelve SOLO las reglas que pasan OOS.
    Además exige min_coverage_oos operaciones mínimas en OOS.
    """
    if df_reglas_is_ok.empty:
        return df_reglas_is_ok.copy()

    direction = str(direction).lower().strip()
    if direction not in {"long", "short"}:
        raise ValueError("direction debe ser 'long' o 'short'.")

    sign = 1.0 if direction == "long" else -1.0
    oos_pass_pct_01 = _pct_to_float01(oos_pass_pct)

    ret_oos = _ensure_return_series_simple(dataset_oos, return_col=return_col)
    ret_oos_vals = ret_oos.to_numpy(dtype=float)

    valid_mask_oos = np.isfinite(ret_oos_vals)
    if not valid_mask_oos.any():
        raise ValueError("No hay returns válidos en OOS.")

    valid_returns_oos = ret_oos_vals[valid_mask_oos]

    rows = list(
        df_reglas_is_ok[["indicators", "bin_labels", "coverage", "indices", "target_promedio", "regla"]].itertuples(
            index=False
        )
    )
    n_input = len(rows)

    all_rules = [r.regla for r in rows]
    clause_cache, rule_clauses_map = _build_clause_mask_cache(dataset=dataset_oos, rules=all_rules, valid_mask=valid_mask_oos)

    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    base_seed = int(random_state if random_state is not None else 42)

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_oos_batch_fast)(
            rows_batch=batch,
            clause_cache=clause_cache,
            rule_clauses_map=rule_clauses_map,
            n_rows_oos=len(dataset_oos),
            ret_oos_vals=ret_oos_vals,
            valid_returns_oos=valid_returns_oos,
            sign=sign,
            n_monkeys=n_monkeys,
            oos_pass_pct_01=oos_pass_pct_01,
            seed_base=base_seed + 100_000 * b,
            min_coverage_oos=min_coverage_oos,
        )
        for b, batch in enumerate(batches)
    )

    passed_rows = [item for sublist in results for item in sublist]
    passed_df = pd.DataFrame(passed_rows).drop_duplicates(subset=["regla"]).reset_index(drop=True)

    if not passed_df.empty:
        ascending = False if direction == "long" else True
        passed_df = passed_df.sort_values(by=["target_promedio", "coverage"], ascending=[ascending, False]).reset_index(
            drop=True
        )

    print("=" * 90)
    print(f"VALIDACIÓN OOS | {direction.upper()}")
    print("=" * 90)
    print(f"Reglas entrada : {n_input}")
    print(f"Reglas aprobadas: {len(passed_df)}")
    print(f"Reglas eliminadas: {n_input - len(passed_df)}")
    print(f"Cláusulas únicas cacheadas: {len(clause_cache)}")
    print(f"min_coverage_oos: {int(min_coverage_oos)}")
    print("=" * 90)

    return passed_df

