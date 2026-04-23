from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from validacion_correlacion_pl import build_rule_return_matrix


def _to_rule_list(rules_obj, name: str) -> List[str]:
    """
    Convierte distintos formatos de entrada a lista de reglas (strings).
    Acepta:
      - list/tuple/set/Series de strings
      - DataFrame con columna 'regla' o 'rule'
    """
    if rules_obj is None:
        return []

    if isinstance(rules_obj, pd.DataFrame):
        if "regla" in rules_obj.columns:
            series = rules_obj["regla"]
        elif "rule" in rules_obj.columns:
            series = rules_obj["rule"]
        else:
            raise ValueError(f"{name}: DataFrame sin columna 'regla' ni 'rule'.")
        rules = series.dropna().astype(str).tolist()
    elif isinstance(rules_obj, (pd.Series, list, tuple, set)):
        rules = pd.Series(list(rules_obj)).dropna().astype(str).tolist()
    else:
        raise ValueError(f"{name}: tipo no soportado ({type(rules_obj)}).")

    return list(pd.unique(pd.Series(rules, dtype=str)))


def select_stable_rules(
    rule_returns: pd.DataFrame,
    winners_list,
    side: str = "auto",
    top_n: int = 25,
    min_ops: int = 100,
    w_sharpe: float = 0.5,
    w_mono: float = 0.3,
    w_mdd: float = 0.2,
    decorrelated_df: Optional[pd.DataFrame] = None,
    decorrelated_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selecciona las N reglas mas estables en su P/L.

    score = w_sharpe * rank_pct(sharpe) + w_mono * monotonicity + w_mdd * (1-rank_pct(mdd))
    """
    if rule_returns is None or rule_returns.empty:
        empty_cols = ["regla", "score", "sharpe", "monotonicity", "max_drawdown", "ops", "mean", "std"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    winners_rules = _to_rule_list(winners_list, "winners_list")
    if len(winners_rules) == 0:
        empty_cols = ["regla", "score", "sharpe", "monotonicity", "max_drawdown", "ops", "mean", "std"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    decor_rules = None
    if decorrelated_df is not None and not decorrelated_df.empty:
        if "regla" not in decorrelated_df.columns:
            raise ValueError("decorrelated_df no contiene columna 'regla'.")
        decor_rules = decorrelated_df["regla"].dropna().astype(str).unique().tolist()

    if decor_rules is not None and decorrelated_only:
        rules_to_eval = sorted(set(winners_rules).intersection(decor_rules))
    else:
        rules_to_eval = sorted(set(winners_rules))

    if len(rules_to_eval) == 0:
        empty_cols = ["regla", "score", "sharpe", "monotonicity", "max_drawdown", "ops", "mean", "std"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    rules_present = [r for r in rules_to_eval if r in rule_returns.columns]
    if len(rules_present) == 0:
        raise ValueError("Ninguna regla de winners_list esta en rule_returns.")

    R = rule_returns[rules_present].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    A = R.to_numpy(dtype=float)  # T x N
    _, _n = A.shape

    # Operaciones: barras con retorno != 0 (equivale a regla activa)
    ops = (A != 0.0).sum(axis=0)
    ok = ops >= int(min_ops)
    if not ok.any():
        empty_cols = ["regla", "score", "sharpe", "monotonicity", "max_drawdown", "ops", "mean", "std"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    A = A[:, ok]
    ops = ops[ok]
    use_cols = [c for i, c in enumerate(rules_present) if ok[i]]
    t_len, _n = A.shape

    side_norm = str(side).lower().strip()
    if side_norm in ("long", "short"):
        sgn = 1.0 if side_norm == "long" else -1.0
        A = sgn * A
    else:
        # AUTO: orienta cada regla segun mejor Sharpe entre A y -A
        mean0 = A.mean(axis=0)
        std0 = A.std(axis=0, ddof=0)
        std0[std0 < 1e-12] = 1e-12
        sharpe0 = mean0 / std0

        mean1 = (-A).mean(axis=0)
        std1 = (-A).std(axis=0, ddof=0)
        std1[std1 < 1e-12] = 1e-12
        sharpe1 = mean1 / std1

        flip = sharpe1 > sharpe0
        A[:, flip] *= -1.0

    mean = A.mean(axis=0)
    std = A.std(axis=0, ddof=0)
    std[std < 1e-12] = 1e-12
    sharpe = mean / std

    csum = A.cumsum(axis=0)
    t = np.arange(t_len, dtype=float)
    t_center = t - t.mean()
    t_norm = np.sqrt((t_center**2).sum())
    X = csum - csum.mean(axis=0)
    num = t_center @ X
    den = t_norm * np.sqrt((X**2).sum(axis=0))
    mono = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    monotonicity = np.clip(mono, 0.0, 1.0)

    run_max = np.maximum.accumulate(csum, axis=0)
    drawdown = csum - run_max
    mdd = -drawdown.min(axis=0)

    sharpe_pct = pd.Series(sharpe).rank(pct=True).to_numpy()
    mdd_pct = pd.Series(mdd).rank(pct=True).to_numpy()
    mdd_score = 1.0 - mdd_pct

    score = (
        w_sharpe * sharpe_pct
        + w_mono * monotonicity
        + w_mdd * mdd_score
    )

    ranking_df = pd.DataFrame(
        {
            "regla": use_cols,
            "score": score,
            "sharpe": sharpe,
            "monotonicity": monotonicity,
            "max_drawdown": mdd,
            "ops": ops,
            "mean": mean,
            "std": std,
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)

    if decor_rules is not None and not decorrelated_only:
        ranking_df.insert(1, "is_decorrelated", ranking_df["regla"].isin(decor_rules).astype(int))

    n_select = min(int(top_n), len(ranking_df))
    best_df = ranking_df.head(n_select).copy().reset_index(drop=True)
    return best_df, ranking_df


def run_pl_stability_selection(
    winners_long,
    winners_short,
    *,
    data: Optional[pd.DataFrame] = None,
    rule_returns_long: Optional[pd.DataFrame] = None,
    rule_returns_short: Optional[pd.DataFrame] = None,
    rules_long_df: Optional[pd.DataFrame] = None,
    rules_short_df: Optional[pd.DataFrame] = None,
    decorrelated_long_df: Optional[pd.DataFrame] = None,
    decorrelated_short_df: Optional[pd.DataFrame] = None,
    return_col: str = "Target",
    top_n_long: int = 25,
    top_n_short: int = 25,
    min_ops: int = 100,
    w_sharpe: float = 0.5,
    w_mono: float = 0.3,
    w_mdd: float = 0.2,
    side_long: str = "auto",
    side_short: str = "auto",
    decorrelated_only: bool = True,
    chunk_size: int = 4000,
    dtype: str = "float32",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Runner de seleccion de estabilidad para LONG+SHORT.

    Opciones:
    - Si ya tienes `rule_returns_long/short`, los usa directamente.
    - Si no, los construye desde `data` usando `build_rule_return_matrix` y `rules_long_df/rules_short_df`.
      Si `rules_*_df` no se pasan, usa un DataFrame construido desde winners_*.
    """
    winners_long_list = _to_rule_list(winners_long, "winners_long")
    winners_short_list = _to_rule_list(winners_short, "winners_short")

    def _ensure_rules_df(df_rules: Optional[pd.DataFrame], winners_list: List[str], name: str) -> pd.DataFrame:
        if df_rules is not None and not df_rules.empty:
            if "regla" not in df_rules.columns:
                raise ValueError(f"{name} no contiene columna 'regla'.")
            return df_rules.copy()
        return pd.DataFrame({"regla": winners_list})

    if rule_returns_long is None:
        if data is None:
            raise ValueError("Falta `data` para construir `rule_returns_long`.")
        rules_long_eval = _ensure_rules_df(rules_long_df, winners_long_list, "rules_long_df")
        if verbose:
            print(f"[validacion_pl] Construyendo rule_returns_long con {len(rules_long_eval)} reglas...")
        rule_returns_long = build_rule_return_matrix(
            data=data,
            df_rules=rules_long_eval,
            direction="long",
            return_col=return_col,
            chunk_size=chunk_size,
            dtype=dtype,
        )

    if rule_returns_short is None:
        if data is None:
            raise ValueError("Falta `data` para construir `rule_returns_short`.")
        rules_short_eval = _ensure_rules_df(rules_short_df, winners_short_list, "rules_short_df")
        if verbose:
            print(f"[validacion_pl] Construyendo rule_returns_short con {len(rules_short_eval)} reglas...")
        rule_returns_short = build_rule_return_matrix(
            data=data,
            df_rules=rules_short_eval,
            direction="short",
            return_col=return_col,
            chunk_size=chunk_size,
            dtype=dtype,
        )

    best_long, rank_long = select_stable_rules(
        rule_returns=rule_returns_long,
        winners_list=winners_long_list,
        side=side_long,
        top_n=top_n_long,
        min_ops=min_ops,
        w_sharpe=w_sharpe,
        w_mono=w_mono,
        w_mdd=w_mdd,
        decorrelated_df=decorrelated_long_df,
        decorrelated_only=decorrelated_only,
    )

    best_short, rank_short = select_stable_rules(
        rule_returns=rule_returns_short,
        winners_list=winners_short_list,
        side=side_short,
        top_n=top_n_short,
        min_ops=min_ops,
        w_sharpe=w_sharpe,
        w_mono=w_mono,
        w_mdd=w_mdd,
        decorrelated_df=decorrelated_short_df,
        decorrelated_only=decorrelated_only,
    )

    if verbose:
        print(
            f"[validacion_pl] LONG: {len(winners_long_list)} -> {len(best_long)} | "
            f"SHORT: {len(winners_short_list)} -> {len(best_short)}"
        )

    return {
        "best_long": best_long,
        "ranking_long": rank_long,
        "best_short": best_short,
        "ranking_short": rank_short,
        "winners_long_stable": best_long["regla"].tolist() if not best_long.empty else [],
        "winners_short_stable": best_short["regla"].tolist() if not best_short.empty else [],
        "rule_returns_long": rule_returns_long,
        "rule_returns_short": rule_returns_short,
    }

