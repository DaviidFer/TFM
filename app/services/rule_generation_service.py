from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping

import pandas as pd

from app.toolbox.ML_tools import (
    build_decision_tree_rules_multiseed,
    build_quantile_bin_combinations,
    build_rulefit_rules_multiseed,
    run_genetico_rules,
)


# Familias cuyo generador acepta `target_n_rules` y, por tanto, pueden encajar
# en el bucle iterativo de `DeveloperAgent`. Centralizado aquí para que los
# consumidores no tengan que hardcodear el conjunto.
FAMILIES_WITH_RULE_TARGET: tuple[str, ...] = (
    "decision_tree",
    "rulefit",
    "genetico",
    "genetic",
)


def _safe_df(df: pd.DataFrame | None) -> pd.DataFrame:
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def safe_rules_from_df(df: pd.DataFrame | None) -> List[str]:
    """Extrae la columna de regla (`regla` o `rule`) como lista de strings.

    Helper compartido por el resto de servicios y agentes para evitar
    duplicar la misma normalización en varios sitios.
    """
    if df is None or df.empty:
        return []
    col = "regla" if "regla" in df.columns else ("rule" if "rule" in df.columns else None)
    if col is None:
        return []
    return df[col].dropna().astype(str).tolist()


def generate_candidate_rules(
    data_is: pd.DataFrame,
    *,
    families: Iterable[str] = ("decision_tree", "rulefit", "genetico", "quantile"),
    family_params: Mapping[str, Mapping[str, object]] | None = None,
) -> Dict[str, Mapping[str, pd.DataFrame]]:
    """
    Genera reglas candidatas por familia sin notebook.
    """
    out: Dict[str, Mapping[str, pd.DataFrame]] = {}
    fams = {f.strip().lower() for f in families}
    params: MutableMapping[str, Mapping[str, object]] = dict(family_params or {})

    if "decision_tree" in fams:
        dt_params = {
            "min_coverage": 100,
            "max_depth": 2,
            "min_samples_leaf": 100,
            "min_samples_split": 160,
            "target_n_rules": 120,
            "progress_every": 0,
        }
        dt_params.update(params.get("decision_tree", {}))
        long_dt, short_dt = build_decision_tree_rules_multiseed(
            data=data_is,
            **dt_params,
        )
        out["decision_tree"] = {"long": _safe_df(long_dt), "short": _safe_df(short_dt)}

    if "rulefit" in fams:
        rf_params = {
            "min_coverage": 100,
            "top_k_features": 100,
            "n_estimators": 50,
            "tree_depth": 2,
            "min_samples_leaf_tree": 100,
            "max_candidate_rules": 500,
            "target_n_rules": 120,
            "progress_every": 0,
        }
        rf_params.update(params.get("rulefit", {}))
        long_rf, short_rf = build_rulefit_rules_multiseed(
            data=data_is,
            **rf_params,
        )
        out["rulefit"] = {"long": _safe_df(long_rf), "short": _safe_df(short_rf)}

    if "genetico" in fams or "genetic" in fams:
        ga_params = {
            "n_bins": 4,
            "min_coverage": 100,
            "top_k_features": 120,
            "max_atoms": 140,
            "population_size": 90,
            "n_generations": 20,
            "max_rule_len": 2,
            "target_n_rules": 120,
            "progress_every": 0,
        }
        ga_params.update(params.get("genetico", {}))
        long_ga, short_ga = run_genetico_rules(
            data=data_is,
            **ga_params,
        )
        out["genetico"] = {"long": _safe_df(long_ga), "short": _safe_df(short_ga)}

    if "quantile" in fams:
        q_params = {
            "n_bins": 4,
            "combo_size": 2,
            "min_coverage": 100,
        }
        q_params.update(params.get("quantile", {}))
        long_q, short_q = build_quantile_bin_combinations(
            data=data_is,
            **q_params,
        )
        out["quantile"] = {"long": _safe_df(long_q), "short": _safe_df(short_q)}

    return out

