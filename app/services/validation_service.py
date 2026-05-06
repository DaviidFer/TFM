from __future__ import annotations

from typing import Dict, Mapping

import pandas as pd

from app.core.structured_logging import emit_log
from app.validation.correlation import run_pl_correlation_pruning
from app.validation.stability import run_pl_stability_selection
from app.validation.forward import validate_forward_year_profitability
from app.validation.monos import monkey_validate_is_multi, monkey_validate_oos_multi


DEFAULT_VALIDATION_PROFILE: Dict[str, Dict[str, object]] = {
    "split_assumption": {
        "is_data": "provided_by_split_service",
        "oos_data": "provided_by_split_service",
        "holdout_year": 2025,
    },
    "monkey_is": {
        "n_monkeys": 200,
        "is_pass_pct": 90.0,
        "n_jobs": 1,
    },
    "monkey_oos": {
        "n_monkeys": 200,
        "oos_pass_pct": 80.0,
        "min_coverage_oos": 60,
        "n_jobs": 1,
    },
    "correlation_pruning": {
        "corr_threshold": 0.50,
        "min_ops": 50,
        "diagnose": False,
    },
    "forward_validation": {
        "target_year": 2025,
        "min_ops": 30,
        "verbose": False,
    },
    "stability_selection": {
        "top_n_long": 15,
        "top_n_short": 15,
        "min_ops": 50,
        "verbose": False,
    },
}


def _concat_rule_groups(groups: Mapping[str, Mapping[str, pd.DataFrame]], side: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for family, data in groups.items():
        df = data.get(side, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            out[family] = df
    return out


def run_validation_pipeline(
    *,
    data_is: pd.DataFrame,
    data_oos: pd.DataFrame,
    data_2025: pd.DataFrame,
    candidates_by_family: Mapping[str, Mapping[str, pd.DataFrame]],
    validation_profile: Mapping[str, Mapping[str, object]] | None = None,
) -> Dict[str, object]:
    """
    Validacion ligera de Fase 3:
    IS/OOS (monos) -> decorrelacion -> forward -> estabilidad.
    """
    merged_profile: Dict[str, Dict[str, object]] = {
        k: dict(v) for k, v in DEFAULT_VALIDATION_PROFILE.items()
    }
    for section, values in (validation_profile or {}).items():
        merged_profile.setdefault(str(section), {})
        merged_profile[str(section)].update(dict(values))

    emit_log(
        "validation_service",
        "validation_pipeline_started",
        validation_profile=merged_profile,
        candidate_families=list(candidates_by_family.keys()),
    )

    monkey_is_cfg = merged_profile["monkey_is"]
    monkey_oos_cfg = merged_profile["monkey_oos"]
    corr_cfg = merged_profile["correlation_pruning"]
    fwd_cfg = merged_profile["forward_validation"]
    stab_cfg = merged_profile["stability_selection"]

    long_groups = _concat_rule_groups(candidates_by_family, side="long")
    short_groups = _concat_rule_groups(candidates_by_family, side="short")

    long_is = monkey_validate_is_multi(
        dataset_is=data_is,
        rule_dfs=long_groups,
        direction="long",
        n_monkeys=int(monkey_is_cfg["n_monkeys"]),
        is_pass_pct=float(monkey_is_cfg["is_pass_pct"]),
        n_jobs=int(monkey_is_cfg["n_jobs"]),
        min_coverage_is=int(monkey_is_cfg.get("min_coverage_is", 0)),
    ) if long_groups else pd.DataFrame()

    short_is = monkey_validate_is_multi(
        dataset_is=data_is,
        rule_dfs=short_groups,
        direction="short",
        n_monkeys=int(monkey_is_cfg["n_monkeys"]),
        is_pass_pct=float(monkey_is_cfg["is_pass_pct"]),
        n_jobs=int(monkey_is_cfg["n_jobs"]),
        min_coverage_is=int(monkey_is_cfg.get("min_coverage_is", 0)),
    ) if short_groups else pd.DataFrame()

    long_oos = monkey_validate_oos_multi(
        dataset_oos=data_oos,
        df_reglas_is_ok=long_is,
        direction="long",
        n_monkeys=int(monkey_oos_cfg["n_monkeys"]),
        oos_pass_pct=float(monkey_oos_cfg["oos_pass_pct"]),
        n_jobs=int(monkey_oos_cfg["n_jobs"]),
        min_coverage_oos=int(monkey_oos_cfg["min_coverage_oos"]),
    ) if not long_is.empty else pd.DataFrame()

    short_oos = monkey_validate_oos_multi(
        dataset_oos=data_oos,
        df_reglas_is_ok=short_is,
        direction="short",
        n_monkeys=int(monkey_oos_cfg["n_monkeys"]),
        oos_pass_pct=float(monkey_oos_cfg["oos_pass_pct"]),
        n_jobs=int(monkey_oos_cfg["n_jobs"]),
        min_coverage_oos=int(monkey_oos_cfg["min_coverage_oos"]),
    ) if not short_is.empty else pd.DataFrame()

    corr = run_pl_correlation_pruning(
        data_all=data_is,
        df_rules_long=long_oos if not long_oos.empty else None,
        df_rules_short=short_oos if not short_oos.empty else None,
        corr_threshold=float(corr_cfg["corr_threshold"]),
        min_ops=int(corr_cfg["min_ops"]),
        diagnose=bool(corr_cfg["diagnose"]),
    )
    decor_long = corr.get("final_rules_long", pd.DataFrame())
    decor_short = corr.get("final_rules_short", pd.DataFrame())

    forward = validate_forward_year_profitability(
        data_target_year=data_2025,
        df_rules_long=decor_long if not decor_long.empty else None,
        df_rules_short=decor_short if not decor_short.empty else None,
        target_year=int(fwd_cfg["target_year"]),
        min_ops=int(fwd_cfg["min_ops"]),
        verbose=bool(fwd_cfg["verbose"]),
    )
    passed_long = forward.get("passed_long_forward", pd.DataFrame())
    passed_short = forward.get("passed_short_forward", pd.DataFrame())

    stable = run_pl_stability_selection(
        winners_long=passed_long,
        winners_short=passed_short,
        data=data_is,
        rules_long_df=decor_long if not decor_long.empty else None,
        rules_short_df=decor_short if not decor_short.empty else None,
        decorrelated_long_df=decor_long if not decor_long.empty else None,
        decorrelated_short_df=decor_short if not decor_short.empty else None,
        top_n_long=int(stab_cfg["top_n_long"]),
        top_n_short=int(stab_cfg["top_n_short"]),
        min_ops=int(stab_cfg["min_ops"]),
        verbose=bool(stab_cfg["verbose"]),
    )
    emit_log(
        "validation_service",
        "validation_pipeline_completed",
        counts={
            "is_long": int(len(long_is)),
            "is_short": int(len(short_is)),
            "oos_long": int(len(long_oos)),
            "oos_short": int(len(short_oos)),
            "decor_long": int(len(decor_long)),
            "decor_short": int(len(decor_short)),
            "stable_long": int(len(stable.get("winners_long_stable", []))),
            "stable_short": int(len(stable.get("winners_short_stable", []))),
        },
    )

    return {
        "is_long": long_is,
        "is_short": short_is,
        "oos_long": long_oos,
        "oos_short": short_oos,
        "decor_long": decor_long,
        "decor_short": decor_short,
        "forward": forward,
        "stability": stable,
        "validation_profile": merged_profile,
    }

