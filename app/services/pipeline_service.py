from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping
from uuid import uuid4

import pandas as pd

from app.contracts import CandidateRules, DatasetContract, ExperimentConfig, ValidationReport

from .data_service import load_asset_ohlc
from .feature_service import build_features
from .promotion_service import build_promoted_spec, summarize_promotion
from .rule_generation_service import generate_candidate_rules
from .split_service import split_is_oos_holdout
from .target_service import apply_target_to_blocks
from .validation_service import run_validation_pipeline


@dataclass
class OfflinePipelineResult:
    dataset_contract: DatasetContract
    experiment_config: ExperimentConfig
    candidate_rules: CandidateRules
    validation_report: ValidationReport
    promoted_summary: Dict[str, object]
    artifacts_dir: Path


def _safe_rules_from_df(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    col = "regla" if "regla" in df.columns else ("rule" if "rule" in df.columns else None)
    if col is None:
        return []
    return df[col].dropna().astype(str).tolist()


def _export_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def run_offline_pipeline(
    *,
    asset: str = "AAPL",
    asset_csv_path: str = "datos/Stocks/AAPL.csv",
    timeframe: str = "D1",
    families: Iterable[str] = ("decision_tree", "rulefit", "genetico", "quantile", "subgroup"),
    family_params: Mapping[str, Mapping[str, object]] | None = None,
    artifacts_root: str = "app/.tmp/phase3",
) -> OfflinePipelineResult:
    exp_id = f"exp_{asset.lower()}_{uuid4().hex[:8]}"
    artifacts_dir = Path(artifacts_root) / asset.upper() / exp_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ohlc = load_asset_ohlc(asset_csv_path=asset_csv_path)
    dataset_contract = DatasetContract(
        dataset_id=f"ds_{asset.lower()}_{timeframe.lower()}_{uuid4().hex[:6]}",
        asset=asset,
        timeframe=timeframe,
        source_path=str(asset_csv_path),
        rows=len(ohlc),
        start_date=str(ohlc.index.min().date()),
        end_date=str(ohlc.index.max().date()),
        quality_score=1.0,
        metadata={"scope": "phase3_offline"},
    )

    features = build_features(data_ohlc=ohlc)
    _export_df(features.tail(500), artifacts_dir / "features_tail.csv")

    blocks = split_is_oos_holdout(features)
    blocks = apply_target_to_blocks(blocks)

    experiment = ExperimentConfig(
        experiment_id=exp_id,
        asset=asset,
        timeframe=timeframe,
        split_policy="is_oos_holdout_2025",
        model_families=list(families),
        parameters={
            "is_pct": 0.5,
            "oos_pct": 0.5,
            "holdout_year": 2025,
            "lookback_years": 10,
        },
    )

    candidates_by_family = generate_candidate_rules(
        data_is=blocks["data_is"],
        families=families,
        family_params=family_params,
    )
    # export candidatos por familia
    for fam, grp in candidates_by_family.items():
        long_df = grp.get("long", pd.DataFrame())
        short_df = grp.get("short", pd.DataFrame())
        _export_df(long_df, artifacts_dir / f"candidates_long_{fam}.csv")
        _export_df(short_df, artifacts_dir / f"candidates_short_{fam}.csv")

    all_long: list[str] = []
    all_short: list[str] = []
    for grp in candidates_by_family.values():
        all_long.extend(_safe_rules_from_df(grp.get("long", pd.DataFrame())))
        all_short.extend(_safe_rules_from_df(grp.get("short", pd.DataFrame())))

    candidate_rules = CandidateRules(
        experiment_id=exp_id,
        asset=asset,
        long_rules=list(dict.fromkeys(all_long)),
        short_rules=list(dict.fromkeys(all_short)),
        generation_summary={
            "families": list(candidates_by_family.keys()),
            "n_long": len(all_long),
            "n_short": len(all_short),
        },
    )

    validation_out = run_validation_pipeline(
        data_is=blocks["data_is"],
        data_oos=blocks["data_oos"],
        data_2025=blocks["data_2025"],
        candidates_by_family=candidates_by_family,
    )

    stable = validation_out["stability"]
    winners_long_stable = stable.get("winners_long_stable", [])
    winners_short_stable = stable.get("winners_short_stable", [])

    validation_report = ValidationReport(
        experiment_id=exp_id,
        asset=asset,
        passed_long=len(winners_long_stable),
        passed_short=len(winners_short_stable),
        failed_long=max(len(candidate_rules.long_rules) - len(winners_long_stable), 0),
        failed_short=max(len(candidate_rules.short_rules) - len(winners_short_stable), 0),
        notes="phase3 offline pipeline",
        metrics={
            "n_candidates_long": len(candidate_rules.long_rules),
            "n_candidates_short": len(candidate_rules.short_rules),
            "n_stable_long": len(winners_long_stable),
            "n_stable_short": len(winners_short_stable),
        },
    )

    _export_df(validation_out["decor_long"], artifacts_dir / "decor_long.csv")
    _export_df(validation_out["decor_short"], artifacts_dir / "decor_short.csv")
    _export_df(validation_out["forward"].get("passed_long_forward", pd.DataFrame()), artifacts_dir / "forward_long.csv")
    _export_df(validation_out["forward"].get("passed_short_forward", pd.DataFrame()), artifacts_dir / "forward_short.csv")
    _export_df(stable.get("best_long", pd.DataFrame()), artifacts_dir / "stable_long.csv")
    _export_df(stable.get("best_short", pd.DataFrame()), artifacts_dir / "stable_short.csv")

    promoted = build_promoted_spec(
        asset=asset,
        timeframe=timeframe,
        experiment_id=exp_id,
        winners_long_stable=winners_long_stable,
        winners_short_stable=winners_short_stable,
    )
    promoted_summary = summarize_promotion(promoted)

    pd.DataFrame([dataset_contract.to_dict()]).to_csv(artifacts_dir / "dataset_contract.csv", index=False, encoding="utf-8")
    pd.DataFrame([experiment.to_dict()]).to_csv(artifacts_dir / "experiment_config.csv", index=False, encoding="utf-8")
    pd.DataFrame([validation_report.to_dict()]).to_csv(artifacts_dir / "validation_report.csv", index=False, encoding="utf-8")
    pd.DataFrame([promoted_summary]).to_csv(artifacts_dir / "promoted_summary.csv", index=False, encoding="utf-8")

    return OfflinePipelineResult(
        dataset_contract=dataset_contract,
        experiment_config=experiment,
        candidate_rules=candidate_rules,
        validation_report=validation_report,
        promoted_summary=promoted_summary,
        artifacts_dir=artifacts_dir,
    )

