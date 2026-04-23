from __future__ import annotations

from app.services import run_offline_pipeline


def main() -> int:
    print("=== Phase 3 Check ===")
    result = run_offline_pipeline(
        asset="AAPL",
        asset_csv_path="datos/Stocks/AAPL.csv",
        timeframe="D1",
        families=("decision_tree", "rulefit", "genetico", "quantile", "subgroup"),
        family_params={
            # Mantener check razonablemente rápido sin perder cobertura de familias
            "decision_tree": {"target_n_rules": 90},
            "rulefit": {"target_n_rules": 90, "n_estimators": 40, "max_candidate_rules": 350},
            "genetico": {"target_n_rules": 90, "population_size": 70, "n_generations": 15},
            "quantile": {"n_bins": 5, "combo_size": 2, "min_coverage": 120},
            "subgroup": {"n_bins": 5, "min_coverage": 80},
        },
        artifacts_root="app/.tmp/phase3",
    )

    print(f"dataset_id: {result.dataset_contract.dataset_id}")
    print(f"experiment_id: {result.experiment_config.experiment_id}")
    print(f"artifact_dir: {result.artifacts_dir}")
    print(f"candidate_long: {len(result.candidate_rules.long_rules)}")
    print(f"candidate_short: {len(result.candidate_rules.short_rules)}")
    print(f"stable_long: {result.validation_report.passed_long}")
    print(f"stable_short: {result.validation_report.passed_short}")
    print(f"promoted_trader_id: {result.promoted_summary['trader_id']}")

    if len(result.candidate_rules.long_rules) + len(result.candidate_rules.short_rules) == 0:
        raise RuntimeError("No se generaron reglas candidatas en Fase 3.")

    print("Phase 3 check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

