from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .domain import PHASE1_SCOPE


@dataclass(frozen=True)
class ToolboxStage:
    stage_id: str
    description: str
    module_path: str
    callable_name: str
    file_path: Path


TOOLBOX_STAGES: Tuple[ToolboxStage, ...] = (
    ToolboxStage(
        stage_id="features",
        description="Generacion de indicadores y libreria de features",
        module_path="app.toolbox.indicators",
        callable_name="build_feature_library",
        file_path=Path("app/toolbox/indicators/indicators.py"),
    ),
    ToolboxStage(
        stage_id="split",
        description="Particion temporal IS/OOS + holdout",
        module_path="app.toolbox.particion_IS_OOS",
        callable_name="run_particion_is_oos",
        file_path=Path("app/toolbox/particion_IS_OOS/particion.py"),
    ),
    ToolboxStage(
        stage_id="target",
        description="Target open-to-open",
        module_path="app.toolbox.definicion_target",
        callable_name="run_target_para_bloques",
        file_path=Path("app/toolbox/definicion_target/target.py"),
    ),
    ToolboxStage(
        stage_id="rule_quantiles",
        description="Reglas por combinaciones de bins",
        module_path="app.toolbox.ML_tools",
        callable_name="build_quantile_bin_combinations",
        file_path=Path("app/toolbox/ML_tools/quantile_bins.py"),
    ),
    ToolboxStage(
        stage_id="rule_tree",
        description="Reglas extraidas de arboles multi-seed",
        module_path="app.toolbox.ML_tools",
        callable_name="build_decision_tree_rules_multiseed",
        file_path=Path("app/toolbox/ML_tools/decision_tree.py"),
    ),
    ToolboxStage(
        stage_id="rule_rulefit",
        description="Reglas via RuleFit multi-seed",
        module_path="app.toolbox.ML_tools",
        callable_name="build_rulefit_rules_multiseed",
        file_path=Path("app/toolbox/ML_tools/rulefit.py"),
    ),
    ToolboxStage(
        stage_id="rule_genetic",
        description="Reglas via algoritmo genetico",
        module_path="app.toolbox.ML_tools",
        callable_name="run_genetico_rules",
        file_path=Path("app/toolbox/ML_tools/genetico.py"),
    ),
    ToolboxStage(
        stage_id="validate_monos",
        description="Validacion estadistica contra monos aleatorios",
        module_path="app.validation.monos",
        callable_name="monkey_validate_oos_multi",
        file_path=Path("app/validation/monos.py"),
    ),
    ToolboxStage(
        stage_id="validate_corr",
        description="Pruning por correlacion de retornos por regla",
        module_path="app.validation.correlation",
        callable_name="run_pl_correlation_pruning",
        file_path=Path("app/validation/correlation.py"),
    ),
    ToolboxStage(
        stage_id="validate_forward",
        description="Filtro forward anual de rentabilidad",
        module_path="app.validation.forward",
        callable_name="validate_forward_year_profitability",
        file_path=Path("app/validation/forward.py"),
    ),
    ToolboxStage(
        stage_id="validate_stability",
        description="Seleccion final por estabilidad de curva P/L",
        module_path="app.validation.stability",
        callable_name="run_pl_stability_selection",
        file_path=Path("app/validation/stability.py"),
    ),
    ToolboxStage(
        stage_id="execution",
        description="Ejecucion determinista de reglas en runtime",
        module_path="app.toolbox.backtest_eventos",
        callable_name="run_event_backtest",
        file_path=Path("app/toolbox/backtest_eventos/runner.py"),
    ),
)


def validate_phase1_toolbox(root: Path) -> Dict[str, List[str]]:
    """
    Verifica precondiciones minimas de Fase 1:
    - Archivos del toolbox presentes
    - CSVs de activos de muestra disponibles
    """
    missing_files: List[str] = []
    missing_assets: List[str] = []

    for stage in TOOLBOX_STAGES:
        if not (root / stage.file_path).exists():
            missing_files.append(str(stage.file_path))

    for ticker in PHASE1_SCOPE.sample_assets:
        asset_path = root / PHASE1_SCOPE.data_root / f"{ticker}.csv"
        if not asset_path.exists():
            missing_assets.append(str(asset_path))

    return {
        "missing_toolbox_files": missing_files,
        "missing_sample_assets": missing_assets,
    }

