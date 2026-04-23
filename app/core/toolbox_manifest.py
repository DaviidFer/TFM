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
        module_path="indicators",
        callable_name="build_feature_library",
        file_path=Path("indicators/indicators.py"),
    ),
    ToolboxStage(
        stage_id="split",
        description="Particion temporal IS/OOS + holdout",
        module_path="particion_IS_OOS",
        callable_name="run_particion_is_oos",
        file_path=Path("particion_IS_OOS/particion.py"),
    ),
    ToolboxStage(
        stage_id="target",
        description="Target open-to-open",
        module_path="definicion_target",
        callable_name="run_target_para_bloques",
        file_path=Path("definicion_target/target.py"),
    ),
    ToolboxStage(
        stage_id="rule_quantiles",
        description="Reglas por combinaciones de bins",
        module_path="ML_tools",
        callable_name="build_quantile_bin_combinations",
        file_path=Path("ML_tools/quantile_bins.py"),
    ),
    ToolboxStage(
        stage_id="rule_tree",
        description="Reglas extraidas de arboles multi-seed",
        module_path="ML_tools",
        callable_name="build_decision_tree_rules_multiseed",
        file_path=Path("ML_tools/decision_tree.py"),
    ),
    ToolboxStage(
        stage_id="rule_rulefit",
        description="Reglas via RuleFit multi-seed",
        module_path="ML_tools",
        callable_name="build_rulefit_rules_multiseed",
        file_path=Path("ML_tools/rulefit.py"),
    ),
    ToolboxStage(
        stage_id="rule_genetic",
        description="Reglas via algoritmo genetico",
        module_path="ML_tools",
        callable_name="run_genetico_rules",
        file_path=Path("ML_tools/genetico.py"),
    ),
    ToolboxStage(
        stage_id="rule_subgroup",
        description="Subgroup discovery sobre indicadores",
        module_path="ML_tools",
        callable_name="run_subgroup_discovery_rules",
        file_path=Path("ML_tools/subgroup_discovery.py"),
    ),
    ToolboxStage(
        stage_id="validate_monos",
        description="Validacion estadistica contra monos aleatorios",
        module_path="validacion_monos",
        callable_name="monkey_validate_oos_multi",
        file_path=Path("validacion_monos/validacion.py"),
    ),
    ToolboxStage(
        stage_id="validate_corr",
        description="Pruning por correlacion de retornos por regla",
        module_path="validacion_correlacion_pl",
        callable_name="run_pl_correlation_pruning",
        file_path=Path("validacion_correlacion_pl/correlacion.py"),
    ),
    ToolboxStage(
        stage_id="validate_forward",
        description="Filtro forward anual de rentabilidad",
        module_path="validacion_forward",
        callable_name="validate_forward_year_profitability",
        file_path=Path("validacion_forward/forward.py"),
    ),
    ToolboxStage(
        stage_id="validate_stability",
        description="Seleccion final por estabilidad de curva P/L",
        module_path="validacion_estabilidad_pl",
        callable_name="run_pl_stability_selection",
        file_path=Path("validacion_estabilidad_pl/estabilidad.py"),
    ),
    ToolboxStage(
        stage_id="execution",
        description="Ejecucion determinista de reglas en runtime",
        module_path="backtest_eventos",
        callable_name="run_event_backtest",
        file_path=Path("backtest_eventos/runner.py"),
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

