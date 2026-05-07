from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .enums import AgentKind, EventType, TraderLifecycleState, TraderReviewAction


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_dataclass(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _serialize_dataclass(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_dataclass(v) for v in value]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return value.to_dict()
    if is_dataclass(value):
        out = asdict(value)
        for k, v in list(out.items()):
            out[k] = _serialize_dataclass(v)
        return out
    return value


@dataclass(frozen=True)
class DatasetContract:
    dataset_id: str
    asset: str
    timeframe: str
    source_path: str
    rows: int
    start_date: str
    end_date: str
    quality_score: float
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    asset: str
    timeframe: str
    split_policy: str
    model_families: List[str]
    parameters: Dict[str, Any]
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRules:
    experiment_id: str
    asset: str
    long_rules: List[str]
    short_rules: List[str]
    generation_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    experiment_id: str
    asset: str
    passed_long: int
    passed_short: int
    failed_long: int
    failed_short: int
    notes: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotedTraderSpec:
    trader_id: str
    asset: str
    timeframe: str
    long_rules: List[str]
    short_rules: List[str]
    origin_experiment_id: str
    lifecycle_state: TraderLifecycleState = TraderLifecycleState.LIVE
    promoted_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["lifecycle_state"] = self.lifecycle_state.value
        return out


@dataclass(frozen=True)
class TraderLiveMetrics:
    trader_id: str
    as_of: str
    pnl: float
    sharpe_rolling: float
    drawdown_rolling: float
    trade_count: int
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioDecision:
    """
    Decision semanal del PortfolioManagerProcess (modo GA + PSO).

    Toda la trazabilidad del optimizador hibrido va en `metadata` (parametros
    GA/PSO, lambdas, baselines, traders excluidos, etc.).
    """

    decision_id: str
    as_of: str
    selected_traders: List[str]
    weights: Dict[str, float]
    rationale: str = ""
    optimizer_mode: str = "ga_pso"
    target_cash_weight: float = 0.0
    active_universe_size: int = 0
    valid_universe_size: int = 0
    selected_universe_size: int = 0
    fitness: float = 0.0
    sharpe_neto: float = 0.0
    mdd: float = 0.0
    corr_media: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioRebalanceSnapshot:
    """Snapshot persistido del rebalanceo semanal (modo GA + PSO)."""

    rebalance_id: str
    rebalance_date: str
    optimizer_mode: str = "ga_pso"
    active_traders: List[str] = field(default_factory=list)
    selected_traders: List[str] = field(default_factory=list)
    target_weights: Dict[str, float] = field(default_factory=dict)
    target_cash_weight: float = 0.0
    fitness: float = 0.0
    sharpe_neto: float = 0.0
    mdd: float = 0.0
    corr_media: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    forward_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraderDesignProfile:
    """
    Perfil de diseno del trader: metricas de su comportamiento esperado
    medidas en backtest historico (IS / OOS / holdout). Es el "DNI" contra
    el que se compara el comportamiento forward post-promocion.
    """

    trader_id: str
    asset: str
    timeframe: str
    promoted_at: str
    design_start: Optional[str] = None
    design_end: Optional[str] = None
    oos_start: Optional[str] = None
    oos_end: Optional[str] = None
    holdout_start: Optional[str] = None
    holdout_end: Optional[str] = None
    sharpe_design: Optional[float] = None
    sharpe_oos: Optional[float] = None
    sharpe_holdout: Optional[float] = None
    profit_factor_design: Optional[float] = None
    profit_factor_oos: Optional[float] = None
    profit_factor_holdout: Optional[float] = None
    max_drawdown_design: Optional[float] = None
    max_drawdown_oos: Optional[float] = None
    max_drawdown_holdout: Optional[float] = None
    avg_loss_design: Optional[float] = None
    avg_win_design: Optional[float] = None
    winrate_design: Optional[float] = None
    expectancy_design: Optional[float] = None
    max_losing_streak_design: Optional[int] = None
    trades_design: Optional[int] = None
    trades_oos: Optional[int] = None
    trades_holdout: Optional[int] = None
    monthly_trade_frequency_design: Optional[float] = None
    returns_mean_design: Optional[float] = None
    returns_std_design: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraderForwardMetrics:
    """
    Metricas forward post-promocion: lo que el trader hace REALMENTE despues
    de salir a operativa, comparable contra `TraderDesignProfile`.

    Conceptos:
    - shadow_*: backtest forward con sus reglas sobre datos OOS reales.
    - executed_*: lo que ejecuto el broker (subset de las senales que el
      PortfolioManagerProcess termino seleccionando).
    - signal_count: cuantas senales emitio en el periodo forward.
    - pm_selected_count: cuantas senales acabaron seleccionadas por el
      PortfolioManagerProcess (GA+PSO).
    """

    trader_id: str
    asset: str
    timeframe: str
    evaluation_run_id: str
    promoted_at: str
    evaluation_date: str
    forward_start: str
    forward_end: str
    shadow_trades: int = 0
    executed_trades: int = 0
    shadow_pnl: float = 0.0
    executed_pnl: float = 0.0
    shadow_return: Optional[float] = None
    executed_return: Optional[float] = None
    shadow_sharpe: Optional[float] = None
    executed_sharpe: Optional[float] = None
    shadow_profit_factor: Optional[float] = None
    executed_profit_factor: Optional[float] = None
    shadow_max_drawdown: Optional[float] = None
    executed_max_drawdown: Optional[float] = None
    shadow_avg_loss: Optional[float] = None
    shadow_avg_win: Optional[float] = None
    shadow_winrate: Optional[float] = None
    shadow_expectancy: Optional[float] = None
    shadow_losing_streak: Optional[int] = None
    signal_count: int = 0
    pm_selected_count: int = 0
    insufficient_evidence: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraderHealthConfig:
    """
    Configuracion de umbrales de salud que usa HumanResourcesProcess para
    decidir entre `KEEP` y `RETRAINING`. No mezcla nada de cartera, pesos,
    margen ni broker: este componente ya no tiene esas competencias.
    """

    min_forward_trades_for_retraining: int = 10
    retraining_health_threshold: float = 60.0
    max_losing_streak_multiplier: float = 1.5
    max_drawdown_multiplier_retraining: float = 1.5
    min_profit_factor_ratio_retraining: float = 0.75
    min_sharpe_ratio_retraining: float = 0.60

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraderHealthSnapshot:
    """
    Resultado de una revision de salud de un trader: estado anterior,
    nuevo estado, accion (`KEEP` / `RETRAINING`), score y razones.
    """

    trader_id: str
    asset: str
    timeframe: str
    evaluation_run_id: str
    evaluation_date: str
    previous_state: str
    new_state: str
    health_score: float
    action: str
    reasons: List[str] = field(default_factory=list)
    design_profile: Optional[TraderDesignProfile] = None
    forward_metrics: Optional[TraderForwardMetrics] = None
    flags: Dict[str, Any] = field(default_factory=dict)
    retrain_request: Optional["RetrainRequest"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["design_profile"] = _serialize_dataclass(self.design_profile)
        out["forward_metrics"] = _serialize_dataclass(self.forward_metrics)
        out["retrain_request"] = _serialize_dataclass(self.retrain_request)
        return out


@dataclass(frozen=True)
class RetrainRequest:
    request_id: str
    trader_id: str
    asset: str
    timeframe: str
    reason: str
    requested_by: AgentKind
    requested_at: str = field(default_factory=utc_now_iso)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["requested_by"] = self.requested_by.value
        return out


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    event_type: EventType
    producer: AgentKind
    occurred_at: str
    payload: Mapping[str, Any]
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["event_type"] = self.event_type.value
        out["producer"] = self.producer.value
        out["payload"] = dict(self.payload)
        return out
