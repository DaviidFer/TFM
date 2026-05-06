from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from .enums import AgentKind, EventType, RiskAction, TraderLifecycleState


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
    decision_id: str
    as_of: str
    selected_traders: List[str]
    weights: Dict[str, float]
    rationale: str = ""
    model_version: str = ""
    training_run_id: str = ""
    fine_tune_run_id: str = ""
    target_cash_weight: float = 0.0
    active_universe_size: int = 0
    selected_universe_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioModelInfo:
    model_version: str
    mode: str
    checkpoint_path: str
    universe_size: int
    trained_at: str = ""
    fine_tuned_at: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioTrainingRun:
    run_id: str
    run_type: str
    model_version: str
    status: str
    started_at: str
    completed_at: str = ""
    algorithm: str = "ppo"
    seed: int = 0
    device: str = "cpu"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioForwardEvaluation:
    evaluation_id: str
    rebalance_id: str
    benchmark_name: str
    as_of: str
    cumulative_return_1y: float
    sharpe_1y: float
    max_drawdown_1y: float
    curve_points: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioRebalanceSnapshot:
    rebalance_id: str
    rebalance_date: str
    model_version: str
    training_run_id: str = ""
    fine_tune_run_id: str = ""
    active_traders: List[str] = field(default_factory=list)
    selected_traders: List[str] = field(default_factory=list)
    target_weights: Dict[str, float] = field(default_factory=dict)
    target_cash_weight: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    forward_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DesignRiskProfile:
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
    ppo_selected_count: int = 0
    ppo_blocked_count: int = 0
    risk_blocked_count: int = 0
    insufficient_evidence: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskLimitsConfig:
    """
    Limites de riesgo simplificados (modelo binario LIVE / RETRAINING).
    Cualquier metrica forward por debajo del umbral correspondiente colapsa al
    trader a RETRAINING (peso 0, cash). No existe estado intermedio "amber".
    """
    min_forward_trades_for_retraining: int = 10
    retraining_health_threshold: float = 60.0
    max_losing_streak_multiplier: float = 1.5
    max_drawdown_multiplier_retraining: float = 1.5
    min_profit_factor_ratio_retraining: float = 0.75
    min_sharpe_ratio_retraining: float = 0.60
    max_weight_per_trader: float = 0.15
    max_weight_per_asset: float = 0.30
    max_total_exposure: float = 1.0
    max_sector_exposure: Optional[float] = None
    max_portfolio_volatility: Optional[float] = None
    max_portfolio_drawdown: Optional[float] = None
    min_cash_buffer: float = 0.10
    min_broker_margin_level: Optional[float] = None
    emergency_drawdown_stop: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraderHealthSnapshot:
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
    design_profile: Optional[DesignRiskProfile] = None
    forward_metrics: Optional[TraderForwardMetrics] = None
    risk_flags: Dict[str, Any] = field(default_factory=dict)
    retrain_request: Optional["RetrainRequest"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["design_profile"] = _serialize_dataclass(self.design_profile)
        out["forward_metrics"] = _serialize_dataclass(self.forward_metrics)
        out["retrain_request"] = _serialize_dataclass(self.retrain_request)
        return out


@dataclass(frozen=True)
class RiskDecision:
    decision_id: str
    trader_id: str
    as_of: str
    action: str
    reason: str
    triggered_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskAdjustedPortfolioDecision:
    rebalance_id: str
    evaluation_id: str
    original_decision: PortfolioDecision
    approved: bool
    action: str
    adjusted_weights: Dict[str, float]
    original_weights: Dict[str, float]
    forced_cash_weight: float = 0.0
    blocked_traders: List[str] = field(default_factory=list)
    clipped_traders: List[str] = field(default_factory=list)
    scaled_down: bool = False
    reasons: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["original_decision"] = _serialize_dataclass(self.original_decision)
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

