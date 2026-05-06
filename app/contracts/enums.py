from __future__ import annotations

from enum import Enum


class AgentKind(str, Enum):
    DATA = "data_process"
    DEVELOPER = "developer_agent"
    VALIDATION = "validation_agent"
    TRADER = "trader_agent"
    PORTFOLIO = "portfolio_manager"
    RISK = "risk_agent"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    BLOCKED = "blocked"


class TraderLifecycleState(str, Enum):
    LIVE = "live"
    RETRAINING = "retraining"


class RiskAction(str, Enum):
    KEEP = "keep"
    RETRAINING = "retraining"
    APPROVE = "approve"
    APPROVE_WITH_CLIPPING = "approve_with_clipping"
    SCALE_DOWN = "scale_down"
    FORCE_CASH = "force_cash"
    REJECT_PORTFOLIO = "reject_portfolio"
    EMERGENCY_STOP = "emergency_stop"


class EventType(str, Enum):
    DATASET_READY = "dataset_ready"
    DEVELOPMENT_STARTED = "development_started"
    SPLIT_AND_TARGET_READY = "split_and_target_ready"
    CANDIDATE_RULES_READY = "candidate_rules_ready"
    VALIDATION_COMPLETED = "validation_completed"
    TRADER_PROMOTED = "trader_promoted"
    TRADER_STATE_CHANGED = "trader_state_changed"
    TRADER_METRICS_UPDATED = "trader_metrics_updated"
    PORTFOLIO_DECISION = "portfolio_decision"
    RISK_DECISION = "risk_decision"
    RETRAIN_REQUESTED = "retrain_requested"
    RETRAIN_PROCESSED = "retrain_processed"
    PORTFOLIO_TRAINING_RUN = "portfolio_training_run"
    PORTFOLIO_MODEL_UPDATED = "portfolio_model_updated"
    PORTFOLIO_REBALANCE_SNAPSHOT = "portfolio_rebalance_snapshot"
    PORTFOLIO_FORWARD_EVALUATED = "portfolio_forward_evaluated"
    BROKER_ORDER_ROUTED = "broker_order_routed"
    BROKER_ORDER_REJECTED = "broker_order_rejected"
    BROKER_ACCESS_DENIED = "broker_access_denied"

