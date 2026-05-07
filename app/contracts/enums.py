from __future__ import annotations

from enum import Enum


class AgentKind(str, Enum):
    DATA = "data_process"
    DEVELOPER = "developer_agent"
    VALIDATION = "validation_agent"
    TRADER = "trader_agent"
    PORTFOLIO = "portfolio_manager"
    HUMAN_RESOURCES = "human_resources_process"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"
    BLOCKED = "blocked"


class TraderLifecycleState(str, Enum):
    LIVE = "live"
    RETRAINING = "retraining"


class TraderReviewAction(str, Enum):
    """
    Resultado de la revision periodica que hace HumanResourcesProcess sobre un
    trader promovido. Solo dos acciones: el trader sigue valido (`KEEP`) o se
    manda a reentrenamiento (`RETRAINING`).
    """

    KEEP = "keep"
    RETRAINING = "retraining"


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
    TRADER_HEALTH_EVALUATED = "trader_health_evaluated"
    RETRAIN_REQUESTED = "retrain_requested"
    RETRAIN_PROCESSED = "retrain_processed"
    PORTFOLIO_REBALANCE_SNAPSHOT = "portfolio_rebalance_snapshot"
    BROKER_ORDER_ROUTED = "broker_order_routed"
    BROKER_ORDER_REJECTED = "broker_order_rejected"
    BROKER_ACCESS_DENIED = "broker_access_denied"
