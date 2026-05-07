"""Contratos compartidos entre agentes y procesos del sistema."""

from .enums import (
    AgentKind,
    AgentStatus,
    EventType,
    TraderLifecycleState,
    TraderReviewAction,
)
from .models import (
    CandidateRules,
    DatasetContract,
    EventRecord,
    ExperimentConfig,
    PortfolioDecision,
    PortfolioRebalanceSnapshot,
    PromotedTraderSpec,
    RetrainRequest,
    TraderDesignProfile,
    TraderForwardMetrics,
    TraderHealthConfig,
    TraderHealthSnapshot,
    TraderLiveMetrics,
    ValidationReport,
)

__all__ = [
    "AgentKind",
    "AgentStatus",
    "EventType",
    "TraderLifecycleState",
    "TraderReviewAction",
    "DatasetContract",
    "ExperimentConfig",
    "CandidateRules",
    "ValidationReport",
    "PromotedTraderSpec",
    "TraderLiveMetrics",
    "PortfolioDecision",
    "PortfolioRebalanceSnapshot",
    "TraderDesignProfile",
    "TraderForwardMetrics",
    "TraderHealthConfig",
    "TraderHealthSnapshot",
    "RetrainRequest",
    "EventRecord",
]
