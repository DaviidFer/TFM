"""
Servicios cuantitativos que utiliza HumanResourcesProcess para evaluar la
salud de los traders promovidos. Aqui solo viven funciones puras de calculo
de metricas y un servicio que ejecuta el backtest forward post-promocion.
"""

from .forward_backtest_service import ForwardBacktestService, build_trader_design_profile
from .health_scoring import evaluate_trader_health
from .metrics import (
    align_development_and_forward_curves,
    build_metric_comparison_table,
    compute_avg_loss,
    compute_avg_win,
    compute_expectancy,
    compute_losing_streak,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    compute_trade_frequency,
    compute_winrate,
)

__all__ = [
    "ForwardBacktestService",
    "build_trader_design_profile",
    "evaluate_trader_health",
    "compute_sharpe",
    "compute_profit_factor",
    "compute_max_drawdown",
    "compute_winrate",
    "compute_avg_win",
    "compute_avg_loss",
    "compute_expectancy",
    "compute_losing_streak",
    "compute_trade_frequency",
    "align_development_and_forward_curves",
    "build_metric_comparison_table",
]
