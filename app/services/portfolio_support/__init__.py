"""
Servicios de soporte para el `PortfolioManagerProcess`.

Este paquete agrupa los servicios auxiliares que el optimizador de cartera
hibrido (GA + PSO) necesita para operar:

- `PortfolioOHLCRefreshService`: refresco mensual de datos OHLC de los activos
  necesarios para los traders promovidos.
- `UniverseRegistry`: persistencia del universo de traders promovidos.

Nota historica: este paquete se llamaba `portfolio_rl` cuando el sistema usaba
PPO. Tras la refactorizacion a GA+PSO se ha eliminado todo el codigo PPO y el
paquete se ha renombrado a `portfolio_support` para reflejar su rol real.
"""

from .data_refresh import OHLCRefreshResult, PortfolioOHLCRefreshService
from .universe_registry import UniverseRegistry

__all__ = [
    "OHLCRefreshResult",
    "PortfolioOHLCRefreshService",
    "UniverseRegistry",
]
