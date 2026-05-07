"""
Servicios de portfolio compartidos (no PPO).

Originalmente este paquete contenia el stack PPO (PPOTrainer, MaskedPortfolioPolicy,
PPOInferenceService, WeeklyPortfolioEnv, PortfolioDatasetBuilder, etc.). Tras la
refactorizacion a un optimizador hibrido GA + PSO, se ha eliminado todo el codigo
de RL/PPO del proyecto. El paquete se mantiene unicamente para los servicios
auxiliares que NO son PPO y que aun siguen en uso desde el supervisor:

- `PortfolioOHLCRefreshService`: refresco mensual de datos OHLC de los activos
  necesarios para los traders promovidos.
- `UniverseRegistry`: persistencia del universo de traders promovidos.
"""

from .data_refresh import OHLCRefreshResult, PortfolioOHLCRefreshService
from .universe_registry import UniverseRegistry

__all__ = [
    "OHLCRefreshResult",
    "PortfolioOHLCRefreshService",
    "UniverseRegistry",
]
