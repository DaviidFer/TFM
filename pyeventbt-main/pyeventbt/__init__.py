"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0

PyEventBT - Event-Driven Backtesting Framework

A powerful event-driven backtesting and live trading framework for algorithmic trading strategies.

Usage:
    from pyeventbt import Strategy, BarEvent, SignalEvent, Modules, StrategyTimeframes
    from pyeventbt.indicators import ATR, KAMA
    # Or: from pyeventbt import indicators
    
    strategy = Strategy()
    
    @strategy.custom_signal_engine(strategy_id="my_strategy", strategy_timeframes=[StrategyTimeframes.ONE_HOUR])
    def my_strategy(event: BarEvent, modules: Modules):
        # Your strategy logic here
        return []
"""

# Dynamically read version from package metadata (pyproject.toml is the single source of truth)
try:
    from importlib.metadata import version
    __version__ = version("pyeventbt")
except Exception:
    # Fallback for development installations where package metadata might not be available
    __version__ = "0.0.0-dev"

__author__ = "Marti Castany, Alain Porto"
__website__ = "https://github.com/marticastany/pyeventbt"
__license__ = "Apache License, Version 2.0"
__description__ = "Event-driven backtesting and live trading framework for MetaTrader 5"

# Core Strategy Components
from .strategy.strategy import Strategy
from .strategy.core.modules import Modules
from .strategy.core.strategy_timeframes import StrategyTimeframes

# Events
from .events.events import (
    BarEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
)

# Risk Engine Configurations
from .risk_engine.core.configurations.risk_engine_configurations import (
    BaseRiskConfig,
    PassthroughRiskConfig,
)

# Sizing Engine Configurations
from .sizing_engine.core.configurations.sizing_engine_configurations import (
    BaseSizingConfig,
    MinSizingConfig,
    FixedSizingConfig,
    RiskPctSizingConfig,
)

# Indicators (import as submodule for namespacing)
from . import indicators

# Configuration
from .config.configs import Mt5PlatformConfig

# Core Entities
from .core.entities.hyper_parameter import HyperParameter
from .core.entities.variable import Variable

# Portfolio
from .portfolio.portfolio import Portfolio

# Data Provider
from .data_provider.core.entities.bar import Bar
from .data_provider.services.quantdle_data_updater import QuantdleDataUpdater

# Backtest Results
from .backtest.core.backtest_results import BacktestResults

__all__ = [
    # Package info
    "__version__",
    "__author__",
    "__website__",
    "__license__",
    "__description__",
    
    # Core Strategy
    "Strategy",
    "Modules",
    "StrategyTimeframes",
    
    # Events
    "BarEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    
    # Risk Engine
    "BaseRiskConfig",
    "PassthroughRiskConfig",
    
    # Sizing Engine
    "BaseSizingConfig",
    "MinSizingConfig",
    "FixedSizingConfig",
    "RiskPctSizingConfig",
    
    # Indicators (as submodule)
    "indicators",
    
    # Configuration
    "Mt5PlatformConfig",
    
    # Core Entities
    "HyperParameter",
    "Variable",
    
    # Portfolio
    "Portfolio",
    
    # Data
    "Bar",
    "QuantdleDataUpdater",
    
    # Results
    "BacktestResults",
]