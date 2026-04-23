"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel

from pyeventbt.data_provider.core.interfaces.data_provider_interface import IDataProvider
from pyeventbt.execution_engine.core.interfaces.execution_engine_interface import IExecutionEngine
from pyeventbt.portfolio.core.interfaces.portfolio_interface import IPortfolio
from pyeventbt.trading_context.trading_context import TypeContext


class Modules(BaseModel):
    
    TRADING_CONTEXT: TypeContext = TypeContext.BACKTEST
    DATA_PROVIDER: IDataProvider 
    EXECUTION_ENGINE: IExecutionEngine
    PORTFOLIO: IPortfolio
    
    
    class Config:
        arbitrary_types_allowed = True