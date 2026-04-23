"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.strategy.core.modules import Modules
# from pyeventbt.data_provider.core.interfaces.data_provider_interface import IDataProvider
# from pyeventbt.execution_engine.core.interfaces.execution_engine_interface import IExecutionEngine
from pyeventbt.portfolio.portfolio import Portfolio
from pyeventbt.events.events import BarEvent, SignalEvent
from typing import Protocol, Callable

class ISignalEngine(Protocol):
    """
    ISignalEngine is an abstract base class providing an interface for
    all subsequent (inherited) signal engine handling objects.

    This is designed to work both with historic and live data as
    the SignalEngine object is agnostic to the data source, since
    it obtains the data from an events queue.
    """
    

    def generate_signal(self, bar_event: BarEvent, modules: Modules) -> SignalEvent | list[SignalEvent]:
        """
        Provides the mechanisms to calculate the list of signals.

        In derived classes this is used to handle the generation of
        SignalEvent objects based on market data updates.
        """
        raise NotImplementedError("Should implement generate_signal()")
    
    

class SignalEngineGenerator(ISignalEngine):
    
    def __init__(self) -> None:
        self.signal_generator = lambda x: None
        
    def generate_signal(self, bar_event: BarEvent) -> SignalEvent | list[SignalEvent]:
        return self.signal_generator(bar_event)
    
    @staticmethod
    def generate_signal_engine(signal_generator: Callable[[BarEvent, Portfolio], SignalEvent] | list[SignalEvent]):
        signal_engine = SignalEngineGenerator()
        signal_engine.signal_generator = signal_generator
        return signal_engine
    
        