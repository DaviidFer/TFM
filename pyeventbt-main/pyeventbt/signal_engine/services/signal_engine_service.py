"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.strategy.core.modules import Modules

from ..core.interfaces.signal_engine_interface import ISignalEngine
from ..signal_engines.signal_passthrough import SignalPassthrough
from ..signal_engines.signal_ma_crossover import SignalMACrossover
from ..core.configurations.signal_engine_configurations import BaseSignalEngineConfig, MACrossoverConfig
from pyeventbt.events.events import BarEvent
from queue import Queue
import logging

logger = logging.getLogger("pyeventbt")


class SignalEngineService:
    
    def __init__(self, events_queue: Queue, modules: Modules, signal_config: BaseSignalEngineConfig = None) -> None:
        self.events_queue = events_queue
        self.modules = modules
        self.signal_engine = self._get_signal_engine(signal_config)
        
    def _get_signal_engine(self, signal_config: BaseSignalEngineConfig) -> ISignalEngine:
        
        if isinstance(signal_config, MACrossoverConfig):
            return SignalMACrossover(configurations=signal_config, trading_context=self.modules.TRADING_CONTEXT) 
        
        else:
            logger.debug("NO PREDEFINED SINAL ENGINE PROVIDED - USING DECORATED SIGNAL ENGINE")
            return SignalPassthrough()
    
    def set_signal_engine(self, new_signal_engine):
        """Sets the internal signal engine to the signal engine passed by argument

        Args:
            new_signal_engine (ISignalEngine): The signal engine to be used
        """
        def generate_signal(bar_event: BarEvent) -> None:
            
            # We get the signal event from the Generate Signal
            signal_event =  new_signal_engine(bar_event, self.modules)
            
            # Put the SignalEvent in the events queue
            if signal_event is not None:
                # Check if signal event is a list
                if isinstance(signal_event, list):
                    for signal in signal_event:
                        self.events_queue.put(signal)
                else:
                    self.events_queue.put(signal_event)
        
        self.generate_signal = generate_signal
    
    # This method is called from TradingDirector class
    def generate_signal(self, bar_event: BarEvent) -> None:
        
        # We get the signal event from the Generate Signal
        signal_event = self.signal_engine.generate_signal(bar_event, self.modules)

        # Put the SignalEvent in the events queue
        if signal_event is not None:
            # Check if signal event is a list
            if isinstance(signal_event, list):
                for signal in signal_event:
                    self.events_queue.put(signal)
            else:
                self.events_queue.put(signal_event)