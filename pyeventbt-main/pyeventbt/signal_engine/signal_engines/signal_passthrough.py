"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.strategy.core.modules import Modules
from pyeventbt.events.events import BarEvent, SignalEvent
from ..core.interfaces.signal_engine_interface import ISignalEngine

class SignalPassthrough(ISignalEngine):
    
    def generate_signal(self, bar_event: BarEvent, modules: Modules) -> SignalEvent:
        pass