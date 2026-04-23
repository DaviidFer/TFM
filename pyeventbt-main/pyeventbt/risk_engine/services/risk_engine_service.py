"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Callable
from pyeventbt.strategy.core.modules import Modules
from ..core.interfaces.risk_engine_interface import IRiskEngine
from ..risk_engines.passthrough_risk_engine import PassthroughRiskEngine
from ..core.configurations.risk_engine_configurations import BaseRiskConfig, PassthroughRiskConfig
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder
from pyeventbt.events.events import OrderEvent
from queue import Queue
from decimal import Decimal

class RiskEngineService(IRiskEngine):
    
    def __init__(self, events_queue: Queue, risk_config: BaseRiskConfig, modules: Modules) -> None:
        self.events_queue = events_queue
        self.modules = modules
        self.risk_engine = self._get_risk_management_method(risk_config)

    def _get_risk_management_method(self, risk_config: BaseRiskConfig) -> IRiskEngine:
        
        if isinstance(risk_config, PassthroughRiskConfig):
            return PassthroughRiskEngine()
        
        # In case we add more default risk engines in the future...
        
        else:
            return PassthroughRiskEngine()

    def _create_and_put_order_event(self, suggested_order: SuggestedOrder, new_volume: Decimal) -> None:
        
        order_event = OrderEvent(
            symbol=suggested_order.signal_event.symbol,
            time_generated=suggested_order.signal_event.time_generated,
            strategy_id=suggested_order.signal_event.strategy_id,
            volume=new_volume,
            signal_type=suggested_order.signal_event.signal_type,
            order_type=suggested_order.signal_event.order_type,
            order_price=suggested_order.signal_event.order_price,
            sl=suggested_order.signal_event.sl,
            tp=suggested_order.signal_event.tp,
            rollover=suggested_order.signal_event.rollover,
            buffer_data=suggested_order.buffer_data
        )

        # Now put the event into the events queue
        self.events_queue.put(order_event)

    def assess_order(self, suggested_order: SuggestedOrder) -> None:
        # We get the suggested order that will be sent to the risk manager via the portfolio handler
        new_volume = self.risk_engine.assess_order(suggested_order, self.modules)

        # Now we need to generate an OrderEvent and put into the events queue
        if new_volume > 0.0:
            self._create_and_put_order_event(suggested_order, new_volume)
            
    def set_custom_asses_order(self, custom_asses_order: Callable[[SuggestedOrder], float]):
        
        modules = self.modules
        
        def _fn(suggested_order: SuggestedOrder) -> None:
            new_volume = custom_asses_order(suggested_order, modules)
            if new_volume > 0.0:
                self._create_and_put_order_event(suggested_order, new_volume)
                
        self.assess_order = _fn