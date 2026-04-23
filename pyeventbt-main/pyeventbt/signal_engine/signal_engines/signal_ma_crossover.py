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
from ..core.configurations.signal_engine_configurations import MACrossoverConfig
from pyeventbt.data_provider.core.interfaces.data_provider_interface import IDataProvider
from pyeventbt.portfolio.portfolio import Portfolio
from pyeventbt.execution_engine.core.interfaces.execution_engine_interface import IExecutionEngine
from pyeventbt.events.events import BarEvent, SignalEvent
from datetime import datetime

import pyeventbt.trading_context.trading_context as trading_context
import pandas as pd

class SignalMACrossover(ISignalEngine):
    """
    SignalMACrossover generates trading signals based on a moving average crossover strategy.
    """

    def __init__(self, configurations: MACrossoverConfig, trading_context: trading_context.TypeContext = trading_context.TypeContext.BACKTEST) -> None:
        """
        Initializes the SignalMACrossover instance.

        Args:
            configurations (MACrossoverConfig): The configuration parameters for the moving average crossover strategy.
            trading_context (trading_context.TypeContext, optional): The trading context. Defaults to trading_context.TypeContext.BACKTEST.
        """
        self.trading_context = trading_context
        self.strategy_id = configurations.strategy_id
        self.ma_type = configurations.ma_type
        self.signal_timeframe = configurations.signal_timeframe
        self.fast_period = configurations.fast_period
        self.slow_period = configurations.slow_period
    
    def generate_signal(self, bar_event: BarEvent, modules: Modules) -> SignalEvent:
        """
        Generates a trading signal based on the moving average crossover strategy.

        Args:
            bar_event (BarEvent): The bar event containing the latest bar data.
            modules (Modules): The modules containing the necessary data and functionality for generating the signal.

        Returns:
            SignalEvent: The generated trading signal event.
        """
        
        # Check if the bar event is the same as the signal timeframe
        if bar_event.timeframe != self.signal_timeframe:
            return

        symbol = bar_event.symbol

        # Get the needed data to compute the signal
        bars = modules.DATA_PROVIDER.get_latest_bars(symbol, self.signal_timeframe, self.slow_period + 1)
        
        # Get the number of open positions for the strategy in long and short sides
        open_positions = modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(symbol=symbol)

        # In case we still have no data (ie during start of bt where higher tf do not have any data yet) we return None and the Service will handle it.
        # We also check for the lenght. We will at least need 2 rows of dataframe to access the fast_ma.iloc[-2]
        if bars is None or bars.shape[0] < 2:
            return

        # Check the kind of averaging method and compute the moving average indicator values
        if self.ma_type == "SIMPLE":
            fast_ma = bars['close'][-self.fast_period:].mean()
            slow_ma = bars['close'].mean()
        
        elif self.ma_type == "EXPONENTIAL":
            fast_ma = bars['close'].ewm(span=self.fast_period, min_periods=1).mean().iloc[-1]
            slow_ma = bars['close'].ewm(span=self.slow_period, min_periods=1).mean().iloc[-1]
        
        # BUY signal
        if open_positions['LONG'] == 0 and fast_ma > slow_ma:
            if open_positions['SHORT'] > 0:
                # Buy signal but already have short positions. We need to close them first
                modules.EXECUTION_ENGINE.close_strategy_short_positions_by_symbol(symbol)
            signal = "BUY"
        
        # SELL signal
        elif open_positions['SHORT'] == 0 and fast_ma < slow_ma:
            if open_positions['LONG'] > 0:
                # Sell signal but already have long positions. We need to close them first
                modules.EXECUTION_ENGINE.close_strategy_long_positions_by_symbol(symbol)
            signal = "SELL"
        
        else:
            signal = ""
        
        # If there's a signal, return a signal event
        if signal != "":
            if self.trading_context== "BACKTEST":
                time_generated = bar_event.data.name + pd.Timedelta(self.signal_timeframe) # timedelta(minutes=1) 
                #time_generated = pd.to_datetime(data_provider.get_latest_tick(symbol)['time_msc'], unit='ms') # Creating a datetime from timestamp gives UTC hour. Need reconversion.
            else:
                time_generated=datetime.now()  # Computer's time zone
            
            # Getting the last tick
            last_tick = modules.DATA_PROVIDER.get_latest_tick(symbol)
            
            signal_event = SignalEvent(
                symbol=symbol,
                time_generated=time_generated,          # moment of generating the signal
                strategy_id=self.strategy_id,
                forecast=10,                            # Average forecast as this is a discrete signal strategy
                signal_type=signal,                     # BUY or SELL
                order_type="MARKET",                    # MARKET, LIMIT, STOP
                order_price=last_tick['ask'] if signal == "BUY" else last_tick['bid'],
                #sl=0.0,
                #tp=0.0
            )

            return signal_event
