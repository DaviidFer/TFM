"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""


from pyeventbt import (
    Strategy,
    BarEvent,
    SignalEvent,
    Modules,
    StrategyTimeframes,
    PassthroughRiskConfig,
    MinSizingConfig,
    Mt5PlatformConfig,
)
from pyeventbt.indicators.indicators import BollingerBands

from datetime import datetime, time
from decimal import Decimal
import logging
import numpy as np

logger = logging.getLogger("pyeventbt")

# Strategy Configuration
strategy_id = "1234"
strategy = Strategy(logging_level=logging.INFO)

# Timeframes
signal_timeframe = StrategyTimeframes.ONE_HOUR
daily_timeframe = StrategyTimeframes.ONE_DAY

strategy_timeframes = [signal_timeframe, daily_timeframe]

# Trading Configuration
symbols_to_trade = ['EURUSD']
starting_capital = 100000

# Strategy Parameters
bb_period = 20
bb_std_dev = 2.5
close_hour = 21
close_minute = 0
order_placement_hour = 8
order_placement_minute = 0

# Daily tracking
orders_placed_today: dict[str, bool] = {symbol: False for symbol in symbols_to_trade}
current_trading_date: dict[str, datetime] = {symbol: None for symbol in symbols_to_trade}


@strategy.custom_signal_engine(strategy_id=strategy_id, strategy_timeframes=strategy_timeframes)
def bbands_breakout(event: BarEvent, modules: Modules):
    """
    Bollinger Bands Breakout Strategy:
    - Breakout levels: Upper and Lower Bollinger Bands
    - Exit: Close all at 21:00
    """
    
    symbol = event.symbol
    signal_events = []
    
    # Get current time and date
    current_time = event.datetime.time()
    current_date = event.datetime.date()
    
    # Reset daily tracking if new day
    if current_trading_date[symbol] != current_date:
        current_trading_date[symbol] = current_date
        orders_placed_today[symbol] = False
    
    # Get positions and orders
    open_positions = modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(symbol)
    pending_orders = modules.PORTFOLIO.get_number_of_strategy_pending_orders_by_symbol(symbol)
    
    # Close positions and cancel orders at close time
    if current_time >= time(close_hour, close_minute):
        if open_positions['TOTAL'] > 0:
            logger.info(f"{event.datetime} - Closing all positions for {symbol}")
            modules.EXECUTION_ENGINE.close_all_strategy_positions()
        
        if pending_orders['TOTAL'] > 0:
            logger.info(f"{event.datetime} - Cancelling all pending orders for {symbol}")
            modules.EXECUTION_ENGINE.cancel_all_strategy_pending_orders()
        
        return
    
    # Place orders at order placement time
    if (current_time >= time(order_placement_hour, order_placement_minute) and 
        not orders_placed_today[symbol] and 
        pending_orders['TOTAL'] == 0 and
        event.timeframe == signal_timeframe):
        
        # Get bars for Bollinger Bands calculation
        bars_needed = bb_period + 10
        indicator_bars = modules.DATA_PROVIDER.get_latest_bars(symbol, signal_timeframe, bars_needed)
        
        if indicator_bars is None or indicator_bars.height < bars_needed:
            return
        
        # Calculate Bollinger Bands
        close = indicator_bars.select('close').to_numpy().flatten()
        upper, middle, lower = BollingerBands.compute(close, bb_period, bb_std_dev)
        
        current_upper = upper[-1]
        current_lower = lower[-1]
        
        if np.isnan(current_upper) or np.isnan(current_lower):
            return
        
        upper_breakout = Decimal(str(current_upper))
        lower_breakout = Decimal(str(current_lower))
        
        # Time for signal generation
        if modules.TRADING_CONTEXT == "BACKTEST":
            time_generated = event.datetime + signal_timeframe.to_timedelta()
        else:
            time_generated = datetime.now()
        
        # Place BUY STOP order
        signal_events.append(SignalEvent(
            symbol=symbol,
            time_generated=time_generated,
            strategy_id=strategy_id,
            signal_type="BUY",
            order_type="STOP",
            order_price=upper_breakout,
            sl=Decimal(str(0.0)),
            tp=Decimal(str(0.0)),
        ))
        
        # Place SELL STOP order
        signal_events.append(SignalEvent(
            symbol=symbol,
            time_generated=time_generated,
            strategy_id=strategy_id,
            signal_type="SELL",
            order_type="STOP",
            order_price=lower_breakout,
            sl=Decimal(str(0.0)),
            tp=Decimal(str(0.0)),
        ))
        
        orders_placed_today[symbol] = True
    
    return signal_events


# Configure Strategy
strategy.configure_predefined_sizing_engine(MinSizingConfig())
strategy.configure_predefined_risk_engine(PassthroughRiskConfig())

# Backtest Configuration
from_date = datetime(year=2020, month=1, day=1)
to_date = datetime(year=2023, month=12, day=1)
# csv_dir = '/Users/marticastany/Desktop/long_data' # Change it with your own path to the CSV data
csv_dir = None # If you don't have CSV data, you can set this to None

# Launch Backtest
backtest = strategy.backtest(
    strategy_id=strategy_id,
    initial_capital=starting_capital,
    symbols_to_trade=symbols_to_trade,
    csv_dir=csv_dir,
    backtest_name=strategy_id,
    start_date=from_date,
    end_date=to_date,
    export_backtest_parquet=False,
    account_currency='USD'
)

print("Backtest finished")
backtest.plot()

# Example: Running live with MT5
# mt5_config = Mt5PlatformConfig(
#     path="C:\\Program Files\\MetaTrader 5\\terminal64.exe",
#     login=1234,
#     password="1234",
#     server="Demo",
#     timeout=60000,
#     portable=False
# )
# strategy.run_live(
#     mt5_configuration=mt5_config,
#     strategy_id=strategy_id,
#     initial_capital=100000,
#     symbols_to_trade=symbols_to_trade,
#     heartbeat=0.1
# )

