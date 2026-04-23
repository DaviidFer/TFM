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
from pyeventbt.indicators import SMA

from datetime import datetime
from decimal import Decimal
import logging

logger = logging.getLogger("pyeventbt")

# Strategy Configuration
strategy_id = "123456"
strategy = Strategy(logging_level=logging.INFO)

# Timeframes
signal_timeframe = StrategyTimeframes.ONE_DAY

strategy_timeframes = [signal_timeframe]

# Trading Configuration
symbols_to_trade = ['EURUSD']
starting_capital = 100000

# Strategy Parameters
fast_ma_period = 10
slow_ma_period = 30

@strategy.custom_signal_engine(strategy_id=strategy_id, strategy_timeframes=strategy_timeframes)
def ma_crossover_strategy(event: BarEvent, modules: Modules):
    """
    Moving Average Dominance Strategy:
    - Stay long while fast MA is above slow MA
    - Stay short while fast MA is below slow MA
    - Flat (or hold current) when both averages equal
    - Always maintain at most one open position
    """
    
    if event.timeframe != signal_timeframe:
        return
    
    symbol = event.symbol

    signal_events = []
    
    # Get bars for MA calculation
    bars_needed = slow_ma_period + 10
    bars = modules.DATA_PROVIDER.get_latest_bars(symbol, signal_timeframe, bars_needed)
    
    if bars is None or bars.height < bars_needed:
        return
    
    # Calculate moving averages
    close_prices = bars.select('close').to_numpy().flatten()
    fast_ma_values = SMA.compute(close_prices, fast_ma_period)
    slow_ma_values = SMA.compute(close_prices, slow_ma_period)
    
    current_fast_ma = fast_ma_values[-1]
    current_slow_ma = slow_ma_values[-1]
    
    # Determine desired position state
    if current_fast_ma > current_slow_ma:
        desired_position = "LONG"
    elif current_fast_ma < current_slow_ma:
        desired_position = "SHORT"
    else:
        return

    # Check current positions (at current bar time - no lookahead)
    open_positions = modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(symbol)
    
    signal_type = ""
    
    
    # Signal generation
    if open_positions['LONG'] == 0 and desired_position == "LONG":
        if open_positions['SHORT'] > 0:
            modules.EXECUTION_ENGINE.close_strategy_short_positions_by_symbol(symbol)
        signal_type = "BUY"

    if open_positions['SHORT'] == 0 and desired_position == "SHORT":
        if open_positions['LONG'] > 0:
            modules.EXECUTION_ENGINE.close_strategy_long_positions_by_symbol(symbol)
        signal_type = "SELL"
    
    if signal_type == "":
        return
    
    # Time for signal generation (for NEXT bar)
    if modules.TRADING_CONTEXT == "BACKTEST":
        time_generated = event.datetime + signal_timeframe.to_timedelta()
    else:
        time_generated = datetime.now()

    last_tick = modules.DATA_PROVIDER.get_latest_tick(symbol)
    
    # Generate signals based on desired position
    signal_events.append(SignalEvent(
        symbol=symbol,
        time_generated=time_generated,
        strategy_id=strategy_id,
        signal_type=signal_type,
        order_type="MARKET",
        order_price=last_tick['ask'] if signal_type == "BUY" else last_tick['bid'],
        sl=Decimal(str(0.0)),
        tp=Decimal(str(0.0)),
    ))
    
    return signal_events


# Configure Strategy
strategy.configure_predefined_sizing_engine(MinSizingConfig())
strategy.configure_predefined_risk_engine(PassthroughRiskConfig())

# Backtest Configuration
from_date = datetime(year=2020, month=1, day=1)
to_date = datetime(year=2023, month=12, day=1)
# csv_dir = '/Users/marticastany/Desktop/long_data' # Change it with your own path to the CSV data
csv_dir = None # If you don't have CSV data, you can set this to None

# Launch Backtest
backtest = strategy.backtest(
    strategy_id=strategy_id,
    initial_capital=starting_capital,
    symbols_to_trade=symbols_to_trade,
    csv_dir=csv_dir,
    backtest_name=strategy_id,
    start_date=from_date,
    end_date=to_date,
    export_backtest_pickle=False,
    account_currency='USD'
)

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
