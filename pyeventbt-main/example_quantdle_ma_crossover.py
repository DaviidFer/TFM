"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

"""
Example: Using Quantdle Data Updater with MA Crossover Strategy

This example demonstrates how to use the QuantdleDataUpdater to automatically
download and cache market data from Quantdle, and then run a Moving Average
Crossover strategy on that data.
"""

from pyeventbt import (
    Strategy,
    BarEvent,
    SignalEvent,
    Modules,
    StrategyTimeframes,
    PassthroughRiskConfig,
    MinSizingConfig,
    QuantdleDataUpdater,
)
from pyeventbt.indicators import SMA

from datetime import datetime
from decimal import Decimal
import logging
import numpy as np

logger = logging.getLogger("pyeventbt")

# =============================================================================
# STEP 1: CONFIGURE QUANTDLE DATA UPDATER
# =============================================================================

# Your Quantdle API credentials
QUANTDLE_API_KEY = "your_api_key_here"
QUANTDLE_API_KEY_ID = "your_api_key_id_here"

# Where to store/cache CSV files
csv_dir = '/Users/marticastany/Desktop/quantdle_cache'

# Backtest parameters
symbols_to_trade = ['EURUSD']
from_date = datetime(year=2020, month=1, day=1)
to_date = datetime(year=2023, month=12, day=1)

# =============================================================================
# STEP 2: UPDATE LOCAL CSV CACHE WITH QUANTDLE DATA
# =============================================================================

# Initialize the updater with your credentials
updater = QuantdleDataUpdater(
    api_key=QUANTDLE_API_KEY,
    api_key_id=QUANTDLE_API_KEY_ID
)

# Update the CSV cache
updater.update_data(
    csv_dir=csv_dir,
    symbols=symbols_to_trade,
    start_date=from_date,
    end_date=to_date,
    timeframe="1min"  # Standard timeframe
)

print("\n" + "="*80)
print("CSV cache updated! Now running MA Crossover backtest with cached data...")
print("="*80 + "\n")

# =============================================================================
# STEP 3: DEFINE YOUR STRATEGY (MA Crossover)
# =============================================================================

strategy_id = "quantdle_ma_crossover"
strategy = Strategy(logging_level=logging.INFO)

# Timeframes
signal_timeframe = StrategyTimeframes.ONE_DAY
strategy_timeframes = [signal_timeframe]

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


# =============================================================================
# STEP 4: RUN BACKTEST WITH CACHED DATA
# =============================================================================

strategy.configure_predefined_sizing_engine(MinSizingConfig())
strategy.configure_predefined_risk_engine(PassthroughRiskConfig())

# Launch backtest using the CSV cache we just updated
backtest = strategy.backtest(
    strategy_id=strategy_id,
    initial_capital=100000,
    symbols_to_trade=symbols_to_trade,
    csv_dir=csv_dir,  # <-- Use the same directory we updated with Quantdle
    backtest_name=strategy_id,
    start_date=from_date,
    end_date=to_date,
    export_backtest_pickle=False,
    account_currency='USD'
)

print("\nBacktest finished!")
backtest.plot()
