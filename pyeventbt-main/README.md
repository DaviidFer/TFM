# pyeventbt

[![PyPI version](https://badge.fury.io/py/pyeventbt.svg)](https://badge.fury.io/py/pyeventbt)
[![Python versions](https://img.shields.io/pypi/pyversions/pyeventbt.svg)](https://pypi.org/project/pyeventbt/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PyEventBT** is an institutional-grade event-driven backtesting and live trading framework built with Python for the MetaTrader 5 platform.

It provides a complete mock of the MT5 API for an easy transition between backtesting and live trading, allowing traders to easily develop multi-rule, multi-timeframe and multi-instrument strategies.

Whether you're building simple moving average crossovers or complex multi-rule and multi-timeframe strategies, PyEventBT provides the tools you need to develop, test, and deploy with confidence.

Its modular architecture allows you to design your own signal sources, position sizing logic and risk management overlay as independent and interchangeable blocks.

## Installation

```bash
pip install pyeventbt
```

## Quick Start

This is a summarized guide. For the full documentation, please visit <a href="https://pyeventbt.com?utm_source=github&utm_medium=readme" target="_blank">pyeventbt.com</a>.

### 1. Define your Strategy

Create a strategy by decorating a function with `@strategy.custom_signal_engine`.

```python
from pyeventbt import (
    Strategy,
    BarEvent,
    SignalEvent,
    Modules,
    StrategyTimeframes,
    PassthroughRiskConfig,
    MinSizingConfig,
)
from pyeventbt.indicators import SMA
from datetime import datetime

# Create strategy
strategy = Strategy()
strategy_id = "1234" # Must be a string of numbers, equivalent to MT5 Magic Number

@strategy.custom_signal_engine(
    strategy_id=strategy_id,
    strategy_timeframes=[StrategyTimeframes.ONE_HOUR]
)
def my_strategy(event: BarEvent, modules: Modules):
    """Simple moving average crossover strategy"""
    bars = modules.DATA_PROVIDER.get_latest_bars(event.symbol, StrategyTimeframes.ONE_HOUR, 50)
    if bars is None or bars.height < 50:
        return []
    
    close = bars.select('close').to_numpy().flatten()
    fast_ma, slow_ma = SMA.compute(close, 10)[-1], SMA.compute(close, 30)[-1]
    
    open_pos = modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol(event.symbol)
    signal_type = ""
    
    if fast_ma > slow_ma and open_pos['LONG'] == 0:
        if open_pos['SHORT'] > 0:
            modules.EXECUTION_ENGINE.close_strategy_short_positions_by_symbol(event.symbol)
        signal_type = "BUY"
    elif fast_ma < slow_ma and open_pos['SHORT'] == 0:
        if open_pos['LONG'] > 0:
            modules.EXECUTION_ENGINE.close_strategy_long_positions_by_symbol(event.symbol)
        signal_type = "SELL"
    
    if not signal_type:
        return []
    
    tick = modules.DATA_PROVIDER.get_latest_tick(event.symbol)
    return [SignalEvent(
        symbol=event.symbol,
        time_generated=event.datetime,
        strategy_id=strategy_id,
        signal_type=signal_type,
        order_type="MARKET",
        order_price=tick['ask'] if signal_type == "BUY" else tick['bid'],
        sl=0.0,
        tp=0.0,
    )]
```

### 2. Configure and Run Backtest

```python
# Configure risk and sizing
strategy.configure_predefined_sizing_engine(MinSizingConfig())
strategy.configure_predefined_risk_engine(PassthroughRiskConfig())

# Run backtest
backtest = strategy.backtest(
    strategy_id=strategy_id,
    initial_capital=100000,
    symbols_to_trade=['EURUSD'],
    csv_dir=None,
    backtest_name="example",
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    account_currency='USD',
    export_backtest_csv=False,
    export_backtest_parquet=True,
    backtest_results_dir=None,  # Defaults to Desktop/PyEventBT/backtest_results
)

backtest.plot()
```

## Example Strategy

Here is a complete example of a Bollinger Bands breakout strategy:

```python
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
csv_dir = None # '/path/to/your/data' or None for default dataset

# Launch Backtest
backtest = strategy.backtest(
    strategy_id=strategy_id,
    initial_capital=starting_capital,
    symbols_to_trade=symbols_to_trade,
    csv_dir=csv_dir,
    backtest_name=strategy_id,
    start_date=from_date,
    end_date=to_date,
    export_backtest_csv=True
    export_backtest_parquet=False,
    account_currency='USD'
)

print("Backtest finished")
backtest.plot()
```

## Documentation

📚 **Full documentation available at <a href="https://pyeventbt.com?utm_source=github&utm_medium=readme" target="_blank">pyeventbt.com</a>.**

The documentation includes:
- Complete API reference
- Detailed examples and tutorials
- Import patterns and best practices
- Advanced configuration options
- Live trading setup guides

## Features

- 🎯 Event-driven architecture for realistic backtesting
- 📊 Built-in technical indicators (ATR, SMA, RSI, and more)
- 🔄 Seamless transition from backtest to live trading
- 📈 Comprehensive performance metrics and visualization
- ⚙️ Flexible risk and position sizing engines
- 🔌 MetaTrader 5 integration for live trading

## License

Apache 2.0

## Author

Made with ❤️ for the Community by [Martí Castany](https://www.linkedin.com/in/marti-castany/)
