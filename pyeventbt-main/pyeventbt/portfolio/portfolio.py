"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.trading_context.trading_context import TypeContext
from .core.interfaces.portfolio_interface import IPortfolio
from .core.entities.open_position import OpenPosition
from .core.entities.pending_order import PendingOrder
from pyeventbt.execution_engine.core.interfaces.execution_engine_interface import IExecutionEngine
from pyeventbt.events.events import BarEvent
from decimal import Decimal
from datetime import datetime
import pandas as pd
import polars as pl
import os, json
import logging

logger = logging.getLogger("pyeventbt")


class Portfolio(IPortfolio):
    
    def __init__(self, initial_balance: Decimal, execution_engine: IExecutionEngine, trading_context: TypeContext = TypeContext.BACKTEST, base_timeframe: str = '1min'):
        
        self.trading_context = trading_context
        self.EXECUTION = execution_engine
        self._initial_balance: Decimal = initial_balance
        self._balance: Decimal = initial_balance
        self._equity: Decimal = initial_balance
        self._realised_pnl: Decimal = Decimal('0.0')
        self._unrealised_pnl: Decimal = Decimal('0.0')
        self._strategy_positions: tuple[OpenPosition] = ()
        self._strategy_pending_orders: tuple[PendingOrder] = ()   
        self._base_timeframe = base_timeframe

        # Backtest historical data variables
        self._first_flag = True
        self._first_seen_historical_symbol = ""
        self.historical_balance: dict[datetime, Decimal] = {}
        self.historical_equity: dict[datetime, Decimal] = {}

    def _update_portfolio(self, bar_event: BarEvent) -> None:
        """
        Updates the portfolio current state with the last received data. It also checks for pending order fills and SL/TP hits.
        It will be executed at each new bar, so in other methods we don't need to re-update the portfolio to return the last values.
        """
        # Updates fills and info from symbol of the bar event.
        self.EXECUTION._update_values_and_check_executions_and_fills(bar_event)
        
        # Returns ALL positions from all symbols. This will be repetitive but needed in case any position is closed 
        # so we don't have to eliminate items from any list. This implies monitoring positions from ALL THE ACCOUNT, so at a global portfolio level.
        self._strategy_positions = self.EXECUTION._get_strategy_positions()
        self._strategy_pending_orders = self.EXECUTION._get_strategy_pending_orders()
        
        # Updates account values
        self._balance = self.EXECUTION._get_account_balance()
        self._equity = self.EXECUTION._get_account_equity()
        self._realised_pnl = self._balance - Decimal(self._initial_balance)
        self._unrealised_pnl = self._equity - self._balance

        # Updates historical values (if we are in a backtest)
        if self._first_flag:
            self._first_seen_historical_symbol = bar_event.symbol
            self._first_flag = False
        
        if self.trading_context == TypeContext.BACKTEST:
            if bar_event.timeframe == self._base_timeframe:
                if bar_event.symbol == self._first_seen_historical_symbol:  # Only store data for first symbol as others will be redundant
                    self.historical_balance[bar_event.datetime] = self._balance
                    self.historical_equity[bar_event.datetime] = self._equity

    def _update_portfolio_end_of_backtest(self) -> None:
        """
        Updates the portfolio at the end of the backtest to store the last values of the account.
        """
        self._balance = self.EXECUTION._get_account_balance()
        self._equity = self.EXECUTION._get_account_equity()
        self._realised_pnl = self._balance - Decimal(self._initial_balance)
        self._unrealised_pnl = self._equity - self._balance

        if self._realised_pnl >= 0:
            color_code = "\x1b[92;20m"
        else:
            color_code = "\x1b[91;20m"

        logger.warning(f"{color_code}Realised PnL: {self._realised_pnl:.2f}, balance: {self._balance:.2f}, initial balance: {self._initial_balance:.2f}")

        # Access last values fo the dict and update them with closing balance and equity
        if(len(self.historical_balance) != 0):
            self.historical_balance[list(self.historical_balance.keys())[-1]] = self._balance
        if(len(self.historical_equity) != 0):
            self.historical_equity[list(self.historical_equity.keys())[-1]] = self._balance  # At end of backtest, equity equals the balance after closing everything
    
    def get_account_balance(self) -> Decimal:
        """
        Returns the current balance of the account.
        """
        return self._balance
    
    def get_account_equity(self) -> Decimal:
        """
        Returns the current equity of the account.
        """
        return self._equity
    
    def get_account_unrealised_pnl(self) -> Decimal:
        """
        Returns the current unrealised profit and loss of the account.
        """
        return self._unrealised_pnl
    
    def get_account_realised_pnl(self) -> Decimal:
        """
        Returns the realised profit and loss of the account.
        """
        return self._realised_pnl
    
    def get_positions(self, symbol: str = '', ticket: int = None) -> tuple[OpenPosition]:
        """
        Returns a list of OpenPosition objects
        """
        if ticket is None:
            positions = self._strategy_positions
        else:
            positions = tuple(pos for pos in self._strategy_positions if pos.ticket == ticket)  #faster
            #positions = tuple(filter(lambda pos: pos.ticket == ticket, self._strategy_positions))  # Tuple of only one OrderSendResult object
        
        if symbol != '':
            # Filter by symbol. Positions is already filled depending on ticket
            positions = tuple(pos for pos in positions if pos.symbol == symbol)  #faster
            #positions = tuple(filter(lambda pos: pos.symbol == symbol, self._strategy_positions))

        return positions
    
    def get_pending_orders(self, symbol: str = '', ticket: int = None) -> tuple[PendingOrder]:
        """
        Returns a list of PendingOrder objects
        """
        if ticket is None:
            orders = self._strategy_pending_orders
        else:
            orders = tuple(order for order in self._strategy_pending_orders if order.ticket == ticket)

        if symbol != '':
            orders = tuple(order for order in orders if order.symbol == symbol)

        return orders
    
    def get_number_of_strategy_open_positions_by_symbol(self, symbol: str) -> dict[str, int]:
        """
        Returns a dict of format {"LONG": int, "SHORT": int, "TOTAL": int} with the number of open positions for a given symbol.
        """
        longs = 0
        shorts = 0
        for pos in self.get_positions(symbol=symbol):   # Already filtered by magic (only strategy positions)
            if pos.type == "BUY":
                longs += 1
            elif pos.type == "SELL":
                shorts += 1
        
        return {"LONG": longs, "SHORT": shorts, "TOTAL": longs + shorts}
    
    def get_number_of_strategy_pending_orders_by_symbol(self, symbol: str) -> dict[str, int]:
        """
        Returns a dict of format {"BUY_LIMIT": int, "SELL_LIMIT": int, "BUY_STOP": int, "SELL_STOP": int} with the number of pending orders for a given symbol.
        """
        buy_limits = 0
        sell_limits = 0
        buy_stops = 0
        sell_stops = 0

        # 2 3 4 5 are the types of pending orders
        for order in self.get_pending_orders(symbol=symbol):
            if order.type == "BUY_LIMIT":
                buy_limits += 1
            elif order.type == "SELL_LIMIT":
                sell_limits += 1
            elif order.type == "BUY_STOP":
                buy_stops += 1
            elif order.type == "SELL_STOP":
                sell_stops += 1

        return {"BUY_LIMIT": buy_limits, "SELL_LIMIT": sell_limits, "BUY_STOP": buy_stops, "SELL_STOP": sell_stops, "TOTAL": buy_limits + sell_limits + buy_stops + sell_stops}

    def _export_historical_pnl_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with the historical balance and equity of the account.
        """
        df = pd.DataFrame({'BALANCE': self.historical_balance, 'EQUITY': self.historical_equity})
        df.index = pd.to_datetime(df.index)
        df.index.name = 'DATETIME'
        return df
    
    def _export_historical_pnl_to_parquet(self, file_path: str) -> None:
        """
        Exports the historical pnl to a parquet file.
        Handles potential errors during data collection and file writing.
        Rounds decimal values to 2 places and stores them as Float64.
        
        Args:
            file_path: Path where the parquet file should be saved.
        """
        try:
            # Validate that we have data to export
            if not self.historical_balance or not self.historical_equity:
                logger.warning("No historical PnL data found to export to parquet file")
                return

            # Extract and convert data, rounding to 2 decimal places
            datetimes = [dt for dt in self.historical_balance.keys()]
            balance_values = [float(v.quantize(Decimal('0.01'))) for v in self.historical_balance.values()]
            equity_values = [float(v.quantize(Decimal('0.01'))) for v in self.historical_equity.values()]

            # Create DataFrame with proper types for each column
            df = pl.DataFrame(
                {
                    'DATETIME': datetimes,
                    'BALANCE': balance_values,
                    'EQUITY': equity_values
                },
                schema={
                    'DATETIME': pl.Datetime,
                    'BALANCE': pl.Float64,  # Store as double precision float
                    'EQUITY': pl.Float64    # Store as double precision float
                }
            )

            # Ensure the directory for the file path exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            df.write_parquet(
                file=file_path,
                compression='zstd',
                compression_level=10
            )
            logger.debug(f"Successfully exported PnL data to {file_path}")

        except pl.exceptions.PolarsError as e:
            logger.error(f"Error creating Polars DataFrame for PnL data: {str(e)}")
            raise
        except OSError as e:
            logger.error(f"Error writing PnL parquet file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PnL export: {str(e)}")
            raise
    
    def _export_historical_pnl_json(self) -> str:
        """Convert to efficient integers for storage and exact reconstruction.
        - Timestamps as seconds since epoch (sufficient for financial reporting)
        - Decimal values scaled to integers with 4 decimal precision for USD

        TO BEAR IN MIND THAT THE DATETIMES ARE IN THE BROKER TIMEZONE (AMERICA/NY + 7 HOURS)
        """
        scale_factor = 10**4  # 4 decimal places (sufficient for financial values)
        
        serialized = {
            "scale_factor": scale_factor,
            "balance": {dt.strftime("%Y-%m-%dT%H:%M:%S"): int(value * scale_factor) for dt, value in self.historical_balance.items()},
            #int(dt.timestamp()): int(value * scale_factor)
            "equity": {dt.strftime("%Y-%m-%dT%H:%M:%S"): int(value * scale_factor) for dt, value in self.historical_equity.items()}
        }
        try:
            # Convert to JSON string
            serialized = json.dumps(serialized)  #indent=4 (removed as it will be mostly machine read)
        except TypeError as e:
            # Handle the case where the data is not serializable
            logger.error(f"Serialization error: {e}")
            return {}
        
        return serialized

    
    def _export_csv_historical_pnl(self, file_path: str) -> None:
        """
        Exports the historical pnl to a csv file
        
        Args:
            file_path: Path where the CSV file should be saved.
        """
        # Ensure the directory for the file path exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Export to CSV
        df = self._export_historical_pnl_dataframe()
        df.to_csv(file_path)