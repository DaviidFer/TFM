"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from queue import Queue
from pyeventbt.backtest.core.backtest_results import BacktestResults
from pyeventbt.sizing_engine.core.interfaces.sizing_engine_interface import ISizingEngine
from pyeventbt.risk_engine.core.interfaces.risk_engine_interface import IRiskEngine
from pyeventbt.trade_archiver.trade_archiver import TradeArchiver
from pyeventbt.portfolio.core.interfaces.portfolio_interface import IPortfolio
from pyeventbt.events.events import BarEvent, SignalEvent, FillEvent
from datetime import datetime
import pandas as pd
import pickle, os, zipfile, io, logging

logger = logging.getLogger("pyeventbt")

class PortfolioHandler():
    """
    The PortfolioHandler is designed to interact with the 
    backtesting or live trading overall event-driven
    architecture. It exposes two methods, on_signal and
    on_fill, which handle how SignalEvent and FillEvent
    objects are dealt with.

    Each PortfolioHandler contains a Portfolio object,
    which stores the actual Position objects. 

    The PortfolioHandler takes a handle to a PositionSizer
    object which determines a mechanism, based on the current
    Portfolio, as to how to size a new Order.

    The PortfolioHandler also takes a handle to the 
    RiskManager, which is used to modify any generated 
    Orders to remain in line with risk parameters.
    """

    def __init__(self, events_queue: Queue, sizing_engine: ISizingEngine, risk_engine: IRiskEngine, portfolio: IPortfolio, base_timeframe: str = '1min', backtest_results_dir: str = None):
        self.event_queue = events_queue
        self.POSITION_SIZER = sizing_engine
        self.RISK_ENGINE = risk_engine
        self.PORTFOLIO = portfolio
        self.base_timeframe = base_timeframe
        self.backtest_results_dir = backtest_results_dir

        self.TRADE_ARCHIVER = TradeArchiver()


    def process_bar_event(self, bar_event: BarEvent):
        """
        This is called by the backtester or live trading architecture
        to update the PORTFOLIO values (equity, balance, positions, etc).
        """
        # We update the portfolio every minute (or base timeframe). No need to process events of other timeframes as it's redundant
        if bar_event.timeframe != self.base_timeframe:
            return
        
        self.PORTFOLIO._update_portfolio(bar_event)

    def process_signal_event(self, signal_event: SignalEvent):
        """
        This is called by the backtester or live trading architecture
        to form the initial orders from the SignalEvent. 

        These orders are sized by the PositionSizer object and then
        sent to the RiskManager to verify, modify or eliminate.

        Once received from the RiskManager they are converted into
        full OrderEvent objects and sent back to the events queue.
        """
        # Size the signal event and convert it into a Suggested order by the position sizer object   
        suggested_order = self.POSITION_SIZER.get_suggested_order(signal_event)

        # Verify the suggested order by the risk manager object. If all OK, the Risk Engine will create and put an order event to the events queue
        self.RISK_ENGINE.assess_order(suggested_order)
    
    def process_fill_event(self, fill_event: FillEvent):
        """
        This is called by the backtester or live trading architecture
        to update the portfolio with new fill events.
        """
        self.TRADE_ARCHIVER.archive_trade(fill_event)


        # Maybe then NOTIFICATE the user via Telegram of the FILL of a trade?
    
    def _get_default_desktop_path(self) -> str:
        """Get OS-specific Desktop path."""
        import platform
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            return os.path.expanduser('~/Desktop')
        elif system == 'Windows':
            return os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop')
        else:  # Linux and others
            return os.path.expanduser('~/Desktop')
    
    def save_compressed_pickle(self, backtest:tuple[str, str], path:str, password:str = None):
        # Create a bytes buffer to hold the pickle data
        buffer = io.BytesIO()
        
        # Serialize the backtest object to the buffer
        logger.info(f"\x1b[95;20mSerializing backtest data to pickle...")
        pickle.dump(backtest, buffer)

        # Move the buffer's position to the beginning
        buffer.seek(0)

        # Create a zip file and write the buffer content to it
        logger.info(f"\x1b[95;20mSaving serialized data to zip...")
        
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add the pickle data to the zip file
            zip_info = zipfile.ZipInfo('backtest.pkl')
            zip_file.writestr(zip_info, buffer.read())
            
            # Set the password for the zip file
            zip_file.setpassword(password.encode())

    
    def process_backtest_end(self, backtest_name:str, export_backtest_to_csv: bool = False, export_backtest_to_parquet: bool = False) -> BacktestResults:
        
        self.PORTFOLIO._update_portfolio_end_of_backtest()
        
        if export_backtest_to_csv:
            logger.info(f"\x1b[95;20mExporting backtest data to CSV...")
            
            # Determine the base export directory (same logic as parquet)
            if self.backtest_results_dir:
                base_export_dir = self.backtest_results_dir
            else:
                base_export_dir = self._get_default_desktop_path()
            
            # Create backtest directory name with timestamp
            datestring = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            bt_name = backtest_name + "_" + datestring
            
            # Create the full export path: base_dir/PyEventBT/backtest_results_csv/bt_name/
            export_dir = os.path.join(base_export_dir, 'PyEventBT', 'backtest_results_csv', bt_name)
            
            # Define CSV file paths in the export directory
            trades_csv = os.path.join(export_dir, 'historical_trades.csv')
            pnl_csv = os.path.join(export_dir, 'historical_pnl.csv')
            
            # Ensure the export directory exists
            os.makedirs(export_dir, exist_ok=True)
            
            # Export CSV files
            self.TRADE_ARCHIVER.export_csv_trade_archive(trades_csv)
            self.PORTFOLIO._export_csv_historical_pnl(pnl_csv)
        
        # Create a BacktestResults object to return to the backtest architecture
        backtest = BacktestResults(backtest_pnl=self.PORTFOLIO._export_historical_pnl_dataframe(), trades=self.TRADE_ARCHIVER.export_historical_trades_dataframe())

        if export_backtest_to_parquet:
            logger.info(f"\x1b[95;20mExporting backtest data to Parquet...")
            
            # Determine the base export directory
            if self.backtest_results_dir:
                base_export_dir = self.backtest_results_dir
            else:
                base_export_dir = self._get_default_desktop_path()
            
            # Create backtest directory name with timestamp
            datestring = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            bt_name = backtest_name + "_" + datestring
            
            # Create the full export path: base_dir/PyEventBT/backtest_results/bt_name/
            export_dir = os.path.join(base_export_dir, 'PyEventBT', 'backtest_results_parquet', bt_name)
            
            # Define parquet file paths in the export directory
            trades_parquet = os.path.join(export_dir, 'historical_trades.parquet')
            pnl_parquet = os.path.join(export_dir, 'historical_pnl.parquet')

            try:
                # Ensure the export directory exists
                os.makedirs(export_dir, exist_ok=True)
                logger.debug(f"\x1b[95;20mExporting backtest results to: {export_dir}")

                # Export the parquet files directly to the export directory
                self.TRADE_ARCHIVER.export_historical_trades_parquet(trades_parquet)
                self.PORTFOLIO._export_historical_pnl_to_parquet(pnl_parquet)
                
                logger.info(f"\x1b[92;20mBacktest results exported successfully to: {export_dir}")

            except OSError as e:
                logger.error(f"OS error during export process: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during export process: {str(e)}")
                raise
        
        return backtest





