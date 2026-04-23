"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from .core.interfaces.trade_archiver_interface import ITradeArchiver
from pyeventbt.events.events import FillEvent
#from zoneinfo import ZoneInfo
import pandas as pd
import polars as pl
from decimal import Decimal
import os, json
from typing import Type
import logging

logger = logging.getLogger("pyeventbt")

class TradeArchiver(ITradeArchiver):

    def __init__(self):
        self.trade_archive: dict[int, FillEvent] = {}
        self.trade_archive_id = 0
    
    def archive_trade(self, fill_event: FillEvent) -> None:
        # Fill events will be in order of execution, so we can use the id as a key to store them.        
        self.trade_archive_id += 1
        self.trade_archive[self.trade_archive_id] = fill_event

    def get_trade_archive(self) -> dict[int, FillEvent]:
        return self.trade_archive

    def export_historical_trades_dataframe(self, ) -> pd.DataFrame:
        #return pd.DataFrame.from_records([event.model_dump() for _,event in self.get_trade_archive().items()])  # columns in minus, and data like EventTpe.FILL

        # Create the dataframe
        columns_fill_event = ['TYPE', 'DEAL', 'SYMBOL', 'TIME_GENERATED', 'POSITION_ID', 'STRATEGY_ID', 'EXCHANGE', 'VOLUME', 'PRICE', 'SIGNAL_TYPE', 'COMMISSION', 'SWAP', 'FEE', 'GROSS_PROFIT', 'CCY']

        # Accumulate each row's data in this list
        rows_data = []

        # Iterate over the trade archive and append each event as a new row in the dataframe
        for _, fill_event in self.trade_archive.items():
            row_data = {
                'TYPE': fill_event.type.value,  # Assuming type is an enum and you want its value
                'DEAL': fill_event.deal.value,  # Assuming deal is an enum and you want its value
                'SYMBOL': fill_event.symbol,
                'TIME_GENERATED': fill_event.time_generated,
                'POSITION_ID': fill_event.position_id,
                'STRATEGY_ID': fill_event.strategy_id,
                'EXCHANGE': fill_event.exchange,
                'VOLUME': fill_event.volume,
                'PRICE': fill_event.price,
                'SIGNAL_TYPE': fill_event.signal_type.value,  # Assuming signal_type is an enum and you want its value
                'COMMISSION': fill_event.commission,
                'SWAP': fill_event.swap,
                'FEE': fill_event.fee,
                'GROSS_PROFIT': fill_event.gross_profit,
                'CCY': fill_event.ccy
            }
            rows_data.append(row_data) # FIXME This is deprecated in new versions of pandas
        
        # Create the DataFrame once after collecting all rows
        df = pd.DataFrame(rows_data, columns=columns_fill_event)

        return df

    def export_historical_trades_json(self) -> str:
        """
        Export trade history as a native Python dictionary with serializable types.
        Returns a dictionary structure that can be easily pickled or JSON serialized.
        """
    
        # Create the main container dictionary
        trades_data = {}
        
        # Iterate over the trade archive and convert each event to native Python types
        for id, fill_event in self.trade_archive.items():
            
            # Create trade record with all native Python types
            trade_record = {
                "type": fill_event.type.value,
                "deal": fill_event.deal.value,
                "symbol": fill_event.symbol,
                "time_generated": fill_event.time_generated.strftime("%Y-%m-%dT%H:%M:%S"),
                "position_id": str(fill_event.position_id),
                "strategy_id": str(fill_event.strategy_id),
                "exchange": fill_event.exchange,
                "volume": str(fill_event.volume),
                # Scale financial values to integers
                "price": str(fill_event.price.quantize(Decimal('0.00001'))),
                "signal_type": fill_event.signal_type.value,
                "commission": str(fill_event.commission.quantize(Decimal('0.00001'))),
                "swap": str(fill_event.swap),
                "fee": str(fill_event.fee),
                "gross_profit": str(fill_event.gross_profit.quantize(Decimal('0.00001'))),
                "currency": fill_event.ccy
            }
            trades_data[str(id)] = trade_record
        
        try:
            # Convert to JSON string
            serialized = json.dumps(trades_data)
        except TypeError as e:
            # Handle the case where the data is not serializable
            logger.error(f"Serialization error: {e}")
            return {}
        
        return serialized
    
    def export_historical_trades_parquet(self, file_path: str) -> None:
        """
        Export trade history as a parquet file using a list of dictionaries approach.
        Handles potential errors during data collection and file writing.
        Converts Decimal values to strings to avoid Decimal128 overflow issues.
        
        Args:
            file_path: Path where the parquet file should be saved.
        """
        try:
            # Collect all trades in a list of dictionaries
            trades_data = []
            for _, event in self.trade_archive.items():
                trade = {
                    'TYPE': event.type.value,
                    'DEAL': event.deal.value,
                    'SYMBOL': event.symbol,
                    'TIME_GENERATED': event.time_generated,
                    'POSITION_ID': event.position_id,
                    'STRATEGY_ID': event.strategy_id,
                    'EXCHANGE': event.exchange,
                    'VOLUME': float(event.volume.quantize(Decimal('0.01'))),
                    'PRICE': float(event.price.quantize(Decimal('0.000001'))),
                    'SIGNAL_TYPE': event.signal_type.value,
                    'COMMISSION': float(event.commission.quantize(Decimal('0.000001'))),
                    'SWAP': float(event.swap.quantize(Decimal('0.000001'))),    
                    'FEE': float(event.fee.quantize(Decimal('0.000001'))),
                    'GROSS_PROFIT': float(event.gross_profit.quantize(Decimal('0.000001'))),
                    'CCY': event.ccy
                }
                trades_data.append(trade)
            
            if not trades_data:
                logger.warning("No trades found to export to parquet file")
                return

            # Create DataFrame with string type for decimal columns
            df = pl.DataFrame(
                trades_data,
                schema={
                    'TYPE': pl.Utf8,
                    'DEAL': pl.Utf8,
                    'SYMBOL': pl.Utf8,
                    'TIME_GENERATED': pl.Datetime,
                    'POSITION_ID': pl.Int64,
                    'STRATEGY_ID': pl.Int64,
                    'EXCHANGE': pl.Utf8,
                    'VOLUME': pl.Float64,
                    'PRICE': pl.Float64,
                    'SIGNAL_TYPE': pl.Utf8,
                    'COMMISSION': pl.Float64,
                    'SWAP': pl.Float64,
                    'FEE': pl.Float64,
                    'GROSS_PROFIT': pl.Float64,
                    'CCY': pl.Utf8
                }
            )

            # Ensure the directory for the file path exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            df.write_parquet(
                file=file_path,
                compression='zstd',
                compression_level=10
            )
            logger.debug(f"Successfully exported trades to {file_path}")

        except pl.exceptions.PolarsError as e:
            logger.error(f"Error creating Polars DataFrame for trades: {str(e)}")
            raise
        except OSError as e:
            logger.error(f"Error writing trades parquet file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during trade export: {str(e)}")
            raise

    def export_csv_trade_archive(self, file_path: str) -> None:
        """
        Exports the trade archive to a CSV file.
        
        Args:
            file_path: Path where the CSV file should be saved.
        """
        # Ensure the directory for the file path exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Get the trade archive as a DataFrame
        df = self.export_historical_trades_dataframe()

        # Export the trade archive to a CSV file
        df.to_csv(file_path, index=False)

