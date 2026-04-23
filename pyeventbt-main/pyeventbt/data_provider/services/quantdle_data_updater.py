"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

"""
Quantdle Data Updater Service

This service manages CSV data files as a cache for Quantdle data.
It downloads missing data from Quantdle and updates local CSV files,
minimizing API calls and bandwidth usage.
"""

import os
from datetime import datetime
from pathlib import Path
import logging

import polars as pl

logger = logging.getLogger("pyeventbt")


class QuantdleDataUpdater:
    """
    Manages local CSV cache and downloads missing data from Quantdle.
    
    Usage:
        updater = QuantdleDataUpdater(api_key="your_key", api_key_id="your_id")
        updater.update_data(
            csv_dir="/path/to/csv",
            symbols=["EURUSD", "GBPUSD"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
    """
    
    def __init__(self, api_key: str, api_key_id: str):
        """
        Initialize the Quantdle data updater.
        
        Args:
            api_key: Your Quantdle API key
            api_key_id: Your Quantdle API key ID
        """
        try:
            import quantdle as qdl
            self.qdl = qdl
        except ImportError:
            raise ImportError(
                "quantdle package is required. Install it with: pip install quantdle"
            )
        
        self.client = qdl.Client(api_key=api_key, api_key_id=api_key_id)
        logger.info("Quantdle client initialized successfully")
    
    def update_data(
        self,
        csv_dir: str,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1min",
        spread_column: str = "spreadopen"
    ) -> None:
        """
        Update CSV cache with data from Quantdle.
        
        Args:
            csv_dir: Directory where CSV files are stored
            symbols: List of symbols to update (e.g., ["EURUSD", "GBPUSD"])
            start_date: Start date for the data range
            end_date: End date for the data range
            timeframe: Timeframe for the data (default: "1min")
            spread_column: Which spread column to use from Quantdle (default: "spreadopen")
        """
        csv_path = Path(csv_dir)
        csv_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*80}")
        logger.info("QUANTDLE DATA UPDATE SERVICE")
        logger.info(f"{'='*80}")
        logger.info(f"CSV Cache Directory: {csv_dir}")
        logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"{'='*80}\n")
        
        for symbol in symbols:
            self._update_symbol_data(
                csv_path, symbol, start_date, end_date, timeframe, spread_column
            )
        
        logger.info(f"\n{'='*80}")
        logger.info("QUANTDLE DATA UPDATE COMPLETE")
        logger.info(f"{'='*80}\n")
    
    def _update_symbol_data(
        self,
        csv_path: Path,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        spread_column: str
    ) -> None:
        """Update data for a single symbol."""
        csv_file = csv_path / f"{symbol}.csv"
        
        logger.info(f"Processing {symbol}...")
        
        if csv_file.exists():
            # CSV exists - check date range and fill gaps
            logger.info(f"  Found existing CSV for {symbol}")
            self._update_existing_csv(
                csv_file, symbol, start_date, end_date, timeframe, spread_column
            )
        else:
            # CSV doesn't exist - download full range
            logger.info(f"  No existing CSV found for {symbol}")
            self._create_new_csv(
                csv_file, symbol, start_date, end_date, timeframe, spread_column
            )
    
    def _update_existing_csv(
        self,
        csv_file: Path,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        spread_column: str
    ) -> None:
        """Update existing CSV file with missing data."""
        # Read existing CSV efficiently with Polars
        existing_df = pl.read_csv(
            csv_file,
            has_header=False,
            new_columns=["date", "time", "open", "high", "low", "close", "tickvol", "volume", "spread"]
        )
        
        # Parse datetime from date + time columns
        existing_df = existing_df.with_columns([
            (pl.col("date") + " " + pl.col("time"))
            .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M:%S")
            .alias("datetime")
        ])
        
        csv_start = existing_df["datetime"].min()
        csv_end = existing_df["datetime"].max()
        
        logger.info(f"  Existing data range: {csv_start} to {csv_end}")
        
        # Check if we need to download data before the existing range
        need_data_before = start_date < csv_start
        need_data_after = end_date > csv_end
        
        dfs_to_concat = []
        
        if need_data_before:
            logger.info(f"  Downloading data BEFORE existing range: {start_date.date()} to {csv_start.date()}")
            before_df = self._download_from_quantdle(
                symbol, start_date, csv_start, timeframe, spread_column
            )
            if before_df is not None and not before_df.is_empty():
                dfs_to_concat.append(before_df)
                logger.info(f"  Downloaded {len(before_df)} bars before existing data")
        
        # Add existing data
        dfs_to_concat.append(existing_df)
        
        if need_data_after:
            logger.info(f"  Downloading data AFTER existing range: {csv_end.date()} to {end_date.date()}")
            after_df = self._download_from_quantdle(
                symbol, csv_end, end_date, timeframe, spread_column
            )
            if after_df is not None and not after_df.is_empty():
                dfs_to_concat.append(after_df)
                logger.info(f"  Downloaded {len(after_df)} bars after existing data")
        
        if len(dfs_to_concat) > 1:
            # Concatenate and remove duplicates
            updated_df = pl.concat(dfs_to_concat)
            updated_df = updated_df.unique(subset=["datetime"], maintain_order=True)
            updated_df = updated_df.sort("datetime")
            
            # Drop datetime column before saving
            updated_df = updated_df.drop("datetime")
            
            # Save updated CSV
            updated_df.write_csv(csv_file, include_header=False)
            logger.info(f"  ✓ Updated {symbol}.csv ({len(updated_df)} total bars)")
        else:
            logger.info(f"  ✓ No update needed for {symbol}.csv")
    
    def _create_new_csv(
        self,
        csv_file: Path,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        spread_column: str
    ) -> None:
        """Create new CSV file with data from Quantdle."""
        logger.info(f"  Downloading full range: {start_date.date()} to {end_date.date()}")
        
        df = self._download_from_quantdle(
            symbol, start_date, end_date, timeframe, spread_column
        )
        
        if df is not None and not df.is_empty():
            # Drop datetime column before saving
            df = df.drop("datetime")
            
            # Save new CSV
            df.write_csv(csv_file, include_header=False)
            logger.info(f"  ✓ Created {symbol}.csv ({len(df)} bars)")
        else:
            logger.warning(f"  ✗ No data received from Quantdle for {symbol}")
    
    def _download_from_quantdle(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        spread_column: str
    ) -> pl.DataFrame | None:
        """
        Download data from Quantdle and transform to required format.
        
        Returns:
            Polars DataFrame with columns: date, time, open, high, low, close, tickvol, volume, spread, datetime
        """
        try:
            # Convert timeframe to Quantdle format (e.g., "1min" -> "M1")
            quantdle_timeframe = self._convert_to_quantdle_timeframe(timeframe)
            
            # Download data from Quantdle directly as Polars
            df = self.client.download_data(
                symbol=[symbol],  # Quantdle accepts list
                timeframe=quantdle_timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                output_format="polars"
            )
            
            if df is None or df.is_empty():
                return None
            
            # Get the datetime column name (Quantdle uses lowercase 'datetime' as index/column)
            datetime_col = None
            for col_name in ["datetime", "Datetime", "time", "Time"]:
                if col_name in df.columns:
                    datetime_col = col_name
                    break
            
            if datetime_col is None:
                logger.error(f"  No datetime column found in Quantdle data for {symbol}")
                return None
            
            # Ensure datetime column is properly typed
            if df[datetime_col].dtype != pl.Datetime:
                df = df.with_columns([
                    pl.col(datetime_col).str.strptime(pl.Datetime).alias("datetime")
                ])
                datetime_col = "datetime"
            else:
                # Rename to standard 'datetime' if it has a different name
                if datetime_col != "datetime":
                    df = df.rename({datetime_col: "datetime"})
                    datetime_col = "datetime"
            
            # Cast price and volume columns (Quantdle returns strings)
            df = df.with_columns([
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("tickvol").cast(pl.Int64),
                pl.col("volume").cast(pl.Int64),
                pl.col(spread_column).cast(pl.Int64).alias("spread"),
            ])
            
            # Drop unused spread columns
            columns_to_drop = [
                col for col in ['spreadmax', 'spreadopen']
                if col != spread_column and col in df.columns
            ]
            if columns_to_drop:
                df = df.drop(columns_to_drop)
            
            # Create date and time columns in MT5 format
            df = df.with_columns([
                pl.col("datetime").dt.strftime("%Y.%m.%d").alias("date"),
                pl.col("datetime").dt.strftime("%H:%M:%S").alias("time"),
            ])
            
            # Reorder columns to match required format
            df = df.select([
                "date", "time", "open", "high", "low", "close", 
                "tickvol", "volume", "spread", "datetime"
            ])
            
            return df
            
        except Exception as e:
            logger.error(f"  Error downloading data from Quantdle for {symbol}: {e}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return None
    
    def _convert_to_quantdle_timeframe(self, timeframe: str) -> str:
        """
        Convert standard timeframe format to Quantdle format.
        
        Examples:
            "1min" -> "M1"
            "5min" -> "M5"
            "1h" -> "H1"
            "1d" -> "D1"
        """
        timeframe_map = {
            "1min": "M1",
            "5min": "M5",
            "15min": "M15",
            "30min": "M30",
            "1h": "H1",
            "1H": "H1",
            "4h": "H4",
            "4H": "H4",
            "1d": "D1",
            "1D": "D1",
            "1w": "W1",
            "1W": "W1",
        }
        
        return timeframe_map.get(timeframe, timeframe)

