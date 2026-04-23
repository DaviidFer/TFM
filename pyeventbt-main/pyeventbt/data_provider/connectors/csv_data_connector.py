"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.events.events import BarEvent, Bar
from ..core.interfaces.data_provider_interface import IDataProvider
from ..core.configurations.data_provider_configurations import CSVBacktestDataConfig
from pyeventbt.broker.mt5_broker.mt5_simulator_wrapper import Mt5SimulatorWrapper as mt5
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
import polars as pl
import pandas as pd
import os
from functools import lru_cache
import logging

logger = logging.getLogger("pyeventbt")
backtest_logger = logging.getLogger("backtest_info")

# Polars‐side resample aggregators
_AGG_MAP = {
    "open": pl.first,
    "high": pl.max,
    "low": pl.min,
    "close": pl.last,
    "tickvol": pl.sum,
    "volume": pl.sum,
    "spread": pl.first,
}


class CSVDataProvider(IDataProvider):
    
    def __init__(self, configs: CSVBacktestDataConfig) -> None:
        # coloured console handler
        backtest_logger_console_handler = logging.StreamHandler()
        green = "\x1b[92;20m"
        reset = "\x1b[0m"
        special_format = logging.Formatter(f'{green}%(message)s{reset}')
        backtest_logger_console_handler.setFormatter(special_format)
        backtest_logger.addHandler(backtest_logger_console_handler)
        backtest_logger.setLevel(logging.INFO)

        backtest_logger.info("\n--> PyEventBT Docs: https://pyeventbt.com")

        # pre‐backtest logging
        backtest_logger.info("\n+-------------- PYEVENTBT EVENT-DRIVEN BACKTEST INITIATED ----------------------------------+")
        backtest_logger.info(f"| - PRE-BACKTEST CHECKS: Initializing MT5 Simulator Shared Data...")
        backtest_logger.info(f"| - PRE-BACKTEST CHECKS: Starting checks for backtesting with CSV data...")
        backtest_logger.info(f"| - PRE-BACKTEST CHECKS: Data Provider: {__name__}")

        # parse window
        self.backtest_start_timestamp: datetime = configs.backtest_start_timestamp if configs.backtest_start_timestamp else None
        self.backtest_end_timestamp: datetime = configs.backtest_end_timestamp if configs.backtest_end_timestamp else None

        self.csv_dir = configs.csv_path
        self.base_timeframe = configs.base_timeframe
        self.timeframes_list = configs.timeframes_list
        self.continue_backtest = True
        self.close_positions_end_of_data = False
        self.symbol_data_generator: dict[str, any] = {}

        # build aggregator expressions once
        # concise way to turn a mapping of column names to aggregation functions into a list of Polars expressions, each tagged with the original column name.
        self._agg_exprs = [func(col).alias(col) for col, func in _AGG_MAP.items()]

        # validations
        self._check_first_element_in_tf_list_is_base_tf(self.base_timeframe, self.timeframes_list)
        self._check_account_currency_is_supported(configs.account_currency)
        self.account_currency = configs.account_currency

        # auxiliary crosses
        self.auxiliary_symbol_list = self._create_auxiliary_symbol_list(configs.tradeable_symbol_list, self.account_currency)
        self.tradeable_symbols = configs.tradeable_symbol_list
        self.symbol_list = self.tradeable_symbols + self.auxiliary_symbol_list

        # will hold per‐symbol, per‐tf DataFrames
        self.complete_symbol_data_timeframes: dict[str, dict[str, pl.DataFrame]] = {}
        self.latest_index_timeframes: dict[str, dict[str, datetime]] = {symbol: {} for symbol in self.symbol_list}

        # construct the symbol_digits dict cache
        self.symbol_digits: dict[str, int] = {symbol: mt5.symbol_info(symbol).digits for symbol in self.symbol_list}     

        # prepare caches for fast timeframe‐checks
        self._base_timestamps: dict[str, list[datetime]] = {}
        self._base_idx_map: dict[str, dict[datetime, int]] = {}

        self._tf_type_cache = {} 

        # add integer‐minute caches (for integer math)
        self._base_minutes:    dict[str, list[int]]      = {}
        self._base_idx_map_int: dict[str, dict[int,int]] = {}
        self._base_minutes_global:    dict[str, list[int]]      = {}
        self._base_idx_map_int_global: dict[str, dict[int,int]] = {}
        self._base_minutes_day:    dict[str, list[int]]      = {}
        self._base_idx_map_int_day: dict[str, dict[int,int]] = {}

        # load, align, resample
        self._open_convert_csv_files()
        self._populate_symbol_data_with_generators()

        backtest_logger.info("| - PRE-BACKTEST CHECKS: Finished checks for backtesting with CSV data")
        backtest_logger.info("+---------------------------------------------------------------------------------------------+")

    @lru_cache(maxsize=None)
    def _timeframe_to_duration(self, tf: str) -> str:
        """
        MT5 timeframe (e.g. '5min','1H','1W') -> Polars window string ('5m','1h','1w').
        """
        unit_map = {"min": "m", "H": "h", "h": "h", "D": "d", "B": "d", "W": "w", "M": "mo"}
        
        for u, du in unit_map.items():
            if tf.endswith(u):
                qty = int(tf[:-len(u)]) if tf[:-len(u)].isdigit() else 1
                return f"{qty}{du}"
        
        raise ValueError(f"Unsupported timeframe format: {tf}")
    
    @lru_cache(maxsize=None)
    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        time_units = {"min": 1, "H": 60, "h": 60, "D": 1440, "B": 1440, "W": 10080, "M": 43200}
        for unit, dur in time_units.items():
            if timeframe.endswith(unit):
                qty = int(timeframe[:-len(unit)]) if timeframe[:-len(unit)].isdigit() else 1
                return qty * dur
        
        raise ValueError("Unsupported timeframe format")
    
    def _merge_sorted_unique(self, list1: list[datetime], list2: list[datetime]) -> list[datetime]:
        """
        Merge two sorted datetime lists into one sorted unique list in O(n).
        """
        i = j = 0
        merged: list[datetime] = []
        last: datetime | None = None

        # merge until one list is exhausted
        while i < len(list1) and j < len(list2):
            a_item, b_item = list1[i], list2[j]
            if a_item < b_item:
                candidate, i = a_item, i + 1
            elif b_item < a_item:
                candidate, j = b_item, j + 1
            else:  # equal
                candidate, i, j = a_item, i + 1, j + 1

            if candidate != last:
                merged.append(candidate)
                last = candidate

        # append any remaining items from either list
        for tail in (list1[i:], list2[j:]):
            for item in tail:
                if item != last:
                    merged.append(item)
                    last = item

        return merged

    def _check_first_element_in_tf_list_is_base_tf(self, base_timeframe: str, timeframes_list: list) -> None:
        """
        Check if the first element in the timeframes list is the base timeframe.
        """
        if timeframes_list[0] != base_timeframe:
            backtest_logger.warning(
                f"\x1b[93;20m| - PRE-BACKTEST CHECKS: First timeframe must be {base_timeframe}"
            )
            raise Exception(f"First timeframe must be {base_timeframe}")
        
        backtest_logger.info(f"| - PRE-BACKTEST CHECKS: Base TF is first: {base_timeframe}")

    def _check_account_currency_is_supported(self, account_currency: str) -> None:
        """
        Check if the account currency is supported.
        """
        if account_currency not in {"USD", "EUR", "GBP"}:
            backtest_logger.warning(f"\x1b[93;20m| - PRE-BACKTEST CHECKS: Invalid account currency: {account_currency}")
            
            raise Exception(f"Invalid account currency: {account_currency}")
        
        backtest_logger.info(f"| - PRE-BACKTEST CHECKS: Account currency: {account_currency}")

    def _create_auxiliary_symbol_list(self, tradeable_symbols: list[str], account_currency: str) -> list[str]:
        
        all_fx = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF",
                "CADJPY", "CHFJPY", "EURAUD", "EURCAD", "EURCHF", "EURGBP",
                "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF",
                "GBPJPY", "GBPNZD", "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY",
                "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "USDSEK", "USDNOK",
                "USDMXN", "EURMXN", "GBPMXN"]

        aux: list[str] = []
        for symbol in tradeable_symbols:
            info = mt5.symbol_info(symbol)
            for cur in (info.currency_margin, info.currency_profit):
                if cur != account_currency:
                    for fx in all_fx:
                        if fx not in tradeable_symbols and account_currency in fx and cur in fx and fx not in aux:
                            aux.append(fx)
                            break
        backtest_logger.info(f"| - PRE-BACKTEST CHECKS: Added auxiliary symbols {aux}" if aux
            else "| - PRE-BACKTEST CHECKS: No auxiliary symbols needed")
        
        return aux

    def _open_convert_csv_files(self) -> None:
        """
        Lazy-scan each CSV → parse & rename → collect eagerly → 
        dynamic-group 1m on eager DF → align on full index → fill gaps → 
        dynamic-group per requested timeframe → store.
        """
        backtest_logger.info("| - PRE-BACKTEST CHECKS: Loading CSV files:")
        complete_m1: dict[str, pl.DataFrame] = {}
        first_dates: dict[str, datetime] = {}
        comb_index: list[datetime] = []

        # STAGE 1: scan, parse, collect, M1 resample
        for symbol in self.symbol_list:
            fn = os.path.join(self.csv_dir, f"{symbol}.csv")
            backtest_logger.info(f"| - PRE-BACKTEST CHECKS: Loading {symbol}.csv...")

            # 1. lazy read
            lf = pl.scan_csv(fn, has_header=False, new_columns=["date","time","open","high","low","close","tickvol","volume","spread"],)

            # 2. Parse & cast
            lf = lf.with_columns([
                (pl.col("date") + " " + pl.col("time"))
                    .str.strptime(pl.Datetime, "%Y.%m.%d %H:%M:%S")
                    .alias("datetime"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("tickvol").cast(pl.Int64),
                pl.col("volume").cast(pl.Int64),
                pl.col("spread").cast(pl.Int64),
            ]).drop(["date","time"])

            # 3. Filter window (push-down)
            if self.backtest_start_timestamp:
                lf = lf.filter(pl.col("datetime") >= self.backtest_start_timestamp)
            if self.backtest_end_timestamp:
                lf = lf.filter(pl.col("datetime") <= self.backtest_end_timestamp)

            # 4. Collect → sort → rechunk
            raw_df = lf.collect(streaming=True).sort("datetime").rechunk()

            # 5. Resample M1
            m1_df = (raw_df
                .group_by_dynamic("datetime", every="1m", closed="left")
                .agg(self._agg_exprs)
                .sort("datetime")
                .rechunk()
            )

            # 5b) annotate each M1 bar with:
            #    - global minute‐since‐epoch ("minute_idx")
            #    - intra‐day minute‐of‐day ("minute_of_day")
            m1_df = m1_df.with_columns([
                # global minutes‐since‐epoch
                (pl.col("datetime")
                .cast(pl.Int64)
                .floordiv(pl.lit(1_000_000_000 * 60, pl.Int64))
                ).alias("minute_idx"),

                # minute‐of‐day 0–1439, cast before math to avoid overflow
                (
                pl.col("datetime").dt.hour().cast(pl.Int16) * pl.lit(60, pl.Int16)
                + pl.col("datetime").dt.minute().cast(pl.Int16)
                ).alias("minute_of_day")
            ])
            
            complete_m1[symbol] = m1_df

            dts  = m1_df["datetime"].to_list()
            #mins_global  = m1_df["minute_idx"].to_list()
            #mins_of_day  = m1_df["minute_of_day"].to_list()

            # exact minute‐since‐epoch via Pandas Timestamp.value (nanoseconds)
            minute_idx = [pd.Timestamp(dt).value // 60_000_000_000 for dt in dts]
            minute_of_day = [mi % 1440 for mi in minute_idx]        # derive minutes-of-day by modulo 1440

            # These caches are based on the M1 data from CSV
            # We'll update them after all timeframes are created with the actual base timeframe data

            # # integer caches
            # # now cache the *global* minute_idx
            # self._base_minutes[symbol]      = mins
            # self._base_idx_map_int[symbol]  = {m: idx for idx, m in enumerate(mins)}

            first_dates[symbol] = dts[0]
            comb_index = dts if not comb_index else self._merge_sorted_unique(comb_index, dts)

        # STAGE 2: build master index from common start
        latest_start = max(first_dates.values())
        full_idx = pl.DataFrame({"datetime": comb_index}).sort("datetime")

        # STAGE 3: align, fill gaps, then per-TF resample
        for symbol, m1_df in complete_m1.items():
            df = (
                full_idx.join(m1_df, on="datetime", how="left")
                    .sort("datetime")
                    # Step 1: Forward fill the close column first
                    .with_columns([
                        pl.col("close").forward_fill().alias("close_filled")
                    ])
                    # Step 2: Use the filled close values to fill other null columns
                    .with_columns([
                        pl.col("close_filled").alias("close"),
                        pl.col("open").fill_null(pl.col("close_filled").shift(1)),
                        pl.col("high").fill_null(pl.col("close_filled").shift(1)),
                        pl.col("low").fill_null(pl.col("close_filled").shift(1)),
                        pl.col("tickvol").fill_null(1).cast(pl.Int64),
                        pl.col("volume").fill_null(1).cast(pl.Int64),
                        pl.col("spread").fill_null(1).cast(pl.Int64),
                    ])
                    .drop("close_filled")
                    .filter(pl.col("datetime") >= latest_start)
            )

            self.complete_symbol_data_timeframes[symbol] = {}
            for tf in self.timeframes_list:
                # Always resample from the M1 data regardless of what the base timeframe is
                window = self._timeframe_to_duration(tf)
                if tf == "1W":
                    tf_df = (df
                        .group_by_dynamic("datetime", every=window, closed="left")
                        .agg(self._agg_exprs)
                        .with_columns((pl.col("datetime") - pl.duration(days=7)).alias("datetime"))
                        .drop_nulls()
                    )
                else:
                    tf_df = (df
                        .group_by_dynamic("datetime", every=window, closed="left")
                        .agg(self._agg_exprs)
                        .drop_nulls()
                    )

                self.complete_symbol_data_timeframes[symbol][tf] = tf_df

        # Now update caches with the actual base timeframe data
        for symbol in self.symbol_list:
            # Get the actual base timeframe data
            base_df = self.complete_symbol_data_timeframes[symbol][self.base_timeframe]
            
            # Extract datetimes for the base timeframe
            dts = base_df["datetime"].to_list()
            
            # Calculate minute indices based on the actual base timeframe data
            minute_idx = [pd.Timestamp(dt).value // 60_000_000_000 for dt in dts]
            minute_of_day = [mi % 1440 for mi in minute_idx]
            
            # Update caches with the actual base timeframe data
            self._base_timestamps[symbol] = dts
            self._base_idx_map[symbol] = {dt:i for i,dt in enumerate(dts)}
            
            self._base_minutes_global[symbol] = minute_idx
            self._base_idx_map_int_global[symbol] = {mi:i for i,mi in enumerate(minute_idx)}
            
            self._base_minutes_day[symbol] = minute_of_day
            self._base_idx_map_int_day[symbol] = {md:i for i,md in enumerate(minute_of_day)}

    def _get_new_bar_generator(self, symbol: str):

        # 1) Pull digits and compute the scale
        digits = self.symbol_digits[symbol]
        scale = 10 ** digits

        # 2) select + rechunk (so all columns are in contiguous memory slots) + scale & cast price columns to Int64
        df = (
            self.complete_symbol_data_timeframes[symbol][self.base_timeframe]
            .select(["datetime","open","high","low","close","tickvol","volume","spread"])
            .with_columns([
                (
                pl.col(c)
                .mul(pl.lit(scale, pl.Float64))  # scale up by 10^digits (1.1234567 * 1e5)
                .floor()                         # truncate any remaining decimals (112345.0)
                .cast(pl.Int64)                  # convert to integer (112345)
                .alias(c)
                )
                for c in ["open", "high", "low", "close"]
            ] + [
                pl.col("tickvol").cast(pl.Int64),
                pl.col("volume").cast(pl.Int64),
                pl.col("spread").cast(pl.Int64),
            ])
            .rechunk()
        )
        # DEBUG
        # s = df["open"]
        # print("chunks:", s.n_chunks())         # must be 1
        # print("has nulls?", s.null_count() > 0)

        # 2.5) sanity‐check: no nulls in price columns
        try:
            for col in ["open", "high", "low", "close"]:
                nulls = df[col].null_count()
                if nulls > 0:
                    raise ValueError(f"Column '{col}' has {nulls} nulls after truncation")
        except ValueError as err:
            # log the error with context, then re‐raise
            logger.critical(f"[{symbol}@{self.base_timeframe}] Data integrity error: {err}")
            raise

        # 3. zero-copy into memoryviews of int64 (point numpy arrays into the polars arrow columns in memory, and then do a memoryview over those arrays)
        ts_mv = memoryview(df["datetime"].cast(pl.Int64).to_numpy(zero_copy_only=True))
        o_mv  = memoryview(df["open"  ].to_numpy(zero_copy_only=True))
        h_mv  = memoryview(df["high"  ].to_numpy(zero_copy_only=True))
        l_mv  = memoryview(df["low"   ].to_numpy(zero_copy_only=True))
        c_mv  = memoryview(df["close" ].to_numpy(zero_copy_only=True))
        tv_mv = memoryview(df["tickvol"].to_numpy(zero_copy_only=True))
        v_mv  = memoryview(df["volume"].to_numpy(zero_copy_only=True))
        sp_mv = memoryview(df["spread"].to_numpy(zero_copy_only=True))

        # 4. iterate
        length = len(o_mv)
        for i in range(length):
            
            # timestamp → Python datetime
            ts = datetime.fromtimestamp(ts_mv[i] / 1_000_000, tz=timezone.utc).replace(tzinfo=None)

            # fresh Bar with scaled ints + digits
            bar = Bar(
                open=   o_mv[i],
                high=   h_mv[i],
                low=    l_mv[i],
                close=  c_mv[i],
                tickvol=tv_mv[i],
                volume= v_mv[i],
                spread= sp_mv[i],
                digits= digits,
            )

            # emit the event envelope
            yield BarEvent(symbol=symbol, datetime=ts, data=bar, timeframe=self.base_timeframe)

    def _populate_symbol_data_with_generators(self) -> None:
        """
        Populate the symbol data with generators for each symbol in the symbol list.
        This method creates a generator for each symbol that will yield
        new bars as they become available.
        """
        for symbol in self.symbol_list:
            self.symbol_data_generator[symbol] = self._get_new_bar_generator(symbol)
    
    def get_latest_tick(self, symbol: str) -> dict:
        """Returns the latest tick dict."""
        
        # Remember: for last tick, is the only time we are accessing what would be the "current opening bar", taking only
        # its datetime, spread and Open price - No lookahead bias.
        df = self.complete_symbol_data_timeframes[symbol][self.base_timeframe]
        target_bar = df.filter(pl.col("datetime") > self.latest_index_timeframes[symbol][self.base_timeframe]).head(1)
        
        # Handle end of backtest
        end_of_backtest = target_bar.is_empty()
        if end_of_backtest:
            target_bar = df.tail(1)
        
        try:
            ts = target_bar["datetime"][0].timestamp()
        except Exception as e:
            logger.error(f"Error getting timestamp for symbol {symbol}: {e}")
            return {}
        
        # Calculate bid/ask with proper scaling
        scale = Decimal(10) ** (-self.symbol_digits[symbol])
        open_val = df.tail(1)["close"][0] if end_of_backtest else target_bar["open"][0]
        bid = Decimal(str(open_val)).quantize(scale, rounding=ROUND_DOWN)
        ask = (bid + Decimal(str(target_bar["spread"][0])) * scale).quantize(scale, rounding=ROUND_DOWN)
        
        return {
            "time": int(ts),
            "bid": bid,
            "ask": ask,
            "last": Decimal("0.0"),
            "volume": 0,
            "time_msc": int(ts * 1000),
            "flags": 2,
            "volume_real": Decimal("0.0"),
        }
    
    def get_latest_tick_old(self, symbol: str) -> dict:
        """
        Return the latest tick dict using the integer Bar + digits.
        Avoids MT5 API call by deriving point = 10**(-digits).
        """

        
        bar_event = self.get_latest_bar(symbol)  # Returns the last bar inside a BarEvent object
        # 1. event timestamp
        ts: float = bar_event.datetime.timestamp()

        # 2. scaling factor: 10**(-digits)
        digits = bar_event.data.digits
        scale = Decimal(10) ** (-digits)

        # 3. read integer values
        open_i   = bar_event.data.open
        spread_i = bar_event.data.spread

        # 4. reconstruct Decimal prices
        bid = Decimal(open_i) * scale
        ask = bid + (Decimal(spread_i) * scale)

        return {
            "time":       int(ts),           # seconds
            "bid":        bid,               # Decimal
            "ask":        ask,               # Decimal
            "last":       Decimal("0.0"),
            "volume":     0,
            "time_msc":   int(ts * 1000),    # milliseconds
            "flags":      2,
            "volume_real":Decimal("0.0"),
        }

    def get_latest_bar_old_lookahead_bias(self, symbol: str, timeframe: str = None) -> BarEvent:
        """
        Retrieve the most recent CLOSED bar for a given symbol and timeframe.
        Returns a specialized BarEvent object with fixed-point integer prices for maximum performance.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: Optional timeframe string (e.g., "1h"); defaults to base_timeframe if not specified
        
        Returns:
            BarEvent: A performance-optimized bar event containing price/volume data
        """
        # Resolve the requested timeframe, defaulting to base if not specified
        tf = timeframe or self.base_timeframe
        
        # Retrieve the symbol's decimal precision (digits) from our cache
        # and calculate the scaling factor (10^digits) for fixed-point conversion
        digits = self.symbol_digits[symbol]
        scale  = 10 ** digits

        # Get the cached DataFrame for this symbol+timeframe
        df = self.complete_symbol_data_timeframes[symbol][tf]
        
        # Find the newest bar that's not ahead of our current "simulation time"
        # This filter ensures we don't peek into the future during backtesting
        last = (
            df.filter(pl.col("datetime") <= self.latest_index_timeframes[symbol][tf])
            .sort("datetime")  # Ensure chronological order
            .tail(1)           # Take only the most recent row
        )
        
        # Convert to dictionary format for faster field access
        # (avoids DataFrame indexing overhead)
        rec = last.to_dict(as_series=False)
        
        # Extract the timestamp for the event envelope
        ts = rec["datetime"][0]

        # Construct a Bar object with fixed-point integer prices
        # This approach provides several benefits:
        # 1. Avoids floating-point precision issues
        # 2. Enables faster integer math in downstream calculations
        # 3. Reduces memory footprint (especially important in high-frequency scenarios)
        # 4. Preserves exact decimal precision without rounding errors
        bar = Bar(
            # Price fields are scaled to integers (e.g., 1.12345 → 112345 with digits=5)
            open =  int(rec["open"  ][0] * scale),  # Opening price
            high =  int(rec["high"  ][0] * scale),  # Highest price
            low =   int(rec["low"   ][0] * scale),  # Lowest price
            close = int(rec["close" ][0] * scale),  # Closing price
            
            # Non-price fields are simply cast to integers (no scaling)
            tickvol = int(rec["tickvol"][0]),  # Number of price changes
            volume =  int(rec["volume" ][0]),  # Trading volume
            spread =  int(rec["spread" ][0]),  # Bid/ask spread in points
            
            # Store original precision for later reconstruction if needed
            digits= digits,
        )
        
        # Wrap everything in a BarEvent envelope and return
        # This specialized event provides a more memory-efficient representation
        # than the previous pandas Series-based approach
        return BarEvent(symbol=symbol, datetime=ts, data=bar, timeframe=tf)
    
    def get_latest_bar(self, symbol: str, timeframe: str = None) -> BarEvent:
        """
        Retrieve the most recent CLOSED bar for a given symbol and timeframe.
        For higher timeframes (non-base timeframes), returns the last fully formed bar,
        not the current forming one, to avoid future information leakage.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: Optional timeframe string (e.g., "1h"); defaults to base_timeframe if not specified
        
        Returns:
            BarEvent: A performance-optimized bar event containing price/volume data as INTEGERS
        """
        # Resolve the requested timeframe, defaulting to base if not specified
        tf = timeframe or self.base_timeframe
        
        # Retrieve the symbol's decimal precision (digits) from our cache
        digits = self.symbol_digits[symbol]
        scale  = 10 ** digits

        # Get the cached DataFrame for this symbol+timeframe
        df = self.complete_symbol_data_timeframes[symbol][tf]
        
        # For higher timeframes, we need to get the previous (fully formed) bar
        # to avoid returning "future" information
        if tf != self.base_timeframe:
            # Get all bars up to the current simulation time
            filtered_df = df.filter(pl.col("datetime") <= self.latest_index_timeframes[symbol][tf])
            
            if len(filtered_df) >= 2:
                # Take the second-to-last bar (the last fully formed one)
                last = filtered_df.sort("datetime").tail(2).head(1)
            else:
                # If we don't have at least 2 bars, take the only one we have
                last = filtered_df.sort("datetime").tail(1)
        else:
            # For base timeframe, get the most recent bar as before
            last = (
                df.filter(pl.col("datetime") <= self.latest_index_timeframes[symbol][tf])
                .sort("datetime")
                .tail(1)
            )
        
        # Convert to dictionary format for faster field access
        rec = last.to_dict(as_series=False)
        
        # Extract the timestamp for the event envelope
        ts = rec["datetime"][0]

        # Construct a Bar object with fixed-point integer prices
        bar = Bar(
            # Price fields are scaled to integers
            open =  int(rec["open"  ][0] * scale),
            high =  int(rec["high"  ][0] * scale),
            low =   int(rec["low"   ][0] * scale),
            close = int(rec["close" ][0] * scale),
            
            # Non-price fields are simply cast to integers
            tickvol = int(rec["tickvol"][0]),
            volume =  int(rec["volume" ][0]),
            spread =  int(rec["spread" ][0]),
            
            digits= digits,
        )
        
        return BarEvent(symbol=symbol, datetime=ts, data=bar, timeframe=tf)

    def get_latest_bars_pandas(self, symbol: str, timeframe: str = None, N: int = 2) -> pd.DataFrame:
        tf = timeframe or self.base_timeframe
        if symbol not in self.complete_symbol_data_timeframes:
            raise KeyError(f"Symbol '{symbol}' not available")
        N = N if N > 0 else 2
        cutoff = self.latest_index_timeframes.get(symbol, {}).get(tf)
        if cutoff is None:
            return pd.DataFrame()
        pdf = (
            self.complete_symbol_data_timeframes[symbol][tf]
            .filter(pl.col("datetime") <= cutoff)
            .to_pandas()
            .set_index("datetime")
        )
        return pdf.iloc[-N:] if not pdf.empty else pd.DataFrame()
    
    def get_latest_bars(self, symbol: str, timeframe: str = None, N: int = 2) -> pl.DataFrame:
        """
        Retrieve the N most recent bars for a given symbol and timeframe. Data can be FLOAT
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: Optional timeframe string (e.g., "1h"); defaults to base_timeframe if not specified
            N: Number of bars to return (default: 2)
            
        Returns:
            pl.DataFrame: A Polars DataFrame containing the N most recent bars
        """
        tf = timeframe or self.base_timeframe
        
        # Check if symbol exists in our dataset
        if symbol not in self.complete_symbol_data_timeframes:
            raise KeyError(f"Symbol '{symbol}' not available")
        
        # Ensure N is positive
        N = N if N > 0 else 2
        
        # Get the current cutoff timestamp for this symbol
        cutoff = self.latest_index_timeframes.get(symbol, {}).get(tf)
        if cutoff is None:
            return pl.DataFrame()
        
        # Pre-sorted cached DataFrame
        df = self.complete_symbol_data_timeframes[symbol][tf]
        filtered_df = df.filter(pl.col("datetime") <= cutoff)
        
        df_len = len(filtered_df)
        if df_len == 0:
            return pl.DataFrame()
            
        # Fast path for base timeframe
        if tf == self.base_timeframe:
            return filtered_df.tail(N)
        
        # For higher timeframes, exclude the most recent bar
        if df_len <= 1:
            return pl.DataFrame()
        elif df_len <= N + 1:
            return filtered_df.slice(0, df_len - 1)
        else:
            return filtered_df.slice(df_len - N - 1, N)

    def get_latest_bid(self, symbol: str) -> Decimal:
        return self.get_latest_tick(symbol)["bid"]

    def get_latest_ask(self, symbol: str) -> Decimal:
        return self.get_latest_tick(symbol)["ask"]

    def get_latest_datetime(self, symbol: str, timeframe: str = None) -> datetime:
        """
        Get the latest datetime for a given symbol and timeframe.
        This method retrieves the most recent datetime from the symbol's data
        for the specified timeframe, defaulting to the base timeframe if not provided.

        Args:
            symbol (str): The trading symbol (e.g., "EURUSD").
            timeframe (str, optional): The timeframe string (e.g., "1h"). Defaults to None.

        Returns:
            datetime: The most recent datetime for the specified symbol and timeframe.
        """
        bar_event = self.get_latest_bar(symbol, timeframe)
        return bar_event.datetime
    
    def _base_tf_bar_creates_new_tf_bar_old(self, latest_base_tf_time: datetime, timeframe: str, symbol: str) -> bool:
        """
        Returns True exactly when the next base-TF bar
        crosses into a new `timeframe` bucket,
        using integer math that matches Pandas normalize()+floor.
        """
        tfm = self._parse_timeframe_to_minutes(timeframe)

        # 1) find index into our cached lists
        idx = self._base_idx_map.get(symbol, {}).get(latest_base_tf_time)
        if idx is None or idx + 1 >= len(self._base_minutes_global[symbol]):
            return False
        
        # 2) pull the two ints
        mi0 = self._base_minutes_global[symbol][idx]            # global minute_idx
        mi1 = self._base_minutes_global[symbol][idx + 1]
        
        # 3) select bucket floor function (same code, different input range)
        if tfm < 1440:
            # intraday: bucket minute_of_day (= minute_idx % 1440)
            b0 = (mi0 % 1440) // tfm * tfm
            b1 = (mi1 % 1440) // tfm * tfm
        else:
            # daily+: bucket global minute_idx
            b0 = (mi0 // tfm) * tfm
            b1 = (mi1 // tfm) * tfm

        # 4) rollover if floors differ
        return b0 != b1
    
    def _base_tf_bar_creates_new_tf_bar(self, latest_base_tf_time: datetime, timeframe: str, symbol: str) -> bool:
        """
        Determines if the current base timeframe bar should create a new higher timeframe bar.
        More robust handling of gaps and boundaries.
        """
        # Get the minutes value for the current timeframe (e.g., 60 for 1H)
        tfm = self._parse_timeframe_to_minutes(timeframe)
        
        # Get the last time this timeframe was updated for this symbol, defaulting to epoch start if none
        last_tf_time = self.latest_index_timeframes.get(symbol, {}).get(timeframe, datetime(1970, 1, 1))
        
        # Direct datetime comparison based on timeframe type
        if tfm < 1440:  # Less than a day (intraday timeframes)
            # For hourly and sub-hourly, check if hour or minute boundary crossed
            if timeframe.endswith('H') or timeframe.endswith('h'):
                # Hourly timeframe - check hour boundary
                return (latest_base_tf_time.year > last_tf_time.year or
                        latest_base_tf_time.month > last_tf_time.month or
                        latest_base_tf_time.day > last_tf_time.day or
                        latest_base_tf_time.hour > last_tf_time.hour)
            else:
                # Sub-hourly timeframe - check minute boundaries
                minutes_since_hour = latest_base_tf_time.minute
                bucket = minutes_since_hour // tfm
                last_bucket = last_tf_time.minute // tfm
                
                return (latest_base_tf_time.year > last_tf_time.year or
                        latest_base_tf_time.month > last_tf_time.month or
                        latest_base_tf_time.day > last_tf_time.day or
                        latest_base_tf_time.hour > last_tf_time.hour or
                        bucket > last_bucket)
        else:  # Daily and above timeframes
            if timeframe == '1D':
                # Daily timeframe - check day boundary
                return (latest_base_tf_time.year > last_tf_time.year or
                        latest_base_tf_time.month > last_tf_time.month or
                        latest_base_tf_time.day > last_tf_time.day)
            elif timeframe == '1W':
                # Weekly timeframe - check week boundary (simplified)
                current_week = latest_base_tf_time.isocalendar()[1]
                last_week = last_tf_time.isocalendar()[1]
                return (latest_base_tf_time.year > last_tf_time.year or
                        current_week > last_week)
            else:
                # Monthly timeframe
                return (latest_base_tf_time.year > last_tf_time.year or
                        latest_base_tf_time.month > last_tf_time.month)

    def _base_tf_bar_creates_new_tf_bar_f(self, latest_base_tf_time: datetime, timeframe: str, symbol: str) -> bool:
        """
        Determines if the current base timeframe bar should create a new higher timeframe bar.
        Optimized for speed with integer comparisons.
        """
        # Get the last time this timeframe was updated for this symbol
        last_tf_time = self.latest_index_timeframes.get(symbol, {}).get(timeframe, datetime(1970, 1, 1))
        
        # Cache the timeframe categorization
        if timeframe not in self._tf_type_cache:
            tfm = self._parse_timeframe_to_minutes(timeframe)
            if tfm < 60:
                self._tf_type_cache[timeframe] = ("minutes", tfm)
            elif tfm < 1440:
                self._tf_type_cache[timeframe] = ("hours", tfm//60)
            elif timeframe == "1D":
                self._tf_type_cache[timeframe] = ("days", 1)
            elif timeframe == "1W":
                self._tf_type_cache[timeframe] = ("weeks", 1)
            else:
                self._tf_type_cache[timeframe] = ("months", 1)
        
        tf_type, tf_value = self._tf_type_cache[timeframe]
        
        # Special case for weekly timeframes - always check week number
        if tf_type == "weeks":
            latest_week = latest_base_tf_time.isocalendar()[1]
            last_week = last_tf_time.isocalendar()[1]
            return (latest_base_tf_time.year > last_tf_time.year or
                    latest_week > last_week)
        
        # Fast path for non-weekly timeframes: if dates differ, we definitely need a new bar
        if (latest_base_tf_time.year > last_tf_time.year or
            latest_base_tf_time.month > last_tf_time.month or
            latest_base_tf_time.day > last_tf_time.day):
            return True
        
        # Now handle time components based on timeframe type (with same date)
        if tf_type == "minutes":
            curr_minutes = latest_base_tf_time.hour * 60 + latest_base_tf_time.minute
            last_minutes = last_tf_time.hour * 60 + last_tf_time.minute
            return curr_minutes // tf_value > last_minutes // tf_value
        
        elif tf_type == "hours":
            return latest_base_tf_time.hour // tf_value > last_tf_time.hour // tf_value
        
        # These cases are for same-date checks
        elif tf_type == "days" or tf_type == "months":
            return False  # Same date, so no new bar
        
        # Should never reach here
        return False
    
    def update_bars(self) -> list[BarEvent]:
        """
        Drive each symbol's generator (which now yields BarEvent),
        update latest_index_timeframes, and collect events.
        """
        events: list[BarEvent] = []

        for symbol in self.symbol_list:
            try:
                #ts, series = next(self.symbol_data_generator[symbol])
                bar_event: BarEvent = next(self.symbol_data_generator[symbol])
                # TODO: Here check if tickvol, vol and spread are exactly 1, which means we filled a null value (as there was no data originally for that minute)
                # so we should skip this bar (it was created so all symbols have same number of bars, but it is not a real bar)
                if bar_event.data.tickvol == 1 and bar_event.data.volume == 1 and bar_event.data.spread == 1:
                    #logger.info(f"Skipping (filled) bar for {symbol} at {bar_event.datetime} (tickvol, volume, spread = 1)")
                    continue
            
            except StopIteration:
                logger.info(f"End of data for {symbol}")
                self.close_positions_end_of_data = True
                continue

            # ALWAYS emit base-timeframe bar
            # Update per-symbol tracking
            if symbol not in self.latest_index_timeframes:
                self.latest_index_timeframes[symbol] = {}
            self.latest_index_timeframes[symbol][self.base_timeframe] = bar_event.datetime
            
            if symbol in self.tradeable_symbols:
                events.append(bar_event)

            # Check and emit higher-timeframe bars
            for tf in self.timeframes_list:
                if tf == self.base_timeframe:
                    continue
                
                if self._base_tf_bar_creates_new_tf_bar(bar_event.datetime, tf, symbol):
                    self.latest_index_timeframes[symbol][tf] = bar_event.datetime
                    if symbol in self.tradeable_symbols:
                        events.append(self.get_latest_bar(symbol, tf))

        return events