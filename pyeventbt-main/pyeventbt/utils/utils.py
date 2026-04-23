"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pyeventbt.data_provider.core.interfaces.data_provider_interface import IDataProvider
from decimal import Decimal
import platform
from enum import Enum
import os
from functools import lru_cache
import logging

logger = logging.getLogger("PyEventBT") # Using the root logger

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize(string: str, color: TerminalColors = TerminalColors.OKBLUE):
    return f"{color}{string}{TerminalColors.ENDC}"


class LoggerColorFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[92;20m"
    cian = "\x1b[96;20m"
    yellow = "\x1b[93;20m"
    red = "\x1b[91;1;4m"    # Bold and underlined
    reset = "\x1b[0m"
    
    format = "%(asctime)s - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: cian + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class Utils():
    
    def __init__(self) -> None:
        pass

    #BUY(0), SELL(1), BUY_LIMIT(2), SELL_LIMIT(3), BUY_STOP(4), SELL_STOP(5)
    @staticmethod
    @lru_cache(maxsize=10)
    def order_type_str_to_int(order_type:str) -> int:
        if order_type == "BUY":
            return 0
        elif order_type == "SELL":
            return 1
        elif order_type == "BUY_LIMIT":
            return 2
        elif order_type == "SELL_LIMIT":
            return 3
        elif order_type == "BUY_STOP":
            return 4
        elif order_type == "SELL_STOP":
            return 5
        elif order_type == "BUY_STOP_LIMIT":
            return 6
        elif order_type == "SELL_STOP_LIMIT":
            return 7
        elif order_type == "CLOSE_BY":
            return 8
        else:
            return -1
    
    @staticmethod
    @lru_cache(maxsize=10)
    def order_type_int_to_str(order_type:int) -> str:
        if order_type == 0:
            return "BUY"
        elif order_type == 1:
            return "SELL"
        elif order_type == 2:
            return "BUY_LIMIT"
        elif order_type == 3:
            return "SELL_LIMIT"
        elif order_type == 4:
            return "BUY_STOP"
        elif order_type == 5:
            return "SELL_STOP"
        elif order_type == 6:
            return "BUY_STOP_LIMIT"
        elif order_type == 7:
            return "SELL_STOP_LIMIT"
        elif order_type == 8:
            return "CLOSE_BY"
        else:
            return "UNKNOWN"

    @staticmethod
    def check_new_m1_bar_creates_new_tf_bar(latest_bar_time: pd.Timestamp, timeframe: str) -> bool:

        """
        Determines whether a new bar in the timeframe passed is created based
        on the latest bar time and timeframe.

        Args:
            latest_bar_time (pd.Timestamp): The timestamp of the latest m1 bar.
            timeframe (str): The timeframe to check for a new bar.

        Returns:
            bool: True if a new bar is created, False otherwise.

        Raises:
            ValueError: If the timeframe is invalid.
        """

        # Adjust latest_bar_time to be able to detect a new bar in m1.
        # tenemos que ver si se ha creado la vela 00:04 para un tf de 5min. Es decir, la vela del minuto anterior
        # del múltiplo del timeframe. La vela de las 00:004 creada implica 5 velas creadas (00, 01, 02, 03 y 04)
        # Si la última vela es 10:44, ahí se habrá creado la vela de M15 de las 10:30.
        # Lo podriamos hacer con el _is_new_bar pasandole el timestamp + timedelta de 1 min
        # Timedelta aquí NO es un lookahead bias, es para comprobar si es múltiple del timeframe objetivo.
        # Nunca accedemos a datos de esa vela
                            
        latest_bar_time = latest_bar_time + pd.Timedelta('1min')


        # Get the timeframe in seconds
        timeframe_seconds = {'1min': 60, '5min': 300, '15min': 900, '30min': 1800,
                            '1H': 3600, '4H': 14400, '1D': 86400, 'D': 86400, 'B': 86400}.get(timeframe)
        
        if timeframe_seconds is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Convert latest_bar_time to seconds
        if latest_bar_time.unit == 'ns':
            latest_bar_time_seconds = latest_bar_time._value / 1e9
        elif latest_bar_time.unit == 'us':
            latest_bar_time_seconds = latest_bar_time._value / 1e6
        elif latest_bar_time.unit == 'ms':
            latest_bar_time_seconds = latest_bar_time._value / 1e3
        else:
            latest_bar_time_seconds = latest_bar_time._value

        # Check if the latest_bar_time_seconds is a multiple of timeframe_seconds
        return latest_bar_time_seconds % timeframe_seconds == 0

    @staticmethod
    def convert_currency_amount_to_another_currency(amount: Decimal, from_ccy: str, to_ccy: str, data_provider: IDataProvider) -> Decimal:
        """
        Converts the given amount from one currency to another.

        Args:
            amount (Decimal): The amount to be converted.
            from_ccy (str): The currency code of the source currency.
            to_ccy (str): The currency code of the target currency.

        Returns:
            Decimal: The converted amount.

        Raises:
            Exception: If the symbol is not available in the MT5 platform.
        """
        # Convert the currencies to uppercase
        from_ccy = from_ccy.upper()
        to_ccy = to_ccy.upper()

        # If the currencies are the same, return the amount
        if from_ccy == to_ccy:
            return amount
        
        all_fx_symbol = ("AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD", "EURCAD",
                        "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
                        "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "USDSEK", "USDNOK", "USDMXN", "EURMXN")

        # Buscamos el símbolo que relaciona nuestra divisa origen con nuestra divisa destino (list comprehension)
        fx_symbol = [symbol for symbol in all_fx_symbol if from_ccy in symbol and to_ccy in symbol][0]
        fx_symbol_base = fx_symbol[:3]

        # Get the conversion rate
        last_price = data_provider.DATA_PROVIDER.get_latest_bid(fx_symbol)

        # Convert the amount to the new currency and return it
        converted_amount = amount / last_price if fx_symbol_base == to_ccy else amount * last_price
        return converted_amount
    
    @staticmethod
    def get_currency_conversion_multiplier_cfd(from_ccy: str, to_ccy: str, data_provider: IDataProvider) -> Decimal:
        """
        Converts the given amount from one currency to another.

        Args:
            amount (Decimal): The amount to be converted.
            from_ccy (str): The currency code of the source currency.
            to_ccy (str): The currency code of the target currency.

        Returns:
            Decimal: The converted amount.

        Raises:
            Exception: If the symbol is not available in the MT5 platform.
        """
        # Convert the currencies to uppercase
        from_ccy = from_ccy.upper()
        to_ccy = to_ccy.upper()

        # If the currencies are the same, return the amount
        if from_ccy == to_ccy:
            return Decimal(1)
        
        all_fx_symbol = ("AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD", "EURCAD",
                        "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
                        "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY", "USDSEK", "USDNOK", "USDMXN", "EURMXN")

        # Buscamos el símbolo que relaciona nuestra divisa origen con nuestra divisa destino (list comprehension)
        fx_symbol = [symbol for symbol in all_fx_symbol if from_ccy in symbol and to_ccy in symbol][0]
        fx_symbol_base = fx_symbol[:3]

        # Get the conversion rate
        last_price = data_provider.DATA_PROVIDER.get_latest_bid(fx_symbol)

        # Get the adecuate multiplier for currency conversion
        return Decimal(1) / last_price if fx_symbol_base == to_ccy else last_price
    
    @staticmethod
    def get_fx_futures_suffix(symbol: str) -> tuple[str]:
        """
        This function will return the closest current contract that may be in place, and the next contract for use in case the first one is not available.
        
        The following logic is based in that we will use:
            - March contract (H) from starting of december until starting of march
            - June contract (M) from starting of march until starting of june
            - September contract (U) from starting of june until starting of september
            - December contract (Z) from starting of september until starting of december
        
        This is based on the fact that the contracts are usually traded 3 months before the actual month.
        """
        # Get the current month
        current_month = datetime.now().month

        # Define the suffixes for the current and next contract: ie in month is 1 (jan), can only be H, but if month is 3 (mar), can be H or M
        if current_month in [1, 2, 3]:
            suffix = ('H', 'M')
        elif current_month in [4, 5, 6]:
            suffix = ('M', 'U')
        elif current_month in [7, 8, 9]:
            suffix = ('U', 'Z')
        elif current_month in [10, 11, 12]:
            suffix = ('Z', 'H')

        # Return the current and next contract
        return (f"{symbol}_{suffix[0]}", f"{symbol}_{suffix[1]}")

    @staticmethod
    def convert_currency_amount_to_another_currency_futures(amount: Decimal, from_ccy: str, to_ccy: str, data_provider: IDataProvider) -> Decimal:
        """
        Converts the given amount from one currency to another using futures contracts.
        """
        # Convert the currencies to uppercase
        from_ccy = from_ccy.upper()
        to_ccy = to_ccy.upper()

        # If the currencies are the same, return the amount
        if from_ccy == to_ccy:
            return amount
        
        # Working with futures and having the account in USD, one of the from_ccy or to_ccy must be USD
        if from_ccy != "USD" and to_ccy != "USD":
            raise Exception("One of the currencies must be USD to convert using futures contracts.")
        
        # Define the base currency to contract mapping.
        base_ccy_to_contract = {"AUD": "6A", "GBP": "6B", "CAD": "6C", "EUR": "6E", "JPY": "6J", "NZD": "6N", "CHF": "6S"}

        # Get the currency that is not USD and use it to get the correct contract
        if from_ccy != "USD":
            from_contract = base_ccy_to_contract[from_ccy]
        else:
            from_contract = base_ccy_to_contract[to_ccy]

        # Get the futures suffix for the current and next contract
        current_contract, next_contract = Utils.get_fx_futures_suffix(from_contract)

        # Now we'll test if we get any data from the current, and if not, we'll use the next contract
        last_tick = data_provider.DATA_PROVIDER.get_latest_tick(current_contract)
        last_price = last_tick.bid if last_tick is not None else data_provider.DATA_PROVIDER.get_latest_tick(next_contract).bid

        # Convert the amount to the new currency and return it
        converted_amount = amount / last_price if from_ccy == "USD" else amount * last_price

        return converted_amount


    @staticmethod
    def dateprint() -> str:
        """
        Returns the current date and time in the format "dd/mm/yyyy HH:MM:SS.sss".
        The timezone is the same as MT5 Server: "Asia/Nicosia" but with US DST.
        """
        date_time = datetime.now(ZoneInfo("America/New_York")) + pd.Timedelta(hours=7)   # Equivalent to Asia/Nicosia with US-DST timezone
        return date_time.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
    
    @staticmethod
    def cap_forecast(forecast: float) -> float:
        return min(20.0, max(-20.0, forecast))
    


def check_platform_compatibility(raise_exception: bool = True) -> bool:

    if platform.system().lower() != 'windows':
        message = f'PLATFORM_INCOMPATIBILTY: MT5 python package does not support {platform.system()} platform. It needs Windows for LIVE trading.'
        
        if raise_exception:
            raise Exception(message)
        else:
            logger.warning(message)
            return False
    
    return True

def print_percentage_bar(percentage, bar_length=50, additional_message: str = '', end='\r'):
    """
    Prints a percentage bar.

    Parameters:
    percentage (float): The percentage to display (0 to 100).
    bar_length (int): The length of the bar in characters.
    """
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")

    # Calculate the number of filled and empty slots in the bar
    filled_length = int(bar_length * (percentage / 100))
    empty_length = bar_length - filled_length

    # Create the bar
    bar = '█' * filled_length + '-' * empty_length

    # Print the bar with the percentage
    print(f"[{bar}] {percentage:.2f}% ({additional_message})", end=end)
