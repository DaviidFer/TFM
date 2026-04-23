"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pyeventbt.utils.utils import check_platform_compatibility

try:
    if check_platform_compatibility(raise_exception=False):
        import MetaTrader5 as mt5
    else:
        mt5 = None
except ImportError:
    mt5 = None

from pyeventbt.broker.mt5_broker.core.entities.order_send_result import OrderSendResult
from pyeventbt.utils.utils import Utils
from datetime import datetime
import os
import time
from dotenv import load_dotenv, find_dotenv
import logging
from pyeventbt.config import Mt5PlatformConfig

logger = logging.getLogger("pyeventbt")
account_logger = logging.getLogger("account_info")



class LiveMT5Broker():

    def __init__(self, symbol_list: list, config: Mt5PlatformConfig):
        """
        Initializes the platform connector object.
        """

        account_logger_console_handler = logging.StreamHandler()
        green = "\x1b[92;20m"
        reset = "\x1b[0m"
        special_format = logging.Formatter(f'{green}%(message)s{reset}')
        account_logger_console_handler.setFormatter(special_format)
        account_logger.addHandler(account_logger_console_handler)
        account_logger.setLevel(logging.INFO)
        self.config = config
        
        # Search for .env file and load it
        # load_dotenv(find_dotenv())
        
        # Initialize MT5 platform
        # self.initialize_platform()
        self.initialize_platformV2()
        
        # Check for LIVE account initialization
        self._live_account_warning()

        # Print account info
        self._print_account_info()

        # Check if algo trading is enabled
        self._check_algo_trading_enabled()

        # Añadimos los símbolos al MarketWatch
        self._add_symbols_to_marketwatch(symbol_list)


    def initialize_platform(self):
        """Initialize MT5 platform with credentials"""
        # Connect to MT5
        try:
            if mt5.initialize(
                path=os.getenv("MT5_PATH"),
                login=int(os.getenv("MT5_LOGIN")),
                password=os.getenv("MT5_PASSWORD"),
                server=os.getenv("MT5_SERVER"),
                timeout=int(os.getenv("MT5_TIMEOUT")),
                portable=eval(os.getenv("MT5_PORTABLE"))):
                version = mt5.version()
                logger.info(f"MT5 Platform successfully launched. Package: {mt5.__version__}, Terminal: {version[0]}, Build: {version[1]}, Released: {version[2]}")
            else:
                raise Exception(f"{Utils.dateprint()} - There was an error while initializing MT5 Plaform: {mt5.last_error()}")
        
        except Exception as e:
            logger.error(f"There was an error initializing MT5 platform: {mt5.last_error()}, exception: {e}")

    def initialize_platformV2(self):
        """Initialize MT5 platform with credentials"""
        # Connect to MT5
        try:
            if mt5.initialize(
                path=self.config.path,
                login=self.config.login,
                password=self.config.password,
                server=self.config.server,
                timeout=self.config.timeout,
                portable=self.config.portable
                ):

                version = mt5.version()
                logger.info(f"MT5 Platform successfully launched. Package: {mt5.__version__}, Terminal: {version[0]}, Build: {version[1]}, Released: {version[2]}")
            else:
                raise Exception(f"{Utils.dateprint()} - There was an error while initializing MT5 Plaform: {mt5.last_error()}")
        
        except Exception as e:
            logger.error(f"There was an error initializing MT5 platform: {mt5.last_error()}, exception: {e}")

    def _live_account_warning(self) -> None:
        """
        Displays a warning message if a real trading account is detected.
        Prompts the user to confirm if they want to continue.
        If the user chooses not to continue, the program is shut down.
        """
        # Recuperamos el objeto de tipo AccountInfo
        account_info = mt5.account_info()
        
        # Comprobar el tipo de cuenta que se ha lanzado
        if account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO:
            logger.info(f"DEMO account detected. No capital is at risk. Launching...")
            counter = 3
            while counter > 0:
                logger.info(f"TEST DEMO LAUNCH - Launching in {counter} seconds...")
                time.sleep(1)
                counter -= 1
        
        elif account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL:
            counter = 10
            while counter > 0:
                logger.warning(f"ALERT! REAL account detected. Capital is at RISK. Launching in {counter} seconds...")
                time.sleep(1)
                counter -= 1
        else:
            logger.info(f"CONTEST account detected.")

    def _print_account_info(self) -> None:
        """
        Prints the account information including account ID, trader name, broker, server, leverage, currency, and balance.
        """
        # Get account info
        acc_info = mt5.account_info()._asdict()

        details = [("Account ID", acc_info['login']),
                    ("Account holder", acc_info['name']),
                    ("Broker", acc_info['company']),
                    ("Server", acc_info['server']),
                    ("Leverage", acc_info['leverage']),
                    ("Account currency", acc_info['currency']),
                    ("Account balance", acc_info['balance'])]

        # Calculate the maximum content width
        max_length = 65
        max_content_width = max_length - 4  # Subtracting 4 for the borders and spaces

        # Define the top border with the title, and the bottom one
        title = "+------------ ACCOUNT DETAILS "
        title_padding = "-" * (max_length - len(title) - 1) + "+"
        top_border = title + title_padding
        bottom_border = "+" + "-" * (max_length - 2) + "+"
        
        # Print connection status and account details
        account_logger.info(f"\n{top_border}")
        for label, value in details:
            line = f"| - {label}: {value}"
            if len(line) > max_content_width:
                line = line[:max_content_width - 1] + '…'  # Truncate with ellipsis if necessary
            padded_line = line + ' ' * (max_content_width - len(line)) + '   |'
            account_logger.info(padded_line)
        account_logger.info(f"{bottom_border}\n")

    def _check_algo_trading_enabled(self) -> None:
        """
        Checks if algorithmic trading is enabled.

        Raises:
            Exception: If algorithmic trading is disabled.
        """
        # Vamos a comprobar que el trading algorítmico está activado
        if not mt5.terminal_info().trade_allowed:
            raise Exception(f"Algorithmic trading is disabled. Please enable it MANUALLY from the MT5 terminal settings.")

    def _add_symbols_to_marketwatch(self, symbols: list) -> None:
        """
        Adds symbols to the MarketWatch if they are not already visible.

        Args:
            symbols (list): List of symbols to be added.

        Returns:
            None
        """
        # 1) Check if the symbol is already visible in the MW
        # 2) If not, add it

        for symbol in symbols:
            if mt5.symbol_info(symbol) is None:
                logger.warning(f"Could not add {symbol} to MarketWatch: {mt5.last_error()}")
                continue
            
            if not mt5.symbol_info(symbol).visible:
                if not mt5.symbol_select(symbol, True):
                    logger.warning(f"Could not add {symbol} to MarketWatch: {mt5.last_error()}")
                else:
                    logger.info(f"Symbol {symbol} has been successfully added to the MarketWatch!")
            else:
                logger.info(f"Symbol {symbol} was already in the MarketWatch.")

    # Let's now create some methos for checking the connection status
    def is_connected(self) -> bool:
        """
        Checks if the platform is connected to the broker.

        Returns:
            bool: True if connected, False otherwise.
        """
        return mt5.terminal_info().connected
    
    def is_closed(self) -> bool:
        """
        Checks if the platform is closed.

        Returns:
            bool: True if closed, False otherwise.
        """
        terminal_info = mt5.terminal_info()

        if terminal_info is None:
            return True
        else:
            return False