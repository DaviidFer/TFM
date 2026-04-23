"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from ..core.interfaces.mt5_broker_interface import IPlatform, IAccountInfo, ITerminalInfo, ISymbol
from ..core.entities.init_credentials import InitCredentials
from ..core.entities.account_info import AccountInfo
from ..core.entities.terminal_info import TerminalInfo
from ..core.entities.symbol_info import SymbolInfo
from ..core.entities.tick import Tick
from ..shared.shared_data import SharedData
from typing import Tuple
import re


class PlatformConnector(IPlatform):

    @staticmethod
    def initialize(path:str='', login:int=0, password:str='', server:int=0, timeout:int=60000, portable:bool=False) -> bool:
        """
        Establish a connection with the MetaTrader 5 terminal.
        
        :param path: The path to the MetaTrader5 terminal executable.
        :type path: str
        :param login: The login ID of the trading account.
        :type login: int
        :param password: The password for the trading account.
        :type password: str
        :param server: The server name for the trading account.
        :type server: int
        :param timeout: The timeout in seconds.
        :type timeout: int
        :param portable: A boolean indicating whether to use the terminal in portable mode.
        :type portable: bool

        :return: A boolean indicating whether the connection was successfully established.
        :rtype: bool
        """
        creds = {
            "path": path,
            "login": login,
            "password": password,
            "server": server,
            "timeout": timeout,
            "portable": portable
            }

        # Simulate a platform launch by returning true if the credentials are valid (can be saved to an InitCredentials object) or false if they are not
        try:
            SharedData.credentials = InitCredentials(**creds)
            SharedData.terminal_info.connected = True
            SharedData.last_error_code = (1, 'success')

            # Updates the login info on the Account Data
            SharedData.account_info.login = SharedData.credentials.login
            SharedData.account_info.server = SharedData.credentials.server
            return True
        
        except Exception as e:
            SharedData.last_error_code = (-6, 'Terminal: Authorization failed')
            print(e)
            return False

    @staticmethod
    def login(login:int=0, password:str='', server:int=0, timeout:int=60000) -> bool:
        """
        Connect to a trading account using specified parameters.

        Args:
            login (int): The user's login ID.
            password (str): The user's password.
            server (int): The server to connect to.
            timeout (int): The timeout for the login request, in milliseconds.

        Returns:
            bool: True in case of a successful connection to the trade account, otherwise False.
        """
        creds = {
            "path": '',
            "login": login,
            "password": password,
            "server": server,
            "timeout": timeout,
            "portable": False
            }
        
        # Simulate a platform login by returning true if the credentials are valid or false if they are not
        try:
            SharedData.credentials = InitCredentials(**creds)
            SharedData.terminal_info.connected = True
            SharedData.last_error_code = (1, 'success')

            # Updates the login info on the Account Data
            SharedData.account_info.login = SharedData.credentials.login
            SharedData.account_info.server = SharedData.credentials.server
            return True
        
        except Exception as e:
            SharedData.last_error_code = (-6, 'Terminal: Authorization failed')
            print(e)
            return False

    @staticmethod
    def shutdown() -> None:
        """
        Close the previously established connection to the MetaTrader 5 terminal.
        """
        SharedData.last_error_code = (1, 'success')
        SharedData.terminal_info.connected = False

    @staticmethod
    def version() -> tuple:
        """
        Return the MetaTrader 5 terminal version.

        Returns:
        tuple: A tuple of (build: int, version: int, date: str).
        """
        SharedData.last_error_code = (1, 'success')
        build = SharedData.terminal_info.build
        return(500, build, '20 Oct 2023')

    @staticmethod
    def last_error() -> tuple:
        """
        Simulate a platform last error by returning a tuple of (code: int, message: str).

        :return: A tuple of (code: int, message: str) representing the last error that occurred on the platform.
        """
        return SharedData.last_error_code


class AccountInfoConnector(IAccountInfo):
    
    @staticmethod
    def account_info() -> AccountInfo:
        """
        Returns the account information object.

        :return: AccountInfo object containing information about the account.
        """
        return SharedData.account_info


class TerminalInfoConnector(ITerminalInfo):

    @staticmethod
    def terminal_info() -> TerminalInfo:
        """
        Returns the TerminalInfo object containing information about the current terminal.

        :return: TerminalInfo object containing information about the current terminal.
        """
        return SharedData.terminal_info


class SymbolConnector(ISymbol):

    @staticmethod
    def symbols_total() -> int:
        """
        Returns the total number of symbols available in the shared data.

        Returns:
        int: The total number of symbols available in the shared data.
        """
        result = len(SharedData.symbol_info)
        SharedData.last_error_code = (1, 'Success')
        return result
    
    @staticmethod
    def symbols_get(group: str = "*") -> Tuple[SymbolInfo, ...]:
        """
        Returns a tuple of SymbolInfo objects for all symbols that match the specified group.

        :param group: A string containing one or more conditions separated by commas. Each condition can be a symbol name or a pattern that may include the wildcard character '*'. If a condition starts with '!', it is treated as an exclusion condition.
        :return: A tuple of SymbolInfo objects for all symbols that match the specified group.
        """
        
        # If the group is NOT a string, return false
        if not isinstance(group, str):
            SharedData.last_error_code = (-2, 'Invalid 1st unnamed argument')
            return False

        # Split the group string into conditions
        conditions = group.split(',')

        # Start with all symbols
        filtered_symbols = list(SharedData.symbol_info.keys())

        # Apply inclusion conditions
        for condition in conditions:
            if '!' not in condition:
                pattern = re.escape(condition).replace('\\*', '.*')
                filtered_symbols = [symbol for symbol in filtered_symbols if re.fullmatch(pattern, symbol)]
        
        # Apply exclusion conditions
        for condition in conditions:
            if '!' in condition:
                condition = condition.replace('!', '')
                pattern = re.escape(condition).replace('\\*', '.*')
                filtered_symbols = [symbol for symbol in filtered_symbols if not re.fullmatch(pattern, symbol)]
    
        # Return the filtered symbol names and their information
        result = tuple(SharedData.symbol_info[symbol] for symbol in filtered_symbols)
        
        # Set the last error code
        SharedData.last_error_code = (1, 'Success')
        
        return result
    
    @staticmethod
    def symbol_info(symbol: str) -> SymbolInfo:
        """
        Returns a SymbolInfo object for the specified symbol.

        :param symbol: The symbol name.
        :return: A SymbolInfo object for the specified symbol.
        """
        # If the symbol is NOT a string, return None
        if not isinstance(symbol, str):
            SharedData.last_error_code = (-2, 'Invalid arguments')
            return None
        
        # If the symbol is NOT in the shared data, return None
        if symbol not in SharedData.symbol_info:
            SharedData.last_error_code = (-4, 'Terminal: Not found')
            return None
        
        # Return the symbol information
        result = SharedData.symbol_info[symbol]
        
        # Set the last error code
        SharedData.last_error_code = (1, 'Success')
        
        return result

    @staticmethod
    def symbol_info_tick(symbol: str) -> Tick:
        """
        NOT FULLY IMPLEMENTED.
        Returns a Tick object for the specified symbol. Needs to be selected in MKT watch but not necessarily visible.

        :param symbol: The symbol name.
        :return: A Tick object for the specified symbol.
        """
        # If the symbol is NOT a string, return None
        if not isinstance(symbol, str):
            SharedData.last_error_code = (-2, 'Invalid arguments')
            return None
        
        # If the symbol is NOT in the shared data, return None. TAMPOCO LO DEVUELVE SI EL SIMBOL NO ESTÁ SELECTED EN MKT WATCH
        if symbol not in SharedData.symbol_info or not SharedData.symbol_info[symbol].select:
            SharedData.last_error_code = (-4, 'Terminal: Not found')
            return None
        
        # # Return the symbol information
        # TODO: RETURN THE TICK OBJECT. So it needs a DATA HANDLER to populate the tick object
        # We will never call this method from MT5 API directly during backtest. We'll use the DP get_latest_tick method, and it's the DP who interacts with the 
        # MT5 API (wrapped or real) if needed.
        
        # Set the last error code
        SharedData.last_error_code = (1, 'Success')
        
        #return result
    
    @staticmethod
    def symbol_select(symbol: str, enable: bool = True) -> bool:
        """
        Selects or deselects a symbol and sets its visibility in the shared data.
        TODO: A symbol cannot be removed if open charts with this symbol are currently present or positions are opened on it.

        Args:
            symbol (str): The symbol to select or deselect.
            enable (bool, optional): Whether to select or deselect the symbol. Defaults to True.

        Returns:
            bool: True if the symbol was successfully selected or deselected, False otherwise.
        """
        
        # If the symbol is NOT a string, return None
        if not isinstance(symbol, str):
            SharedData.last_error_code = (-2, 'Invalid arguments')
            return None
        
        # If the symbol is NOT in the shared data, return False.
        if symbol not in SharedData.symbol_info:
            SharedData.last_error_code = (-1, 'Terminal: Call failed')
            return False
        
        # Set the symbol select and visible value
        if not enable:
            # TODO: A symbol cannot be removed if open charts with this symbol are currently present or positions are opened on it.
            SharedData.symbol_info[symbol].select = enable
            SharedData.symbol_info[symbol].visible = enable
        else:
            SharedData.symbol_info[symbol].select = enable
            SharedData.symbol_info[symbol].visible = enable
        
        # Set the last error code
        SharedData.last_error_code = (1, 'Success')
        return True
    

# There are unimplemented classes in mt5_broker_interface.py for the current simulator connector