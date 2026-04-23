"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""


from . import BaseConfig


class Mt5PlatformConfig(BaseConfig):
    path: str
    """path to the MT5 platform, e.g. C:/Program Files/MT5/terminal64.exe"""
    
    login: int
    """login for the connection to the MT5 platform"""
    
    password: str
    """password for the connection to the MT5 platform"""
    
    server: str
    """server to connect to the MT5 platform"""
    
    timeout: int
    """timeout for the connection to the MT5 platform"""
    
    portable: bool
    """wether the passed MT5 is portable or not"""