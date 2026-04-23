"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from enum import Enum

class AccountCurrencies(str, Enum):
    EUR = 'EUR'
    USD = 'USD'
    GBP = 'GBP'