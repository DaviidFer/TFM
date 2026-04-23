"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from typing import Dict, Any

class ParameterStore:

    __paramters: Dict[str, Any] = {}

    def add_parameter(self, parameter_name: str, value: Any):
        self.__paramters.setdefault(parameter_name, value)

    def get_parameter(self, parameter_name: str):

        return self.__paramters[parameter_name]
    
    def set_parameter(self, parameter_name: str, value: Any):

        try:
            self.__paramters[parameter_name] = value
        except KeyError:
            pass

