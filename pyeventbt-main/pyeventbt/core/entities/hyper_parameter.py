"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel, Field
from typing import Any
from .variable import Variable

class HyperParameterRange(BaseModel):    
    minimum: float | int
    maximum: float | int
    step: float | int = 1

class HyperParameterValues(BaseModel):
    values: list[float | int]

class HyperParameter(Variable):
    range: HyperParameterRange | HyperParameterValues
    