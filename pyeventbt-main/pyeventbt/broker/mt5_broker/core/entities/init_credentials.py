"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel

class InitCredentials(BaseModel):
    path: str
    login: int
    password: str
    server: str
    timeout: int
    portable: bool
