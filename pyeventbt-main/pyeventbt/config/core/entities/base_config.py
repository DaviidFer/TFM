"""
PyEventBT
Documentation: https://pyeventbt.com
GitHub: https://github.com/marticastany/pyeventbt

Author: Marti Castany
Copyright (c) 2025 Marti Castany
Licensed under the Apache License, Version 2.0
"""

from pydantic import BaseModel
import yaml

class BaseConfig(BaseModel):

    @classmethod
    def load_from_yaml(cls, file_path: str = 'config.yaml'):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return cls(**data)

    def save_to_yaml(self, file_path: str = 'config.yaml'):
        with open(file_path, 'w') as file:
            yaml.dump(self.model_dump(), file)