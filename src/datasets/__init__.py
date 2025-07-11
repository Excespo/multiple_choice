from .manager import DatasetManager
from .base import BaseDataset, DatasetRegistry

import importlib
import os

current_dir = os.path.dirname(__file__)

for filename in os.listdir(current_dir):
    if (filename.endswith('.py') and 
        not filename.startswith('__') and 
        filename not in ['base.py', 'manager.py']):
        
        module_name = filename[:-3]  # Remove .py extension
        try:
            importlib.import_module(f'.{module_name}', package=__package__)
        except ImportError as e:
            print(f"Warning: Could not import dataset module {module_name}: {e}")

__all__ = ['DatasetManager', 'BaseDataset', 'DatasetRegistry'] 