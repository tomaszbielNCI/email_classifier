"""
Quick test to verify BaseStrategy import works correctly
"""
import sys
import os

# Add src directory to Python path
project_root = os.path.dirname(__file__)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from core.strategies.base_strategy import BaseStrategy
from core.strategies.chained_strategy import ChainedMultiLabelStrategy

print("BaseStrategy imported successfully")
print(f"ChainedMultiLabelStrategy inherits from BaseStrategy: {issubclass(ChainedMultiLabelStrategy, BaseStrategy)}")
print(f"BaseStrategy is abstract: {BaseStrategy.__bases__}")
