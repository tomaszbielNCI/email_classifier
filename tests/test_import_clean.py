"""
Quick test to verify BaseStrategy import works correctly
"""
from src.core.strategies.base_strategy import BaseStrategy
from src.core.strategies.chained_strategy import ChainedMultiLabelStrategy

print("BaseStrategy imported successfully")
print(f"ChainedMultiLabelStrategy inherits from BaseStrategy: {issubclass(ChainedMultiLabelStrategy, BaseStrategy)}")
print(f"BaseStrategy is abstract: {BaseStrategy.__bases__}")
