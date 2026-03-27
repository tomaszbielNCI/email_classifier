"""
Base Strategy - Abstract interface for multi-label classification strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging


class BaseStrategy(ABC):
    """
    Abstract base class for multi-label classification strategies
    
    Implements Strategy pattern for different multi-label approaches
    """
    
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.results = {}
        self.is_trained = False
        
    @abstractmethod
    def train_models(self, X: pd.DataFrame, y2: pd.Series, 
                    y3: pd.Series, y4: pd.Series) -> Dict[str, Any]:
        """
        Train models according to strategy
        
        Args:
            X: Training features
            y2: Type 2 labels
            y3: Type 3 labels  
            y4: Type 4 labels
            
        Returns:
            Dictionary with training results
        """
        pass
    
    @abstractmethod
    def predict(self, x: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions according to strategy
        
        Args:
            x: Features for prediction
            
        Returns:
            Dictionary with predictions for each level
        """
        pass
    
    @abstractmethod
    def evaluate(self, x_test: pd.DataFrame, y2_test: pd.Series,
                y3_test: pd.Series, y4_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate strategy performance
        
        Args:
            x_test: Test features
            y2_test: Test Type 2 labels
            y3_test: Test Type 3 labels
            y4_test: Test Type 4 labels
            
        Returns:
            Dictionary with evaluation results
        """
        pass
    
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return self.__class__.__name__
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        if hasattr(self, '__doc__'):
            return self.__doc__.strip()
        return "No description available"
    
    @staticmethod
    def validate_inputs(x: pd.DataFrame, y2: pd.Series,
                      y3: pd.Series, y4: pd.Series) -> bool:
        """
        Validate input data
        
        Args:
            x: Features
            y2: Type 2 labels
            y3: Type 3 labels
            y4: Type 4 labels
            
        Returns:
            True if valid, False otherwise
        """
        if x is None or len(x) == 0:
            logging.error("Features DataFrame is empty")
            return False
            
        if y2 is None or len(y2) == 0:
            logging.error("Type 2 labels are empty")
            return False
            
        if y3 is None or len(y3) == 0:
            logging.error("Type 3 labels are empty")
            return False
            
        if y4 is None or len(y4) == 0:
            logging.error("Type 4 labels are empty")
            return False
            
        if len(x) != len(y2) or len(x) != len(y3) or len(x) != len(y4):
            logging.error("Features and labels have different lengths")
            return False
            
        return True
    
    def print_strategy_info(self) -> None:
        """Print strategy information"""
        print(f"\nStrategy: {self.get_strategy_name()}")
        print(f"Description: {self.get_strategy_description()}")
        print(f"Model Factory: {self.model_factory.__class__.__name__}")
        print(f"Trained: {self.is_trained}")
    
    def save_results(self, filepath: str) -> None:
        """Save strategy results"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"Strategy results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load strategy results"""
        import json
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        logging.info(f"Strategy results loaded from {filepath}")
        return self.results
