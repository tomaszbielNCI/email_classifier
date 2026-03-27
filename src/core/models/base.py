"""
Base Model - Abstract interface for all ML models
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union, Optional, Any
import logging


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models
    
    Implements the Template Method pattern for consistent model interface
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_trained = False
        self.model = None
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
                y: Union[pd.Series, np.ndarray]) -> None:
        """
        Train the model on the provided data
        
        Args:
            X: Training features
            y: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def print_results(self, y_true: Union[pd.Series, np.ndarray], 
                         y_pred: np.ndarray) -> None:
        """
        Print evaluation results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        pass
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name
    
    def get_model_params(self) -> dict:
        """Get model parameters"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def get_model_info(self) -> dict:
        """Get comprehensive model information"""
        return {
            'name': self.get_model_name(),
            'type': self.__class__.__bases__[0].__name__ if self.__class__.__bases__ else 'BaseModel',
            'description': getattr(self, '__doc__', '').strip() or 'No description available',
            'parameters': self.get_model_params(),
            'is_trained': self.is_trained
        }
    
    def set_model_params(self, **params) -> None:
        """Set model parameters"""
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        import joblib
        joblib.dump(self.model, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        import joblib
        self.model = joblib.load(filepath)
        self.is_trained = True
        logging.info(f"Model loaded from {filepath}")


class MultiLabelModel(BaseModel):
    """
    Base class for multi-label classification models
    """
    
    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        self.chain_accuracy = {}
        
    def calculate_chain_accuracy(self, y_true_dict: dict, y_pred_dict: dict) -> dict:
        """
        Calculate chain accuracy for multi-label classification
        
        Args:
            y_true_dict: Dictionary of true labels for each level
            y_pred_dict: Dictionary of predicted labels for each level
            
        Returns:
            Dictionary with chain accuracy metrics
        """
        chain_accuracy = {}
        
        # Type 2 accuracy
        if 'type2' in y_true_dict and 'type2' in y_pred_dict:
            acc_type2 = np.mean(y_true_dict['type2'] == y_pred_dict['type2'])
            chain_accuracy['type2'] = acc_type2
        
        # Type 2 + Type 3 accuracy (both must be correct)
        if ('type2' in y_true_dict and 'type3' in y_true_dict and 
            'type2' in y_pred_dict and 'type3' in y_pred_dict):
            correct_type2_3 = ((y_true_dict['type2'] == y_pred_dict['type2']) & 
                           (y_true_dict['type3'] == y_pred_dict['type3']))
            acc_type2_3 = np.mean(correct_type2_3)
            chain_accuracy['type2_3'] = acc_type2_3
        
        # Type 2 + Type 3 + Type 4 accuracy (all must be correct)
        if ('type2' in y_true_dict and 'type3' in y_true_dict and 'type4' in y_true_dict and
            'type2' in y_pred_dict and 'type3' in y_pred_dict and 'type4' in y_pred_dict):
            correct_all = ((y_true_dict['type2'] == y_pred_dict['type2']) & 
                        (y_true_dict['type3'] == y_pred_dict['type3']) &
                        (y_true_dict['type4'] == y_pred_dict['type4']))
            acc_all = np.mean(correct_all)
            chain_accuracy['type2_3_4'] = acc_all
        
        self.chain_accuracy = chain_accuracy
        return chain_accuracy
