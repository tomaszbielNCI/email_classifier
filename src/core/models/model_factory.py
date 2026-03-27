"""
Model Factory - Factory Method Pattern Implementation
"""

from .base import BaseModel, MultiLabelModel
from .random_forest import RandomForestModel
from .xgboost import XGBoostModel
from typing import Dict, Any
import logging


class ModelFactory:
    """
    Factory for creating ML models
    
    Implements Factory Method pattern for dynamic model creation
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance based on type
        
        Args:
            model_type: Type of model to create
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is unknown
        """
        model_type = model_type.lower()
        
        if model_type == "random_forest":
            logging.info(f"Creating RandomForest model with params: {kwargs}")
            return RandomForestModel(**kwargs)
        elif model_type == "xgboost":
            logging.info(f"Creating XGBoost model with params: {kwargs}")
            return XGBoostModel(**kwargs)
        elif model_type == "lightgbm":
            # TODO: Implement LightGBMModel
            raise NotImplementedError("LightGBM model not yet implemented")
        elif model_type == "logistic_regression":
            # TODO: Implement LogisticRegressionModel
            raise NotImplementedError("Logistic Regression model not yet implemented")
        else:
            available_models = ["random_forest", "xgboost"]
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available models: {available_models}")
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models
        
        Returns:
            Dictionary with model information
        """
        return {
            "random_forest": {
                "name": "Random Forest",
                "description": "Ensemble of decision trees",
                "parameters": ["n_estimators", "max_depth", "n_jobs"],
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "n_jobs": -1
                }
            },
            "xgboost": {
                "name": "XGBoost",
                "description": "Optimized gradient boosting",
                "parameters": ["n_estimators", "learning_rate", "max_depth"],
                "default_params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6
                }
            }
        }
    
    @staticmethod
    def validate_params(model_type: str, params: dict) -> bool:
        """
        Validate parameters for a specific model type
        
        Args:
            model_type: Type of model
            params: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        available = ModelFactory.get_available_models()
        if model_type not in available:
            return False
        
        valid_params = available[model_type]["parameters"]
        for param in params:
            if param not in valid_params:
                logging.warning(f"Invalid parameter '{param}' for model '{model_type}'")
                return False
        
        return True
