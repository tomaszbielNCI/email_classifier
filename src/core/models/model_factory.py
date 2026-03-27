"""
Model Factory - Factory Method Pattern Implementation
"""

from .base import BaseModel, MultiLabelModel
from .random_forest import RandomForestModel
from .xgboost import XGBoostModel
from .extended_models import EXTENDED_MODEL_REGISTRY, get_extended_model_info
from typing import Dict, Any
import logging


class ModelFactory:
    """
    Factory for creating ML models
    
    Implements Factory Method pattern for dynamic model creation
    """
    
    @staticmethod
    def create_model(model_type: str, random_state: int = 42, **kwargs) -> BaseModel:
        """
        Create a model instance based on type
        
        Args:
            model_type: Type of model to create
            random_state: Random state for reproducibility
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is unknown
        """
        model_type = model_type.lower()
        
        # Core models
        if model_type == "random_forest":
            logging.info(f"Creating RandomForest model with params: {kwargs}")
            return RandomForestModel(random_state=random_state, **kwargs)
        elif model_type == "xgboost":
            logging.info(f"Creating XGBoost model with params: {kwargs}")
            return XGBoostModel(random_state=random_state, **kwargs)
        
        # Extended models
        elif model_type in EXTENDED_MODEL_REGISTRY:
            logging.info(f"Creating extended model '{model_type}' with params: {kwargs}")
            model_class = EXTENDED_MODEL_REGISTRY[model_type]
            return model_class(random_state=random_state, **kwargs)
        
        # Placeholder models (not yet implemented)
        elif model_type == "lightgbm":
            raise NotImplementedError("LightGBM model not yet implemented")
        elif model_type == "catboost":
            raise NotImplementedError("CatBoost model not yet implemented")
        else:
            available_models = ModelFactory.get_available_model_names()
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available models: {available_models}")
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models
        
        Returns:
            Dictionary with model information
        """
        # Core models
        core_models = {
            "random_forest": {
                "name": "Random Forest",
                "description": "Ensemble of decision trees",
                "parameters": ["n_estimators", "max_depth", "n_jobs"],
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "n_jobs": -1
                },
                "category": "core"
            },
            "xgboost": {
                "name": "XGBoost",
                "description": "Optimized gradient boosting",
                "parameters": ["n_estimators", "learning_rate", "max_depth"],
                "default_params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6
                },
                "category": "core"
            }
        }
        
        # Add extended models
        extended_models = get_extended_model_info()
        for name, info in extended_models.items():
            info["category"] = "extended"
        
        # Combine all models
        all_models = {**core_models, **extended_models}
        
        return all_models
    
    @staticmethod
    def get_available_model_names() -> list:
        """
        Get list of available model names
        
        Returns:
            List of model names
        """
        models_info = ModelFactory.get_available_models()
        return list(models_info.keys())
    
    @staticmethod
    def get_models_by_category() -> Dict[str, list]:
        """
        Get models grouped by category
        
        Returns:
            Dictionary with categories as keys and model lists as values
        """
        models_info = ModelFactory.get_available_models()
        categories = {}
        
        for name, info in models_info.items():
            category = info.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append(name)
        
        return categories
    
    @staticmethod
    def get_core_models() -> list:
        """Get list of core models"""
        return ["random_forest", "xgboost"]
    
    @staticmethod
    def get_extended_models() -> list:
        """Get list of extended models"""
        return list(EXTENDED_MODEL_REGISTRY.keys())
    
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
        
        valid_params = available[model_type].get("parameters", [])
        for param in params:
            if param not in valid_params:
                logging.warning(f"Invalid parameter '{param}' for model '{model_type}'")
                return False
        
        return True
