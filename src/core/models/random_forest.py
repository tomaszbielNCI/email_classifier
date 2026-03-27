"""
Random Forest Model - Concrete implementation of BaseModel
"""

from .base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import logging
from typing import Optional


class RandomForestModel(BaseModel):
    """
    Random Forest model implementation
    
    Uses ensemble of decision trees for classification
    """
    
    def __init__(self, random_state: int = 42, n_estimators: int = 100, 
                 max_depth: Optional[int] = None, n_jobs: int = -1):
        super().__init__(random_state)
        
        # Initialize Random Forest with parameters
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            n_jobs=n_jobs,
            class_weight='balanced'
        )
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        
        logging.info(f"RandomForest initialized with n_estimators={n_estimators}, "
                   f"max_depth={max_depth}, n_jobs={n_jobs}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the Random Forest model
        
        Args:
            X: Training features
            y: Training labels
        """
        logging.info(f"Training RandomForest on {X.shape[0]} samples, {X.shape[1]} features")
        
        self.model.fit(X, y)
        self.is_trained = True
        
        logging.info(f"RandomForest trained successfully")
        if hasattr(X, 'columns'):
            logging.info(f"Feature importance: {dict(zip(X.columns, self.model.feature_importances_))}")
        else:
            logging.info(f"Feature importance: {self.model.feature_importances_}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using Random Forest
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        logging.info(f"RandomForest predictions made for {len(predictions)} samples")
        return predictions
    
    def print_results(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """
        Print comprehensive evaluation results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n🌲 **Random Forest Results:**")
        print(f"   Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print(f"\n📊 **Classification Report:**")
        print(classification_report(y_true, y_pred))
        
        # Confusion matrix
        print(f"\n🔢 **Confusion Matrix:**")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print(f"\n🎯 **Top 10 Important Features:**")
            feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            importance_pairs = sorted(zip(feature_names, self.model.feature_importances_), 
                                 key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(importance_pairs[:10]):
                print(f"   {i+1:2d}. {feature}: {importance:.4f}")
    
    def get_feature_importance(self) -> dict:
        """Get feature importance as dictionary"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip([f"feature_{i}" for i in range(len(self.model.feature_importances_))], 
                          self.model.feature_importances_))
        return {}
    
    def get_oob_score(self) -> float:
        """Get Out-of-Bag score"""
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        return 0.0
