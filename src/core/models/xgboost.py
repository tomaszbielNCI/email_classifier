"""
XGBoost Model - Concrete implementation of BaseModel
"""

from .base import BaseModel
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import logging


class XGBoostModel(BaseModel):
    """
    XGBoost model implementation
    
    Uses optimized gradient boosting for classification
    """
    
    def __init__(self, random_state: int = 42, n_estimators: int = 100,
                 learning_rate: float = 0.1, max_depth: int = 6,
                 subsample: float = 0.8, colsample_bytree: float = 0.8):
        super().__init__(random_state)
        
        # Initialize XGBoost with parameters
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            tree_method='hist'
        )
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        logging.info(f"XGBoost initialized with n_estimators={n_estimators}, "
                   f"learning_rate={learning_rate}, max_depth={max_depth}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the XGBoost model
        
        Args:
            X: Training features
            y: Training labels
        """
        logging.info(f"Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Handle label encoding for XGBoost
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        self.model.fit(X, y_encoded)
        self.label_encoder = le
        self.is_trained = True
        
        logging.info(f"XGBoost trained successfully")
        logging.info(f"Best iteration: {self.model.best_iteration}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using XGBoost
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array (decoded to original labels)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions_encoded = self.model.predict(X)
        
        # Decode predictions back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        logging.info(f"XGBoost predictions made for {len(predictions)} samples")
        return predictions
    
    def print_results(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """
        Print comprehensive evaluation results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n🚀 **XGBoost Results:**")
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
    
    def get_training_score(self) -> dict:
        """Get training evaluation scores"""
        if hasattr(self.model, 'best_score'):
            return {
                'best_score': self.model.best_score,
                'best_iteration': self.model.best_iteration
            }
        return {}
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
