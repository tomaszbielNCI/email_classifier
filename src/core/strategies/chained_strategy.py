"""
Chained Multi-Outputs Strategy - Design Decision 1
"""

from .base_strategy import BaseStrategy
from ..models.model_factory import ModelFactory
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
import logging


class ChainedMultiLabelStrategy(BaseStrategy):
    """
    Chained Multi-Outputs Strategy Implementation
    
    Design Decision 1: One model instance assesses:
    1. Type 2
    2. Type 2 + Type 3  
    3. Type 2 + Type 3 + Type 4
    """
    
    def __init__(self, model_factory):
        super().__init__(model_factory)
        self.model_type2 = None
        self.model_type2_3 = None
        self.model_type2_3_4 = None
        
    def train_models(self, X: pd.DataFrame, y2: pd.Series, 
                    y3: pd.Series, y4: pd.Series) -> Dict[str, Any]:
        """
        Train chained multi-output models
        
        Args:
            X: Training features
            y2: Type 2 labels
            y3: Type 3 labels
            y4: Type 4 labels
            
        Returns:
            Dictionary with training results
        """
        logging.info("Training Chained Multi-Outputs Strategy")
        self.print_strategy_info()
        
        if not self.validate_inputs(X, y2, y3, y4):
            raise ValueError("Invalid input data")
        
        # Create combined labels for chaining
        y_type2_3 = self._combine_labels(y2, y3)
        y_type2_3_4 = self._combine_labels(y2, y3, y4)
        
        # Train three models with same features but different targets
        logging.info("Training Type 2 model...")
        self.model_type2 = self.model_factory.create_model("random_forest")
        self.model_type2.train(X, y2)
        
        logging.info("Training Type 2+3 model...")
        self.model_type2_3 = self.model_factory.create_model("random_forest")
        self.model_type2_3.train(X, y_type2_3)
        
        logging.info("Training Type 2+3+4 model...")
        self.model_type2_3_4 = self.model_factory.create_model("random_forest")
        self.model_type2_3_4.train(X, y_type2_3_4)
        
        self.is_trained = True
        
        # Store training results
        self.results = {
            'strategy': 'chained_multi_outputs',
            'models_trained': 3,
            'training_samples': len(X),
            'features': X.shape[1],
            'type2_classes': len(np.unique(y2)),
            'type2_3_classes': len(np.unique(y_type2_3)),
            'type2_3_4_classes': len(np.unique(y_type2_3_4))
        }
        
        logging.info("Chained Multi-Outputs training completed")
        return self.results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using chained approach
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary with predictions for each level
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        logging.info(f"Making chained predictions for {len(X)} samples")
        
        # Make predictions with all three models
        pred_type2 = self.model_type2.predict(X)
        pred_type2_3 = self.model_type2_3.predict(X)
        pred_type2_3_4 = self.model_type2_3_4.predict(X)
        
        # Extract individual predictions from combined ones
        pred_type3 = self._extract_individual_labels(pred_type2_3, pred_type2)
        pred_type4 = self._extract_individual_labels(pred_type2_3_4, pred_type2, pred_type3)
        
        predictions = {
            'type2': pred_type2,
            'type2_3': pred_type2_3,
            'type2_3_4': pred_type2_3_4,
            'type3': pred_type3,
            'type4': pred_type4
        }
        
        logging.info("Chained predictions completed")
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y2_test: pd.Series,
                y3_test: pd.Series, y4_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate chained strategy performance
        
        Args:
            X_test: Test features
            y2_test: Test Type 2 labels
            y3_test: Test Type 3 labels
            y4_test: Test Type 4 labels
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        logging.info("Evaluating Chained Multi-Outputs Strategy")
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate individual accuracies
        from sklearn.metrics import accuracy_score
        
        acc_type2 = accuracy_score(y2_test, predictions['type2'])
        acc_type2_3 = accuracy_score(
            self._combine_labels(y2_test, y3_test), 
            predictions['type2_3']
        )
        acc_type2_3_4 = accuracy_score(
            self._combine_labels(y2_test, y3_test, y4_test),
            predictions['type2_3_4']
        )
        
        # Calculate chain accuracy (all must be correct)
        correct_all = (
            (y2_test == predictions['type2']) &
            (y3_test == predictions['type3']) &
            (y4_test == predictions['type4'])
        )
        acc_chain_all = np.mean(correct_all)
        
        # Calculate propagated accuracies
        acc_type3_given_type2 = np.mean(
            (y2_test == predictions['type2']) & 
            (y3_test == predictions['type3'])
        )
        
        acc_type4_given_type2_3 = np.mean(
            (y2_test == predictions['type2']) & 
            (y3_test == predictions['type3']) &
            (y4_test == predictions['type4'])
        )
        
        evaluation_results = {
            'strategy': 'chained_multi_outputs',
            'individual_accuracies': {
                'type2': acc_type2,
                'type3': acc_type3_given_type2,
                'type4': acc_type4_given_type2_3
            },
            'combined_accuracies': {
                'type2_3': acc_type2_3,
                'type2_3_4': acc_type2_3_4
            },
            'chain_accuracy': acc_chain_all,
            'test_samples': len(X_test),
            'detailed_metrics': self._calculate_detailed_metrics(
                y2_test, y3_test, y4_test, predictions
            )
        }
        
        logging.info(f"Chained evaluation completed: Chain Accuracy = {acc_chain_all:.4f}")
        return evaluation_results
    
    def _combine_labels(self, *labels) -> pd.Series:
        """Combine multiple labels into single series"""
        combined = []
        for i in range(len(labels[0])):
            label_parts = [str(label.iloc[i]) for label in labels]
            combined_label = '_'.join(label_parts)
            combined.append(combined_label)
        return pd.Series(combined)
    
    def _extract_individual_labels(self, combined_labels: np.ndarray, 
                                  *previous_labels) -> np.ndarray:
        """Extract individual labels from combined predictions"""
        individual_labels = []
        for combined_label in combined_labels:
            parts = str(combined_label).split('_')
            if len(parts) >= 2:
                individual_labels.append(parts[1])  # Second part
            else:
                individual_labels.append(parts[0])  # Fallback
        return np.array(individual_labels)
    
    def _calculate_detailed_metrics(self, y2_true: pd.Series, y3_true: pd.Series,
                                y4_true: pd.Series, predictions: Dict) -> Dict:
        """Calculate detailed evaluation metrics"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        detailed_metrics = {}
        
        # Type 2 metrics
        detailed_metrics['type2'] = {
            'classification_report': classification_report(y2_true, predictions['type2'], output_dict=True),
            'confusion_matrix': confusion_matrix(y2_true, predictions['type2']).tolist()
        }
        
        # Type 3 metrics (conditional on Type 2)
        type2_correct = y2_true == predictions['type2']
        type3_metrics = classification_report(
            y3_true[type2_correct], 
            predictions['type3'][type2_correct], 
            output_dict=True
        )
        detailed_metrics['type3_given_type2'] = {
            'classification_report': type3_metrics,
            'samples': type2_correct.sum()
        }
        
        # Type 4 metrics (conditional on Type 2+3)
        type2_3_correct = (
            (y2_true == predictions['type2']) & 
            (y3_true == predictions['type3'])
        )
        type4_metrics = classification_report(
            y4_true[type2_3_correct],
            predictions['type4'][type2_3_correct],
            output_dict=True
        )
        detailed_metrics['type4_given_type2_3'] = {
            'classification_report': type4_metrics,
            'samples': type2_3_correct.sum()
        }
        
        return detailed_metrics
    
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return "ChainedMultiLabelStrategy"
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return ("Design Decision 1: Chained Multi-Outputs Strategy. "
                "One model instance assesses Type 2, Type 2+3, and Type 2+3+4. "
                "Each level depends on previous level accuracy.")
