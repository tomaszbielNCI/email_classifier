"""
Hierarchical Multi-Label Strategy - Design Decision 2
"""

from .base_strategy import BaseStrategy
from ..models.model_factory import ModelFactory
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
import logging


class HierarchicalMultiLabelStrategy(BaseStrategy):
    """
    Hierarchical Multi-Label Strategy Implementation
    
    Design Decision 2: Multiple model instances with data filtering
    - Type 2 model → filters data for Type 3 models
    - Type 3 models (one per Type 2 class) → filters data for Type 4 models
    - Type 4 models (one per Type 2+3 combination)
    """
    
    def __init__(self, model_factory):
        super().__init__(model_factory)
        self.type2_model = None
        self.type3_models = {}  # One model per Type 2 class
        self.type4_models = {}  # One model per Type 2+3 combination
        self.type2_classes = []
        self.type3_classes = []
        
    def train_models(self, X: pd.DataFrame, y2: pd.Series, 
                    y3: pd.Series, y4: pd.Series) -> Dict[str, Any]:
        """
        Train hierarchical multi-label models
        
        Args:
            X: Training features
            y2: Type 2 labels
            y3: Type 3 labels
            y4: Type 4 labels
            
        Returns:
            Dictionary with training results
        """
        logging.info("Training Hierarchical Multi-Label Strategy")
        self.print_strategy_info()
        
        if not self.validate_inputs(X, y2, y3, y4):
            raise ValueError("Invalid input data")
        
        # Store unique classes
        self.type2_classes = np.unique(y2)
        self.type3_classes = np.unique(y3)
        
        # Step 1: Train Type 2 model on all data
        logging.info("Training Type 2 model...")
        self.type2_model = self.model_factory.create_model("random_forest")
        self.type2_model.train(X, y2)
        
        # Step 2: Train Type 3 models - one per Type 2 class
        logging.info("Training Type 3 models (one per Type 2 class)...")
        for class_2 in self.type2_classes:
            # Filter data for this Type 2 class
            mask = y2 == class_2
            X_filtered = X[mask]
            y3_filtered = y3[mask]
            
            if len(X_filtered) > 1:  # Minimum samples for training
                logging.info(f"  Training Type 3 model for Type 2 class '{class_2}' "
                           f"({len(X_filtered)} samples)")
                
                model_type3 = self.model_factory.create_model("random_forest")
                model_type3.train(X_filtered, y3_filtered)
                self.type3_models[class_2] = model_type3
            else:
                logging.warning(f"  Skipping Type 3 model for class '{class_2}' "
                              f"- insufficient samples ({len(X_filtered)})")
        
        # Step 3: Train Type 4 models - one per Type 2+3 combination
        logging.info("Training Type 4 models (one per Type 2+3 combination)...")
        for class_2 in self.type2_classes:
            for class_3 in self.type3_classes:
                combination_key = f"{class_2}_{class_3}"
                
                # Filter data for this Type 2+3 combination
                mask = (y2 == class_2) & (y3 == class_3)
                X_filtered = X[mask]
                y4_filtered = y4[mask]
                
                if len(X_filtered) > 1:  # Minimum samples for training
                    logging.info(f"  Training Type 4 model for combination "
                               f"'{combination_key}' ({len(X_filtered)} samples)")
                    
                    model_type4 = self.model_factory.create_model("random_forest")
                    model_type4.train(X_filtered, y4_filtered)
                    self.type4_models[combination_key] = model_type4
                else:
                    logging.warning(f"  Skipping Type 4 model for combination "
                                  f"'{combination_key}' - insufficient samples "
                                  f"({len(X_filtered)})")
        
        self.is_trained = True
        
        # Store training Results
        self.results = {
            'strategy': 'hierarchical_multi_label',
            'models_trained': 1 + len(self.type3_models) + len(self.type4_models),
            'training_samples': len(X),
            'features': X.shape[1],
            'type2_classes': len(self.type2_classes),
            'type3_classes': len(self.type3_classes),
            'type3_models_created': len(self.type3_models),
            'type4_models_created': len(self.type4_models),
            'model_distribution': {
                'type2': 1,
                'type3': len(self.type3_models),
                'type4': len(self.type4_models)
            }
        }
        
        logging.info("Hierarchical Multi-Label training completed")
        return self.results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using hierarchical approach
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary with predictions for each level
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        logging.info(f"Making hierarchical predictions for {len(X)} samples")
        
        # Step 1: Predict Type 2
        pred_type2 = self.type2_model.predict(X)
        
        # Step 2: Predict Type 3 using appropriate Type 3 model
        pred_type3 = np.zeros(len(X), dtype=object)
        for i, type2_pred in enumerate(pred_type2):
            if type2_pred in self.type3_models:
                # Use specific Type 3 model for this Type 2 class
                model_type3 = self.type3_models[type2_pred]
                if hasattr(X, 'iloc'):
                    pred_type3[i] = model_type3.predict(X.iloc[[i]])[0]
                else:
                    pred_type3[i] = model_type3.predict(X[i:i+1])[0]
            else:
                # Fallback prediction
                pred_type3[i] = "unknown"
        
        # Step 3: Predict Type 4 using appropriate Type 4 model
        pred_type4 = np.zeros(len(X), dtype=object)
        for i, (type2_pred, type3_pred) in enumerate(zip(pred_type2, pred_type3)):
            combination_key = f"{type2_pred}_{type3_pred}"
            if combination_key in self.type4_models:
                # Use specific Type 4 model for this combination
                model_type4 = self.type4_models[combination_key]
                if hasattr(X, 'iloc'):
                    pred_type4[i] = model_type4.predict(X.iloc[[i]])[0]
                else:
                    pred_type4[i] = model_type4.predict(X[i:i+1])[0]
            else:
                # Fallback prediction
                pred_type4[i] = "unknown"
        
        predictions = {
            'type2': pred_type2,
            'type3': pred_type3,
            'type4': pred_type4
        }
        
        logging.info("Hierarchical predictions completed")
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y2_test: pd.Series,
                y3_test: pd.Series, y4_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate hierarchical strategy performance
        
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
        
        logging.info("Evaluating Hierarchical Multi-Label Strategy")
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate individual accuracies
        from sklearn.metrics import accuracy_score
        
        # Type 2 accuracy
        acc_type2 = accuracy_score(y2_test, predictions['type2'])
        
        # Type 3 accuracy (only where Type 2 was predicted correctly)
        type2_correct = y2_test == predictions['type2']
        type3_mask = (predictions['type3'] != 'unknown') & type2_correct
        if type3_mask.sum() > 0:
            acc_type3 = accuracy_score(
                y3_test[type3_mask], 
                predictions['type3'][type3_mask]
            )
        else:
            acc_type3 = 0.0
        
        # Type 4 accuracy (only where Type 2+3 were predicted correctly)
        type2_3_correct = type2_correct & (predictions['type3'] != 'unknown') & (
            y3_test == predictions['type3']
        )
        type4_mask = (predictions['type4'] != 'unknown') & type2_3_correct
        if type4_mask.sum() > 0:
            acc_type4 = accuracy_score(
                y4_test[type4_mask], 
                predictions['type4'][type4_mask]
            )
        else:
            acc_type4 = 0.0
        
        # Calculate overall hierarchical accuracy
        all_correct = (
            (y2_test == predictions['type2']) &
            (predictions['type3'] != 'unknown') &
            (y3_test == predictions['type3']) &
            (predictions['type4'] != 'unknown') &
            (y4_test == predictions['type4'])
        )
        acc_hierarchical = np.mean(all_correct)
        
        evaluation_results = {
            'strategy': 'hierarchical_multi_label',
            'individual_accuracies': {
                'type2': acc_type2,
                'type3': acc_type3,
                'type4': acc_type4
            },
            'hierarchical_accuracy': acc_hierarchical,
            'test_samples': len(X_test),
            'model_coverage': {
                'type2_predictions': len(predictions['type2']),
                'type3_predictions': (predictions['type3'] != 'unknown').sum(),
                'type4_predictions': (predictions['type4'] != 'unknown').sum()
            },
            'detailed_metrics': self._calculate_detailed_metrics(
                y2_test, y3_test, y4_test, predictions
            )
        }
        
        logging.info(f"Hierarchical evaluation completed: "
                   f"Hierarchical Accuracy = {acc_hierarchical:.4f}")
        return evaluation_results
    
    def _calculate_detailed_metrics(self, y2_true: pd.Series, y3_true: pd.Series,
                                y4_true: pd.Series, predictions: Dict) -> Dict:
        """Calculate detailed evaluation metrics"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        detailed_metrics = {}
        
        # Type 2 metrics
        detailed_metrics['type2'] = {
            'classification_report': classification_report(y2_true, predictions['type2'], output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y2_true, predictions['type2']).tolist()
        }
        
        # Type 3 metrics (only valid predictions)
        type3_valid_mask = np.array(predictions['type3']) != 'unknown'
        if type3_valid_mask.sum() > 0:
            y3_true_filtered = np.array(y3_true)[type3_valid_mask]
            pred_type3_filtered = np.array(predictions['type3'])[type3_valid_mask]
            
            detailed_metrics['type3'] = {
                'classification_report': classification_report(
                    y3_true_filtered, 
                    pred_type3_filtered, 
                    output_dict=True,
                    zero_division=0
                ),
                'valid_predictions': type3_valid_mask.sum(),
                'confusion_matrix': confusion_matrix(
                    y3_true_filtered, 
                    pred_type3_filtered
                ).tolist()
            }
        else:
            detailed_metrics['type3'] = {
                'classification_report': {},
                'valid_predictions': 0,
                'confusion_matrix': []
            }
        
        # Type 4 metrics (only valid predictions)
        type4_valid_mask = predictions['type4'] != 'unknown'
        if type4_valid_mask.sum() > 0:
            detailed_metrics['type4'] = {
                'classification_report': classification_report(
                    y4_true[type4_valid_mask], 
                    predictions['type4'][type4_valid_mask], 
                    output_dict=True
                ),
                'valid_predictions': type4_valid_mask.sum(),
                'confusion_matrix': confusion_matrix(
                    y4_true[type4_valid_mask], 
                    predictions['type4'][type4_valid_mask]
                ).tolist()
            }
        else:
            detailed_metrics['type4'] = {
                'classification_report': {},
                'valid_predictions': 0,
                'confusion_matrix': []
            }
        
        return detailed_metrics
    
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return "HierarchicalMultiLabelStrategy"
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return ("Design Decision 2: Hierarchical Multi-Label Strategy. "
                "Multiple model instances with data filtering. "
                "Type 2 model -> filters data for Type 3 models. "
                "Type 3 models (one per Type 2 class) -> filters data for Type 4 models. "
                "Each level uses filtered data from previous levels.")
