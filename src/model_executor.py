"""
Model Executor - Modular execution of winning model with CatBoost
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
import time
import json
from pathlib import Path

# CatBoost import (optional - will handle import gracefully)
try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost not installed. Install with: pip install catboost")

from model_trainer import ModelTrainer


class ModelExecutor:
    """
    Modular executor for winning model with enhanced capabilities
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_trainer = ModelTrainer(random_state=random_state)
        self.winning_model = None
        self.winning_model_name = None
        self.execution_history = {}
        
    def add_catboost_to_available_models(self) -> None:
        """Add CatBoost to available models if not already present"""
        if not CATBOOST_AVAILABLE:
            logging.warning("CatBoost not available - skipping")
            return
            
        # Get current models
        models = self.model_trainer.get_available_models()
        
        # Add CatBoost if not present
        if 'catboost' not in models:
            # Extend the get_available_models method
            original_get_models = self.model_trainer.get_available_models
            
            def get_models_with_catboost():
                all_models = original_get_models()
                all_models['catboost'] = {
                    'model': catboost.CatBoostClassifier,
                    'params': {
                        'random_state': self.random_state,
                        'verbose': False,
                        'iterations': 100
                    },
                    'description': 'CatBoost - gradient boosting with categorical features'
                }
                return all_models
            
            self.model_trainer.get_available_models = get_models_with_catboost
            logging.info("CatBoost added to available models")
    
    def execute_winning_model(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        models_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute model comparison and return winning model results
        """
        
        # Add CatBoost to available models
        self.add_catboost_to_available_models()
        
        # Default models to compare (including CatBoost)
        if models_to_compare is None:
            models_to_compare = [
                'random_forest', 'gradient_boosting', 'xgboost', 
                'lightgbm', 'logistic_regression', 'catboost'
            ]
        
        logging.info(f"Starting model comparison with: {models_to_compare}")
        
        # Train all models
        start_time = time.time()
        results = self.model_trainer.train_multiple_models(
            X_train, y_train, models_to_compare, X_test, y_test
        )
        
        # Get model comparison
        comparison = self.model_trainer.get_model_comparison()
        
        # Find winning model (best accuracy)
        best_idx = comparison['val_accuracy'].idxmax()
        winning_model_info = comparison.loc[best_idx]
        
        self.winning_model_name = winning_model_info['model_name']
        self.winning_model = self.model_trainer.models[self.winning_model_name]
        
        execution_time = time.time() - start_time
        
        # Prepare results
        execution_results = {
            'execution_time': execution_time,
            'winning_model': {
                'name': self.winning_model_name,
                'accuracy': winning_model_info['val_accuracy'],
                'training_time': winning_model_info['training_time'],
                'description': self._get_model_description(self.winning_model_name)
            },
            'model_comparison': comparison.to_dict('records'),
            'detailed_results': results,
            'feature_importance': self._get_feature_importance_safe()
        }
        
        # Store execution history
        self.execution_history[time.time()] = execution_results
        
        logging.info(f"Winning model: {self.winning_model_name} with accuracy: {winning_model_info['val_accuracy']:.4f}")
        
        return execution_results
    
    def predict_with_winning_model(
        self,
        X_new: Union[pd.DataFrame, np.ndarray, List[str]],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions using the winning model
        """
        
        if self.winning_model is None:
            raise ValueError("No winning model available. Run execute_winning_model first.")
        
        # Handle different input types
        if isinstance(X_new, list):
            # Assume list of text strings - need processing
            X_processed = self._process_text_input(X_new)
        else:
            X_processed = X_new
        
        # Make predictions
        start_time = time.time()
        
        if return_probabilities and hasattr(self.winning_model, 'predict_proba'):
            predictions = self.winning_model.predict(X_processed)
            probabilities = self.winning_model.predict_proba(X_processed)
            prediction_time = time.time() - start_time
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'prediction_time': prediction_time,
                'model_used': self.winning_model_name,
                'confidence_scores': np.max(probabilities, axis=1)
            }
        else:
            predictions = self.winning_model.predict(X_processed)
            prediction_time = time.time() - start_time
            
            return {
                'predictions': predictions,
                'prediction_time': prediction_time,
                'model_used': self.winning_model_name
            }
    
    def _process_text_input(self, text_list: List[str]) -> np.ndarray:
        """
        Process text input (placeholder - should integrate with pipeline)
        """
        # This is a simplified version - in real implementation,
        # should use the full pipeline preprocessing
        logging.warning("Text processing simplified - use full pipeline for production")
        
        # For now, return dummy features
        return np.random.rand(len(text_list), 10)
    
    def _get_model_description(self, model_name: str) -> str:
        """Get model description"""
        models = self.model_trainer.get_available_models()
        if model_name in models:
            return models[model_name]['description']
        return "Unknown model"
    
    def _get_feature_importance_safe(self) -> Optional[np.ndarray]:
        """Get feature importance safely"""
        try:
            return self.model_trainer.get_feature_importance(self.winning_model_name)
        except Exception as e:
            logging.warning(f"Could not get feature importance: {e}")
            return None
    
    def save_execution_report(self, filepath: str) -> None:
        """Save execution report to file"""
        report = self.generate_execution_report()
        
        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"Execution report saved to {filepath}")
    
    def save_with_timestamp(self, base_path: str = "results") -> Tuple[str, str]:
        """Save report and model with unique timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if not exists
        results_dir = Path(base_path)
        results_dir.mkdir(exist_ok=True)
        
        # Generate unique filenames
        report_path = results_dir / f"execution_report_{timestamp}.json"
        model_path = results_dir / f"winning_model_{timestamp}.pkl"
        
        # Save files
        self.save_execution_report(str(report_path))
        self.save_winning_model(str(model_path))
        
        return str(report_path), str(model_path)
    
    def save_winning_model(self, filepath: str) -> None:
        if self.winning_model is None:
            raise ValueError("No winning model to save")
        
        model_data = {
            'model': self.winning_model,
            'model_name': self.winning_model_name,
            'random_state': self.random_state,
            'execution_history': self.execution_history
        }
        
        joblib.dump(model_data, filepath)
        logging.info(f"Winning model saved to {filepath}")
    
    def load_winning_model(self, filepath: str) -> None:
        """Load winning model"""
        model_data = joblib.load(filepath)
        
        self.winning_model = model_data['model']
        self.winning_model_name = model_data['model_name']
        self.random_state = model_data['random_state']
        self.execution_history = model_data.get('execution_history', {})
        
        logging.info(f"Winning model loaded from {filepath}")
    
    def generate_execution_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report"""
        if not self.execution_history:
            return {"error": "No execution history available"}
        
        latest_execution = max(self.execution_history.keys())
        results = self.execution_history[latest_execution]
        
        return {
            'execution_summary': {
                'total_models_tested': len(results['model_comparison']),
                'winning_model': results['winning_model'],
                'total_execution_time': results['execution_time']
            },
            'model_rankings': results['model_comparison'],
            'feature_analysis': {
                'has_feature_importance': results['feature_importance'] is not None,
                'top_features': results['feature_importance'][:5] if results['feature_importance'] is not None else None
            },
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        winning_acc = results['winning_model']['accuracy']
        
        if winning_acc > 0.95:
            recommendations.append("Excellent model performance! Consider deployment.")
        elif winning_acc > 0.85:
            recommendations.append("Good performance. Consider hyperparameter tuning.")
        else:
            recommendations.append("Moderate performance. Consider feature engineering.")
        
        if results['winning_model']['name'] == 'catboost':
            recommendations.append("CatBoost selected - excellent for categorical features.")
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, 
        n_classes=3, 
        n_features=10, 
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize executor
    executor = ModelExecutor()
    
    # Execute model comparison
    print("🚀 Starting model execution...")
    results = executor.execute_winning_model(X_train, y_train, X_test, y_test)
    
    # Display results
    print(f"\n🏆 Winning Model: {results['winning_model']['name']}")
    print(f"📊 Accuracy: {results['winning_model']['accuracy']:.4f}")
    print(f"⏱️ Training Time: {results['winning_model']['training_time']:.4f}s")
    print(f"🔍 Description: {results['winning_model']['description']}")
    
    # Generate report
    report = executor.generate_execution_report()
    print(f"\n📋 Execution Report:")
    print(f"Total Models Tested: {report['execution_summary']['total_models_tested']}")
    print(f"Recommendations: {', '.join(report['recommendations'])}")
    
    # Save report and model with timestamp
    report_path, model_path = executor.save_with_timestamp()
    print(f"\n💾 Report saved to: {report_path}")
    print(f"💾 Model saved to: {model_path}")
    
    # Test predictions
    print(f"\n🔮 Making predictions...")
    predictions = executor.predict_with_winning_model(X_test[:5])
    print(f"Predictions: {predictions['predictions']}")
    print(f"Confidence: {predictions.get('confidence_scores', 'N/A')}")
