"""
Chained Multi-Outputs Evaluator - Evaluation for Design Decision 1
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from sklearn.metrics import accuracy_score, classification_report


class ChainedMultiLabelEvaluator:
    """
    Evaluator for Chained Multi-Outputs Strategy
    
    Calculates accuracy for:
    1. Type 2 only
    2. Type 2 + Type 3 (both must be correct)
    3. Type 2 + Type 3 + Type 4 (all must be correct)
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_chained_performance(self, y_true_dict: Dict[str, pd.Series], 
                                      y_pred_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate chained multi-label performance
        
        Args:
            y_true_dict: Dictionary with true labels for each level
            y_pred_dict: Dictionary with predictions for each level
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        logging.info("Evaluating Chained Multi-Outputs Performance")
        
        # Extract true labels
        y2_true = y_true_dict.get('type2', pd.Series())
        y3_true = y_true_dict.get('type3', pd.Series())
        y4_true = y_true_dict.get('type4', pd.Series())
        
        # Extract predictions
        pred_type2 = y_pred_dict.get('type2', np.array([]))
        pred_type2_3 = y_pred_dict.get('type2_3', np.array([]))
        pred_type2_3_4 = y_pred_dict.get('type2_3_4', np.array([]))
        
        # Calculate individual level accuracies
        results = {}
        
        # Type 2 accuracy
        if len(y2_true) > 0 and len(pred_type2) > 0:
            acc_type2 = accuracy_score(y2_true, pred_type2)
            results['type2_accuracy'] = acc_type2
            results['type2_report'] = classification_report(y2_true, pred_type2, output_dict=True, zero_division=0)
            logging.info(f"Type 2 Accuracy: {acc_type2:.4f}")
        
        # Type 2 + Type 3 accuracy (both must be correct)
        if len(y2_true) > 0 and len(y3_true) > 0 and len(pred_type2_3) > 0:
            # Extract individual predictions from combined
            pred_type3 = self._extract_type3_from_combined(pred_type2_3, pred_type2)
            
            # Both Type 2 and Type 3 must be correct
            type2_correct = y2_true == pred_type2
            type3_correct = y3_true == pred_type3
            type2_3_correct = type2_correct & type3_correct
            
            acc_type2_3 = np.mean(type2_3_correct)
            results['type2_3_accuracy'] = acc_type2_3
            
            # Calculate Type 3 metrics
            type3_mask = type2_correct  # Only evaluate Type 3 where Type 2 is correct
            if type3_mask.sum() > 0:
                # Convert to numpy arrays to avoid pandas indexing issues
                y3_true_masked = np.array(y3_true)[type3_mask]
                pred_type3_masked = np.array(pred_type3)[type3_mask]
                
                # Filter out 'unknown' predictions
                valid_mask = pred_type3_masked != 'unknown'
                if valid_mask.sum() > 0:
                    y3_true_filtered = y3_true_masked[valid_mask]
                    pred_type3_filtered = pred_type3_masked[valid_mask]
                    
                    type3_report = classification_report(
                        y3_true_filtered, 
                        pred_type3_filtered, 
                        output_dict=True,
                        zero_division=0
                    )
                    results['type3_given_type2_accuracy'] = np.mean(
                        accuracy_score(y3_true_filtered, pred_type3_filtered)
                    )
                    results['type3_given_type2_report'] = type3_report
            
            logging.info(f"Type 2+3 Accuracy: {acc_type2_3:.4f}")
        
        # Type 2 + Type 3 + Type 4 accuracy (all must be correct)
        if len(y2_true) > 0 and len(y3_true) > 0 and len(y4_true) > 0 and len(pred_type2_3_4) > 0:
            # Extract individual predictions from combined
            pred_type3 = self._extract_type3_from_combined(pred_type2_3_4, pred_type2)
            pred_type4 = self._extract_type4_from_combined(pred_type2_3_4, pred_type2, pred_type3)
            
            # All three levels must be correct
            type2_correct = y2_true == pred_type2
            type3_correct = y3_true == pred_type3
            type4_correct = y4_true == pred_type4
            all_correct = type2_correct & type3_correct & type4_correct
            
            acc_type2_3_4 = np.mean(all_correct)
            results['type2_3_4_accuracy'] = acc_type2_3_4
            
            # Calculate Type 4 metrics
            type2_3_correct = type2_correct & type3_correct
            type4_mask = type2_3_correct  # Only evaluate Type 4 where Type 2+3 are correct
            if type4_mask.sum() > 0:
                type4_report = classification_report(
                    y4_true[type4_mask], 
                    pred_type4[type4_mask], 
                    output_dict=True
                )
                results['type4_given_type2_3_accuracy'] = np.mean(
                    accuracy_score(y4_true[type4_mask], pred_type4[type4_mask])
                )
                results['type4_given_type2_3_report'] = type4_report
            
            logging.info(f"Type 2+3+4 Accuracy: {acc_type2_3_4:.4f}")
        
        # Chain dependency analysis
        results['chain_dependency_analysis'] = self._analyze_chain_dependencies(results)
        
        # Overall summary
        results['summary'] = {
            'strategy': 'chained_multi_outputs',
            'samples_evaluated': len(y2_true),
            'chain_levels': ['type2', 'type2_3', 'type2_3_4'],
            'dependency_pattern': 'Each level depends on previous levels'
        }
        
        self.evaluation_results = results
        logging.info("Chained evaluation completed")
        return results
    
    @staticmethod
    def _extract_type3_from_combined(combined_labels: np.ndarray, 
                                  pred_type2: np.ndarray) -> np.ndarray:
        """Extract Type 3 predictions from Type 2+3 combined labels"""
        type3_predictions = []
        for i, combined_label in enumerate(combined_labels):
            parts = str(combined_label).split('_')
            if len(parts) >= 2:
                type3_predictions.append(parts[1])
            else:
                # Fallback - use Type 2 prediction
                type3_predictions.append(pred_type2[i])
        return np.array(type3_predictions)
    
    @staticmethod
    def _extract_type4_from_combined(combined_labels: np.ndarray, 
                                  pred_type2: np.ndarray, 
                                  pred_type3: np.ndarray) -> np.ndarray:
        """Extract Type 4 predictions from Type 2+3+4 combined labels"""
        type4_predictions = []
        for i, combined_label in enumerate(combined_labels):
            parts = str(combined_label).split('_')
            if len(parts) >= 3:
                type4_predictions.append(parts[2])
            else:
                # Fallback - use Type 3 prediction
                type4_predictions.append(pred_type3[i] if i < len(pred_type3) else pred_type2[i])
        return np.array(type4_predictions)
    
    @staticmethod
    def _analyze_chain_dependencies(results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how accuracy propagates through the chain"""
        dependency_analysis = {}
        
        if 'type2_accuracy' in results and 'type2_3_accuracy' in results:
            # Type 3 accuracy cannot exceed Type 2 accuracy
            type2_acc = results['type2_accuracy']
            type2_3_acc = results['type2_3_accuracy']
            
            dependency_analysis['type3_max_possible'] = type2_acc
            dependency_analysis['type3_actual'] = type2_3_acc
            dependency_analysis['type3_dependency_loss'] = type2_acc - type2_3_acc
            
            if 'type2_3_4_accuracy' in results:
                # Type 4 accuracy cannot exceed Type 2+3 accuracy
                type2_3_4_acc = results['type2_3_4_accuracy']
                
                dependency_analysis['type4_max_possible'] = type2_3_acc
                dependency_analysis['type4_actual'] = type2_3_4_acc
                dependency_analysis['type4_dependency_loss'] = type2_3_acc - type2_3_4_acc
        
        # Chain efficiency analysis
        if all(k in results for k in ['type2_accuracy', 'type2_3_accuracy', 'type2_3_4_accuracy']):
            chain_efficiency = results['type2_3_4_accuracy'] / results['type2_accuracy']
            dependency_analysis['chain_efficiency'] = chain_efficiency
            
            # Classify efficiency
            if chain_efficiency >= 0.8:
                dependency_analysis['efficiency_rating'] = 'Excellent'
            elif chain_efficiency >= 0.6:
                dependency_analysis['efficiency_rating'] = 'Good'
            elif chain_efficiency >= 0.4:
                dependency_analysis['efficiency_rating'] = 'Fair'
            else:
                dependency_analysis['efficiency_rating'] = 'Poor'
        
        return dependency_analysis
    
    @staticmethod
    def print_evaluation_summary(results: Dict[str, Any]) -> None:
        """Print comprehensive evaluation summary"""
        print(f"\nChained Multi-Outputs Evaluation Summary:")
        print(f"Strategy: {results['summary']['strategy']}")
        print(f"Samples: {results['summary']['samples_evaluated']}")
        
        if 'type2_accuracy' in results:
            print(f"\nType 2 Accuracy: {results['type2_accuracy']:.4f}")
        
        if 'type2_3_accuracy' in results:
            print(f"Type 2+3 Accuracy: {results['type2_3_accuracy']:.4f}")
        
        if 'type2_3_4_accuracy' in results:
            print(f"Type 2+3+4 Accuracy: {results['type2_3_4_accuracy']:.4f}")
        
        if 'chain_dependency_analysis' in results:
            analysis = results['chain_dependency_analysis']
            print(f"\nChain Dependency Analysis:")
            print(f"   Type 3 max possible: {analysis.get('type3_max_possible', 0):.4f}")
            print(f"   Type 3 actual: {analysis.get('type3_actual', 0):.4f}")
            print(f"   Type 3 dependency loss: {analysis.get('type3_dependency_loss', 0):.4f}")
            
            if 'type4_max_possible' in analysis:
                print(f"   Type 4 max possible: {analysis['type4_max_possible']:.4f}")
                print(f"   Type 4 actual: {analysis.get('type4_actual', 0):.4f}")
                print(f"   Type 4 dependency loss: {analysis.get('type4_dependency_loss', 0):.4f}")
            
            print(f"   Chain Efficiency: {analysis.get('chain_efficiency', 0):.4f}")
            print(f"   Efficiency Rating: {analysis.get('efficiency_rating', 'Unknown')}")
    
    def save_evaluation_results(self, filepath: str) -> None:
        """Save evaluation results to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        logging.info(f"Evaluation results saved to {filepath}")
