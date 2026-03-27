"""
Hierarchical Multi-Label Evaluator - Evaluation for Design Decision 2
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from sklearn.metrics import accuracy_score, classification_report


class HierarchicalMultiLabelEvaluator:
    """Evaluator for Hierarchical Multi-Label Strategy"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_hierarchical_performance(self, y_true_dict: Dict[str, pd.Series], 
                                           y_pred_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate hierarchical multi-label performance"""
        logging.info("Evaluating Hierarchical Multi-Label Performance")
        
        # Extract labels and predictions
        y2_true = y_true_dict.get('type2', pd.Series())
        y3_true = y_true_dict.get('type3', pd.Series())
        y4_true = y_true_dict.get('type4', pd.Series())
        
        pred_type2 = y_pred_dict.get('type2', np.array([]))
        pred_type3 = y_pred_dict.get('type3', np.array([]))
        pred_type4 = y_pred_dict.get('type4', np.array([]))
        
        results = {}
        
        # Type 2 accuracy
        if len(y2_true) > 0 and len(pred_type2) > 0:
            acc_type2 = accuracy_score(y2_true, pred_type2)
            results['type2_accuracy'] = acc_type2
            results['type2_report'] = classification_report(y2_true, pred_type2, output_dict=True)
            logging.info(f"Type 2 Accuracy: {acc_type2:.4f}")
        
        # Type 3 accuracy (conditional on Type 2)
        if len(y3_true) > 0 and len(pred_type3) > 0:
            type2_correct = y2_true == pred_type2
            type3_valid = (pred_type3 != 'unknown') & type2_correct
            
            if type3_valid.sum() > 0:
                acc_type3 = accuracy_score(y3_true[type3_valid], pred_type3[type3_valid])
                results['type3_given_type2_accuracy'] = acc_type3
                results['type3_coverage'] = type3_valid.sum() / len(y2_true)
                logging.info(f"Type 3 Accuracy: {acc_type3:.4f}")
            else:
                results['type3_given_type2_accuracy'] = 0.0
                results['type3_coverage'] = 0.0
        
        # Type 4 accuracy (conditional on Type 2+3)
        if len(y4_true) > 0 and len(pred_type4) > 0:
            type2_correct = y2_true == pred_type2
            type3_correct = (y3_true == pred_type3) & (pred_type3 != 'unknown')
            type4_valid = (pred_type4 != 'unknown') & type2_correct & type3_correct
            
            if type4_valid.sum() > 0:
                acc_type4 = accuracy_score(y4_true[type4_valid], pred_type4[type4_valid])
                results['type4_given_type2_3_accuracy'] = acc_type4
                results['type4_coverage'] = type4_valid.sum() / len(y2_true)
                logging.info(f"Type 4 Accuracy: {acc_type4:.4f}")
            else:
                results['type4_given_type2_3_accuracy'] = 0.0
                results['type4_coverage'] = 0.0
        
        # Overall hierarchical accuracy
        all_correct = (
            (y2_true == pred_type2) &
            (pred_type3 != 'unknown') &
            (y3_true == pred_type3) &
            (pred_type4 != 'unknown') &
            (y4_true == pred_type4)
        )
        acc_hierarchical = np.mean(all_correct)
        results['hierarchical_accuracy'] = acc_hierarchical
        logging.info(f"Hierarchical Accuracy: {acc_hierarchical:.4f}")
        
        # Model coverage analysis
        results['model_coverage'] = {
            'type2_predictions': len(pred_type2),
            'type3_valid_predictions': (pred_type3 != 'unknown').sum(),
            'type4_valid_predictions': (pred_type4 != 'unknown').sum(),
            'type3_coverage_rate': results.get('type3_coverage', 0.0),
            'type4_coverage_rate': results.get('type4_coverage', 0.0)
        }
        
        # Summary
        results['summary'] = {
            'strategy': 'hierarchical_multi_label',
            'samples_evaluated': len(y2_true),
            'hierarchical_levels': ['type2', 'type3_given_type2', 'type4_given_type2_3'],
            'coverage_pattern': 'Each level depends on previous level predictions'
        }
        
        self.evaluation_results = results
        return results
    
    @staticmethod
    def print_evaluation_summary(results: Dict[str, Any]) -> None:
        """Print evaluation summary"""
        print(f"\nHierarchical Multi-Label Evaluation Summary:")
        print(f"Strategy: {results['summary']['strategy']}")
        print(f"Samples: {results['summary']['samples_evaluated']}")
        
        if 'type2_accuracy' in results:
            print(f"Type 2 Accuracy: {results['type2_accuracy']:.4f}")
        
        if 'type3_given_type2_accuracy' in results:
            print(f"Type 3 Accuracy: {results['type3_given_type2_accuracy']:.4f}")
            print(f"Type 3 Coverage: {results['type3_coverage']:.4f}")
        
        if 'type4_given_type2_3_accuracy' in results:
            print(f"Type 4 Accuracy: {results['type4_given_type2_3_accuracy']:.4f}")
            print(f"Type 4 Coverage: {results['type4_coverage']:.4f}")
        
        if 'hierarchical_accuracy' in results:
            print(f"Hierarchical Accuracy: {results['hierarchical_accuracy']:.4f}")
        
        if 'model_coverage' in results:
            coverage = results['model_coverage']
            print(f"Coverage Analysis:")
            print(f"   Type 3 valid predictions: {coverage['type3_valid_predictions']}")
            print(f"   Type 4 valid predictions: {coverage['type4_valid_predictions']}")
            print(f"   Type 3 coverage rate: {coverage['type3_coverage_rate']:.4f}")
            print(f"   Type 4 coverage rate: {coverage['type4_coverage_rate']:.4f}")
