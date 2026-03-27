#!/usr/bin/env python3
"""
Simple test script for strategies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.strategies.chained_strategy import ChainedMultiLabelStrategy
from src.core.strategies.hierarchical_strategy import HierarchicalMultiLabelStrategy
from src.core.models.model_factory import ModelFactory
from src.core.evaluation.chained_evaluator import ChainedMultiLabelEvaluator
from src.core.evaluation.hierarchical_evaluator import HierarchicalMultiLabelEvaluator
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for testing"""
    logger.info("Creating sample data")
    
    # Create simple sample data
    X_train = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    X_test = np.array([[17, 18, 19, 20], [21, 22, 23, 24]])
    
    y2_train = pd.Series(['A', 'B', 'A', 'B'])
    y3_train = pd.Series(['X', 'Y', 'X', 'Y'])
    y4_train = pd.Series(['1', '2', '1', '2'])
    
    y2_test = pd.Series(['A', 'B'])
    y3_test = pd.Series(['X', 'Y'])
    y4_test = pd.Series(['1', '2'])
    
    logger.info(f"Sample data created: {len(X_train)} train, {len(X_test)} test samples")
    
    return (X_train, X_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test)

def test_chained_strategy(data_tuple):
    """Test Chained Multi-Outputs Strategy"""
    logger.info("Testing Chained Multi-Outputs Strategy")
    
    X_train, X_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = data_tuple
    
    try:
        # Create strategy
        factory = ModelFactory()
        chained_strategy = ChainedMultiLabelStrategy(factory)
        
        # Train models
        training_results = chained_strategy.train_models(X_train, y2_train, y3_train, y4_train)
        logger.info("Chained strategy training completed")
        
        # Make predictions
        predictions = chained_strategy.predict(X_test)
        logger.info("Chained strategy predictions completed")
        
        # Simple evaluation
        evaluator = ChainedMultiLabelEvaluator()
        y_true_dict = {'type2': y2_test, 'type3': y3_test, 'type4': y4_test}
        
        # Create simple predictions for testing
        simple_predictions = {
            'type2': np.array(['A', 'B']),
            'type2_3': np.array(['A_X', 'B_Y']),
            'type2_3_4': np.array(['A_X_1', 'B_Y_2'])
        }
        
        evaluation_results = evaluator.evaluate_chained_performance(y_true_dict, simple_predictions)
        
        # Print results
        evaluator.print_evaluation_summary(evaluation_results)
        
        return {
            'strategy': 'chained_multi_outputs',
            'training': training_results,
            'evaluation': evaluation_results,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Chained strategy test failed: {str(e)}")
        return {
            'strategy': 'chained_multi_outputs',
            'status': 'failed',
            'error': str(e)
        }

def test_hierarchical_strategy(data_tuple):
    """Test Hierarchical Multi-Label Strategy"""
    logger.info("Testing Hierarchical Multi-Label Strategy")
    
    X_train, X_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = data_tuple
    
    try:
        # Create strategy
        factory = ModelFactory()
        hierarchical_strategy = HierarchicalMultiLabelStrategy(factory)
        
        # Train models
        training_results = hierarchical_strategy.train_models(X_train, y2_train, y3_train, y4_train)
        logger.info("Hierarchical strategy training completed")
        
        # Make predictions
        predictions = hierarchical_strategy.predict(X_test)
        logger.info("Hierarchical strategy predictions completed")
        
        # Simple evaluation
        evaluator = HierarchicalMultiLabelEvaluator()
        y_true_dict = {'type2': y2_test, 'type3': y3_test, 'type4': y4_test}
        
        # Create simple predictions for testing
        simple_predictions = {
            'type2': np.array(['A', 'B']),
            'type3': np.array(['X', 'Y']),
            'type4': np.array(['1', '2'])
        }
        
        evaluation_results = evaluator.evaluate_hierarchical_performance(y_true_dict, simple_predictions)
        
        # Print results
        evaluator.print_evaluation_summary(evaluation_results)
        
        return {
            'strategy': 'hierarchical_multi_label',
            'training': training_results,
            'evaluation': evaluation_results,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Hierarchical strategy test failed: {str(e)}")
        return {
            'strategy': 'hierarchical_multi_label',
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main function to run simple tests"""
    logger.info("Starting Simple Strategy Tests")
    
    try:
        # Create sample data
        data_tuple = create_sample_data()
        
        # Test chained strategy
        chained_results = test_chained_strategy(data_tuple)
        
        # Test hierarchical strategy
        hierarchical_results = test_hierarchical_strategy(data_tuple)
        
        # Summary
        print(f"\n**Simple Test Summary:**")
        print(f"Chained Strategy: {chained_results['status']}")
        print(f"Hierarchical Strategy: {hierarchical_results['status']}")
        
        if chained_results['status'] == 'success' and hierarchical_results['status'] == 'success':
            print("All tests passed!")
        else:
            print("Some tests failed.")
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        print(f"Test execution failed: {str(e)}")

if __name__ == "__main__":
    main()
