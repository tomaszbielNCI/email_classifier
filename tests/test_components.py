#!/usr/bin/env python3
"""
Test Individual Components
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.models.model_factory import ModelFactory
from src.core.strategies.chained_strategy import ChainedMultiLabelStrategy
from src.core.strategies.hierarchical_strategy import HierarchicalMultiLabelStrategy
from src.core.evaluation.chained_evaluator import ChainedMultiLabelEvaluator
from src.core.evaluation.hierarchical_evaluator import HierarchicalMultiLabelEvaluator
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_factory():
    """Test Model Factory"""
    logger.info("Testing Model Factory")
    
    try:
        # Test factory
        factory = ModelFactory()
        
        # Test available models
        available = factory.get_available_models()
        print("Available Models:")
        for model_type, info in available.items():
            print(f"  {model_type}: {info['description']}")
        
        # Test model creation
        rf_model = factory.create_model("random_forest", n_estimators=50)
        xgb_model = factory.create_model("xgboost", learning_rate=0.05)
        
        print(f"PASSED Model Factory test passed")
        print(f"   Created RandomForest: {rf_model.get_model_name()}")
        print(f"   Created XGBoost: {xgb_model.get_model_name()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model Factory test failed: {str(e)}")
        return False

def test_strategies():
    """Test Strategies"""
    logger.info("Testing Strategies")
    
    try:
        # Create factory
        factory = ModelFactory()
        
        # Test chained strategy
        chained = ChainedMultiLabelStrategy(factory)
        print(f"PASSED Chained Strategy: {chained.get_strategy_name()}")
        print(f"   Description: {chained.get_strategy_description()}")
        
        # Test hierarchical strategy
        hierarchical = HierarchicalMultiLabelStrategy(factory)
        print(f"PASSED Hierarchical Strategy: {hierarchical.get_strategy_name()}")
        print(f"   Description: {hierarchical.get_strategy_description()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Strategies test failed: {str(e)}")
        return False

def test_evaluators():
    """Test Evaluators"""
    logger.info("Testing Evaluators")
    
    try:
        # Create dummy data
        y2_true = pd.Series(['A', 'B', 'A', 'B', 'A'])
        y3_true = pd.Series(['X', 'Y', 'X', 'Y', 'X'])
        y4_true = pd.Series(['1', '2', '1', '2', '1'])
        
        y2_pred = np.array(['A', 'B', 'A', 'B', 'B'])
        y3_pred = np.array(['X', 'Y', 'X', 'Y', 'X'])
        y4_pred = np.array(['1', '2', '1', '2', '1'])
        
        # Test chained evaluator
        chained_eval = ChainedMultiLabelEvaluator()
        y_true_dict = {'type2': y2_true, 'type3': y3_true, 'type4': y4_true}
        y_pred_dict = {
            'type2': y2_pred,
            'type2_3': np.array(['A_X', 'B_Y', 'A_X', 'B_Y', 'B_X']),
            'type2_3_4': np.array(['A_X_1', 'B_Y_2', 'A_X_1', 'B_Y_2', 'B_X_1'])
        }
        
        chained_results = chained_eval.evaluate_chained_performance(y_true_dict, y_pred_dict)
        print(f"PASSED Chained Evaluator test passed")
        print(f"   Type 2 Accuracy: {chained_results['type2_accuracy']:.4f}")
        
        # Test hierarchical evaluator
        hierarchical_eval = HierarchicalMultiLabelEvaluator()
        y_pred_dict_hierarchical = {
            'type2': y2_pred,
            'type3': y3_pred,
            'type4': y4_pred
        }
        
        hierarchical_results = hierarchical_eval.evaluate_hierarchical_performance(y_true_dict, y_pred_dict_hierarchical)
        print(f"PASSED Hierarchical Evaluator test passed")
        print(f"   Type 2 Accuracy: {hierarchical_results['type2_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluators test failed: {str(e)}")
        return False

def test_imports():
    """Test all imports"""
    logger.info("Testing Imports")
    
    try:
        # Test core imports
        from src.core.models.base import BaseModel, MultiLabelModel
        from src.core.models.random_forest import RandomForestModel
        from src.core.models.xgboost import XGBoostModel
        from src.core.models.model_factory import ModelFactory
        
        # Test strategy imports
        from src.core.strategies.base_strategy import BaseStrategy
        from src.core.strategies.chained_strategy import ChainedMultiLabelStrategy
        from src.core.strategies.hierarchical_strategy import HierarchicalMultiLabelStrategy
        
        # Test evaluation imports
        from src.core.evaluation.chained_evaluator import ChainedMultiLabelEvaluator
        from src.core.evaluation.hierarchical_evaluator import HierarchicalMultiLabelEvaluator
        
        print(f"PASSED All imports successful")
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {str(e)}")
        return False

def main():
    """Main function to test all components"""
    logger.info("Starting Component Tests")
    
    tests = [
        ("Imports", test_imports),
        ("Model Factory", test_model_factory),
        ("Strategies", test_strategies),
        ("Evaluators", test_evaluators)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        results[test_name] = test_func()
        
        if results[test_name]:
            print(f"PASSED {test_name} test")
        else:
            print(f"FAILED {test_name} test")
    
    # Summary
    print(f"\nTest Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is ready to use.")
    else:
        print("Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
