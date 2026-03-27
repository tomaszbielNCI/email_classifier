#!/usr/bin/env python3
"""
Run Multi-Label Strategies Comparison
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.strategies.chained_strategy import ChainedMultiLabelStrategy
from src.core.strategies.hierarchical_strategy import HierarchicalMultiLabelStrategy
from src.core.models.model_factory import ModelFactory
from src.core.evaluation.chained_evaluator import ChainedMultiLabelEvaluator
from src.core.evaluation.hierarchical_evaluator import HierarchicalMultiLabelEvaluator
from src.core.preprocessing.data_selector import DataSelector
from src.core.preprocessing.text_preprocessor import TextPreprocessor
from src.core.preprocessing.vectorizer import Vectorizer
from src.core.preprocessing.sampler import Sampler
from src.core.preprocessing.data_splitter import DataSplitter
import pandas as pd
import numpy as np
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data():
    """Prepare data for multi-label strategies"""
    logger.info("Preparing data for multi-label strategies")
    
    # Load and preprocess data
    data_selector = DataSelector("data/raw/AppGallery.csv")
    data = data_selector.load_data()
    data = data_selector.clean_data_types()
    data = data_selector.rename_columns()
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    data = preprocessor.preprocess_dataframe(data, "Ticket Summary", "Interaction content")
    
    # Vectorize
    vectorizer = Vectorizer()
    X = vectorizer.fit_transform_text(data, "ts")
    
    # Get labels (filter to match X shape)
    y2 = data["Type 2"].iloc[:X.shape[0]]
    y3 = data["Type 3"].iloc[:X.shape[0]]
    y4 = data["Type 4"].iloc[:X.shape[0]]
    
    # Split data
    splitter = DataSplitter()
    X_train, X_test, y2_train, y2_test = splitter.basic_split(X, y2, test_size=0.2)
    # Use same split indices for y3 and y4 to maintain consistency
    train_indices = X_train.index if hasattr(X_train, 'index') else range(len(X_train))
    test_indices = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
    
    if hasattr(y3, 'iloc'):
        y3_train = y3.iloc[train_indices]
        y3_test = y3.iloc[test_indices]
        y4_train = y4.iloc[train_indices]
        y4_test = y4.iloc[test_indices]
    else:
        y3_train = y3[train_indices]
        y3_test = y3[test_indices]
        y4_train = y4[train_indices]
        y4_test = y4[test_indices]
    
    # Sample training data
    sampler = Sampler()
    # Skip sampling due to small dataset size
    X_train_balanced = X_train
    y2_train_balanced = y2_train
    y3_train_balanced = y3_train
    y4_train_balanced = y4_train
    
    logger.info(f"Data prepared: {len(X_train_balanced)} training samples, {len(X_test)} test samples")
    
    return (X_train_balanced, X_test, y2_train_balanced, y2_test, 
            y3_train_balanced, y3_test, y4_train_balanced, y4_test)

def run_chained_strategy(data_tuple):
    """Run Chained Multi-Outputs Strategy"""
    logger.info("Running Chained Multi-Outputs Strategy")
    
    X_train, X_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = data_tuple
    
    # Create strategy
    factory = ModelFactory()
    chained_strategy = ChainedMultiLabelStrategy(factory)
    
    # Train models
    training_results = chained_strategy.train_models(X_train, y2_train, y3_train, y4_train)
    
    # Make predictions
    predictions = chained_strategy.predict(X_test)
    
    # Evaluate
    evaluator = ChainedMultiLabelEvaluator()
    y_true_dict = {'type2': y2_test.reset_index(drop=True), 'type3': y3_test.reset_index(drop=True), 'type4': y4_test.reset_index(drop=True)}
    evaluation_results = evaluator.evaluate_chained_performance(y_true_dict, predictions)
    
    # Print results
    evaluator.print_evaluation_summary(evaluation_results)
    
    return {
        'strategy': 'chained_multi_outputs',
        'training': training_results,
        'evaluation': evaluation_results,
        'predictions': {k: v.tolist() for k, v in predictions.items()}
    }

def run_hierarchical_strategy(data_tuple):
    """Run Hierarchical Multi-Label Strategy"""
    logger.info("Running Hierarchical Multi-Label Strategy")
    
    X_train, X_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = data_tuple
    
    # Create strategy
    factory = ModelFactory()
    hierarchical_strategy = HierarchicalMultiLabelStrategy(factory)
    
    # Train models
    training_results = hierarchical_strategy.train_models(X_train, y2_train, y3_train, y4_train)
    
    # Make predictions
    predictions = hierarchical_strategy.predict(X_test)
    
    # Evaluate
    evaluator = HierarchicalMultiLabelEvaluator()
    y_true_dict = {'type2': y2_test, 'type3': y3_test, 'type4': y4_test}
    evaluation_results = evaluator.evaluate_hierarchical_performance(y_true_dict, predictions)
    
    # Print results
    evaluator.print_evaluation_summary(evaluation_results)
    
    return {
        'strategy': 'hierarchical_multi_label',
        'training': training_results,
        'evaluation': evaluation_results,
        'predictions': {k: v.tolist() for k, v in predictions.items()}
    }

def compare_strategies(chained_results, hierarchical_results):
    """Compare both strategies"""
    logger.info("Comparing strategies")
    
    print("\n🏆 **STRATEGY COMPARISON:**")
    
    # Extract key metrics
    chained_acc = chained_results['evaluation'].get('type2_3_4_accuracy', 0)
    hierarchical_acc = hierarchical_results['evaluation'].get('hierarchical_accuracy', 0)
    
    print(f"\n📊 **Overall Accuracy:**")
    print(f"🔗 Chained Multi-Outputs: {chained_acc:.4f}")
    print(f"🏗️ Hierarchical Multi-Label: {hierarchical_acc:.4f}")
    
    # Determine winner
    if chained_acc > hierarchical_acc:
        winner = "Chained Multi-Outputs"
        print(f"\n🎯 **Winner: {winner}** (higher overall accuracy)")
    elif hierarchical_acc > chained_acc:
        winner = "Hierarchical Multi-Label"
        print(f"\n🎯 **Winner: {winner}** (higher overall accuracy)")
    else:
        winner = "Tie"
        print(f"\n🎯 **Result: {winner}** (equal accuracy)")
    
    # Detailed comparison
    print(f"\n📈 **Detailed Analysis:**")
    
    # Chained strategy details
    chained_eval = chained_results['evaluation']
    print(f"🔗 Chained Strategy:")
    print(f"   Type 2: {chained_eval.get('type2_accuracy', 0):.4f}")
    print(f"   Type 2+3: {chained_eval.get('type2_3_accuracy', 0):.4f}")
    print(f"   Type 2+3+4: {chained_eval.get('type2_3_4_accuracy', 0):.4f}")
    
    # Hierarchical strategy details
    hierarchical_eval = hierarchical_results['evaluation']
    print(f"🏗️ Hierarchical Strategy:")
    print(f"   Type 2: {hierarchical_eval.get('type2_accuracy', 0):.4f}")
    print(f"   Type 3 (given Type 2): {hierarchical_eval.get('type3_given_type2_accuracy', 0):.4f}")
    print(f"   Type 4 (given Type 2+3): {hierarchical_eval.get('type4_given_type2_3_accuracy', 0):.4f}")
    print(f"   Overall: {hierarchical_eval.get('hierarchical_accuracy', 0):.4f}")
    
    # Model complexity comparison
    print(f"\n🤖 **Model Complexity:**")
    chained_models = chained_results['training']['models_trained']
    hierarchical_models = hierarchical_results['training']['models_trained']
    
    print(f"🔗 Chained Strategy: {chained_models} models")
    print(f"🏗️ Hierarchical Strategy: {hierarchical_models} models")
    
    if chained_models < hierarchical_models:
        print(f"✅ Chained Strategy is simpler (fewer models)")
    elif hierarchical_models < chained_models:
        print(f"✅ Hierarchical Strategy is simpler (fewer models)")
    else:
        print(f"⚖️ Both strategies have equal complexity")

def main():
    """Main function to run strategy comparison"""
    logger.info("Starting Multi-Label Strategy Comparison")
    
    try:
        # Prepare data
        data_tuple = prepare_data()
        
        # Run chained strategy
        chained_results = run_chained_strategy(data_tuple)
        
        # Run hierarchical strategy
        hierarchical_results = run_hierarchical_strategy(data_tuple)
        
        # Compare strategies
        compare_strategies(chained_results, hierarchical_results)
        
        # Save results
        results = {
            'chained_strategy': chained_results,
            'hierarchical_strategy': hierarchical_results,
            'comparison': {
                'winner': 'chained' if chained_results['evaluation'].get('type2_3_4_accuracy', 0) > 
                         hierarchical_results['evaluation'].get('hierarchical_accuracy', 0) else 'hierarchical'
            }
        }
        
        with open("results/strategy_comparison.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Strategy comparison completed and saved to results/strategy_comparison.json")
        
    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
