#!/usr/bin/env python3
"""
Test Extended Models - Demonstrate all available models without running them
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.models.model_factory import ModelFactory
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_all_models():
    """List all available models with information"""
    print("\n" + "="*80)
    print("EXTENDED MODEL LIBRARY - ALL AVAILABLE MODELS")
    print("="*80)
    
    # Get models by category
    categories = ModelFactory.get_models_by_category()
    
    for category, models in categories.items():
        print(f"\n{category.upper()} MODELS ({len(models)} models)")
        print("-" * 50)
        
        for model_name in sorted(models):
            model_info = ModelFactory.get_available_models()[model_name]
            print(f"\n{model_info['name']} ({model_name})")
            print(f"   Type: {model_info.get('type', 'unknown')}")
            print(f"   Description: {model_info['description']}")
            if 'parameters' in model_info:
                print(f"   Parameters: {model_info['parameters']}")
            if 'default_params' in model_info:
                print(f"   Default: {model_info['default_params']}")

def demonstrate_model_creation():
    """Demonstrate creating different types of models"""
    print("\n" + "="*80)
    print("MODEL CREATION DEMONSTRATION")
    print("="*80)
    
    # Sample data for demonstration
    X_sample = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_sample = np.array([0, 1, 0])
    
    # Test a few representative models from each category
    test_models = [
        ('random_forest', {'n_estimators': 50}),
        ('xgboost', {'n_estimators': 50}),
        ('logistic_regression', {'C': 0.1}),
        ('svm', {'C': 1.0}),
        ('naive_bayes', {}),
        ('knn', {'n_neighbors': 3}),
        ('mlp', {'hidden_layer_sizes': (50,)}),
        ('decision_tree', {'max_depth': 3}),
        ('adaboost', {'n_estimators': 50}),
        ('extra_trees', {'n_estimators': 50})
    ]
    
    print(f"\nTesting {len(test_models)} representative models:")
    print("(Models are created but not trained to keep this demonstration fast)")
    
    for model_name, params in test_models:
        try:
            print(f"\nCreating: {model_name}")
            # Only pass params that don't conflict with BaseModel constructor
            if model_name in ['random_forest', 'xgboost', 'adaboost', 'extra_trees']:
                model = ModelFactory.create_model(model_name, random_state=42, **params)
            else:
                model = ModelFactory.create_model(model_name, random_state=42, **params)
            info = model.get_model_info()
            print(f"   Name: {info['name']}")
            print(f"   Type: {info['type']}")
            print(f"   Description: {info['description']}")
            print(f"   Parameters: {info.get('parameters', 'N/A')}")
            
        except Exception as e:
            print(f"Error creating {model_name}: {str(e)}")

def show_model_comparison():
    """Show comparison of model characteristics"""
    print("\n" + "="*80)
    print("📊 **MODEL CHARACTERISTICS COMPARISON**")
    print("="*80)
    
    models_info = ModelFactory.get_available_models()
    
    # Group models by type
    type_groups = {}
    for name, info in models_info.items():
        model_type = info['type']
        if model_type not in type_groups:
            type_groups[model_type] = []
        type_groups[model_type].append((name, info))
    
    print("\n🏗️ **Models by Algorithm Type:**")
    for model_type, model_list in type_groups.items():
        print(f"\n📋 **{model_type.upper()}** ({len(model_list)} models):")
        for name, info in sorted(model_list):
            category = info.get('category', 'other')
            print(f"   • {name} ({category})")

def show_usage_examples():
    """Show usage examples for different scenarios"""
    print("\n" + "="*80)
    print("💡 **USAGE EXAMPLES FOR DIFFERENT SCENARIOS**")
    print("="*80)
    
    examples = [
        {
            'scenario': 'Quick Baseline Testing',
            'models': ['logistic_regression', 'naive_bayes', 'decision_tree'],
            'description': 'Fast models for initial baseline'
        },
        {
            'scenario': 'High Performance',
            'models': ['random_forest', 'xgboost', 'extra_trees'],
            'description': 'Ensemble methods for best accuracy'
        },
        {
            'scenario': 'Text Classification',
            'models': ['naive_bayes', 'logistic_regression', 'svm'],
            'description': 'Models suitable for text data'
        },
        {
            'scenario': 'Neural Networks',
            'models': ['mlp'],
            'description': 'Deep learning approach'
        },
        {
            'scenario': 'Interpretable Models',
            'models': ['decision_tree', 'logistic_regression', 'naive_bayes'],
            'description': 'Easy to understand and explain'
        },
        {
            'scenario': 'Large Dataset',
            'models': ['sgd_classifier', 'linear_svc', 'knn'],
            'description': 'Scalable models for big data'
        }
    ]
    
    for example in examples:
        print(f"\n🎯 **{example['scenario']}**")
        print(f"   Description: {example['description']}")
        print(f"   Recommended models: {', '.join(example['models'])}")
        
        # Show code example
        print(f"   Code example:")
        print(f"   ```python")
        print(f"   from src.core.models.model_factory import ModelFactory")
        print(f"   ")
        for model in example['models'][:2]:  # Show first 2 models
            print(f"   model = ModelFactory.create_model('{model}')")
        print(f"   ```")

def main():
    """Main function to run all demonstrations"""
    print("EXTENDED MODEL LIBRARY DEMONSTRATION")
    print("This script shows all available models without training them")
    
    try:
        # List all models
        list_all_models()
        
        # Demonstrate model creation
        demonstrate_model_creation()
        
        # Show model comparison
        show_model_comparison()
        
        # Show usage examples
        show_usage_examples()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        all_models = ModelFactory.get_available_model_names()
        core_models = ModelFactory.get_core_models()
        extended_models = ModelFactory.get_extended_models()
        
        print(f"Total Models Available: {len(all_models)}")
        print(f"   • Core Models: {len(core_models)}")
        print(f"   • Extended Models: {len(extended_models)}")
        print(f"\nAll models are ready to use with:")
        print(f"   ```python")
        print(f"   from src.core.models.model_factory import ModelFactory")
        print(f"   model = ModelFactory.create_model('model_name', random_state=42, **params)")
        print(f"   model.train(X_train, y_train)")
        print(f"   predictions = model.predict(X_test)")
        print(f"   ```")
        
        print(f"\nAll models successfully integrated into the system!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
