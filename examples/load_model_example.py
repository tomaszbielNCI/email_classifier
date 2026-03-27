"""
Example: How to load and use saved model and results
"""

import joblib
import json
from pathlib import Path

def load_results(results_dir="results"):
    """Load latest results"""
    results_path = Path(results_dir)
    
    # Find latest files
    json_files = list(results_path.glob("execution_report_*.json"))
    pkl_files = list(results_path.glob("winning_model_*.pkl"))
    
    if not json_files or not pkl_files:
        print("No results found!")
        return None, None
    
    # Get latest files
    latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
    latest_pkl = max(pkl_files, key=lambda x: x.stat().st_mtime)
    
    # Load JSON report
    with open(latest_json, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # Load PKL model
    model_data = joblib.load(latest_pkl)
    
    return report, model_data

def print_summary(report, model_data):
    """Print readable summary"""
    print("=" * 60)
    print("🏆 MODEL EXECUTION SUMMARY")
    print("=" * 60)
    
    # Winner info
    winner = report['execution_summary']['winning_model']
    print(f"🥇 Winning Model: {winner['name']}")
    print(f"📊 Accuracy: {winner['accuracy']:.4f}")
    print(f"⏱️ Training Time: {winner['training_time']:.4f}s")
    print(f"🔍 Description: {winner['description']}")
    
    print(f"\n📈 Model Rankings ({report['execution_summary']['total_models_tested']} models):")
    print("-" * 60)
    
    for i, model in enumerate(report['model_rankings'], 1):
        print(f"{i}. {model['model_name']:15} | {model['val_accuracy']:.4f} | {model['training_time']:.4f}s")
    
    # Feature importance
    if report['feature_analysis']['has_feature_importance']:
        print(f"\n🎯 Top Features: {report['feature_analysis']['top_features']}")
    
    # Recommendations
    print(f"\n💡 Recommendations: {', '.join(report['recommendations'])}")
    
    print("=" * 60)

def make_predictions(model_data, X_new):
    """Make predictions with loaded model"""
    model = model_data['model']
    model_name = model_data['model_name']
    
    print(f"\n🔮 Making predictions with {model_name}...")
    
    # Make predictions
    predictions = model.predict(X_new)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_new)
        confidence = probabilities.max(axis=1)
        return predictions, confidence
    else:
        return predictions, None

if __name__ == "__main__":
    # Load latest results
    report, model_data = load_results()
    
    if report and model_data:
        # Print summary
        print_summary(report, model_data)
        
        # Example predictions (using dummy data)
        import numpy as np
        X_new = np.random.rand(5, 10)  # 5 samples, 10 features
        
        predictions, confidence = make_predictions(model_data, X_new)
        
        print(f"\n🎯 Sample Predictions: {predictions}")
        if confidence is not None:
            print(f"📊 Confidence: {confidence}")
    else:
        print("No results to load!")
